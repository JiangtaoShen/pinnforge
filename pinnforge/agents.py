"""Non-block agents: authoring a task, building its reference, distilling papers.

These run on the same runtime adapters as blocks, so whichever harness the
operator has is the one that does the work. They differ from a block in
scope only: no GPU budget, no `evals.jsonl`, no summary contract — they
produce files that the framework then checks.

Every one of them is started with `PINNFORGE_ROLE` set, so a harness plugin
installed for the operator stays silent inside them, as it does in blocks.

An agent runs in a process group of its own, so a signal to `pinnforge`
never reaches it. Whatever ends a session — the agent finishing, its time
limit, Ctrl-C, SIGTERM from `tmux kill-session` — the group is stopped and
confirmed gone before `_drive` returns or raises: an agent that outlived its
session would go on writing into a directory the framework is about to hand
to someone else (for the author, the one the answers are unsealed back into).
Its process group is recorded on disk while it runs, so a later session can
refuse to start while an orphan from a crashed one is still alive.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from pinnforge import oscompat, paths, private
from pinnforge.config import AgentConfig
from pinnforge.run import lifecycle
from pinnforge.runtime import registry as runtime_registry
from pinnforge.runtime.base import clean_env

PROMPTS = paths.prompts_dir()


@dataclass
class AgentRun:
    ok: bool
    log_path: Path
    duration_s: float
    returncode: int | None
    timed_out: bool = False


def _path(p) -> str:
    """A local path as a prompt names it: forward slashes, which every shell an
    agent may type into — bash (Git Bash on Windows), PowerShell, cmd — reads
    the same way, where a backslash is an escape to bash."""
    return Path(p).as_posix()


def _render(name: str, **fields: str) -> str:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text


# The process group of the agent running right now, if any — what a second
# stop signal kills outright (`kill_live`).
_LIVE: dict = {"pgid": None}


def live_pgid() -> int | None:
    return _LIVE["pgid"]


def group_alive(pgid) -> bool:
    """Does process group `pgid` have a member left?"""
    try:
        pgid = int(pgid)
    except (TypeError, ValueError):
        return False
    if pgid <= 0 or pgid == oscompat.own_pgid():
        return False
    return oscompat.group_alive(pgid)


def end_group(pgid, grace: float = 10.0) -> bool:
    """TERM what is left of group `pgid`, KILL it after `grace`; True once it is gone."""
    for sig, wait in ((signal.SIGTERM, grace), (oscompat.SIGKILL, 5.0)):
        if not group_alive(pgid):
            return True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            oscompat.killpg(int(pgid), sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            if not group_alive(pgid):
                return True
            time.sleep(0.05)
    return not group_alive(pgid)


def kill_live() -> None:
    """The second stop signal: the running agent's group, killed outright."""
    pgid = _LIVE["pgid"]
    if group_alive(pgid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            oscompat.killpg(int(pgid), oscompat.SIGKILL)


def read_record(path: Path) -> dict:
    try:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _drive(
    prompt: str,
    *,
    cwd: Path,
    agent: AgentConfig,
    log_path: Path,
    label: str,
    env: dict[str, str] | None = None,
    role: str = "author",
    timeout_s: float | None = None,
    record: Path | None = None,
    on_start=None,
) -> AgentRun:
    """Run one agent session to its end, and make sure nothing of it is left running.

    `record`, if given, holds the agent's process group while it runs; it is
    removed once the group is confirmed gone, so a record that survives says
    an agent may still be alive. `on_start(pgid)`, if given, is told the group
    as soon as it exists — a run's lock records it there, so a driver that
    dies outright leaves the next one a group it can stop.
    """
    runtime = runtime_registry.get_runtime(agent.runtime)
    started = time.time()
    handle = None
    pgid = None
    timed_out = False
    rc = None
    try:
        # Between starting the agent and being ready to stop it, a stop signal
        # waits: torn in half here it would leave a group nobody knows about.
        with lifecycle.critical():
            handle = runtime.start(
                block_id=label,
                cwd=cwd,
                prompt=prompt,
                model=runtime_registry.resolve_model(runtime, agent.model),
                log_path=log_path,
                env=clean_env({**(env or {}), "PINNFORGE_ROLE": role}),
                command=agent.command or None,
                runtime_options=agent.runtime_options,
                max_turns=agent.max_turns,
            )
            pgid = handle._pgid()
            _LIVE["pgid"] = pgid
            if record is not None:
                Path(record).parent.mkdir(parents=True, exist_ok=True)
                Path(record).write_text(json.dumps({
                    "pgid": pgid, "pid": getattr(handle.process, "pid", None),
                    "label": label, "cwd": str(cwd), "driver": os.getpid(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }) + "\n", encoding="utf-8")
            if on_start is not None:
                on_start(pgid)
        rc = handle.wait(timeout=timeout_s)
        if rc is None and timeout_s is not None:
            # Past its time limit: stopped the way a Ctrl-C would stop it, so
            # a harness that checkpoints can leave its work in a sane state.
            timed_out = True
            handle.interrupt()
            rc = handle.returncode
    except BaseException:
        # Same reason as the block loop: the agent has its own session, so a
        # Ctrl-C here would leave it running and still writing files.
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.interrupt()
        raise
    finally:
        if handle is not None:
            with lifecycle.critical():
                # The CLI exiting proves nothing about the shells and solvers
                # under it: the group is what has to be gone.
                gone = end_group(pgid) if pgid is not None else True
                handle.close()
                if gone:
                    _LIVE["pgid"] = None
                    if record is not None:
                        Path(record).unlink(missing_ok=True)
    return AgentRun(
        ok=(rc == 0 and not timed_out),
        log_path=log_path,
        duration_s=time.time() - started,
        returncode=rc,
        timed_out=timed_out,
    )


def author_task(
    name: str,
    *,
    intake: str,
    sources: list[Path],
    mode: str,
    agent: AgentConfig,
    template: Path,
    log_path: Path,
    root: Path | None = None,
    timeout_s: float | None = None,
    record: Path | None = None,
) -> AgentRun:
    """Write `tasks/<name>/`'s public half from the intake, adapting `template`."""
    root = root or paths.project_root()
    dest = paths.tasks_dir(root) / name
    dest.mkdir(parents=True, exist_ok=True)
    prompt = _render(
        "task_author.md",
        dest=_path(dest),
        intake=intake.strip(),
        sources=", ".join(str(s) for s in sources) or "none — the description above is all",
        mode=mode.strip(),
        template_task=_path(template),
        validate_cmd=f"pinnforge task validate {name} --no-reference",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return _drive(
        prompt,
        cwd=dest,
        agent=agent,
        log_path=log_path,
        label=f"task-author:{name}",
        env={"PINNFORGE_ROOT": str(root)},
        role="author",
        timeout_s=timeout_s,
        record=record,
    )


def build_reference(
    name: str,
    *,
    agent: AgentConfig,
    python: Path,
    uv: str,
    gpu: str,
    tol: float,
    budget_s: float,
    log_path: Path,
    root: Path | None = None,
    record: Path | None = None,
) -> AgentRun:
    """Compute the task's `ref_values.npy`, in its private directory, with a classical solver."""
    root = root or paths.project_root()
    task_dir = paths.tasks_dir(root) / name
    home = private.locate(task_dir)  # the configured private directory, if there is one
    workdir = home / "reference"
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = _render(
        "reference_builder.md",
        workdir=_path(workdir),
        task_dir=_path(task_dir),
        private_dir=_path(home),
        python=_path(python),
        uv=_path(uv),
        gpu=gpu,
        budget_min=f"{budget_s / 60:.0f}",
        tol=f"{tol:g}",
        check_cmd=f"pinnforge task reference {name} --check",
    )
    return _drive(
        prompt,
        cwd=workdir,
        agent=agent,
        log_path=log_path,
        label=f"reference:{name}",
        env={"PINNFORGE_ROOT": str(root), "CUDA_VISIBLE_DEVICES": gpu},
        role="reference",
        timeout_s=budget_s,
        record=record,
    )


def ingest_reference(
    name: str,
    *,
    intake: str,
    user_data: Path,
    agent: AgentConfig,
    python: Path,
    uv: str,
    budget_s: float,
    log_path: Path,
    root: Path | None = None,
    record: Path | None = None,
) -> AgentRun:
    """Turn the operator's data into `eval_grid.csv` + the private `ref_values.npy`."""
    root = root or paths.project_root()
    task_dir = paths.tasks_dir(root) / name
    home = private.locate(task_dir)  # the configured private directory, if there is one
    workdir = home / "reference"
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = _render(
        "reference_ingest.md",
        intake=intake.strip(),
        user_data=str(user_data),
        python=_path(python),
        uv=_path(uv),
        workdir=_path(workdir),
        task_dir=_path(task_dir),
        private_dir=_path(home),
        budget_min=f"{budget_s / 60:.0f}",
    )
    return _drive(
        prompt,
        cwd=workdir,
        agent=agent,
        log_path=log_path,
        label=f"reference:{name}",
        env={"PINNFORGE_ROOT": str(root)},
        role="reference",
        timeout_s=budget_s,
        record=record,
    )


def distill_kb1(
    sources: Path,
    instruction: str,
    *,
    agent: AgentConfig,
    root: Path | None = None,
) -> AgentRun:
    """Turn papers into kb1 nodes and index them.

    Into the workspace's own `kb1/`: the corpus that ships with the tool is
    copied there first, so the result is the whole corpus plus the new
    nodes — never an edit to an installed package.
    """
    root = root or paths.project_root()
    kb1 = seed_corpus(root)
    prompt = _render(
        "kb1_distill.md",
        kb1=_path(kb1),
        sources=_path(sources),
        instruction=instruction.strip() or "Distil every paper in the sources.",
    )
    log = root / "logs" / f"kb1-distill.{int(time.time())}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return _drive(
        prompt,
        cwd=root,
        agent=agent,
        log_path=log,
        label="kb1-distill",
        env={"PINNFORGE_ROOT": str(root)},
        role="kb1",
    )


def seed_corpus(root: Path) -> Path:
    """The workspace's `kb1/`, created as a copy of the packaged corpus if absent."""
    import shutil

    override = Path(root) / "kb1"
    if not override.is_dir():
        packaged = Path(paths.__file__).resolve().parent / "kb1"
        if packaged.is_dir():
            shutil.copytree(packaged, override)
        else:
            override.mkdir(parents=True)
    return override
