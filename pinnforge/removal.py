"""Deleting a run or a task: everything it owns, and nothing while it is in use.

`pinnforge run delete` and `pinnforge task delete` first build a plan (`run_plan`,
`task_plan`): what will be removed, how large it is, and what stands in the
way. Only once the operator has seen the plan does `carry_out` remove it.

What a run owns: its folder under `runs/`, and its score ledger, which lives
outside it. The ledger has to go too, because run ids are reused: the next
run of the task takes the highest number left, so a ledger that stayed would
charge the old run's scores against the new run's cap. With a private
directory configured, all runs share one ledger, so only this run's lines
are taken out of it.

What a task owns: its folder under `tasks/` (the public package, its
reference answers in `private/`, its anchor, its authoring record). It also
owns its answers wherever else they are kept: in `.pinnforge/sealed/` while
an author session holds them away, or in the configured private directory.

Nothing is deleted while it is in use: a run with a live driver or score
service, or a task being authored, or one that a live run is scoring against.
Links are removed, never followed. A run's `.venv` is a junction to the
workspace's environment, and deleting through it would empty the workspace.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import oscompat, paths, private
from pinnforge.run import lifecycle

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
AGENT_RECORD = "agent.json"  # task_new.py: an author session's process group


@dataclass
class Plan:
    what: str                                     # "run demo_5", "task demo"
    paths: list[Path] = field(default_factory=list)
    ledger_lines: tuple[Path, str] | None = None  # a shared ledger, and the queue to take out
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    runs: list[Plan] = field(default_factory=list)  # a task's runs, deleted with it on request


def size(path: Path) -> int:
    """Bytes under `path`, not counting what its links point at."""
    if oscompat.is_link(path):
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not oscompat.is_link(Path(dirpath) / d)]
        for f in filenames:
            p = Path(dirpath) / f
            if not oscompat.is_link(p):
                with contextlib.suppress(OSError):
                    total += p.stat().st_size
    return total


def _owned(parent: Path, name: str, kind: str) -> Path:
    """`parent/name`, checked: a plain name, an existing directory, not a link."""
    if not NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a {kind} name")
    path = parent / name
    if oscompat.is_link(path):
        raise ValueError(f"{path} is a link; remove it by hand if that is what you mean")
    if not path.is_dir():
        raise FileNotFoundError(f"no {kind} {name!r} in {parent}")
    return path


# ─────────────────────────────── runs ───────────────────────────────


def run_plan(root: Path, run_id: str) -> Plan:
    from pinnforge import scoring

    root = Path(root)
    run = _owned(paths.runs_dir(root), run_id, "run")
    plan = Plan(f"run {run_id}", [run])
    if (who := lifecycle.holder(run)) is not None:
        pid = who.get("pid")
        plan.blockers.append(f"{run_id} is running (driver pid {pid}); stop it first: "
                             f"pinnforge run stop {run_id}")
    if scoring.standalone_serving(run):
        plan.blockers.append(f"a score service is serving {run_id}; stop it first")
    private_dir = private.configured(root)
    ledger = scoring.ledger_path(root, run, private_dir)
    if private_dir is not None:
        if ledger.is_file():
            plan.ledger_lines = (ledger, run_id)
    elif ledger.is_file() and ledger.parent != run / scoring.QUEUE:
        plan.paths.append(ledger)
    return plan


# ─────────────────────────────── tasks ───────────────────────────────


def _runs_of(root: Path, name: str) -> list[str]:
    from pinnforge import scoring

    return [r for r in paths.list_runs(root, task=name)
            if scoring.run_task(paths.runs_dir(root) / r) == name]


def task_plan(root: Path, name: str, with_runs: bool = False) -> Plan:
    from pinnforge import agents
    from pinnforge.task import examples

    root = Path(root)
    task = _owned(paths.tasks_dir(root), name, "task")
    plan = Plan(f"task {name}", [task])
    sealed = root / paths.STATE_DIR / "sealed" / name
    if sealed.exists() and not oscompat.is_link(sealed):
        plan.paths.append(sealed)
    lock = root / paths.STATE_DIR / "locks" / f"anchor-{name}.lock"
    if lock.is_file():
        plan.paths.append(lock)
    private_dir = private.configured(root)
    if private_dir is not None:
        outside = Path(private_dir) / name
        if outside.is_dir() and not oscompat.is_link(outside):
            plan.paths.append(outside)
            plan.notes.append(f"its answers in the private directory go too: {outside}")

    # in use: being authored, by a pipeline or an agent it left running
    authoring = task / ".authoring" / "lock"
    if authoring.is_file() and _locked(authoring):
        plan.blockers.append(f"{name} is being authored; let it finish or stop it first")
    for rec in (sealed / "private" / "logs" / AGENT_RECORD,
                private.locate(task) / "logs" / AGENT_RECORD):
        info = agents.read_record(rec)
        if info and agents.group_alive(info.get("pgid")):
            plan.blockers.append(f"an author agent of {name} is still running "
                                 f"(process group {info.get('pgid')}); stop it first")

    runs = _runs_of(root, name)
    live = [r for r in runs if lifecycle.holder(paths.runs_dir(root) / r) is not None]
    for r in live:
        plan.blockers.append(f"run {r} is scoring against {name}'s answers; stop it first: "
                             f"pinnforge run stop {r}")
    if with_runs:
        plan.runs = [run_plan(root, r) for r in runs if r not in live]
    elif runs:
        plan.notes.append(
            f"{len(runs)} run(s) were made from it ({', '.join(runs)}): they keep their own "
            "copy of the task, but can no longer be resumed or scored. Delete them too with "
            "--with-runs.")
    if name == examples.DEMO or any(e.get("name") == name for e in examples.catalog()):
        plan.notes.append(f"a packaged example: `pinnforge examples add {name}` brings it back")
    return plan


def _locked(path: Path) -> bool:
    """Is some process holding the lock on `path`?"""
    try:
        fd = oscompat.open_file(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        oscompat.lock(fd, block=False)
    except BlockingIOError:
        return True
    except OSError:
        return False
    else:
        oscompat.unlock(fd)
        return False
    finally:
        os.close(fd)


# ─────────────────────────────── removing ───────────────────────────────


def _writable_retry(func, path, _exc) -> None:
    """A read-only file (Windows refuses those) made writable, and removed again."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _unlink_link(path: Path) -> None:
    """Remove a link itself: a junction or a directory symlink by rmdir, else unlink."""
    try:
        os.rmdir(path)
    except OSError:
        os.unlink(path)


def remove_tree(path: Path) -> None:
    """Delete `path` and all under it, removing every link found without following it."""
    path = Path(path)
    if oscompat.is_link(path):
        _unlink_link(path)
        return
    if path.is_file():
        try:
            path.unlink()
        except PermissionError:
            _writable_retry(os.unlink, path, None)
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=True):
        for d in list(dirnames):
            if oscompat.is_link(Path(dirpath) / d):
                _unlink_link(Path(dirpath) / d)  # e.g. a run's .venv junction
                dirnames.remove(d)
        for f in filenames:
            if oscompat.is_link(Path(dirpath) / f):
                os.unlink(Path(dirpath) / f)
    kw = {"onexc": _writable_retry} if sys.version_info >= (3, 12) else {"onerror": _writable_retry}
    shutil.rmtree(path, **kw)


def _drop_queue(ledger: Path, queue: str) -> None:
    """Take one run's lines out of a ledger every run shares, in place and whole."""
    kept = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(line).get("queue") == queue:
                continue
        except ValueError:
            pass
        kept.append(line)
    tmp = ledger.with_name(f".{ledger.name}.tmp")
    tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    oscompat.replace(tmp, ledger)


def carry_out(plan: Plan) -> list[str]:
    """Remove what `plan` lists, its runs first; what could not be removed, said."""
    if plan.blockers:
        raise RuntimeError("; ".join(plan.blockers))
    errors: list[str] = []
    for sub in plan.runs:
        errors += carry_out(sub)
    for path in plan.paths:
        try:
            remove_tree(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    if plan.ledger_lines is not None:
        try:
            _drop_queue(*plan.ledger_lines)
        except OSError as e:
            errors.append(f"{plan.ledger_lines[0]}: {e}")
    return errors
