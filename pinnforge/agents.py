"""Non-block agents: authoring a task, and distilling papers into kb1.

These run on the same runtime adapters as blocks, so whichever harness the
operator has is the one that does the work. They differ from a block in
scope only: no GPU budget, no `evals.jsonl`, no summary contract — they
produce files that the validator then checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pinnforge import paths
from pinnforge.config import AgentConfig
from pinnforge.runtime import registry as runtime_registry
from pinnforge.runtime.base import clean_env

PROMPTS = paths.prompts_dir()


@dataclass
class AgentRun:
    ok: bool
    log_path: Path
    duration_s: float
    returncode: int | None


def _render(name: str, **fields: str) -> str:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text


def _drive(
    prompt: str,
    *,
    cwd: Path,
    agent: AgentConfig,
    log_path: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> AgentRun:
    runtime = runtime_registry.get_runtime(agent.runtime)
    started = time.time()
    handle = runtime.start(
        block_id=label,
        cwd=cwd,
        prompt=prompt,
        model=agent.model or runtime.default_model,
        log_path=log_path,
        env=clean_env(env or {}),
        command=agent.command or None,
        runtime_options=agent.runtime_options,
    )
    try:
        rc = handle.wait()
    except BaseException:
        # Same reason as the block loop: the agent has its own session, so a
        # Ctrl-C here would leave it running and still writing files.
        handle.interrupt()
        raise
    handle.close()
    return AgentRun(
        ok=(rc == 0), log_path=log_path, duration_s=time.time() - started, returncode=rc
    )


def author_task(
    name: str,
    sources: Path,
    instruction: str,
    *,
    agent: AgentConfig,
    template_task: str = "ldc",
    root: Path | None = None,
) -> AgentRun:
    """Build `tasks/<name>/` from reference material plus a prompt."""
    root = root or paths.project_root()
    dest = paths.tasks_dir(root) / name
    dest.mkdir(parents=True, exist_ok=True)
    template = paths.tasks_dir(root) / template_task
    prompt = _render(
        "task_author.md",
        dest=str(dest),
        sources=str(sources),
        instruction=instruction.strip() or "Build the task package from the sources.",
        template_task=str(template),
        validate_cmd=f"pinnforge task validate {name}",
        smoke_cmd=f"pinnforge task smoke {name}",
    )
    log = paths.project_root(root) / "logs" / f"task-author.{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return _drive(
        prompt,
        cwd=root,
        agent=agent,
        log_path=log,
        label=f"task-author:{name}",
        env={"PINNFORGE_ROOT": str(root)},
    )


def distill_kb1(
    sources: Path,
    instruction: str,
    *,
    agent: AgentConfig,
    root: Path | None = None,
) -> AgentRun:
    """Turn papers into kb1 nodes and index them."""
    root = root or paths.project_root()
    kb1 = paths.kb1_dir(root)
    kb1.mkdir(parents=True, exist_ok=True)
    prompt = _render(
        "kb1_distill.md",
        kb1=str(kb1),
        sources=str(sources),
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
    )
