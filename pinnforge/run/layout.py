"""Building a run directory.

A run root is a self-contained working directory that *looks like* the
single-task project the block charter was written against:

    runs/ks_1/
      block.md            rendered charter
      run.yaml            frozen config
      state.json          orchestrator checkpoint
      task/               copy of tasks/ks/ — frozen for the run's lifetime
      kb1/  -> ../../kb1  shared corpus, read-only to blocks
      .venv -> ../../.venv
      blocks/bNN/         per-block workspaces
      blocks/kb2/         block summaries
      logs/               agent transcripts

Blocks are started with this as their working directory, so `task/eval.py`,
`kb1/INDEX.md` and `blocks/kb2/bNN.md` all resolve without the charter
knowing anything about the multi-task layout above it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pinnforge import charter, paths
from pinnforge.config import DEFAULT_WALL_BUDGET, RunConfig
from pinnforge.task import contract, registry
from pinnforge.types import RunState, utcnow


def _link(target: Path, link: Path) -> None:
    """Point `link` at `target`, replacing a stale link if present."""
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        if link.is_symlink():
            link.unlink()
        else:
            return  # a real directory was put there deliberately; leave it
    link.symlink_to(target, target_is_directory=True)


def create_run(cfg: RunConfig, root: Path | None = None, run_id: str | None = None) -> Path:
    """Materialise a new run directory and return its path."""
    root = root or paths.project_root()
    task = registry.get_task(cfg.task, root)

    report = contract.validate(task.path)
    if not report.ok:
        names = ", ".join(name for name, ok, _ in report.failures)
        raise ValueError(f"task {cfg.task!r} fails the contract: {names}")

    # A task's eval.py declares the per-block budget it was tuned for. Adopt
    # it unless the caller asked for something else — the charter quotes this
    # number, so getting it from anywhere but the task is how the two drift.
    if cfg.sandbox.wall_budget_s == DEFAULT_WALL_BUDGET:
        declared = contract.wall_budget(task.path)
        if declared:
            cfg.sandbox.wall_budget_s = declared

    run_id = run_id or paths.next_run_id(cfg.task, root)
    run = paths.run_root(run_id, root)
    if run.exists():
        raise FileExistsError(f"run directory already exists: {run}")

    run.mkdir(parents=True)
    paths.blocks_dir(run).mkdir()
    paths.kb2_dir(run).mkdir()
    paths.logs_dir(run).mkdir()

    registry.install(task, run)
    _link(paths.kb1_dir(root), run / "kb1")
    venv = root / ".venv"
    if venv.is_dir():
        _link(venv, run / ".venv")
    # The charter tells a block it may extend the stack with `uv add` "if the
    # pins stay intact" — a sentence that needs the pins to be reachable from
    # the working directory it says it in. They are linked, not copied, for the
    # same reason `.venv` is: there is one environment, and a run that edited a
    # private copy of the manifest would be describing packages nobody
    # installed.
    for manifest in ("pyproject.toml", "uv.lock"):
        src = root / manifest
        if src.is_file():
            _link(src, run / manifest)

    charter.write_charter(run, cfg, root)
    cfg.save(paths.config_path(run))

    state = RunState(task=cfg.task, run_id=run_id, created_at=utcnow())
    state.save(paths.state_path(run))
    return run


def load_run(run_id: str, root: Path | None = None) -> tuple[Path, RunConfig, RunState]:
    run = paths.run_root(run_id, root)
    if not paths.config_path(run).is_file():
        raise FileNotFoundError(f"no run at {run}")
    return run, RunConfig.load(paths.config_path(run)), RunState.load(paths.state_path(run))


def existing_block_ids(run: Path) -> list[str]:
    """`bNN` directories present, sorted. `kb2` and bookkeeping files excluded."""
    blocks = paths.blocks_dir(run)
    if not blocks.is_dir():
        return []
    out = [
        p.name
        for p in blocks.iterdir()
        if p.is_dir() and p.name != "kb2" and p.name.startswith("b") and p.name[1:].isdigit()
    ]
    return sorted(out)


def next_block_id(run: Path) -> str:
    ids = existing_block_ids(run)
    highest = max((int(b[1:]) for b in ids), default=-1)
    return f"b{highest + 1:02d}"
