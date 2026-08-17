"""The task library — `tasks/<name>/`, one directory per problem.

Several tasks coexist; a run names the one it wants. Installing a task copies
it into the run directory rather than referencing it, so editing the library
mid-run cannot change what a running block is solving.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from pinnforge import paths
from pinnforge.task import contract


@dataclass(frozen=True)
class TaskInfo:
    name: str
    path: Path
    title: str
    train_time_s: float | None
    wall_budget_s: float | None
    data_files: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return contract.validate(self.path).ok


def _title_of(task_dir: Path) -> str:
    problem = task_dir / "problem.md"
    if not problem.is_file():
        return ""
    first = problem.read_text(encoding="utf-8").splitlines()
    return first[0].lstrip("# ").strip() if first else ""


def describe(task_dir: Path) -> TaskInfo:
    data = tuple(
        sorted(
            p.name
            for p in task_dir.iterdir()
            if p.is_file() and p.name not in contract.CONTRACT_FILES
        )
    )
    return TaskInfo(
        name=task_dir.name,
        path=task_dir,
        title=_title_of(task_dir),
        train_time_s=contract.train_time(task_dir) if (task_dir / "eval.py").is_file() else None,
        wall_budget_s=contract.wall_budget(task_dir) if (task_dir / "eval.py").is_file() else None,
        data_files=data,
    )


def list_tasks(root: Path | None = None) -> list[TaskInfo]:
    base = paths.tasks_dir(root)
    if not base.is_dir():
        return []
    out = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "problem.md").is_file():
            out.append(describe(child))
    return out


def get_task(name: str, root: Path | None = None) -> TaskInfo:
    task_dir = paths.tasks_dir(root) / name
    if not task_dir.is_dir():
        known = ", ".join(t.name for t in list_tasks(root)) or "none"
        raise KeyError(f"unknown task {name!r}. Available: {known}")
    return describe(task_dir)


def install(task: TaskInfo, run_root: Path) -> Path:
    """Copy a task package into `<run>/task/`, minus the scoring truth.

    A copy, not a link: the run must stay reproducible even if the library
    entry is edited or swapped afterwards.

    Only files are copied, never subdirectories — which is what keeps
    `private/` (the answers) and `.anchor/` (the cached b00) out of the run
    without either of them having to be named here. A task may ship as many
    public data files as its physics needs; geometry, level sets and initial
    conditions all ride along.
    """
    dest = run_root / "task"
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(task.path.iterdir()):
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    return dest


def runs_of(name: str, root: Path | None = None) -> list[Path]:
    """Runs of this task, newest first — the library's other copy of it."""
    runs = (root or paths.project_root()) / "runs"
    return sorted(
        (d for d in runs.glob(f"{name}_*") if (d / "task" / "problem.md").is_file()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def recover_from_run(name: str, root: Path | None = None) -> tuple[Path, Path] | None:
    """Rebuild `tasks/<name>/` from the newest run of it.

    A run froze the task it was given, so re-running a task the library has
    lost costs a copy rather than an authoring session — and, more to the
    point, gives back the *same* definition, which is what makes the new runs
    comparable with the old ones.

    What comes back is the public half only. `private/` — the answers — is
    never copied into a run, so it cannot be recovered from one; that is the
    split working, not a gap. The caller has to say so.
    """
    runs = runs_of(name, root)
    if not runs:
        return None
    src = runs[0] / "task"
    dest = (root or paths.project_root()) / "tasks" / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return dest, runs[0]


def fingerprint(task_dir: Path) -> dict[str, str]:
    """md5 of each contract file — what a cached b00 anchor is keyed on.

    Data files are deliberately excluded: they are large and are covered by
    `eval.py`'s own reading of them, whereas a changed `baseline.py` means
    cached params came from other code and a changed `eval.py` means the
    cached score is on another metric.
    """
    out = {}
    for name in contract.CONTRACT_FILES:
        p = task_dir / name
        if p.is_file():
            out[name] = hashlib.md5(p.read_bytes()).hexdigest()
    return out
