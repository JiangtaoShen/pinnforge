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
from pinnforge.task.view import VIEW_FILE

# The shared evaluation machinery, and the name it is vendored under in a run.
KIT_SOURCE = Path(__file__).resolve().parent / "evalkit.py"
KIT_FILE = contract.KIT_MODULE + ".py"
FINGERPRINT_SCHEMA = 2


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
    # The shared evaluation machinery travels with the run, frozen and hashed
    # like the rest of `task/`, so upgrading the tool cannot change how an
    # existing run evaluates. Harmless for a task that does not use it.
    shutil.copy2(KIT_SOURCE, dest / KIT_FILE)
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
        # the vendored machinery belongs to the run, not to the package
        if f.is_file() and f.name != KIT_FILE:
            shutil.copy2(f, dest / f.name)
    return dest, runs[0]


def fingerprint_v1(task_dir: Path) -> dict[str, str]:
    """md5 of each contract file — how anchors were keyed before schema 2.

    Kept so an anchor cached under it is still recognised: a task whose three
    files have not changed has not changed, whichever scheme measured it.
    """
    out = {}
    for name in contract.CONTRACT_FILES:
        p = task_dir / name
        if p.is_file():
            out[name] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(task_dir: Path) -> dict:
    """What a cached b00 anchor is keyed on: everything that decides its score.

    * `baseline.py` — the params came from this code;
    * `eval.py`, and the shared machinery when the task uses it — the score
      is on this metric, produced by this evaluation;
    * the public data files — the grid it was scored on, the geometry and
      initial conditions it trained on.

    `problem.md` is left out: it is prose a block reads, and b00 is measured
    without an agent, so rewording it cannot move the number. Keying on it
    used to mean a typo fix cost a GPU re-measurement.
    """
    if not (task_dir / "eval.py").is_file():
        return {}
    files = {}
    for name in ("baseline.py", "eval.py"):
        p = task_dir / name
        if p.is_file():
            files[name] = file_md5(p)
    if contract.KIT_MODULE in (task_dir / "eval.py").read_text(encoding="utf-8"):
        vendored = task_dir / KIT_FILE
        files[KIT_FILE] = file_md5(vendored if vendored.is_file() else KIT_SOURCE)
    # view.json decides what the dashboard draws, not what anything scores
    data = {
        p.name: file_md5(p)
        for p in sorted(task_dir.iterdir())
        if p.is_file() and p.name not in contract.CONTRACT_FILES
        and p.name not in (KIT_FILE, VIEW_FILE)
    }
    return {"schema": FINGERPRINT_SCHEMA, "files": files, "data": data}


def same_package(recorded: dict | None, task_dir: Path) -> bool:
    """Does a recorded fingerprint describe the package at `task_dir` today?"""
    if not recorded:
        return False
    if "schema" not in recorded:
        return recorded == fingerprint_v1(task_dir)
    return recorded == fingerprint(task_dir)


def fingerprint_changes(recorded: dict | None, task_dir: Path) -> list[str]:
    """Which parts of the package differ from a recorded fingerprint."""
    if not recorded:
        return ["no fingerprint recorded"]
    if "schema" not in recorded:
        now = fingerprint_v1(task_dir)
        return [f for f in sorted(set(recorded) | set(now)) if recorded.get(f) != now.get(f)]
    now = fingerprint(task_dir)
    out = []
    for part in ("files", "data"):
        a, b = recorded.get(part) or {}, now.get(part) or {}
        out += [f for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
    return out
