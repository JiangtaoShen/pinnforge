"""b00 — the control node every later block is measured against.

b00 is not a subagent product: it is `task/baseline.py` copied verbatim and
evaluated as-is. Measuring it costs a full GPU run, and the number only
depends on the task package, so it is cached per task and restored on later
runs — keyed on the md5s of the three contract files, because a different
`baseline.py` means cached params came from other code and a different
`eval.py` means the cached score is on another metric.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pinnforge import paths
from pinnforge.task import registry
from pinnforge.types import EvalRecord, best_score

ANCHOR_FILES = ("b00_v01.py", "b00_v01.pkl", "evals.jsonl", ".budget")


@dataclass
class AnchorResult:
    rRMSE: float | None
    restored: bool
    detail: str = ""


def cache_dir(task: str, root: Path | None = None) -> Path:
    return (root or paths.project_root()) / "tasks" / task / ".anchor"


def save(run: Path, task: str, root: Path | None = None) -> Path:
    """Cache the run's measured b00 so the next run of this task is free."""
    dest = cache_dir(task, root)
    dest.mkdir(parents=True, exist_ok=True)
    b00 = paths.block_dir(run, "b00")
    for name in ANCHOR_FILES:
        src = b00 / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    summary = paths.summary_path(run, "b00")
    if summary.is_file():
        shutil.copy2(summary, dest / "b00.md")
    recs = EvalRecord.read_all(b00 / "evals.jsonl")
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "task": task,
                "rRMSE": best_score(recs),
                "fingerprint": registry.fingerprint(run / "task"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dest


def restore(run: Path, task: str, root: Path | None = None) -> AnchorResult | None:
    """Install a cached b00 if it matches this task package, else None."""
    src = cache_dir(task, root)
    meta_path = src / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != registry.fingerprint(run / "task"):
        return AnchorResult(
            None, False, "cached anchor was measured against different contract files"
        )
    dest = paths.block_dir(run, "b00")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ANCHOR_FILES:
        f = src / name
        if f.is_file():
            shutil.copy2(f, dest / name)
    if (src / "b00.md").is_file():
        paths.kb2_dir(run).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "b00.md", paths.summary_path(run, "b00"))
    return AnchorResult(meta.get("rRMSE"), True, "restored from cache")


def measure(run: Path, root: Path | None = None, gpu: int = 0) -> AnchorResult:
    """Copy the baseline into b00 and evaluate it once — smoke first, free."""
    dest = paths.block_dir(run, "b00")
    dest.mkdir(parents=True, exist_ok=True)
    candidate = dest / "b00_v01.py"
    shutil.copy2(run / "task" / "baseline.py", candidate)

    python = paths.venv_python(root)
    rel = str(candidate.relative_to(run))
    smoke = subprocess.run(
        [str(python), "task/eval.py", rel, "--smoke"],
        cwd=str(run),
        capture_output=True,
        text=True,
        check=False,  # the record is the verdict; the exit status is inspected below
    )
    if smoke.returncode != 0:
        return AnchorResult(None, False, f"smoke failed: {smoke.stderr.strip()[-400:]}")

    full = subprocess.run(
        [str(python), "task/eval.py", rel, "--gpu", str(gpu)],
        cwd=str(run),
        check=False,
        capture_output=True,
        text=True,
    )
    recs = EvalRecord.read_all(paths.evals_path(run, "b00"))
    score = best_score(recs)
    if full.returncode != 0 and score is None:
        return AnchorResult(None, False, f"eval failed: {full.stderr.strip()[-400:]}")
    return AnchorResult(score, False, "measured on GPU")


def harvest(task: str, root: Path | None = None) -> AnchorResult | None:
    """Recover this task's anchor from an earlier run of it, if one measured it.

    A run keeps a complete b00 — the candidate, its params, the eval records,
    the spent budget and the summary — so a task whose cache was cleared, or
    which arrived without one, does not need a GPU to get its anchor back. The
    frozen `task/` copy in the run is what makes it safe: harvesting only from
    a run whose contract files fingerprint to today's package is the same
    guard `restore` applies, asked of a different source.

    Newest run first, since a task is usually re-measured because it changed.
    """
    root = root or paths.project_root()
    want = registry.fingerprint(root / "tasks" / task)
    if not want:
        return None
    runs = sorted(
        (d for d in (root / "runs").glob(f"{task}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if registry.fingerprint(run / "task") != want:
            continue
        b00 = paths.block_dir(run, "b00")
        if not all((b00 / n).is_file() for n in ANCHOR_FILES):
            continue
        score = best_score(EvalRecord.read_all(b00 / "evals.jsonl"))
        if score is None:
            continue
        save(run, task, root)
        return AnchorResult(score, True, f"harvested from {run.name}")
    return None


def ensure(run: Path, task: str, root: Path | None = None, gpu: int = 0) -> AnchorResult:
    """The anchor, by the cheapest route that is still this task's anchor.

    Cache, then the run library, then a GPU — each step guarded by the same
    fingerprint, so a cheaper source is never a different measurement.
    """
    restored = restore(run, task, root)
    if restored is not None and restored.restored:
        return restored
    if (found := harvest(task, root)) is not None:
        installed = restore(run, task, root)
        if installed is not None and installed.restored:
            return AnchorResult(found.rRMSE, True, found.detail)
    result = measure(run, root, gpu)
    if result.rRMSE is not None:
        save(run, task, root)
    return result
