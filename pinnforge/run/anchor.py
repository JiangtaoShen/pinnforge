"""b00 — the control node every later block is measured against.

b00 is not a subagent product: it is `task/baseline.py` copied verbatim and
evaluated as-is. Measuring it costs a full GPU run, and the number only
depends on the task package and the hardware, so it is cached per task and
restored on later runs — keyed on the package's fingerprint (the baseline,
the evaluation, the public data), on the reference it was scored against,
and on the class of hardware it ran on. A different `baseline.py` means the
cached params came from other code; a different `eval.py` means the score is
on another metric; a different card means a wall-clock-bounded training run
got a different number of steps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from pinnforge import gpus, oscompat, paths, private
from pinnforge.task import registry
from pinnforge.types import RECORD_KEYS, EvalRecord, best_score

ANCHOR_FILES = ("b00_v01.py", "b00_v01.pkl", "evals.jsonl", ".budget")


@dataclass
class AnchorResult:
    rRMSE: float | None
    restored: bool
    detail: str = ""


def cache_dir(task: str, root: Path | None = None) -> Path:
    return (root or paths.project_root()) / "tasks" / task / ".anchor"


@contextmanager
def _locked(task: str, root: Path | None = None):
    """Hold one task's anchor cache against every other reader and writer.

    Two runs of a task with nothing cached each measure b00 and each save it;
    copied file by file into one directory, the cache could end up pairing
    one run's params with the other's score. The lock lives in the
    workspace's state directory: a file in the task directory would be copied
    into every run as if it were task data.
    """
    locks = paths.state_dir(root) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with open(locks / f"anchor-{task}.lock", "a", encoding="utf-8") as f:
        oscompat.lock(f)
        yield


def _copy_records(src: Path, dest: Path) -> None:
    """Copy b00's `evals.jsonl`, keeping only the fields of today's record format.

    The cached anchors were measured when a record still carried an `MSE`;
    copied verbatim, that key travelled from the cache into every new run's
    b00, where a block can read it — and an MSE beside an rRMSE, with the
    public grid's row count, is the reference's L2 norm. Whatever is not a
    current record field is dropped on the way, and so is any line that is
    not a record at all.
    """
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(json.dumps({k: v for k, v in rec.items() if k in RECORD_KEYS}))
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(f"{ln}\n" for ln in out), encoding="utf-8")
    oscompat.replace(tmp, dest)


def _copy_anchor_file(src: Path, dest: Path) -> None:
    if src.name == "evals.jsonl":
        _copy_records(src, dest)
    else:
        shutil.copy2(src, dest)


def clear(task: str, root: Path | None = None) -> None:
    """Drop the cached anchor, so the next run measures it again."""
    with _locked(task, root):
        shutil.rmtree(cache_dir(task, root), ignore_errors=True)


def reference_digest(task: str, root: Path | None = None) -> str:
    """One hash over the task's answers, wherever they are kept.

    Empty when the answers are not readable from here — a hardened
    deployment keeps them another user's — in which case it is not compared.
    """
    root = root or paths.project_root()
    home = private.locate(paths.tasks_dir(root) / task)
    try:
        files = sorted(p for p in home.iterdir() if p.is_file())
    except OSError:
        return ""
    h = hashlib.md5()
    for f in files:
        try:
            h.update(f.name.encode() + b"\0" + f.read_bytes())
        except OSError:
            return ""
    return h.hexdigest() if files else ""


def hardware(gpu: int | str | None) -> dict:
    """The hardware class a measurement on `gpu` runs on (CPU if none is visible)."""
    inv = gpus.inventory()
    if not inv or gpu is None:
        return gpus.hw_class([])
    return gpus.hw_class([int(gpu)], inv)


def save(run: Path, task: str, root: Path | None = None, hw: dict | None = None) -> Path:
    """Cache the run's measured b00 so the next run of this task is free.

    Built beside the cache and swapped in whole, so a reader sees the old
    anchor or the new one, never half of each.
    """
    dest = cache_dir(task, root)
    b00 = paths.block_dir(run, "b00")
    recs = EvalRecord.read_all(b00 / "evals.jsonl")
    meta = {
        "task": task,
        "rRMSE": best_score(recs),
        "fingerprint": registry.fingerprint(run / "task"),
        "reference": reference_digest(task, root),
    }
    if hw:
        meta["hw"] = hw
    with _locked(task, root):
        new = Path(tempfile.mkdtemp(prefix=".anchor.new-", dir=dest.parent))
        try:
            for name in ANCHOR_FILES:
                src = b00 / name
                if src.is_file():
                    _copy_anchor_file(src, new / name)
            summary = paths.summary_path(run, "b00")
            if summary.is_file():
                shutil.copy2(summary, new / "b00.md")
            (new / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            new.chmod(0o755)
            old = dest.with_name(f".anchor.old-{os.getpid()}")
            if dest.exists():
                dest.rename(old)
            new.rename(dest)
            shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(new, ignore_errors=True)
    return dest


def read_meta(task: str, root: Path | None = None) -> dict | None:
    path = cache_dir(task, root) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def staleness(meta: dict, task_dir: Path, task: str, root: Path | None, hw: dict | None) -> str:
    """Why a cached anchor does not apply here, or "" if it does."""
    changed = registry.fingerprint_changes(meta.get("fingerprint"), task_dir)
    if changed and not registry.same_package(meta.get("fingerprint"), task_dir):
        return f"measured against a different package ({', '.join(changed)})"
    ref = reference_digest(task, root)
    if meta.get("reference") and ref and meta["reference"] != ref:
        return "measured against a different reference"
    if hw:
        # No class recorded is not a match: it is not knowing. A cache from
        # before classes were recorded, restored onto whatever runs next, is
        # how a CPU-measured b00 ends up under a run on A6000s.
        if not meta.get("hw"):
            return f"measured on unrecorded hardware (this run is on {hw})"
        if meta["hw"] != hw:
            return f"measured on different hardware ({meta['hw']} vs {hw})"
    return ""


def restore(
    run: Path, task: str, root: Path | None = None, hw: dict | None = None
) -> AnchorResult | None:
    """Install a cached b00 if it matches this task package, else None."""
    with _locked(task, root):
        src = cache_dir(task, root)
        meta = read_meta(task, root)
        if meta is None:
            return None
        if why := staleness(meta, run / "task", task, root, hw):
            return AnchorResult(None, False, f"cached anchor was {why}")
        dest = paths.block_dir(run, "b00")
        dest.mkdir(parents=True, exist_ok=True)
        for name in ANCHOR_FILES:
            f = src / name
            if f.is_file():
                _copy_anchor_file(f, dest / name)
        if (src / "b00.md").is_file():
            paths.kb2_dir(run).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "b00.md", paths.summary_path(run, "b00"))
    return AnchorResult(meta.get("rRMSE"), True, "restored from cache")


def _run_group(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `eval.py` in a session of its own, and take the whole session down
    if this process is interrupted.

    `eval.py` trains in a worker it spawns, so killing `eval.py` alone left
    that worker running on the card — holding the GPU lock, with `.budget`
    pre-charged and never reconciled — after a Ctrl-C or a `run stop` during
    the anchor measurement.
    """
    proc = oscompat.popen_group(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        out, err = oscompat.communicate(proc)
    except BaseException:
        pgid = proc.pid
        for sig, grace in ((signal.SIGTERM, 10.0), (oscompat.SIGKILL, 5.0)):
            with suppress(ProcessLookupError, PermissionError):
                oscompat.killpg(pgid, sig)
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        with suppress(ProcessLookupError, PermissionError):
            oscompat.killpg(pgid, oscompat.SIGKILL)  # the worker may outlive eval.py
        raise
    finally:
        oscompat.forget_group(proc.pid)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def measure(run: Path, root: Path | None = None, gpu: int | str | None = 0) -> AnchorResult:
    """Copy the baseline into b00 and evaluate it once — smoke first, free.

    `gpu=None` measures on the CPU: `eval.py` still takes `--gpu 0`, and
    `JAX_PLATFORMS=cpu` keeps it off any card that is merely hidden.
    """
    dest = paths.block_dir(run, "b00")
    dest.mkdir(parents=True, exist_ok=True)
    candidate = dest / "b00_v01.py"
    shutil.copy2(run / "task" / "baseline.py", candidate)

    python = paths.venv_python(root)
    rel = candidate.relative_to(run).as_posix()
    # the record is the verdict; the exit status is inspected below
    smoke = _run_group([str(python), "task/eval.py", rel, "--smoke"], run)
    if smoke.returncode != 0:
        return AnchorResult(None, False, f"smoke failed: {smoke.stderr.strip()[-400:]}")

    if gpu is not None:
        # `--gpu N` must name the card nvidia-smi calls N — the pool, and the
        # hardware class this anchor is filed under, are in its (PCI) order,
        # which is not CUDA's default. Blocks get the same setting.
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    else:
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    full = _run_group([str(python), "task/eval.py", rel, "--gpu", str(gpu or 0)], run, env)
    recs = EvalRecord.read_all(paths.evals_path(run, "b00"))
    score = best_score(recs)
    if full.returncode != 0 and score is None:
        return AnchorResult(None, False, f"eval failed: {full.stderr.strip()[-400:]}")
    return AnchorResult(score, False, "measured on GPU" if gpu is not None else "measured on CPU")


def harvest(task: str, root: Path | None = None, hw: dict | None = None) -> AnchorResult | None:
    """Recover this task's anchor from an earlier run of it, if one measured it.

    A run keeps a complete b00 — the candidate, its params, the eval records,
    the spent budget and the summary — so a task whose cache was cleared, or
    which arrived without one, does not need a GPU to get its anchor back. The
    frozen `task/` copy in the run is what makes it safe: harvesting only from
    a run whose contract files fingerprint to today's package is the same
    guard `restore` applies, asked of a different source.

    Newest run first, since a task is usually re-measured because it changed.
    A run records the hardware class it ran on; one measured on another class
    is not this machine's anchor — and neither is one that records none, when
    the class wanted is known: it is filed under the class it recorded, never
    under the one asked for.
    """
    root = root or paths.project_root()
    lib = root / "tasks" / task
    if not registry.fingerprint(lib):
        return None
    runs = sorted(
        (d for d in (root / "runs").glob(f"{task}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if registry.fingerprint(run / "task") != registry.fingerprint(lib):
            continue
        run_hw = _run_hw(run)
        if hw and run_hw != hw:
            continue
        b00 = paths.block_dir(run, "b00")
        if not all((b00 / n).is_file() for n in ANCHOR_FILES):
            continue
        score = best_score(EvalRecord.read_all(b00 / "evals.jsonl"))
        if score is None:
            continue
        save(run, task, root, hw=run_hw)
        return AnchorResult(score, True, f"harvested from {run.name}")
    return None


def _run_hw(run: Path) -> dict | None:
    cfg = paths.config_path(run)
    if not cfg.is_file():
        return None
    import yaml

    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return (data.get("hw") or None) if isinstance(data, dict) else None


def ensure(
    run: Path,
    task: str,
    root: Path | None = None,
    gpu: int | str | None = 0,
    hw: dict | None = None,
    rebuild: bool = False,
) -> AnchorResult:
    """The anchor, by the cheapest route that is still this task's anchor.

    Cache, then the run library, then a GPU — each step guarded by the same
    fingerprint, reference and hardware class, so a cheaper source is never a
    different measurement. `rebuild` goes straight to the GPU: an operator
    who asked for a new measurement did not ask for an old run's.
    """
    hw = hw or hardware(gpu)
    if not rebuild:
        restored = restore(run, task, root, hw)
        if restored is not None and restored.restored:
            return restored
        if (found := harvest(task, root, hw)) is not None:
            installed = restore(run, task, root, hw)
            if installed is not None and installed.restored:
                # What was installed, not what was harvested: another run's
                # save may have replaced the cache in between.
                return AnchorResult(installed.rRMSE, True, found.detail)
    result = measure(run, root, gpu)
    if result.rRMSE is not None:
        save(run, task, root, hw=hw)
    return result


def rekey(
    run: Path, task: str, root: Path | None = None, hw: dict | None = None
) -> AnchorResult:
    with _locked(task, root):
        return _rekey(run, task, root, hw)


def _rekey(
    run: Path, task: str, root: Path | None = None, hw: dict | None = None
) -> AnchorResult:
    """Carry a cached anchor over to a re-fingerprinted package, without a GPU.

    Allowed only when the baseline itself is unchanged: the cached params are
    then still the baseline's params, and re-scoring them through today's
    evaluation on the CPU must reproduce the recorded score exactly. If it
    does, nothing about the measurement changed but its key; if it does not,
    the evaluation changed what it computes and the anchor has to be measured
    again.

    `run` is a scratch run with today's package installed and a score daemon
    serving it. `hw` is the class of the card this machine would measure on:
    a cache that records no class is stamped with it — on the assumption,
    said out loud, that it was measured here — when it is a GPU's, and left
    without one otherwise.
    """
    import sys

    meta = read_meta(task, root)
    src = cache_dir(task, root)
    if meta is None or not (src / "b00_v01.pkl").is_file():
        return AnchorResult(None, False, "no cached anchor with params to rekey")
    base = run / "task" / "baseline.py"
    recorded = meta.get("fingerprint") or {}
    want = (recorded.get("files") or recorded).get("baseline.py")
    if want != registry.file_md5(base):
        return AnchorResult(
            None, False, "baseline.py changed — the cached params are not its params"
        )

    b00 = paths.block_dir(run, "b00")
    b00.mkdir(parents=True, exist_ok=True)
    for name in ANCHOR_FILES:
        if (src / name).is_file():
            _copy_anchor_file(src / name, b00 / name)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "FORGE_TASK": task,
        "FORGE_RUN_ID": run.name,
        "FORGE_BLOCK": "b00",
        "FORGE_CANDIDATE": "blocks/b00/b00_v01.py",
        "FORGE_SEED": "0",
        "FORGE_SMOKE": "0",
    }
    proc = subprocess.run(
        [str(paths.venv_python(root)), "task/eval.py", "--worker-eval",
         str(b00 / "b00_v01.py"), str(b00 / "b00_v01.pkl")],
        cwd=str(run), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        out = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        out = {}
    score = out.get("rRMSE")
    if score is None:
        err = out.get("error") or proc.stderr[-400:]
        print(err, file=sys.stderr)
        return AnchorResult(None, False, f"re-scoring failed: {str(err)[-200:]}")
    if score != meta.get("rRMSE"):
        return AnchorResult(
            None, False,
            f"re-scored {score!r} != recorded {meta.get('rRMSE')!r} — the evaluation "
            "changed what it computes; measure the anchor again",
        )
    meta["fingerprint"] = registry.fingerprint(run / "task")
    meta["reference"] = reference_digest(task, root)
    detail = "rekeyed: re-scored identically on the CPU"
    if not meta.get("hw") and hw and hw.get("platform") == "gpu":
        meta["hw"] = hw
        detail += (f"; the cache recorded no hardware class, so it is assumed to have "
                   f"been measured on this machine's {hw.get('gpu')}")
    # The cache's own records go through the same whitelist as a restore:
    # rekeying is how an old cache is brought up to today's format.
    if (src / "evals.jsonl").is_file():
        _copy_records(src / "evals.jsonl", src / "evals.jsonl")
    tmp = src / f".meta.{os.getpid()}.json"
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(src / "meta.json")
    return AnchorResult(score, True, detail)
