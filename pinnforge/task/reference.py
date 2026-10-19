"""The reference: fetched from the operator, or built — and then checked here.

A reference the reference-builder agent computed is not accepted on its word.
`check()` reads the refinement levels it left behind and asks the task's own
metric how far each level is from the next:

    e_k = rRMSE(level k  scored against level k+1)

A converging solver makes these shrink by a steady factor r**p (r the
refinement ratio, p the order); from the last two the order is observed and
the error of the finest level estimated the way Richardson extrapolation
does, e_{K-1} / (r**p - 1), never less than what the order supports. The
check then re-runs the two coarsest levels itself, because a table of
numbers is only evidence if the program that made it reproduces it.

Everything the checker reads sits in the task's `private/`, and so does the
full verdict (`private/reference/verdict.json`). What leaves it is only what
the framework computed itself — pass or fail, the level-to-level differences
under the task's metric, the observed order, the error estimate, whether the
re-run reproduced — in `.authoring/reference.json`, for the operator and the
dashboard. Nothing an agent that saw the reference wrote (its method, its
resolutions, its notes, its solver's output) and nothing about the operator's
data (where it came from, what it is) is ever written outside `private/`, and
nothing about the reference at all reaches the task package a block reads.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from pinnforge import paths

DEFAULT_TOL = 1e-3
VERDICT = "verdict.json"  # the full verdict, under private/reference/
SOURCE = "source.json"  # where the operator's data came from, under private/reference/
# The keys of a verdict the framework computed itself; only these are published.
PUBLIC_KEYS = ("source", "verdict", "reasons", "tol", "differences", "order", "estimate",
               "reproducible", "ratio", "n_levels", "rows", "ref_columns",
               "accepted_by_operator")
ROUNDOFF = 1e-10  # below this, successive levels differ by roundoff, not truncation
# The refinement ratios a convergence study may claim: below 1.5 the levels are
# too close to tell an order from noise; above 4 the estimate's cap (a factor
# r**4 - 1 under the last difference) would claim more than any level shows.
MIN_RATIO, MAX_RATIO = 1.5, 4.0


# ─────────────────────────────── task module ───────────────────────────────


def _literal(evalsrc: str, name: str):
    """A module-level literal of `eval.py`, read without executing it."""
    tree = ast.parse(evalsrc)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def ref_columns(task_dir: Path) -> tuple[str, ...] | None:
    src = (Path(task_dir) / "eval.py").read_text(encoding="utf-8")
    cols = _literal(src, "REF_COLUMNS")
    return tuple(cols) if cols else None


def fields(task_dir: Path) -> tuple[str, ...]:
    src = (Path(task_dir) / "eval.py").read_text(encoding="utf-8")
    return tuple(_literal(src, "FIELDS") or ())


def load_metric(task_dir: Path, into: Path):
    """Import the task's `eval.py` from a copy of its public files in `into`."""
    for f in Path(task_dir).iterdir():
        if f.is_file():
            shutil.copy2(f, into / f.name)
    spec = importlib.util.spec_from_file_location("pinnforge_refcheck", into / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ─────────────────────────────── operator data ───────────────────────────────


def fetch(source: str, dest_dir: Path) -> tuple[Path, str]:
    """Copy or download the operator's reference data into `dest_dir`.

    Returns the local path and a sha256 of what arrived, which is recorded so
    the reference's provenance survives the operator's copy moving on.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "://" in source and not source.startswith("file://"):
        name = Path(urllib.request.urlparse(source).path).name or "reference.data"
        out = dest_dir / name
        req = urllib.request.Request(source, headers={"User-Agent": "pinnforge"})
        with urllib.request.urlopen(req, timeout=120) as resp, out.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    else:
        src = Path(paths.local_path(source)).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"no reference data at {src}")
        out = dest_dir / src.name
        if src.is_dir():
            shutil.copytree(src, out, dirs_exist_ok=True)
        else:
            shutil.copy2(src, out)
    return out, _digest(out)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        with f.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────── the check ───────────────────────────────


def _levels(workdir: Path) -> list[tuple[int, Path]]:
    found = []
    for f in (workdir / "levels").glob("L*.npy"):
        m = re.fullmatch(r"L(\d+)\.npy", f.name)
        if m:
            found.append((int(m.group(1)), f))
    return sorted(found)


def _ratio(meta: dict, reasons: list[str]) -> float | None:
    """The refinement ratio levels.json claims, if it is one the check can rest on.

    The estimate's cap is a factor r**4 - 1 below the last difference, so a
    ratio of 1 divides by zero, one below 1 turns the cap negative and an
    absurdly large one claims a cap nothing supports. What the agent wrote is
    never repeated back: only that it is not a usable ratio.
    """
    raw = meta.get("ratio")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        reasons.append("levels.json must state the refinement ratio as a number")
        return None
    ratio = float(raw)
    if not math.isfinite(ratio) or not MIN_RATIO <= ratio <= MAX_RATIO:
        reasons.append(f"levels.json: the refinement ratio must lie between {MIN_RATIO:g} "
                       f"and {MAX_RATIO:g}")
        return None
    return ratio


def _meta(workdir: Path, reasons: list[str]) -> dict:
    path = workdir / "levels.json"
    if not path.is_file():
        reasons.append("levels.json was not written")
        return {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        reasons.append("levels.json is not readable JSON")
        return {}
    if not isinstance(meta, dict):
        reasons.append("levels.json is not a JSON object")
        return {}
    return meta


def check(
    task_dir: Path,
    private_home: Path,
    *,
    tol: float = DEFAULT_TOL,
    python: Path | None = None,
    rerun: bool = True,
    rerun_timeout: float = 1800.0,
) -> dict:
    """Verify a built reference converged. Returns the full verdict (`public()` cuts it)."""
    import numpy as np

    task_dir, private_home = Path(task_dir), Path(private_home)
    workdir = private_home / "reference"
    reasons: list[str] = []
    verdict: dict = {"source": "built", "tol": tol, "reasons": reasons}

    def fail() -> dict:
        verdict["verdict"] = "fail"
        return verdict

    meta = _meta(workdir, reasons)
    verdict["method"] = meta.get("method", "")
    verdict["horizon"] = meta.get("horizon")
    listed = meta.get("levels") if isinstance(meta.get("levels"), list) else []
    verdict["levels"] = [
        {k: lv.get(k) for k in ("level", "resolution", "dof", "wall_s")}
        for lv in listed if isinstance(lv, dict)
    ]
    levels = _levels(workdir)
    verdict["n_levels"] = len(levels)
    ratio = None if reasons else _ratio(meta, reasons)
    if ratio is not None:
        verdict["ratio"] = ratio
    if reasons:
        return fail()

    if len(levels) < 3:
        # the count only: where the levels live is the private directory's business
        reasons.append(f"{len(levels)} refinement level(s) in levels/; need 3")
        return fail()
    cols = ref_columns(task_dir)
    names = fields(task_dir)
    if not cols or not names:
        reasons.append("eval.py declares no REF_COLUMNS / FIELDS")
        return fail()
    missing = [f for f in names if f not in cols]
    if missing:
        reasons.append(f"FIELDS {missing} are not among REF_COLUMNS {list(cols)}")
        return fail()

    with tempfile.TemporaryDirectory(prefix="pinnforge-refcheck-",
                                     ignore_cleanup_errors=True) as tmp:
        mod_dir = Path(tmp) / "task"
        mod_dir.mkdir()
        mod = load_metric(task_dir, mod_dir)
        grid = np.loadtxt(mod.GRID_CSV, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        rows = mod.predict_rows(grid) if callable(getattr(mod, "predict_rows", None)) else (
            slice(None)
        )
        expected = (grid.shape[0], len(cols))
        arrays = []
        for k, f in levels:
            try:
                a = np.load(f, allow_pickle=False)
            except (OSError, ValueError):
                reasons.append(f"L{k}: not a readable .npy array")
                continue
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.shape != expected:
                reasons.append(f"L{k}: shape {a.shape}, expected {expected}")
            elif a.dtype != np.float64:
                reasons.append(f"L{k}: dtype {a.dtype}, expected float64")
            elif not np.all(np.isfinite(a)):
                reasons.append(f"L{k}: non-finite values")
            arrays.append(a)
        if reasons:
            return fail()

        ref_dir = Path(tmp) / "ref"
        ref_dir.mkdir()
        diffs = []
        for fine, coarse in zip(arrays[1:], arrays[:-1], strict=True):
            np.save(ref_dir / mod.REF_VALUES, fine)
            pred = {f: coarse[rows, cols.index(f)] for f in names}
            out = mod.score_predictions(pred, ref_dir)
            if out.get("error") or out.get("rRMSE") is None:
                reasons.append("the task's metric refused a level (details in the "
                               "private verdict)")
                verdict["details"] = str(out.get("error"))[:2000]
                return fail()
            try:
                diffs.append(float(out["rRMSE"]))
            except (TypeError, ValueError):
                reasons.append("the task's metric returned a non-number for a level")
                return fail()
    if not all(math.isfinite(d) for d in diffs):
        # a 0/0 metric (a field that is zero on every scored row) gives NaN
        reasons.append("the task's metric gave a non-finite difference between two levels")
        return fail()
    verdict["differences"] = diffs

    last, prev = diffs[-1], diffs[-2]
    if any(d == 0.0 for d in diffs):
        reasons.append("two levels are identical — they were not computed at different resolutions")
    order = None
    if last > ROUNDOFF and prev > ROUNDOFF and last < prev:
        order = math.log(prev / last) / math.log(ratio)
    verdict["order"] = order
    plateau = max(ROUNDOFF, 1e-3 * tol)
    if last <= ROUNDOFF:
        estimate = last  # converged to roundoff
    elif max(last, prev) <= plateau:
        # A converged solver stops improving where its arithmetic does: the
        # finest levels differ at a floor far below the tolerance, and which
        # of the two is larger is noise.
        estimate = max(last, prev)
    elif last >= prev:
        reasons.append(
            f"not converging: the finest two levels differ by {last:.2e}, "
            f"the previous two by {prev:.2e}"
        )
        estimate = last
    else:
        # Richardson: error(finest) ≈ e_last / (r^p - 1). An order below 1
        # makes that larger than e_last, as it should — a slowly converging
        # solver is further from the limit than its last step suggests. A
        # large order makes it tiny, which is only credible if the solver's
        # formal order is that large — so never claim more than a factor
        # 1/(r^4 - 1), a fourth-order method's worth, below the last difference.
        factor = ratio**order - 1.0
        estimate = max(last / factor, last / (ratio**4 - 1.0))
    verdict["estimate"] = estimate

    finest = arrays[-1]
    ref_file = private_home / "ref_values.npy"
    if not ref_file.is_file():
        reasons.append("ref_values.npy was not written")
    else:
        try:
            stored = np.load(ref_file, allow_pickle=False)
        except (OSError, ValueError):
            stored = None
            reasons.append("ref_values.npy is not a readable .npy array")
        if stored is not None:
            if stored.ndim == 1:
                stored = stored.reshape(-1, 1)
            if stored.shape != finest.shape:
                reasons.append(f"ref_values.npy has shape {stored.shape}, expected {finest.shape}")
            elif not np.array_equal(stored, finest):
                reasons.append("ref_values.npy is not the finest level")

    if rerun and python is not None:
        verdict["reproducible"] = _rerun(workdir, python, levels[:2], rerun_timeout, reasons,
                                         verdict)

    if estimate > tol:
        reasons.append(f"estimated error {estimate:.2e} exceeds the tolerance {tol:.0e}")
    verdict["verdict"] = "fail" if reasons else "pass"
    return verdict


def _rerun(workdir: Path, python: Path, levels, timeout: float, reasons: list[str],
           verdict: dict) -> bool:
    import numpy as np

    solve = workdir / "solve.py"
    if not solve.is_file():
        reasons.append("solve.py is missing — the levels cannot be reproduced")
        return False
    with tempfile.TemporaryDirectory(prefix="pinnforge-rerun-",
                                     ignore_cleanup_errors=True) as tmp:
        for k, stored in levels:
            out = Path(tmp) / f"L{k}.npy"
            try:
                proc = subprocess.run(
                    [str(python), str(solve), "--level", str(k), "--out", str(out)],
                    cwd=str(workdir), capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                reasons.append(f"re-running level {k} took longer than {timeout:.0f} s")
                return False
            if proc.returncode != 0 or not out.is_file():
                # the solver's own output stays private: it is the agent's text
                verdict["details"] = proc.stderr[-4000:]
                reasons.append(f"re-running level {k} failed (exit {proc.returncode}; its "
                               "output is in the private verdict)")
                return False
            a, b = np.load(out), np.load(stored)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-12):
                reasons.append(f"re-running level {k} gave different numbers")
                return False
    return True


def check_user(task_dir: Path, private_home: Path, ingest: dict) -> dict:
    """What can be checked of a reference the operator supplied.

    `ingest` is the grid's shape as the framework derived it from the files
    (never the ingest agent's own account), so it may be published.
    """
    import numpy as np

    reasons: list[str] = []
    verdict = {"source": "operator", "reasons": reasons, "estimate": None,
               "rows": ingest.get("rows"), "ref_columns": ingest.get("ref_columns")}
    ref = private_home / "ref_values.npy"
    grid = Path(task_dir) / "eval_grid.csv"
    if not ref.is_file() or not grid.is_file():
        reasons.append("ref_values.npy or eval_grid.csv was not written")
    else:
        try:
            a = np.load(ref, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            a = None
            reasons.append("ref_values.npy is not a readable .npy array")
        if a is not None:
            n = grid_rows(grid)
            if a.ndim not in (1, 2) or a.shape[0] != n:
                reasons.append(f"{a.shape[0] if a.ndim else 0} reference rows for {n} grid rows")
            if a.dtype != np.float64 or not np.all(np.isfinite(a)):
                reasons.append("the reference must be finite float64")
    verdict["verdict"] = "fail" if reasons else "pass"
    return verdict


def grid_rows(grid: Path) -> int:
    """Data rows of a grid file: every non-blank line after the header."""
    with Path(grid).open("rb") as fh:
        next(fh, None)
        return sum(1 for line in fh if line.strip())


# ─────────────────────────────── the verdict ───────────────────────────────


def public(verdict: dict) -> dict:
    """The part of a verdict that may leave `private/`: what the framework computed."""
    return {k: verdict[k] for k in PUBLIC_KEYS if k in verdict}


def record(verdict: dict, private_home: Path, public_path: Path) -> None:
    """Write the full verdict under `private/`, and its public part beside the pipeline."""
    workdir = Path(private_home) / "reference"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / VERDICT).write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    Path(public_path).write_text(json.dumps(public(verdict), indent=2) + "\n",
                                 encoding="utf-8")


def remember_source(private_home: Path, location: str) -> None:
    """Keep where the operator's data is under `private/` — never in a public record.

    The authoring state, the intake the author agent reads and the pipeline
    log are all outside `private/`; a path in any of them is a path to the
    answers.
    """
    if "://" not in location:
        location = str(Path(location).expanduser().resolve())
    workdir = Path(private_home) / "reference"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / SOURCE).write_text(json.dumps({"location": location}) + "\n", encoding="utf-8")


def source_location(private_home: Path, recorded: str = "") -> str:
    """Where the operator's data is: the private record, else (older state) `recorded`."""
    f = Path(private_home) / "reference" / SOURCE
    if f.is_file():
        return json.loads(f.read_text(encoding="utf-8"))["location"]
    if recorded and recorded != "operator":
        return recorded
    raise FileNotFoundError(f"no record of the operator's reference data in {f.parent}")


# ─────────────────────────────── the ingested grid ───────────────────────────────

GRID_PIN = "grid.json"  # under private/reference/: the ingested grid's sha256
GRID_COPY = "eval_grid.ingested.csv"  # and the grid itself, to put back


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pin_grid(task_dir: Path, private_home: Path) -> str:
    """Remember the ingested `eval_grid.csv` exactly, with the answers it lines up with.

    The reference is row-for-row against this file: a grid re-saved, sorted
    or rounded afterwards misaligns every score without failing any of them.
    """
    grid = Path(task_dir) / "eval_grid.csv"
    workdir = Path(private_home) / "reference"
    workdir.mkdir(parents=True, exist_ok=True)
    sha = sha256_file(grid)
    shutil.copy2(grid, workdir / GRID_COPY)
    (workdir / GRID_PIN).write_text(json.dumps({"sha256": sha}) + "\n", encoding="utf-8")
    return sha


def pinned_grid(private_home: Path) -> str:
    f = Path(private_home) / "reference" / GRID_PIN
    try:
        return str(json.loads(f.read_text(encoding="utf-8")).get("sha256") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def grid_drift(task_dir: Path, private_home: Path, *, restore: bool = False) -> str:
    """Why `eval_grid.csv` is no longer the ingested grid, or "" if it is.

    With `restore`, the ingested copy is put back over whatever is there.
    """
    pin = pinned_grid(private_home)
    if not pin:
        return "no record of the ingested grid — re-run the ingest (`--resume --redo reference`)"
    grid = Path(task_dir) / "eval_grid.csv"
    if grid.is_file() and sha256_file(grid) == pin:
        return ""
    why = "eval_grid.csv is no longer the grid the reference was ingested on"
    copy = Path(private_home) / "reference" / GRID_COPY
    if restore and copy.is_file() and sha256_file(copy) == pin and not (
        grid.is_dir() and not grid.is_symlink()
    ):
        grid.unlink(missing_ok=True)  # never write through a link someone left in its place
        shutil.copy2(copy, grid)
        why += " (the ingested grid has been put back)"
    return why


# ─────────────────────────────── solver environment ───────────────────────────────


def solver_env(state_dir: Path, uv: str) -> Path:
    """The reference agent's own Python — never the environment blocks train in."""
    env = Path(state_dir) / "refenv"
    python = paths.venv_bin(env)
    if not python.is_file():
        subprocess.run([uv, "venv", "--python", "3.12", str(env)], check=True,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
        subprocess.run([uv, "pip", "install", "--python", str(python), "numpy", "scipy"],
                       check=True, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return python
