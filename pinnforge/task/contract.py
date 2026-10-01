"""The task contract — what makes a directory a valid PINNForge task.

The framework touches a task only through this interface, so a package that
honours it plugs in without framework changes. Validation is mechanical and
runs before any GPU time is spent: a task that disagrees with itself (a wall
budget in `problem.md` that the frozen header contradicts, a smoke literal
that changes the line length) wastes a whole run before anyone notices.

`problem.md` is also checked for *shape*. The prose paradigm is the point of
the corpus — every task states the same seven sections in the same order, so
a block agent reading its second task already knows where to look.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import private

CONTRACT_FILES = ("problem.md", "baseline.py", "eval.py")

# Where the scoring truth lives. A directory rather than a list of filenames:
# a name-based rule silently stops covering the answers the moment a task
# calls them something else, which is exactly how the old charter ended up
# forbidding a file that no longer existed.
PRIVATE_DIR = "private"

# The seven sections, in order. Nothing added, dropped, renamed or reordered.
REQUIRED_SECTIONS = (
    "## Equation",
    "## Domain",
    "## Boundary conditions",
    "## Initial condition",
    "## Scoring",
    "## Environment",
    "## Time budget",
)

# Verbatim in every task. Whitespace is normalised before comparing, so the
# line wrapping is free but the words are not.
INTRO = (
    "Solve the PDE with a genuine physics-informed neural network (PINN). "
    "The space/time derivatives entering the PDE residual come from automatic "
    "differentiation of the neural network(s). Classical methods may solve it "
    "efficiently, but the value of this task is probing the limits of PINN itself."
)

ENVIRONMENT_BODY = (
    "The GPU pool and the per-block budget are stated in your charter "
    "(`block.md`). PINNs must be built on the frozen JAX stack pinned in "
    "`pyproject.toml`; extensions via `uv add` if the pins survive."
)

# The wording every task used before the hardware moved into the charter.
# Still accepted, because rewriting a task's `problem.md` would only churn
# its fingerprint; it no longer describes the machine a run is on, so new
# tasks state the neutral form above.
LEGACY_ENVIRONMENT_BODY = (
    "Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack "
    "pinned in `pyproject.toml`; extensions via `uv add` if the pins survive."
)
ENVIRONMENT_BODIES = (ENVIRONMENT_BODY, LEGACY_ENVIRONMENT_BODY)

# A task whose `eval.py` delegates to the shared machinery names it here.
KIT_MODULE = "_forgekit"

_TRAIN_TIME_LITERAL = re.compile(r'TRAIN_TIME_LITERAL\s*=\s*"([^"]+)"')
_SMOKE_TRAIN_TIME = re.compile(r'SMOKE_TRAIN_TIME\s*=\s*"([^"]+)"')
_BUDGET_DEFAULT = re.compile(r'FORGE_WALL_BUDGET"\s*,\s*"([\d.]+)"')
_TRAIN_TIME_VALUE = re.compile(r"TRAIN_TIME\s*=\s*([\d.]+)")


@dataclass
class ValidationReport:
    task: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"task: {self.task}", ""]
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
        lines.append("")
        lines.append("ALL CHECKS PASSED" if self.ok else f"{len(self.failures)} CHECK(S) FAILED")
        return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.split())


def validate(task_dir: Path, reference: bool = True) -> ValidationReport:
    """Check one task directory against the contract. Never raises.

    `reference=False` checks everything but the answers — what a task author
    can check before the reference stage has filled `private/`.
    """
    rep = ValidationReport(task=task_dir.name)

    for name in CONTRACT_FILES:
        rep.check(f"{name} present", (task_dir / name).is_file())
    if rep.failures:
        return rep  # nothing else is meaningful without the three files

    problem = (task_dir / "problem.md").read_text(encoding="utf-8")
    baseline = (task_dir / "baseline.py").read_text(encoding="utf-8")
    evalsrc = (task_dir / "eval.py").read_text(encoding="utf-8")

    _check_problem(rep, problem)
    _check_baseline(rep, baseline)
    _check_eval(rep, evalsrc)
    _check_data(rep, task_dir, reference, evalsrc)
    _check_sealed(rep, task_dir, problem, baseline, evalsrc)
    _check_view(rep, task_dir)
    if reference:
        _check_reference(rep, task_dir, evalsrc)
    _check_agreement(rep, problem, baseline, evalsrc)
    return rep


# ─────────────────────── nothing of the answers in the package ───────────────────────

# A task package is everything a block is handed. It states the problem — the
# equation, the domain, the conditions, how it is scored — and nothing about
# its solution: not a closed form, not values, not where the answers are.
_SOLUTION_WORDS = re.compile(
    r"\b(?:exact|analytic(?:al)?|closed[- ]form|true|manufactured)\s+solution"
    r"|ground[- ]truth|discreti[sz]ation\s+error",
    re.IGNORECASE,
)
# "No closed-form solution is known", "there is no exact solution": a sentence
# that says a solution does not exist does not give one away.
_NEGATED = re.compile(r"\b(?:no|not|nor|without|never)\b(?:\W+\w+){0,2}\W*$", re.IGNORECASE)
_DENIED_AFTER = re.compile(r"^\W*(?:is|are|was|remains)\s+(?:not\b|unknown\b|unavailable\b)",
                           re.IGNORECASE)
# `private` as a path: next to a separator, or a string that is just the name.
# "Kept private to this module" is prose, not a path.
_PRIVATE_PATH = re.compile(r"(?:[/\\]private\b|\bprivate[/\\])")
_DATA_SUFFIXES = (".csv", ".npy", ".npz", ".mat", ".h5", ".hdf5", ".vtu", ".vtk", ".pkl",
                  ".dat", ".txt", ".json")
_FILE_NAME = re.compile(r"[\w.\-/\\]+")


def _names_private(text: str) -> bool:
    return text.strip() == PRIVATE_DIR or _PRIVATE_PATH.search(text) is not None


def _check_view(rep: ValidationReport, task_dir: Path) -> None:
    """`view.json`, if the task has one: what the dashboard's field window draws."""
    from pinnforge.task import view

    path = task_dir / view.VIEW_FILE
    if not path.is_file():
        rep.check("view.json (optional) — the dashboard picks a default view", True)
        return
    spec = view.load(path)
    coords, fields = view.task_axes(task_dir)
    shown, problems = view.check(spec, coords, fields)
    rep.check("view.json draws the task's own coordinates and output heads",
              shown is not None, "; ".join(problems[:3]))
    titles = " ".join(p.get("title", "") for p in (shown or {}).get("panels", []))
    m = _solution_words(titles)
    rep.check("view.json says nothing about the solution", m is None,
              f"mentions {m.group(0)!r}" if m else "")


def _solution_words(text: str):
    """The first mention of a solution that is not a denial of one, or None."""
    for m in _SOLUTION_WORDS.finditer(text):
        if not (_NEGATED.search(text[max(0, m.start() - 40) : m.start()])
                or _DENIED_AFTER.search(text[m.end() : m.end() + 40])):
            return m
    return None


def _file_name(s: str) -> str | None:
    """`s` as the name of a data file, or None for a pattern, a suffix or a sentence."""
    if not s.lower().endswith(_DATA_SUFFIXES) or not _FILE_NAME.fullmatch(s):
        return None
    name = Path(s).name
    stem = name.rsplit(".", 1)[0]
    if not stem or stem.startswith("."):
        return None  # ".csv": a suffix, not a file
    return name


def _strings(src: str) -> list[str]:
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)]


def _check_sealed(rep: ValidationReport, task_dir: Path, problem: str, baseline: str,
                  evalsrc: str) -> None:
    for name, text in (("problem.md", problem), ("baseline.py", baseline), ("eval.py", evalsrc)):
        m = _solution_words(text)
        rep.check(f"{name} says nothing about the solution", m is None,
                  f"mentions {m.group(0)!r}" if m else "")
    public = {f.name for f in task_dir.iterdir() if f.is_file()}
    strings = _strings(baseline)
    private_refs = [s for s in strings if _names_private(s) or "ref_values" in s]
    rep.check("baseline.py never names the answers", not private_refs,
              ", ".join(map(repr, private_refs[:3])))
    unknown = sorted({s for s in strings if (n := _file_name(s)) and n not in public})
    rep.check("baseline.py reads only the task's public files", not unknown,
              ", ".join(unknown[:3]))
    rep.check("eval.py never names private/",
              not any(_names_private(s) for s in _strings(evalsrc)))


def _affine(a, b, tol: float) -> bool:
    """Is `a` (nearly) α·b + β? A copy, a rescaled copy, or a rounded one."""
    import numpy as np

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return False
    a, b = a[m], b[m]
    if np.std(b) == 0 or np.std(a) == 0:
        return False
    design = np.column_stack([b, np.ones_like(b)])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return bool(np.linalg.norm(a - design @ coef) <= tol * np.linalg.norm(a - a.mean()))


# ─────────────────────── reading what a package publishes ───────────────────────

# Files no text or table can be read from; everything else at the top of a
# task directory is a candidate for both.
_BINARY_SUFFIXES = (".npy", ".npz", ".pkl", ".pickle", ".pyc", ".mat", ".h5", ".hdf5", ".vtu",
                    ".vtk", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".gz", ".zip", ".tar",
                    ".bz2", ".xz", ".so", ".pt", ".pth", ".ckpt", ".msgpack")
_TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".dat")
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?!\.?\w)")
_MAX_TEXT = 64 * 1024 * 1024  # a file larger than this is not read as text


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _split(line: str, delim: str | None) -> list[str]:
    return [c.strip() for c in (line.split(delim) if delim else line.split())]


def _table(text: str):
    """(header names or None, numeric columns) of a delimited table, or None if ragged."""
    import numpy as np

    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith(("#", "%"))]
    if not lines:
        return None
    first = lines[0]
    delim = next((d for d in (",", "\t", ";") if d in first), None)
    head = _split(first, delim)
    names = None
    if not all(_is_number(c) for c in head if c):
        names, lines = head, lines[1:]  # a header: the first line is not all numbers
    if not lines:
        return names, []
    width = len(_split(lines[0], delim))
    cells = [_split(ln, delim) for ln in lines]
    if any(len(c) != width for c in cells):
        return None
    cols = []
    for j in range(width):
        try:
            col = np.array([float(c[j]) for c in cells], dtype=np.float64)
        except ValueError:
            continue  # a non-numeric column: a label, a name
        label = names[j] if names and j < len(names) and names[j] else f"column {j}"
        cols.append((label, col))
    return names, cols


def _numbers(text: str):
    import numpy as np

    vals = []
    for t in _NUMBER.findall(text):
        try:
            vals.append(float(t))
        except ValueError:
            continue
    return np.array(vals, dtype=np.float64)


def _json_arrays(obj, name: str = ""):
    """Every list of numbers in a JSON document, as (name, array)."""
    import numpy as np

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_arrays(v, f"{name}.{k}" if name else str(k))
    elif isinstance(obj, list) and obj:
        try:
            a = np.asarray(obj, dtype=np.float64)
        except (TypeError, ValueError):
            a = None
        if a is not None and a.ndim >= 1:
            yield (name or "array", a)
        else:
            for i, v in enumerate(obj):
                yield from _json_arrays(v, f"{name}[{i}]")


def _file_arrays(f: Path):
    """(label, array) for the numeric content of one public file; nothing if it has none."""
    import json

    import numpy as np

    suffix = f.suffix.lower()
    try:
        if suffix == ".npy":
            yield "", np.load(f, allow_pickle=False)
            return
        if suffix == ".npz":
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    yield k, z[k]
            return
        if suffix in _BINARY_SUFFIXES or f.stat().st_size > _MAX_TEXT:
            return
        text = f.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError, EOFError):
        return
    if suffix == ".json":
        with contextlib.suppress(ValueError, RecursionError):
            yield from list(_json_arrays(json.loads(text)))
        yield "numbers", _numbers(text)
    elif suffix in _TABLE_SUFFIXES:
        table = _table(text)
        if table is None:
            yield "numbers", _numbers(text)  # ragged: every number, in order
        else:
            yield from table[1]
            if suffix != ".csv" and not table[1]:
                yield "numbers", _numbers(text)


def _vectors(a, rows: int):
    """The 1-D readings of an array that could line up with a `rows`-long reference.

    A column of a table; a whole array flattened, either way round, when its
    size is a multiple of `rows` (a 2-D field on the grid, a transposed table).
    """
    import numpy as np

    a = np.asarray(a)
    if a.ndim == 0 or not a.size or not np.issubdtype(a.dtype, np.number):
        return
    a = a.astype(np.float64)
    if a.ndim == 1:
        yield "", a
    else:
        lead = a.reshape(a.shape[0], -1)
        for j in range(lead.shape[1]):
            yield f"[:, {j}]", lead[:, j]
    if (a.size % rows == 0 and a.size > rows) or (a.ndim > 1 and a.size == rows):
        for order in ("C", "F") if a.ndim > 1 else ("C",):
            flat = a.ravel(order=order)
            by_col = flat.reshape(rows, -1)
            for j in range(by_col.shape[1]):
                yield f" flattened ({order}) column {j}", by_col[:, j]
            for i, row in enumerate(flat.reshape(-1, rows)):
                yield f" flattened ({order}) row {i}", row


def _offsets(n: int, rows: int):
    """(stride, offset) pairs whose every-k-th subsample of `rows` has length n."""
    for k in range(2, 9):
        for off in range(k):
            if len(range(off, rows, k)) == n:
                yield k, off


def _grid_columns(task_dir: Path, n_in: int):
    grid = task_dir / "eval_grid.csv"
    if not grid.is_file() or n_in <= 0:
        return []
    return [col for _, col in list(_file_arrays(grid))[:n_in]]


def published_copies(task_dir: Path, ref, n_in: int, only: tuple[str, ...] = ()) -> list[str]:
    """Where a public file of `task_dir` holds the reference as data.

    Every top-level file is read for numbers — tables column by column, arrays
    whichever way round they fit, JSON and text tables too — and every column
    is compared with every reference column: a copy, a rescaled or rounded
    copy, or an every-k-th subsample of one. The grid's coordinates are set
    aside first, wherever they are repeated, and so are columns of three
    values or fewer (masks, flags), which cannot be a field.
    """
    import numpy as np

    ref = np.asarray(ref, dtype=np.float64).reshape(np.shape(ref)[0], -1)
    rows = ref.shape[0]
    coords = _grid_columns(task_dir, n_in)
    found = []
    for f in sorted(task_dir.iterdir()):
        if not f.is_file() or (only and f.name not in only):
            continue
        skip = n_in if f.name == "eval_grid.csv" else 0
        for label, arr in _file_arrays(f):
            if skip:
                skip -= 1
                continue
            for sub, col in _vectors(arr, rows):
                if col.size < 10 or len(np.unique(col[np.isfinite(col)])) <= 3:
                    continue
                where = f"{f.name}:{label or 'array'}{sub}"
                if col.size == rows:
                    if any(_affine(col, c, 1e-9) for c in coords if c.size == rows):
                        continue  # a coordinate, wherever it is repeated
                    hit = next((k for k in range(ref.shape[1])
                                if _affine(col, ref[:, k], 1e-2)), None)
                else:
                    hit = next((k for k in range(ref.shape[1]) for s, o in
                                _offsets(col.size, rows)
                                if _affine(col, ref[o::s, k], 1e-2)
                                and not any(_affine(col, c[o::s], 1e-9) for c in coords)),
                               None)
                if hit is not None:
                    found.append(f"{where} ~ reference column {hit}")
                    break
    return found


def _significant(token: str) -> int:
    mantissa = re.split(r"[eE]", token)[0]
    digits = re.sub(r"\D", "", mantissa).lstrip("0")
    if "." not in mantissa:
        digits = digits.rstrip("0")  # 1000 is one significant digit
    return len(digits)


def _matches(tokens: list[str], values) -> int:
    """How many printed numbers equal a reference value at the precision they are printed to."""
    import numpy as np

    if not tokens or not len(values):
        return 0
    x = np.array([float(t) for t in tokens])
    unit = []
    for t in tokens:
        mantissa, _, exp = t.lower().partition("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        unit.append(10.0 ** (int(exp or 0) - decimals))
    tol = 0.5 * np.array(unit) * (1 + 1e-9) + 4 * np.finfo(np.float64).eps * np.abs(x)
    lo = np.searchsorted(values, x - tol, side="left")
    ok = lo < len(values)
    ok[ok] = values[lo[ok]] <= (x + tol)[ok]
    return int(ok.sum())


def printed_values(text: str, values) -> tuple[int, int]:
    """(numbers with ≥4 significant digits, how many of them are reference values)."""
    tokens = [t for t in _NUMBER.findall(text) if _significant(t) >= 4]
    return len(tokens), _matches(tokens, values)


def _texts(task_dir: Path, only: tuple[str, ...] = ()):
    """(name, text) of what a block reads as text: the package's text files, a table's header."""
    for f in sorted(task_dir.iterdir()):
        if not f.is_file() or (only and f.name not in only):
            continue
        suffix = f.suffix.lower()
        if suffix in _BINARY_SUFFIXES:
            continue
        try:
            if f.stat().st_size > _MAX_TEXT:
                continue
            if suffix in (".csv", ".tsv"):
                with f.open(encoding="utf-8", errors="ignore") as fh:
                    head = fh.readline()
                if not all(_is_number(c) for c in re.split(r"[,\t;\s]+", head.strip()) if c):
                    yield f"{f.name} (header)", head
                continue
            yield f.name, f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue


def printed_copies(task_dir: Path, ref, only: tuple[str, ...] = ()) -> list[str]:
    """Which public texts print the reference.

    A text is flagged when at least ten of its precise numbers (four or more
    significant digits) are reference values at the precision they are
    printed to, and they are at least half of its precise numbers: a dumped
    table trips it, however it was printed; a handful of problem constants
    or a baseline's hyperparameters do not.
    """
    import numpy as np

    vals = np.asarray(ref, dtype=np.float64).ravel()
    vals = np.unique(vals[np.isfinite(vals)])
    found = []
    for name, text in _texts(task_dir, only):
        precise, hits = printed_values(text, vals)
        if hits >= PRINTED_MIN and hits >= 0.5 * precise:
            found.append(f"{name}: {hits} of its {precise} precise numbers are reference values")
    return found


PRINTED_MIN = 10


def _input_columns(evalsrc: str) -> int:
    import ast

    try:
        return int(ast.literal_eval(re.search(r"^INPUT_COLUMNS\s*=\s*(.+)$", evalsrc,
                                              re.MULTILINE).group(1).split("#")[0]))
    except (AttributeError, ValueError, SyntaxError, TypeError):
        return 0


def _check_no_answers_published(rep: ValidationReport, task_dir: Path, arr, evalsrc: str) -> None:
    """No public file of the task carries the reference: not as data, not as printed numbers.

    Only the package itself is read — what a block is handed. The authoring
    records beside it are the framework's and the author's (an author never
    sees the answers), and a training log is full of numbers that are not.
    """
    import numpy as np

    ref = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
    copies = published_copies(task_dir, ref, _input_columns(evalsrc))
    rep.check("no public data file holds the reference", not copies, "; ".join(copies[:3]))
    printed = printed_copies(task_dir, ref)
    rep.check("no public text prints reference values", not printed, "; ".join(printed[:3]))


_REF_VALUES = re.compile(r'^REF_VALUES\s*=\s*"([^"]+)"', re.MULTILINE)


def _check_reference(rep: ValidationReport, task_dir: Path, evalsrc: str) -> None:
    """The answers line up with the grid, row for row, and are real numbers.

    Only checked when both halves are present — their absence is
    `_check_data`'s finding, not this one's.
    """
    m = _REF_VALUES.search(evalsrc)
    ref = private.locate(task_dir) / (m.group(1) if m else "ref_values.npy")
    grid = task_dir / "eval_grid.csv"
    if not (ref.is_file() and grid.is_file()):
        return
    import numpy as np

    with grid.open("rb") as fh:
        rows = sum(1 for _ in fh) - 1
    try:
        arr = np.load(ref, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as e:
        rep.check("reference is a readable .npy array", False, type(e).__name__)
        return
    rep.check(
        "reference has one row per eval_grid.csv row",
        arr.ndim >= 1 and arr.shape[0] == rows,
        f"{arr.shape[0] if arr.ndim else '?'} vs {rows}",
    )
    rep.check(
        "reference is finite float64",
        arr.dtype == np.float64 and bool(np.isfinite(arr).all()),
        str(arr.dtype),
    )
    if arr.ndim >= 1 and arr.shape[0] == rows:
        _check_no_answers_published(rep, task_dir, np.asarray(arr), evalsrc)


def _ref_name(evalsrc: str) -> str:
    m = _REF_VALUES.search(evalsrc or "")
    return m.group(1) if m else "ref_values.npy"


def grid_problem(grid: Path) -> str | None:
    """What is wrong with an eval grid as a file, or None.

    A header naming the columns, then rows of plain numbers, every row as
    wide as the header and every value finite — what `eval.py` (np.loadtxt)
    and the score service read. Said in the framework's words: the file may
    have been written by an agent that saw the answers.
    """
    import numpy as np

    try:
        with Path(grid).open(encoding="utf-8") as fh:
            header = [h.strip() for h in fh.readline().split(",")]
    except (OSError, UnicodeDecodeError):
        return "eval_grid.csv is not readable UTF-8 text"
    if not header or not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", h) for h in header):
        return "eval_grid.csv must start with a header line of column names"
    try:
        rows = np.loadtxt(grid, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except ValueError:
        return ("eval_grid.csv rows must be plain decimal numbers (no text, no Python reprs "
                "such as np.float64(...)), as many per row as the header names")
    if rows.shape[0] < 2:
        return "eval_grid.csv has fewer than two rows"
    if rows.shape[1] != len(header):
        return f"eval_grid.csv rows have {rows.shape[1]} values for {len(header)} header names"
    if not np.isfinite(rows).all():
        return "eval_grid.csv holds non-finite values"
    return None


def _check_data(rep: ValidationReport, task_dir: Path, reference: bool = True,
                evalsrc: str = "") -> None:
    """The public/private split the score service rests on."""
    grid = task_dir / "eval_grid.csv"
    rep.check("ships eval_grid.csv (public: where to predict)", grid.is_file())
    if grid.is_file():
        why = grid_problem(grid)
        rep.check("eval_grid.csv is a header and rows of plain numbers", why is None, why or "")
    if not reference:
        return
    # A deployment with an external private directory keeps the answers
    # there, not in the library; check the place the score service will read.
    # The answers themselves, not merely a directory with something in it: a
    # private/ holding only logs scores nothing.
    home = private.locate(task_dir)
    shown = f"{PRIVATE_DIR}/" if home == task_dir / PRIVATE_DIR else f"{home}/"
    ref = _ref_name(evalsrc)
    rep.check(
        f"ships the scoring truth under {PRIVATE_DIR}/",
        (home / ref).is_file(),
        f"{shown}{ref}" + ("" if (home / ref).is_file() else " is missing"),
    )


def _check_problem(rep: ValidationReport, problem: str) -> None:
    title = problem.splitlines()[0] if problem.strip() else ""
    rep.check(
        "title is '# <ABBREV> — <full name>'",
        bool(re.match(r"^#\s+[A-Z0-9]+\s+—\s+\S", title)),
        title[:60],
    )

    found = [s for s in REQUIRED_SECTIONS if s in problem]
    rep.check(
        "all seven sections present",
        len(found) == len(REQUIRED_SECTIONS),
        f"{len(found)}/{len(REQUIRED_SECTIONS)}",
    )
    order = [problem.index(s) for s in REQUIRED_SECTIONS if s in problem]
    rep.check("sections in canonical order", order == sorted(order))

    extra = [
        line.strip()
        for line in problem.splitlines()
        if line.startswith("## ") and line.strip() not in REQUIRED_SECTIONS
    ]
    rep.check("no extra top-level sections", not extra, ", ".join(extra))
    rep.check(
        "no '###' subsections",
        "\n### " not in problem,
        f"{problem.count(chr(10) + '### ')} found",
    )
    rep.check("intro paragraph is verbatim", INTRO in _norm(problem))
    rep.check(
        "environment section is verbatim",
        any(body in _norm(problem) for body in ENVIRONMENT_BODIES),
    )
    # normalised: the corpus wraps prose, so "Lower is\nbetter." is the same
    # sentence as "Lower is better."
    flat = _norm(problem)
    rep.check(
        "scoring states rRMSE, lower is better",
        "rRMSE" in flat and "Lower is better" in flat,
    )


def _check_baseline(rep: ValidationReport, baseline: str) -> None:
    has = lambda pat: re.search(pat, baseline, re.MULTILINE) is not None  # noqa: E731
    rep.check("defines train()", has(r"^def train\("))
    rep.check("defines predict_fn()", has(r"^def predict_fn\("))
    rep.check("eval gate present", "FORGE_EVAL_TOKEN" in baseline)
    rep.check(
        "no RNG in the dataset path",
        "default_rng" not in baseline and "np.random" not in baseline,
    )
    rep.check("frozen header marker present", "DO NOT MODIFY" in baseline)


def _check_eval(rep: ValidationReport, evalsrc: str) -> None:
    lit = _TRAIN_TIME_LITERAL.search(evalsrc)
    smoke = _SMOKE_TRAIN_TIME.search(evalsrc)
    rep.check("declares TRAIN_TIME_LITERAL", lit is not None)
    rep.check("declares SMOKE_TRAIN_TIME", smoke is not None)
    if lit and smoke:
        rep.check(
            "smoke replacement has the same length",
            len(lit.group(1)) == len(smoke.group(1)),
            f"{len(lit.group(1))} vs {len(smoke.group(1))}",
        )
    rep.check("reads FORGE_WALL_BUDGET", "FORGE_WALL_BUDGET" in evalsrc)
    rep.check("logs the rRMSE field", "rRMSE" in evalsrc)
    # A task on the shared machinery gets the budget counter and the queue
    # client from it; it still has to say which grid columns and fields it
    # files, because those are the parts of the protocol that are its own.
    kit = KIT_MODULE in evalsrc
    rep.check(
        "maintains the .budget counter",
        kit or ".budget" in evalsrc,
        f"via {KIT_MODULE}" if kit else "",
    )
    # The scoring split: the half that executes candidate code must not be
    # the half that reads the answers.
    rep.check(
        "defines score_predictions() for the score service",
        re.search(r"^def score_predictions\(", evalsrc, re.MULTILINE) is not None,
    )
    rep.check(
        "submits predictions instead of scoring inline",
        kit or "submit_score" in evalsrc,
        f"via {KIT_MODULE}" if kit else "",
    )
    if kit:
        for name in ("GRID_CSV", "INPUT_COLUMNS", "FIELDS", "TRAIN_WALL_S", "SMOKE_WALL_S",
                     "EVAL_WALL_S", "BUDGET"):
            rep.check(
                f"declares {name} for {KIT_MODULE}",
                re.search(rf"^{name}\s*=", evalsrc, re.MULTILINE) is not None,
            )


def _check_agreement(
    rep: ValidationReport, problem: str, baseline: str, evalsrc: str
) -> None:
    """Cross-file agreement — where task swaps historically went wrong."""
    lit = _TRAIN_TIME_LITERAL.search(evalsrc)
    if lit:
        rep.check(
            "TRAIN_TIME literal appears exactly once in baseline",
            baseline.count(lit.group(1)) == 1,
            f"'{lit.group(1)}' x{baseline.count(lit.group(1))}",
        )
        want = lit.group(1).split("=")[-1].strip()
        rep.check(
            "problem.md states the same TRAIN_TIME",
            f"TRAIN_TIME = {want}" in _norm(problem),
            want,
        )
    m = _TRAIN_TIME_VALUE.search(problem.split("## Time budget")[-1]) if problem else None
    rep.check("time budget section quotes a number", m is not None, m.group(1) if m else "")


def train_time(task_dir: Path) -> float | None:
    """Per-run wall in seconds, read from `eval.py`'s frozen-header literal."""
    m = _TRAIN_TIME_LITERAL.search((task_dir / "eval.py").read_text(encoding="utf-8"))
    if not m:
        return None
    try:
        return float(m.group(1).split("=")[-1].strip())
    except ValueError:
        return None


def wall_budget(task_dir: Path) -> float | None:
    """Per-block GPU budget default declared by the task's `eval.py`."""
    m = _BUDGET_DEFAULT.search((task_dir / "eval.py").read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None
