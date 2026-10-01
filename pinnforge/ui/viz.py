"""Snapshots of predicted fields, for the dashboard's field animation.

The score service already holds every prediction a run makes — it has to, to
score it. When a prediction sets a new best, for the run or for its block,
the service keeps a small copy under `runs/<id>/.viz/`: the field values at a
fixed subsample of the grid, as float16. The dashboard plays them back in
order, so an operator can watch the solution sharpen as the blocks improve.

Only predictions are kept — the block's own output. Never the reference, and
never an error map, which would be the reference by subtraction: the dashboard
is reachable from a block's shell (`curl localhost`), so nothing it serves may
be derived from the answers.

The grid subsample is written once per run, from the score service's own
snapshot of the task, not from anything a block could have edited.

The score service may run as a more privileged user than the blocks, and it
writes inside a directory the blocks can write to. So every write here goes
through directory descriptors opened without following links: `.viz/` and
`frames/` must be real directories owned by this process's user, files are
created exclusively under temporary names and renamed into place, and the
index is appended to only if it is a regular file with a single link. If
anything is not what the archiver made, archiving stops — with a warning,
never an exception into the daemon.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_POINTS = 40_000
VIZ_DIR = ".viz"
FRAMES_DIR = "frames"
INDEX = "index.jsonl"
INDEX_CAP = 20_000_000
MAX_FRAMES = 10**7

BLOCK = re.compile(r"^b\d{2,3}$", re.ASCII)
PART = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# an open refused for one of these means something was planted where we write
_PLANTED = {errno.ELOOP, errno.ENXIO, errno.EISDIR, errno.ENOTDIR, errno.EMLINK,
            errno.EACCES, errno.EPERM, errno.EEXIST}


class Refused(Exception):
    """Something under `.viz/` is not what the archiver made."""


def _open_dir(name: str, parent: int | None, *, create: bool, owned: bool) -> int:
    """A descriptor on directory `name` (under `parent`), never through a link."""
    if create:
        try:
            os.mkdir(name, 0o755, dir_fd=parent)
        except FileExistsError:
            pass
        except OSError as e:
            raise Refused(f"cannot create {name}: {e.strerror}") from None
    try:
        fd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=parent)
    except OSError as e:
        raise Refused(f"{name} is not a directory ({e.strerror})") from None
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode) or (owned and st.st_uid != os.geteuid()):
        os.close(fd)
        raise Refused(f"{name} is not a directory this process owns")
    return fd


def _write_atomic(dir_fd: int, name: str, write: Callable) -> None:
    """Create `name` in `dir_fd` whole: a fresh temp file, then a rename over it."""
    tmp = f".tmp-{secrets.token_hex(8)}-{name}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC, 0o644,
                     dir_fd=dir_fd)
    except OSError as e:
        if e.errno in _PLANTED:
            raise Refused(f"cannot create {tmp}: {e.strerror}") from None
        raise
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        # rename replaces whatever is at `name` — a planted link included — never
        # writing through it
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp, dir_fd=dir_fd)
        raise


def _append(dir_fd: int, name: str, text: str) -> None:
    """Append to `name` in `dir_fd` — only if it is a regular file with one link."""
    try:
        fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK
                     | _CLOEXEC, 0o644, dir_fd=dir_fd)
    except OSError as e:
        if e.errno in _PLANTED:
            raise Refused(f"cannot append to {name}: {e.strerror}") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise Refused(f"{name} is not a regular, singly linked file")
        data = text.encode("utf-8")
        while data:
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)


def _read_index(dir_fd: int) -> str:
    """The index's text ('' if there is none yet), read the same careful way."""
    try:
        fd = os.open(INDEX, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC, dir_fd=dir_fd)
    except FileNotFoundError:
        return ""
    except OSError as e:
        raise Refused(f"cannot read {INDEX}: {e.strerror}") from None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise Refused(f"{INDEX} is not a regular, singly linked file")
        return fh.read(INDEX_CAP).decode("utf-8", errors="replace")


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _block(value) -> str | None:
    return value if isinstance(value, str) and BLOCK.fullmatch(value) else None


def _candidate(value) -> str | None:
    """A relative path of plain names (`blocks/b01/b01_a.py`), '' for none, else None."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or len(value) > 1024:
        return None
    parts = value.split("/")
    if any(p in ("", ".", "..") or not PART.fullmatch(p) for p in parts):
        return None
    return value


def _seed(value):
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) < 2**63:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 18:
        return value
    return None


class Archiver:
    """Called by the score daemon after each successful score."""

    def __init__(self, run: Path, coords, input_columns: int, names: list[str],
                 step: int | None = None):
        """`coords`: the rows predictions are filed at (grid[predict_rows]).

        `step`: the coordinate column the view puts on a slider (time, or the
        sliced axis), if any — the sample then keeps whole slices of it.
        """
        import numpy as np

        self.run = Path(run)
        self.dir = self.run / VIZ_DIR
        self.lock = threading.Lock()
        self.disabled = False
        n = coords.shape[0]
        self.dims = min(int(input_columns), 3)
        self.idx = sample_rows(coords, self.dims, step)
        self.names = list(names)
        self.best_run: float | None = None
        self.best_block: dict[str, float] = {}
        self.count = 0
        sample = coords[self.idx, : self.dims].astype(np.float32)
        try:
            with self._dirs(frames=False) as (viz, _):
                _write_atomic(viz, "grid.npz",
                              lambda fh: np.savez(fh, coords=sample, n=np.array(n)))
                self._replay(_read_index(viz))
        except Refused as e:
            self._refuse(e)
        except Exception:
            logger.warning("field archive unavailable for %s", self.run, exc_info=True)
            self.disabled = True

    # ── where it writes ──

    @contextmanager
    def _dirs(self, frames: bool) -> Iterator[tuple[int, int | None]]:
        """(`.viz`, `frames` or None) as descriptors, each checked, closed after."""
        fds: list[int] = []
        try:
            run = _open_dir(str(self.run), None, create=False, owned=False)
            fds.append(run)
            viz = _open_dir(VIZ_DIR, run, create=True, owned=True)
            fds.append(viz)
            fr = None
            if frames:
                fr = _open_dir(FRAMES_DIR, viz, create=True, owned=True)
                fds.append(fr)
            yield viz, fr
        finally:
            for fd in fds:
                os.close(fd)

    def _refuse(self, why) -> None:
        self.disabled = True
        logger.warning("field archive disabled for %s: %s", self.run, why)

    def write_json(self, name: str, data: dict) -> None:
        """A small JSON file beside the frames (the grid's description)."""
        if self.disabled:
            return
        text = (json.dumps(data) + "\n").encode("utf-8")
        try:
            with self._dirs(frames=False) as (viz, _):
                _write_atomic(viz, name, lambda fh: fh.write(text))
        except Refused as e:
            self._refuse(e)
        except Exception:
            logger.warning("could not write %s for %s", name, self.run, exc_info=True)

    # ── what it keeps ──

    def _replay(self, text: str) -> None:
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if not isinstance(r, dict):
                continue
            frame = _num(r.get("frame"))
            if frame is not None and 0 <= frame < MAX_FRAMES:
                self.count = max(self.count, int(frame) + 1)
            s = _num(r.get("rRMSE"))
            b = _block(r.get("block"))
            if s is None or b is None:
                continue
            if self.best_run is None or s < self.best_run:
                self.best_run = s
            if b not in self.best_block or s < self.best_block[b]:
                self.best_block[b] = s

    def __call__(self, req: dict, result: dict, payload: Path) -> None:
        if self.disabled:
            return
        try:
            self._archive(req, result, payload)
        except Refused as e:
            self._refuse(e)
        except Exception:  # a frame not kept is a frame not kept; scoring goes on
            logger.warning("prediction not archived for %s", self.run, exc_info=True)

    def _load(self, payload: Path) -> dict | None:
        import numpy as np

        try:
            fd = os.open(payload, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC)
        except (OSError, ValueError):
            return None
        with os.fdopen(fd, "rb") as fh:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                return None
            with np.load(fh, allow_pickle=False) as data:
                return {
                    k: np.asarray(data[k], dtype=np.float64).reshape(-1)
                    for k in data.files
                    if k in self.names or not self.names
                }

    def _archive(self, req: dict, result: dict, payload: Path) -> None:
        import numpy as np

        if not isinstance(req, dict) or req.get("smoke"):
            return
        score = _num(result.get("rRMSE"))
        if score is None:
            return
        block = _block(req.get("block"))
        candidate = _candidate(req.get("candidate", ""))
        if block is None or candidate is None:
            return  # not a name the dashboard could show safely: no frame
        new_run = self.best_run is None or score < self.best_run
        new_block = block not in self.best_block or score < self.best_block[block]
        if not (new_run or new_block):
            return
        fields = self._load(payload)
        if not fields:
            return
        n = None
        for arr in fields.values():
            n = arr.shape[0] if n is None else n
        if n is None or n <= int(self.idx[-1]):
            return
        with self.lock, self._dirs(frames=True) as (viz, frames):
            frame = self.count
            sampled = {k: v[self.idx].astype(np.float16) for k, v in fields.items()}
            _write_atomic(frames, f"{frame:05d}.npz", lambda fh: np.savez(fh, **sampled))
            entry = {
                "frame": frame,
                "block": block,
                "candidate": candidate,
                "seed": _seed(req.get("seed")),
                "rRMSE": score,
                "run_best": new_run,
            }
            _append(viz, INDEX, json.dumps(entry) + "\n")
            self.count += 1
            if new_run:
                self.best_run = score
            if new_block:
                self.best_block[block] = score


MAX_STEPS = 12


def _thin(points, budget: int):
    """Indices of at most about `budget` of `points` (m × d), spread evenly.

    Points on a lattice — every combination of a few coordinate values, with
    holes allowed where a body sits — keep every k-th line in each direction,
    so they are still a lattice the page can draw as an image. Anything else
    keeps every k-th point.
    """
    import numpy as np

    m, d = points.shape
    if m <= budget:
        return np.arange(m)
    lines = [np.unique(points[:, j]) for j in range(d)]
    if float(np.prod([len(u) for u in lines], dtype=float)) <= 2 * m:
        # thin the direction with the most lines left, one step at a time
        k = [1] * d
        while m / float(np.prod(k)) > budget:
            j = max(range(d), key=lambda j: len(lines[j]) / k[j])
            k[j] += 1
        keep = np.ones(m, dtype=bool)
        for j, u in enumerate(lines):
            keep &= (np.searchsorted(u, points[:, j]) % k[j]) == 0
        return np.flatnonzero(keep)
    return np.arange(0, m, max(1, -(-m // budget)))


def sample_rows(coords, dims: int, step: int | None, limit: int = MAX_POINTS):
    """Which rows of the scored grid to keep, at most about `limit` of them.

    With a slider coordinate, up to MAX_STEPS of its values, evenly spaced,
    each kept as a slice thinned the same way, so every slice is the same set
    of points; a global stride over a (t, x, y) grid hit a different scatter
    in each slice and drew stripes.
    """
    import numpy as np

    pts = np.asarray(coords[:, :dims], dtype=np.float64)
    if step is None or not 0 <= step < dims:
        return _thin(pts, limit)
    space = [j for j in range(dims) if j != step]
    values = pts[:, step]
    steps = np.unique(values)
    count = min(len(steps), MAX_STEPS)
    picks = np.unique(np.linspace(0, len(steps) - 1, count).round().astype(int))
    per = max(1, limit // len(picks))
    kept = []
    for s in steps[picks]:
        rows = np.flatnonzero(values == s)
        kept.append(rows[_thin(pts[rows][:, space], per)] if space else rows)
    return np.sort(np.concatenate(kept))


def step_column(task_dir: Path, header: list[str], names: list[str]) -> int | None:
    """The coordinate a run's view puts on a slider, as a column index, or None.

    The task's view.json when it is valid; otherwise the dashboard's default
    for three coordinates — a time coordinate on the slider, else the last
    one sliced.
    """
    from pinnforge.task import view

    coords = header[:3]
    shown, _ = view.check(view.load(Path(task_dir) / view.VIEW_FILE), coords, names)
    if shown:
        name = shown.get("time") or (shown.get("slice") or {}).get("axis")
        return coords.index(name) if name in coords else None
    if len(coords) < 3:
        return None
    timed = [i for i, c in enumerate(coords) if c.lower() in ("t", "time")]
    return timed[0] if timed else len(coords) - 1


def archiver_for(run: Path, metric) -> Archiver | None:
    """An archiver for `run`, from the score daemon's snapshot of its task."""
    import numpy as np

    mod = getattr(metric, "module", None)
    if mod is None or not hasattr(mod, "GRID_CSV"):
        return None
    try:
        grid = np.loadtxt(mod.GRID_CSV, delimiter=",", skiprows=1, dtype=np.float64)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
        rows = mod.predict_rows(grid) if callable(getattr(mod, "predict_rows", None)) else (
            slice(None)
        )
        cols = int(getattr(mod, "INPUT_COLUMNS", min(grid.shape[1], 2)))
        names = list(getattr(mod, "FIELDS", ()) or ())
        with open(mod.GRID_CSV, encoding="utf-8") as fh:
            header = [h.strip() for h in fh.readline().split(",")]
        # the snapshot's own directory holds its view.json, beside its grid
        step = step_column(Path(mod.GRID_CSV).parent, header[: min(cols, 3)], names)
        arch = Archiver(run, grid[rows], cols, names, step=step)
        arch.write_json("grid.json", {
            "coords": header[: arch.dims],
            "fields": names,
            "points": len(arch.idx),
            "of": int(grid[rows].shape[0]),
            "dims": arch.dims,
        })
        return None if arch.disabled else arch
    except Exception:
        logger.warning("no field archive for %s", run, exc_info=True)
        return None
