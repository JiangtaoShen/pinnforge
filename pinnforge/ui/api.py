"""What the dashboard shows, assembled from the files a workspace already keeps.

Every answer here is built from a known file in a known place — a run's
`state.json`, its `evals.jsonl`s, its summaries, its `.viz/` — and never from
a path a request names. Names in a request (a run id, a block id, a task) are
checked against a strict pattern before they touch the filesystem, and every
file is opened through `safe.open_regular`: a block can plant a symlink, a
hard link or a FIFO in its own workspace, and the dashboard must neither be
the thing that follows it into `private/` nor hang on it.

A block can also write whatever it likes into those files, so nothing read
from them is trusted to have the shape the framework gave it: numbers are
normalised to finite floats (or None), strings to capped strings, maps that
are not maps to empty ones, and only whitelisted fields of `run.yaml` are
served. Agent-written text passes through `feeds.redact`.

Nothing here reads `tasks/*/private/`, a run's `.scoreq/`, or an authoring
agent's private log. The only facts about a reference that reach a page are
the scalars in `.authoring/reference.json`.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import re
import stat
import subprocess
import time
from pathlib import Path

from pinnforge import __version__, gpus, oscompat, paths
from pinnforge.run import lifecycle
from pinnforge.ui import safe

logger = logging.getLogger(__name__)

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
BLOCK = re.compile(r"^b\d{2,3}$", re.ASCII)
TEXT_CAP = 200_000
JSONL_CAP = 20_000_000
STR_CAP = 300
NPZ_CAP = 50_000_000
NPZ_MEMBER_CAP = 8 * 2**20  # one archived array is ~80 kB; a few MB is already absurd
NPZ_MAX_MEMBERS = 64


class NotFound(LookupError):
    pass


def _name(value: str, pattern=NAME) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value or ""):
        raise NotFound(value)
    return value


# Set by the server: the workspace every served file must resolve inside.
ROOT_REAL: str | None = None
FORBIDDEN = safe.FORBIDDEN


def _allowed(path: Path) -> bool:
    """Inside the workspace, and nowhere near the answers — after every link."""
    return safe.allowed(path, ROOT_REAL)


def _read_bytes(path: Path, cap: int = TEXT_CAP) -> bytes | None:
    """A regular, singly linked file's bytes, never through a link; else None."""
    return safe.read_bytes(path, cap, ROOT_REAL)


def _regular(path: Path) -> bool:
    return safe.is_regular(path, ROOT_REAL)


def _read_text(path: Path, cap: int = TEXT_CAP) -> str | None:
    data = _read_bytes(path, cap)
    return None if data is None else data.decode("utf-8", errors="replace")


def _json(path: Path):
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict]:
    text = _read_text(path, cap=JSONL_CAP)
    out = []
    for line in (text or "").splitlines():
        try:
            r = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


_LOADER: list = []


def _yaml_loader():
    """A SafeLoader that refuses aliases: `run.yaml` is block-writable, and a
    few nested aliases expand into billions of nodes."""
    if not _LOADER:
        import yaml

        class NoAliasLoader(yaml.SafeLoader):
            def compose_node(self, parent, index):
                if self.check_event(yaml.AliasEvent):
                    raise yaml.composer.ComposerError(
                        None, None, "YAML aliases are not accepted here",
                        self.peek_event().start_mark)
                return super().compose_node(parent, index)

        _LOADER.append(NoAliasLoader)
    return _LOADER[0]


def _yaml(path: Path) -> dict:
    import yaml

    text = _read_text(path)
    try:
        data = yaml.load(text, Loader=_yaml_loader()) if text else None
    except (yaml.YAMLError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


# ── normalising what a block may have written ──


def _num(value) -> float | None:
    """A finite float, or None — a string, a bool, NaN or infinity is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(value, default: int = 0) -> int:
    f = _num(value)
    return int(f) if f is not None and abs(f) < 2**53 else default


def _str(value, default: str = "", cap: int = STR_CAP) -> str:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return str(value) if _num(value) is not None else default
    return value[:cap] if isinstance(value, str) else default


def _gpu_list(value) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(g) for g in value[:64]
            if isinstance(g, int) and not isinstance(g, bool) and 0 <= g < 4096]


def _redacted(value, cap: int) -> str | None:
    from pinnforge.ui import feeds

    if value is None or value == "" or value is False:
        return None
    return feeds.redact(str(value)[: cap * 4])[:cap]


# ─────────────────────────────── gpus ───────────────────────────────

_gpu_cache: dict = {"t": 0.0, "data": []}


def gpu_stats() -> list[dict]:
    """Utilisation, memory and temperature per GPU, at most once a second."""
    now = time.time()
    if now - _gpu_cache["t"] < 1.0:
        return _gpu_cache["data"]
    data = []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
             "power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5, check=False,
        )
        for line in out.stdout.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6 or not p[0].isdigit():
                continue

            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None

            data.append({"index": int(p[0]), "name": p[1], "util": num(p[2]),
                         "mem_used": num(p[3]), "mem_total": num(p[4]),
                         "temp": num(p[5]), "power": num(p[6]) if len(p) > 6 else None})
    except (OSError, subprocess.SubprocessError):
        data = []
    _gpu_cache.update(t=now, data=data)
    return data


def _pid_alive(pid) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid < 2**31:
        return False
    return oscompat.pid_alive(pid)


def _leases(pool: list[int]) -> dict[int, str]:
    """Which run holds each card, from the machine-wide lease files.

    Anyone on the machine can write there, so each lease is read like a
    block's file — no links, no FIFOs, at most a few kB — and one that is not
    a JSON object naming a live pid is ignored.
    """
    held = {}
    for g in pool:
        names = []
        try:
            found = sorted(lifecycle.LEASE_DIR.glob(f"gpu{int(g)}.*.json"))
        except (OSError, TypeError, ValueError):
            found = []
        for p in found:
            raw = safe.read_bytes(p, 4096, check_path=False)
            try:
                info = json.loads(raw) if raw else None
            except (ValueError, RecursionError):
                continue
            if not isinstance(info, dict) or not _pid_alive(info.get("pid")):
                continue
            where = info.get("run")
            name = Path(where[:4096]).name if isinstance(where, str) else ""
            if NAME.fullmatch(name):
                names.append(name)
        if names:
            held[g] = ", ".join(names)
    return held


# ─────────────────────────────── runs ───────────────────────────────


def _run_dir(root: Path, run_id: str) -> Path:
    run = paths.runs_dir(root) / _name(run_id)
    if run.is_symlink() or not (run / "run.yaml").is_file():
        raise NotFound(run_id)
    return run


def _live(run: Path) -> bool:
    """Is somebody driving `run`? (`lifecycle.holder`, but never hanging on a FIFO.)"""
    fd = safe.open_regular(run / lifecycle.LOCK_FILE, ROOT_REAL, single_link=False)
    if fd is None:
        return False
    try:
        oscompat.lock(fd, block=False)
    except BlockingIOError:
        return True
    except OSError:
        # A filesystem without flock (some NFS): fall back to the recorded pid.
        try:
            with os.fdopen(os.dup(fd), "rb") as fh:
                info = json.loads(fh.read(4096) or b"{}")
        except (OSError, ValueError, RecursionError):
            return False
        return isinstance(info, dict) and _pid_alive(info.get("pid"))
    else:
        oscompat.unlock(fd)
        return False
    finally:
        os.close(fd)


def _records(run: Path, block: str) -> list[dict]:
    return _jsonl(run / "blocks" / block / "evals.jsonl")


def _block_ids(run: Path) -> list[str]:
    d = run / "blocks"
    if d.is_symlink() or not d.is_dir():
        return []
    try:
        return sorted(p.name for p in d.iterdir()
                      if BLOCK.fullmatch(p.name) and not p.is_symlink() and p.is_dir())
    except OSError:
        return []


def _usage(run: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in _jsonl(run / "blocks" / "run_usage.jsonl"):
        b = _str(r.get("block"), cap=16)
        if not BLOCK.fullmatch(b):
            continue
        cur = out.setdefault(b, {"tokens": 0, "cost_usd": 0.0, "tool_uses": 0, "segments": 0,
                                 "duration_ms": 0, "model": ""})
        cur["tokens"] += _int(r.get("tokens"))
        cur["tool_uses"] += _int(r.get("tool_uses"))
        cur["cost_usd"] += _num(r.get("cost_usd")) or 0.0
        cur["duration_ms"] += _int(r.get("duration_ms"))
        cur["segments"] += 1
        cur["model"] = _str(r.get("model_resolved")) or _str(r.get("model")) or cur["model"]
    for cur in out.values():  # sums of finite numbers can still overflow to inf
        cur["cost_usd"] = _num(cur["cost_usd"]) or 0.0
    return out


def _wall(r: dict) -> float:
    return _num(r.get("wall_s")) or _num(r.get("train_s")) or 0.0


def _spent(recs: list[dict], budget_file: Path) -> float:
    spent = sum(_wall(r) for r in recs if not r.get("smoke"))
    text = _read_text(budget_file, 64)
    if text:
        with contextlib.suppress(ValueError):
            charged = float(text.strip() or 0)
            if math.isfinite(charged):
                spent = max(spent, charged)
    return spent if math.isfinite(spent) else 0.0


def _state(run: Path) -> tuple[dict, dict]:
    """`state.json`, and its per-block map — each a dict whatever the file holds."""
    state = _dict(_json(run / "state.json"))
    blocks = {b: _dict(s) for b, s in _dict(state.get("blocks")).items() if isinstance(b, str)}
    return state, blocks


def _agent(cfg: dict) -> dict:
    agent = _dict(cfg.get("agent"))
    return {"runtime": _str(agent.get("runtime")), "model": _str(agent.get("model"))}


def _hw(cfg: dict) -> dict:
    out = {}
    for k, v in list(_dict(cfg.get("hw")).items())[:32]:
        if not isinstance(k, str):
            continue
        if isinstance(v, str):
            out[k[:64]] = v[:200]
        elif isinstance(v, bool):
            out[k[:64]] = v
        elif _num(v) is not None:
            out[k[:64]] = _num(v)
    return out


def _scores(recs: list[dict]) -> list[float]:
    """The scored (not smoke, not diagnostic) rRMSEs among `recs`, finite floats only."""
    out = []
    for r in recs:
        if r.get("smoke") or r.get("diag"):
            continue
        s = _num(r.get("rRMSE"))
        if s is not None:
            out.append(s)
    return out


def run_summary(root: Path, run: Path) -> dict:
    cfg = _yaml(run / "run.yaml")
    state, bstates = _state(run)
    best = None
    best_block = None
    n_evals = 0
    anchor = None
    scored = []
    for b in _block_ids(run):
        for r in _records(run, b):
            score = _num(r.get("rRMSE"))
            if r.get("smoke") or r.get("diag") or score is None:
                continue
            n_evals += 1
            if b == "b00":
                anchor = score if anchor is None else min(anchor, score)
                continue
            scored.append((_str(r.get("ts")), score))
            if best is None or score < best:
                best, best_block = score, b
    trend, running = [], anchor
    for _, score in sorted(scored):
        running = score if running is None else min(running, score)
        trend.append(running)
    if anchor is not None:
        trend.insert(0, anchor)
    if len(trend) > 48:
        step = len(trend) / 48
        trend = [trend[int(i * step)] for i in range(47)] + [trend[-1]]
    blocks = [b for b in _block_ids(run) if b != "b00"]
    done = sum(1 for b in blocks if bstates.get(b, {}).get("status") in ("done", "short"))
    agent = _agent(cfg)
    return {
        "id": run.name,
        "task": _str(cfg.get("task")),
        "status": _str(state.get("status"), "?"),
        "live": _live(run),
        "blocks_total": _int(cfg.get("blocks")),
        "blocks_done": done,
        "blocks_started": len(blocks),
        "anchor": anchor,
        "best": best,
        "best_block": best_block,
        "trend": trend,
        "evals": n_evals,
        "agent": f"{agent['runtime']} / {agent['model'] or 'default'}",
        "gpus": _gpu_list(_dict(cfg.get("sandbox")).get("gpus")),
        "created": _str(state.get("created_at")),
        "updated": _str(state.get("updated_at")),
    }


def _unreadable(run_id: str) -> dict:
    """The card of a run whose record could not be read: shown, not fatal."""
    return {"id": run_id, "task": "", "status": "unreadable", "live": False,
            "blocks_total": 0, "blocks_done": 0, "blocks_started": 0, "anchor": None,
            "best": None, "best_block": None, "trend": [], "evals": 0, "agent": "",
            "gpus": [], "created": "", "updated": "", "unreadable": True}


def _gpu_slot(value):
    n = _num(value)
    if n is not None:
        return int(n) if abs(n) < 2**31 else None
    return _str(value, cap=32) or None


def run_detail(root: Path, run_id: str) -> dict:
    run = _run_dir(root, run_id)
    cfg = _yaml(run / "run.yaml")
    _, bstates = _state(run)
    usage = _usage(run)
    sandbox = _dict(cfg.get("sandbox"))
    budget = _num(sandbox.get("wall_budget_s")) or 0.0
    evals = []
    cum = 0.0
    all_recs = []
    for b in _block_ids(run):
        for r in _records(run, b):
            all_recs.append((b, r))
    all_recs.sort(key=lambda br: _str(br[1].get("ts")))
    for b, r in all_recs:
        if r.get("smoke"):
            continue
        cum += _wall(r)
        evals.append({
            "block": b,
            "candidate": Path(_str(r.get("candidate"), cap=4096)).name[:200],
            "rRMSE": _num(r.get("rRMSE")),
            "wall_s": _num(r.get("wall_s")),
            "cum_wall": round(cum, 1) if math.isfinite(cum) else None,
            "ts": _str(r.get("ts"), cap=64) or None,
            "diag": bool(r.get("diag")),
            "error": _redacted(r.get("error"), 300),
            "gpu": _gpu_slot(r.get("gpu")),
        })
    blocks = []
    for b in _block_ids(run):
        recs = _records(run, b)
        scores = _scores(recs)
        bs = bstates.get(b, {})
        u = usage.get(b, {})
        blocks.append({
            "id": b,
            "status": _str(bs.get("status")) or (
                _anchor_status(run, bool(scores)) if b == "b00" else "—"),
            "spent": round(_spent(recs, run / "blocks" / b / ".budget"), 1),
            "budget": budget if b != "b00" else None,
            "evals": len(scores),
            "best": min(scores, default=None),
            "summary": _regular(run / "blocks" / "kb2" / f"{b}.md"),
            "model": _str(bs.get("model_resolved")) or u.get("model") or _str(bs.get("model")),
            "tokens": u.get("tokens", 0),
            "cost_usd": round(u.get("cost_usd", 0.0), 4),
            "tool_uses": u.get("tool_uses", 0),
            "duration_ms": u.get("duration_ms", 0),
            "started": _str(bs.get("started_at"), cap=64) or None,
            "completed": _str(bs.get("completed_at"), cap=64) or None,
            "exit_reason": _redacted(bs.get("exit_reason"), STR_CAP) or "",
            "integrity": _integrity(run, b),
        })
    out = run_summary(root, run)
    out.update({
        "config": {
            "agent": _agent(cfg),
            "gpus": _gpu_list(sandbox.get("gpus")),
            "wall_budget_s": budget,
            "hw": _hw(cfg),
            "task": _str(cfg.get("task")),
            "blocks": _int(cfg.get("blocks")),
            # what the header's chips show on hover
            "path": str(run),
            "task_title": _problem_title(run / "task" / "problem.md"),
            "train_s": _train_time(run),
            "task_path": str(task_dir) if NAME.fullmatch(_str(cfg.get("task"))) and (
                task_dir := Path(root) / "tasks" / _str(cfg.get("task"))).is_dir() else "",
        },
        "blocks": blocks,
        "evals": evals,
        "frames": _frames(run),
        "grid": (grid := _grid(run)),
        "view": _view(run, grid),
    })
    return out


def _anchor_status(run: Path, scored: bool) -> str:
    """`lifecycle.anchor_status`, through `_live` (which never hangs on a FIFO)."""
    if scored:
        return "done"
    return "measuring" if _live(run) else "missing"


def _integrity(run: Path, block: str) -> str:
    d = run / ".integrity"
    try:
        files = sorted(d.glob(f"{block}.*.result.json")) if (
            d.is_dir() and not d.is_symlink()) else []
    except OSError:
        files = []
    verdicts = [_dict(_json(f)) for f in files]
    if not verdicts:
        return "unchecked"
    return "clean" if all(v.get("clean", True) for v in verdicts) else "violated"


def _frames(run: Path) -> list[dict]:
    out = []
    for r in _jsonl(run / ".viz" / "index.jsonl"):
        frame = _num(r.get("frame"))
        block = _str(r.get("block"), cap=16)
        if frame is None or frame < 0 or frame != int(frame) or frame >= 10**7:
            continue
        out.append({
            "frame": int(frame),
            "block": block if BLOCK.fullmatch(block) else "",
            "candidate": _str(r.get("candidate"), cap=200),
            "rRMSE": _num(r.get("rRMSE")),
            "run_best": r.get("run_best") is True,
        })
    return out


def _grid(run: Path) -> dict | None:
    g = _json(run / ".viz" / "grid.json")
    if not isinstance(g, dict):
        return None

    def names(value) -> list[str]:
        return [v[:64] for v in value[:32] if isinstance(v, str)] if isinstance(value, list) else []

    return {"coords": names(g.get("coords")), "fields": names(g.get("fields")),
            "points": _int(g.get("points")), "of": _int(g.get("of")),
            "dims": min(max(_int(g.get("dims"), 2), 1), 3)}


def _view(run: Path, grid: dict | None) -> dict | None:
    """The task's view.json as the run froze it, checked against what is archived.

    The run directory is the block's to write, so the file is read like any
    other untrusted one and kept only if it is a valid view of the archived
    coordinates and output heads; otherwise the page draws its default.
    """
    from pinnforge.task import view
    from pinnforge.ui import feeds

    if not grid:
        return None
    raw = _read_bytes(run / "task" / view.VIEW_FILE, cap=view.MAX_BYTES + 1)
    if raw is None or len(raw) > view.MAX_BYTES:
        return None
    try:
        spec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    shown, _ = view.check(spec, grid.get("coords") or [], grid.get("fields") or [])
    if shown:
        for panel in shown["panels"]:
            panel["title"] = feeds.redact(panel["title"])
    return shown


def summary_md(root: Path, run_id: str, block: str) -> dict:
    from pinnforge.ui import feeds

    run = _run_dir(root, run_id)
    text = _read_text(run / "blocks" / "kb2" / f"{_name(block, BLOCK)}.md")
    if text is None:
        raise NotFound(block)
    return {"block": block, "markdown": feeds.redact(text)}


def _log_index(p: Path) -> int:
    part = p.name.split(".")[1]
    return int(part) if part.isascii() and part.isdigit() and len(part) < 10 else -1


def activity(root: Path, run_id: str, block: str | None, after: int | None) -> dict:
    """The newest events of a block's agent, from its latest transcript."""
    from pinnforge.ui import feeds

    run = _run_dir(root, run_id)
    cfg = _yaml(run / "run.yaml")
    runtime = _agent(cfg)["runtime"]
    if block is None:
        _, bstates = _state(run)
        running = [b for b, s in bstates.items()
                   if BLOCK.fullmatch(b) and s.get("status") == "running"]
        ids = [b for b in _block_ids(run) if b != "b00"]
        block = running[-1] if running else (ids[-1] if ids else None)
    if block is None:
        return {"block": None, "events": [], "offset": 0, "log": None}
    _name(block, BLOCK)
    logs_dir = run / "logs"
    try:
        logs = sorted((p for p in logs_dir.glob(f"{block}.*.log") if not p.is_symlink()),
                      key=_log_index) if not logs_dir.is_symlink() else []
    except OSError:
        logs = []
    if not logs:
        return {"block": block, "events": [], "offset": 0, "log": None}
    log = logs[-1]
    if not _allowed(log):
        return {"block": block, "events": [], "offset": 0, "log": None}
    events, offset = feeds.tail(log, after, runtime, base=str(run.resolve()),
                                root_real=ROOT_REAL)
    return {"block": block, "log": log.name, "events": events, "offset": offset}


def _npz(raw: bytes, max_members: int = NPZ_MAX_MEMBERS) -> dict:
    """The arrays of an `.npz`, refusing anything that would expand past a few MB.

    Each member's declared size is checked before it is read (the zip reader
    never inflates past it), and each `.npy` header's shape before any array
    is allocated: a block can write `.viz/`, and a small file can declare a
    huge array.
    """
    import zipfile

    import numpy as np
    from numpy.lib import format as npf

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as e:
        raise NotFound("not an archive") from e
    out = {}
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > max_members:
            raise NotFound("too many members")
        for info in infos:
            if info.file_size > NPZ_MEMBER_CAP or not info.filename.endswith(".npy"):
                raise NotFound("member too large")
        for info in infos:
            try:
                data = zf.read(info)
                if len(data) > NPZ_MEMBER_CAP:
                    raise NotFound("member too large")
                fp = io.BytesIO(data)
                version = npf.read_magic(fp)
                if version == (1, 0):
                    shape, fortran, dtype = npf.read_array_header_1_0(fp)
                elif version == (2, 0):
                    shape, fortran, dtype = npf.read_array_header_2_0(fp)
                else:
                    raise NotFound("unknown npy version")
                if dtype.hasobject or dtype.kind not in "biuf":
                    raise NotFound("not a numeric array")
                count = math.prod(shape)
                if count * dtype.itemsize > len(data) - fp.tell():
                    raise NotFound("array larger than its data")
                arr = np.frombuffer(data, dtype=dtype, count=count, offset=fp.tell())
                arr = arr.reshape(shape, order="F" if fortran else "C")
            except NotFound:
                raise
            except Exception as e:  # a malformed member is a missing frame
                raise NotFound("malformed member") from e
            out[info.filename[: -len(".npy")]] = arr
    return out


def frame(root: Path, run_id: str, n: int) -> dict:
    """One archived prediction, with the grid subsample, for the field view."""
    import numpy as np

    from pinnforge.ui import viz

    run = _run_dir(root, run_id)
    grid_raw = _read_bytes(run / ".viz" / "grid.npz", cap=NPZ_CAP)
    frame_raw = _read_bytes(run / ".viz" / "frames" / f"{int(n):05d}.npz", cap=NPZ_CAP)
    if grid_raw is None or frame_raw is None:
        raise NotFound(str(n))
    grid = _npz(grid_raw)
    if "coords" not in grid:
        raise NotFound(str(n))
    coords = np.asarray(grid["coords"], dtype=np.float64)
    if coords.ndim != 2 or not 0 < coords.shape[0] <= viz.MAX_POINTS or not (
        1 <= coords.shape[1] <= 3
    ):
        raise NotFound(str(n))
    fields = _npz(frame_raw)
    npts = coords.shape[0]
    coords = np.where(np.isfinite(coords), coords, 0.0)
    out_fields = {}
    for k, v in list(fields.items())[:32]:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1 or v.shape[0] != npts:
            continue
        v = np.where(np.isfinite(v), v, 0.0)
        out_fields[k[:64]] = {
            "min": float(v.min()), "max": float(v.max()),
            "values": [round(float(x), 5) for x in v],
        }
    return {
        "frame": int(n),
        "coords": [[round(float(x), 6) for x in coords[:, d]] for d in range(coords.shape[1])],
        "fields": out_fields,
    }


# ─────────────────────────────── tasks ───────────────────────────────


def _authoring_status(task_dir: Path) -> dict | None:
    """`authoring.status`, read like every other file here (no links, no FIFOs)."""
    from pinnforge.task import authoring

    path = authoring.state_path(task_dir)
    if not os.path.lexists(path):
        return None
    data = _json(path)
    if not isinstance(data, dict):
        return {"stage": "?", "status": "unreadable", "finalized": False}
    return data


def _train_time(run: Path) -> float | None:
    """How long each evaluation trains: the task's frozen `TRAIN_TIME`, from its `eval.py`."""
    from pinnforge.task import contract

    value = contract.train_time_in(_read_text(run / "task" / "eval.py") or "")
    return value if value is not None and math.isfinite(value) and value > 0 else None


def _problem_title(problem_md: Path) -> str:
    """The title line of a `problem.md`, without its `#`."""
    from pinnforge.ui import feeds

    problem = _read_text(problem_md, 4000)
    first = problem.splitlines()[0] if problem and problem.splitlines() else ""
    return feeds.redact(first.lstrip("# ").strip())[:200]


def task_summary(root: Path, task_dir: Path) -> dict:
    from pinnforge.task import authoring

    st = _authoring_status(task_dir)
    title = _problem_title(task_dir / "problem.md")
    anchor = _dict(_json(task_dir / ".anchor" / "meta.json"))
    out = {
        "name": task_dir.name,
        "title": title,
        "anchor": _num(anchor.get("rRMSE")),
        "ready": st is None or bool(st.get("finalized")),
    }
    if st is not None:
        mode = _str(st.get("mode"), "build")
        out["authoring"] = {
            "mode": _str(st.get("mode")) or None,
            "stage": _str(st.get("stage")) or None,
            "status": _str(st.get("status")) or None,
            "finalized": bool(st.get("finalized")),
            "plan": authoring.plan_for(mode),
            "stages": {
                k[:64]: {x: (_redacted(_dict(v).get(x), 500) if x == "note"
                             else _str(_dict(v).get(x)) or None)
                         for x in ("status", "started", "ended", "note")}
                for k, v in list(_dict(st.get("stages")).items())[:32] if isinstance(k, str)
            },
        }
        ref = _json(task_dir / authoring.STATE_DIR / "reference.json")
        if isinstance(ref, dict) and ref:
            from pinnforge.task import reference

            # Only what the framework computed — an older record may still
            # carry an agent's account of the reference, which is not served.
            out["reference"] = reference.public(ref)
    return out


def task_detail(root: Path, name: str) -> dict:
    from pinnforge.ui import feeds

    task_dir = paths.tasks_dir(root) / _name(name)
    if task_dir.is_symlink() or (
        not (task_dir / "problem.md").is_file() and not (task_dir / ".authoring").is_dir()
    ):
        raise NotFound(name)
    out = task_summary(root, task_dir)
    problem = _read_text(task_dir / "problem.md", 60_000)
    out["problem"] = feeds.redact(problem) if problem is not None else None
    return out


def author_activity(root: Path, name: str, after: int | None) -> dict:
    """The author agent's transcript — public: it never sees the answers."""
    from pinnforge.task import authoring
    from pinnforge.ui import feeds

    task_dir = paths.tasks_dir(root) / _name(name)
    st = _authoring_status(task_dir) or {}
    runtime = _str(_dict(_dict(st.get("inputs")).get("agent")).get("runtime"))
    log = task_dir / authoring.STATE_DIR / "author.log"
    if not _allowed(log):
        return {"events": [], "offset": after or 0}
    events, offset = feeds.tail(log, after, runtime, base=str(task_dir.resolve()),
                                root_real=ROOT_REAL)
    return {"events": events, "offset": offset}


# ─────────────────────────────── overview ───────────────────────────────


def overview(root: Path) -> dict:
    runs = []
    for run_id in reversed(paths.list_runs(root)):
        try:
            runs.append(run_summary(root, _run_dir(root, run_id)))
        except NotFound:
            continue
        except Exception:  # one unreadable run must not take the page down
            logger.warning("run %s unreadable", run_id, exc_info=True)
            runs.append(_unreadable(run_id))
    tasks = []
    tdir = paths.tasks_dir(root)
    if tdir.is_dir():
        for t in sorted(tdir.iterdir()):
            try:
                if t.is_dir() and not t.is_symlink() and NAME.fullmatch(t.name) and (
                    (t / "problem.md").is_file() or (t / ".authoring").is_dir()
                ):
                    tasks.append(task_summary(root, t))
            except Exception:
                logger.warning("task %s unreadable", t.name, exc_info=True)
                tasks.append({"name": t.name, "title": "", "anchor": None, "ready": False,
                              "authoring": {"mode": None, "stage": "?", "status": "unreadable",
                                            "finalized": False, "plan": [], "stages": {}}})
    pool = gpus.inventory()
    return {
        "app": "pinnforge",
        "version": __version__,
        "workspace": str(root),
        "name": Path(root).name,
        "gpus": [{"index": g.index, "name": g.name, "memory_mb": g.memory_mb} for g in pool],
        # the cards this workspace's runs may use; the others are shown, dimmed
        "pool": gpus.workspace_pool(root, inv=pool),
        "leases": _leases([g.index for g in pool]),
        "runs": runs,
        "tasks": tasks,
    }


def fingerprint(root: Path) -> dict[str, float]:
    """mtimes of everything the pages show — what the event stream watches.

    `lstat`, not `stat`: the mtime of whatever a planted link points at is
    not this workspace's business.
    """
    sig: dict[str, float] = {}

    def see(key: str, p: Path) -> None:
        with contextlib.suppress(OSError, ValueError):
            st = os.lstat(p)
            if stat.S_ISREG(st.st_mode):
                sig[key] = st.st_mtime

    for run_id in paths.list_runs(root):
        run = paths.runs_dir(root) / run_id
        see(f"run:{run_id}:state", run / "state.json")
        see(f"run:{run_id}:viz", run / ".viz" / "index.jsonl")
        for b in _block_ids(run):
            see(f"run:{run_id}:{b}", run / "blocks" / b / "evals.jsonl")
            see(f"run:{run_id}:{b}:budget", run / "blocks" / b / ".budget")
            see(f"run:{run_id}:{b}:kb2", run / "blocks" / "kb2" / f"{b}.md")
        logs = run / "logs"
        with contextlib.suppress(OSError):
            if logs.is_dir() and not logs.is_symlink():
                for f in logs.glob("b*.log"):
                    see(f"run:{run_id}:log:{f.name}", f)
    tdir = paths.tasks_dir(root)
    with contextlib.suppress(OSError):
        if tdir.is_dir():
            for t in tdir.iterdir():
                see(f"task:{t.name}", t / ".authoring" / "state.json")
                see(f"task:{t.name}:author", t / ".authoring" / "author.log")
                see(f"task:{t.name}:anchor", t / ".anchor" / "meta.json")
    return sig
