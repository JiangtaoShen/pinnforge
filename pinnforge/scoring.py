"""Score service — the one component that reads a task's scoring truth.

rRMSE is computed here, not inside `tasks/<task>/eval.py`. `eval.py` trains
a candidate and predicts on a public grid, then files a score request; this
service loads the predictions and the reference field, compares them with
the task's own metric, and files the result back. It never executes agent
code, so the reference may live where the agent's processes cannot read it.

That matters because the alternative is a rule. The charter tells a block
"never read `task/<reference>`" and blocks have honoured it — but a rule
that names a file goes stale the moment a task swap changes the filename,
and nothing about a rule survives an agent that decides otherwise. Moving
the truth behind a process boundary turns "the answers are unreadable" into
a property of the operating system.

Wire protocol, all under `<run>/.scoreq/` (created on demand):

    req/<id>.npz    predictions, written first
    req/<id>.json   request metadata — its arrival marks the request ready
    res/<id>.json   score record, written atomically (tmp + rename)
    scored.alive    daemon heartbeat (mtime)
    scored.lock     held by a live daemon for as long as it serves

The client removes the whole exchange once it has read the answer — request,
payload and response — so a finished run leaves an empty queue behind. The
daemon skips requests that already have a response, so a re-scan before the
client gets there is idempotent and no locking is needed; `_collect` sweeps
whatever a client died holding.

Everything under `.scoreq/` is writable by the block being scored, so the
daemon treats it as hostile ground: it never follows a symlink there, never
reads anything but a regular file, and writes only through fresh temporary
names it created itself. A planted link would otherwise turn the daemon's
own write access — which in a hardened deployment reaches the answers — into
the block's. For the same reason the per-block cap is counted in a ledger
outside the queue: a count the block could delete is not a cap — and it is
keyed on what the daemon knows, not on what a request says. The in-process
daemon is told which block is dispatched and refuses requests naming any
other; a per-run cap bounds the standalone one, which cannot know.

The metric itself is loaded once, when the daemon starts, from a private
snapshot of the library task. Loading it from `tasks/<task>/eval.py` on every
request meant a block could edit the library file (same user, two directories
up) and be scored by code of its own choosing; a snapshot taken before the
first dispatch is out of its reach. The snapshot also has to agree with the
copy frozen into the run: a library edited since the run was created would
otherwise change the run's metric halfway through it.

Reference resolution goes through `pinnforge.private`: `<private>/<task>/`
when a private directory is configured, otherwise `tasks/<task>/private/`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pinnforge import oscompat

QUEUE = ".scoreq"
PROTOCOL = 1  # bumped when the request/response shape changes
# Client: how long to wait for a result. This MUST stay below the smallest
# `EVAL_WALL_S` any task's eval.py enforces, or the scoring worker is killed
# at that wall before the client can give up — and the caller sees "hung or
# runaway process" instead of "scorer unavailable", which points at the
# candidate rather than at the missing daemon.
# `tests/test_scoring.py::test_client_gives_up_before_the_eval_wall` pins it.
WAIT_S = float(os.environ.get("FORGE_SCORE_WAIT_S", "90"))
POLL_S = 0.5  # queue scan / result poll interval
STALE_S = 3600.0  # daemon: age at which leftover queue files are collected
ALIVE_FRESH_S = 10.0  # a heartbeat younger than this means someone is serving
MAX_SCORES = 256  # per-block cap on scored requests (oracle guard)
# Per-run backstop. The block a request names is the requester's say-so —
# it writes the request file itself — so a per-block cap alone can be
# renewed by naming a new block each time. The in-process daemon knows which
# block is dispatched and refuses the rest; a standalone `pinnforge scored`
# cannot know, and this is what bounds it.
MAX_RUN_SCORES = 4096
NOT_DISPATCHED = "not the block being run"
# Library files that are prose or candidate code, not metric: an edit to them
# does not change how a run is scored, so it is not drift.
NOT_METRIC = ("problem.md", "baseline.py")
DRIFT_ENV = "PINNFORGE_ALLOW_METRIC_DRIFT"
LEDGER_KEYS = ("task", "run", "block", "candidate", "seed", "smoke")

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


def queue_dirs(run: Path) -> tuple[Path, Path, Path]:
    q = Path(run) / QUEUE
    req, res = q / "req", q / "res"
    req.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return q, req, res


def scorer_alive(run: Path) -> bool:
    """True if a score daemon's heartbeat is fresh."""
    try:
        mt = (Path(run) / QUEUE / "scored.alive").stat().st_mtime
    except OSError:
        return False
    return time.time() - mt < ALIVE_FRESH_S


def standalone_serving(run: Path) -> bool:
    """Is a daemon other than this process serving `run` right now?

    Asked of the daemon's lock, not its heartbeat. A heartbeat outlives the
    process that wrote it by up to ten seconds, so a run resumed right after
    a crash found its predecessor's heartbeat still fresh, deferred to a
    daemon that no longer existed, and filed every evaluation into a queue
    nobody answered. A lock dies with its holder.
    """
    lock = Path(run) / QUEUE / "scored.lock"
    try:
        fd = oscompat.open_file(lock, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        # Unreadable (another user's, in an odd mode): fall back to the
        # heartbeat, which is the only other evidence there is.
        return scorer_alive(run)
    try:
        oscompat.lock(fd, block=False)
    except BlockingIOError:
        return True
    except OSError:
        return scorer_alive(run)
    else:
        oscompat.unlock(fd)
        return False
    finally:
        os.close(fd)


# ───────────────────────── client (eval.py side) ─────────────────────────


def submit_score(run: Path, meta: dict[str, Any], pred_path: Path, wait_s: float = WAIT_S) -> dict:
    """File a score request and block for the result.

    `meta` carries task/run/block/candidate/seed/smoke; `pred_path` is an
    `.npz` of the predicted fields. Returns the result dict — `rRMSE` on
    success, `error` set otherwise. A timeout reports the scorer as
    unavailable rather than raising, so a block still gets a logged record.
    """
    _, req, res = queue_dirs(run)
    rid = uuid.uuid4().hex
    payload, request, answer = req / f"{rid}.npz", req / f"{rid}.json", res / f"{rid}.json"
    payload.write_bytes(Path(pred_path).read_bytes())
    request.write_text(
        json.dumps({**meta, "pred": payload.name, "protocol": PROTOCOL}), encoding="utf-8"
    )
    deadline = time.time() + wait_s
    try:
        while time.time() < deadline:
            if answer.is_file():
                return json.loads(answer.read_text(encoding="utf-8"))
            time.sleep(POLL_S)
        return {
            "rRMSE": None,
            "error": "scorer unavailable — start `pinnforge scored` "
            "(`pinnforge run start` serves it in-process)",
        }
    finally:
        # The answer goes too, not just the request. The daemon's idempotency
        # rests on a response outliving its request, not on it outliving the
        # client that consumed it — and `_collect` only sweeps what is already
        # an hour old, while the daemon stops with the run. Leaving them behind
        # littered every run directory with the results of its last hour: 37
        # orphaned files in the longest one here.
        for f in (payload, request, answer):
            with suppress(OSError):
                f.unlink()


# ───────────────────────── safe file access ─────────────────────────


def _open_regular(path: Path) -> int:
    """Open `path` read-only, refusing links and anything but a regular file.

    A hard link too: what a client files has one name, and a second is one a
    block gave it — on Windows, where no privilege guards hard links as Linux's
    `protected_hardlinks` does, to the answers as easily as anything else.
    """
    fd = oscompat.open_file(path, os.O_RDONLY)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise OSError(f"{path.name} is not a regular, singly linked file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_json(path: Path) -> Any:
    fd = _open_regular(path)
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_atomic(dest: Path, text: str) -> None:
    """Write through a fresh temporary created here, then rename over `dest`.

    `mkstemp` creates with O_EXCL, so a name planted in advance cannot be
    followed; the rename replaces a planted `dest` link instead of writing
    through it.
    """
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        oscompat.replace(tmp, dest)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = oscompat.open_file(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _beat(path: Path) -> None:
    """Touch the heartbeat without following a link planted in its place."""
    try:
        fd = oscompat.open_file(path, os.O_WRONLY | os.O_CREAT, 0o644)
    except OSError:
        with suppress(OSError):
            path.unlink()
        fd = oscompat.open_file(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.utime(fd if os.utime in os.supports_fd else path)  # Windows: by name, once opened
    finally:
        os.close(fd)


# ───────────────────────── the metric ─────────────────────────


class MetricDrift(RuntimeError):
    """The library task no longer matches the copy frozen into the run."""


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_task(run: Path) -> str:
    """Which task a run (or a `<task>_0` scratch run) is of."""
    cfg = Path(run) / "run.yaml"
    if cfg.is_file():
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        if data.get("task"):
            return str(data["task"])
    return re.sub(r"_\d+$", "", Path(run).name)


def _import_module(path: Path, task: str):
    spec = importlib.util.spec_from_file_location(f"pinnforge_score_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not callable(getattr(module, "score_predictions", None)):
        raise AttributeError(f"tasks/{task}/eval.py must define score_predictions()")
    return module


def _import_metric(path: Path, task: str):
    return _import_module(path, task).score_predictions


def _load_metric(root: Path, task: str):
    """Import the task's own `score_predictions()` from `tasks/<task>/eval.py`.

    The metric stays where it always lived — in the task package — and the
    service simply calls the half of it that touches the reference. It is
    loaded from the **library**, never from a run's copy: a run directory is
    writable by the block, and a metric the scored party can edit is not a
    metric. The code is not secret (knowing *how* you are scored is fair);
    it just must not be theirs to change.
    """
    path = Path(root) / "tasks" / task / "eval.py"
    if not path.is_file():
        raise FileNotFoundError(f"no task '{task}' in {root / 'tasks'}")
    return _import_metric(path, task)


class Metric:
    """A task's scoring half, snapshotted once for the daemon's lifetime.

    The library task's files are copied into a directory this process owns,
    and the module is imported from there — so its module-level paths
    (`GRID_CSV`, masks, level sets) resolve against the snapshot too.
    """

    def __init__(self, root: Path, run: Path, task: str):
        self.task = task
        lib = Path(root) / "tasks" / task
        if not (lib / "eval.py").is_file():
            raise FileNotFoundError(f"no task '{task}' in {Path(root) / 'tasks'}")
        frozen = Path(run) / "task"
        drift = []
        if (frozen / "eval.py").is_file():
            for f in sorted(lib.iterdir()):
                if not f.is_file() or f.name in NOT_METRIC:
                    continue
                twin = frozen / f.name
                if twin.is_file() and _md5(twin) != _md5(f):
                    drift.append(f.name)
        if drift and os.environ.get(DRIFT_ENV) != "1":
            raise MetricDrift(
                f"tasks/{task}/ changed since {Path(run).name} was created "
                f"({', '.join(drift)}): scoring it now would change the run's metric "
                f"halfway through. Restore the library copy or start a new run "
                f"({DRIFT_ENV}=1 overrides)"
            )
        self._dir = Path(tempfile.mkdtemp(prefix=f"pinnforge-metric-{task}-"))
        for f in lib.iterdir():
            if f.is_file():
                shutil.copy2(f, self._dir / f.name)
        self.module = _import_module(self._dir / "eval.py", task)
        self.score_predictions = self.module.score_predictions

    def close(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)


# What the score service ever says: an rRMSE — or, when there is none, why,
# in words worked out from the request and the public grid alone. Whatever
# else a task's metric computes stays inside (an MSE beside the rRMSE gives the
# reference's norm away), and so do the metric's own words when it fails: it
# ran against the answers. The details go to the task's private/logs.
ANSWER_KEYS = ("rRMSE", "error")
UNSCORABLE = "the predictions could not be scored"


def _public_shape(metric: Metric | None) -> tuple[tuple[str, ...], int | None]:
    """The fields a submission must hold, and how many values each — public facts."""
    import numpy as np

    mod = getattr(metric, "module", None)
    names = tuple(getattr(mod, "FIELDS", ()) or ())
    n = None
    grid_csv = getattr(mod, "GRID_CSV", None)
    if grid_csv is not None:
        try:
            grid = np.loadtxt(grid_csv, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
            rows = mod.predict_rows(grid) if callable(getattr(mod, "predict_rows", None)) \
                else slice(None)
            n = int(grid[rows].shape[0])
        except Exception:  # an odd grid is the metric's to refuse, not this check's
            n = None
    return names, n


def _submission_problem(fields: dict, names: tuple[str, ...], n: int | None) -> str | None:
    import numpy as np

    if names and (missing := [f for f in names if f not in fields]):
        return f"the predictions lack {missing}; the task scores {list(names)}"
    for key, value in fields.items():
        if n is not None and value.size != n:
            return f"{key} has {value.size} values; the task's grid has {n} scored points"
        if not np.all(np.isfinite(value)):
            return f"{key} holds non-finite values"
    return None


def _answer(raw) -> tuple[dict, str | None]:
    """The metric's output reduced to the answer, and what was held back."""
    import math

    r = raw.get("rRMSE") if isinstance(raw, dict) else None
    if isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) \
            and not (isinstance(raw, dict) and raw.get("error")):
        return {"rRMSE": float(r)}, None
    detail = raw.get("error") if isinstance(raw, dict) else raw
    return {"error": UNSCORABLE}, f"metric returned {str(detail)[:2000]!r}"


def _private_note(root: Path, private_dir: Path | None, req: dict, detail: str) -> None:
    """Keep a failure's details with the answers, where only the operator reads them."""
    from pinnforge import private

    try:
        home = private.reference_dir(Path(root), str(req.get("task", "")), private_dir)
        logs = Path(home) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           **{k: req.get(k) for k in LEDGER_KEYS}, "detail": detail})
        _append_line(logs / "score-errors.jsonl", line)
    except Exception:
        logger.warning("could not record a scoring failure privately")


def score_request(
    root: Path,
    private_dir: Path | None,
    req_dir: Path,
    req: dict,
    metric: Metric | None = None,
) -> dict:
    """Compare one request's predictions against the reference. No agent code runs.

    Returns `{"rRMSE": x}`, or `{"error": reason}` — nothing else, ever.
    """
    import numpy as np

    from pinnforge import private

    task, pred = str(req.get("task", "")), str(req.get("pred", ""))
    if not _NAME_RE.fullmatch(task) or Path(pred).name != pred or not pred:
        return {"error": f"malformed request (task={task!r}, pred={pred!r})"}
    if int(req.get("protocol") or 1) > PROTOCOL:
        return {"error": f"score request protocol {req.get('protocol')} is newer than this "
                         f"service ({PROTOCOL}) — upgrade the pinnforge that serves scores"}
    if metric is not None and task != metric.task:
        return {"error": f"request for task {task!r} filed in a run of {metric.task!r}"}

    score_predictions = metric.score_predictions if metric else _load_metric(root, task)
    ref_dir = private.reference_dir(Path(root), task, private_dir)
    fd = _open_regular(req_dir / pred)
    with os.fdopen(fd, "rb") as fh:
        data = np.load(fh, allow_pickle=False)
        if not hasattr(data, "files"):
            return {"error": "predictions must be an .npz of named fields"}
        with data:
            fields = {k: np.asarray(data[k], dtype=np.float64).reshape(-1) for k in data.files}
    if problem := _submission_problem(fields, *_public_shape(metric)):
        return {"error": problem}
    try:
        raw = score_predictions(fields, ref_dir)
    except Exception as e:
        _private_note(root, private_dir, req, f"{type(e).__name__}: {e}")
        return {"error": UNSCORABLE}
    answer, held = _answer(raw)
    if held:
        _private_note(root, private_dir, req, held)
    return answer


# ───────────────────────── daemon ─────────────────────────


def ledger_path(root: Path, run: Path, private_dir: Path | None) -> Path:
    """Where the per-block cap is counted.

    Never inside the queue: deleting `.scoreq/scores.jsonl` used to reset the
    cap on the next restart. A private deployment keeps it with the answers;
    otherwise it goes under the workspace's `.pinnforge/`. A scratch run (a
    smoke or an anchor measured in a temporary directory) counts in its own
    queue, which vanishes with it.
    """
    if private_dir is not None:
        return Path(private_dir) / "scores.jsonl"
    run = Path(run).resolve()
    runs = (Path(root) / "runs").resolve()
    if run.parent == runs:
        return Path(root) / ".pinnforge" / "scores" / f"{run.name}.jsonl"
    return run / QUEUE / "scores.jsonl"


class Dispatch:
    """Which block the score service answers for — the driver's word, not the request's.

    The driver sets `block` while a block is dispatched (and to "b00" while
    the anchor is measured) and clears it in between. A request naming any
    other block is refused, so a block cannot open a fresh score cap by
    filing requests under made-up block ids.
    """

    def __init__(self, block: str | None = None):
        self.block = block


def serve(
    root: Path,
    run: Path,
    private_dir: Path | None = None,
    once: bool = False,
    max_scores: int = MAX_SCORES,
    stop: Any = None,
    on_scored: Any = None,
    dispatch: Dispatch | None = None,
    max_run_scores: int = MAX_RUN_SCORES,
) -> None:
    """Answer queued requests until stopped: heartbeat, score, ledger, GC.

    Run standalone via `pinnforge scored` — as a dedicated user when the
    reference is private — or let `pinnforge run start` serve it in a thread
    when no standalone daemon is alive. `on_scored(req, result, payload)` is
    called after each successful score, before its answer is written — while
    the client, still waiting, has not yet removed the payload.

    Each pass is guarded on its own. The queue is the block's to write — and
    to delete — and a pass that raised used to end the thread silently, after
    which every evaluation came back "scorer unavailable".
    """
    root, run = Path(root), Path(run)
    private_dir = Path(private_dir) if private_dir else None
    ledger = ledger_path(root, run, private_dir)
    counts = _replay_counts(ledger)
    legacy = Path(run) / QUEUE / "scores.jsonl"
    if legacy != ledger:
        for key, n in _replay_counts(legacy).items():
            counts[key] = max(counts.get(key, 0), n)

    q, req_dir, res_dir = queue_dirs(run)
    lock_fd = _hold_lock(q / "scored.lock")
    metric: Metric | None = None
    metric_error: str | None = None
    try:
        metric = Metric(root, run, run_task(run))
    except Exception as e:
        metric_error = f"scoring unavailable: {type(e).__name__}: {e}"
        logger.error("%s", metric_error)
    if on_scored is None and metric is not None and _is_real_run(root, run):
        # Keep the best predictions for the dashboard's field animation.
        from pinnforge.ui import viz

        on_scored = viz.archiver_for(run, metric)

    limits = {"max_scores": max_scores, "max_run_scores": max_run_scores}
    last_failure = None
    try:
        while True:
            try:
                # Recreated every pass: a block that deletes `.scoreq/` must
                # not leave the daemon heartbeating into a directory that is gone.
                q, req_dir, res_dir = queue_dirs(run)
                if lock_fd is None or not _still_held(lock_fd, q / "scored.lock"):
                    if lock_fd is not None:
                        os.close(lock_fd)
                    lock_fd = _hold_lock(q / "scored.lock")
                _beat(q / "scored.alive")
                _serve_pass(root, run, private_dir, req_dir, res_dir, ledger, counts, limits,
                            metric, metric_error, on_scored, dispatch)
                _collect(req_dir, res_dir)
                last_failure = None
            except Exception as e:
                # once per distinct failure, not twice a second for as long as it lasts
                if repr(e) != last_failure:
                    logger.exception("score service: a pass over %s failed; carrying on", run)
                last_failure = repr(e)
            if once or (stop is not None and stop.is_set()):
                return
            time.sleep(POLL_S)
    finally:
        if metric is not None:
            metric.close()
        if lock_fd is not None:
            os.close(lock_fd)


def _serve_pass(root, run, private_dir, req_dir, res_dir, ledger, counts, limits,
                metric, metric_error, on_scored, dispatch) -> None:
    """One scan of the queue: answer every request that has no answer yet."""
    me = Path(run).name
    for jf in sorted(req_dir.glob("*.json")):
        if not _NAME_RE.fullmatch(jf.stem):
            continue
        out = res_dir / jf.name
        if out.exists() or out.is_symlink():
            continue
        try:
            req = _read_json(jf)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue  # half-written (or not a request); the next scan decides
        if not isinstance(req, dict):
            continue

        block = str(req.get("block") or "")
        # The cap is keyed on what the daemon knows — the run it serves — and
        # the block, which is checked against the dispatch where it can be.
        key = (me, block)
        run_total = sum(n for (r, _), n in counts.items() if r == me)
        if metric_error:
            result = {"error": metric_error}
        elif dispatch is not None and block != dispatch.block:
            result = {"error": NOT_DISPATCHED}
        elif counts.get(key, 0) >= limits["max_scores"]:
            result = {"error": f"score cap reached ({limits['max_scores']}) for {me}/{block}"}
        elif run_total >= limits["max_run_scores"]:
            result = {"error": f"score cap reached ({limits['max_run_scores']}) for {me}"}
        else:
            try:
                result = score_request(root, private_dir, req_dir, req, metric=metric)
            except Exception as e:  # one bad request must not kill the daemon
                _private_note(root, private_dir, req, f"{type(e).__name__}: {e}")
                result = {"error": UNSCORABLE}
            if not result.get("error"):
                counts[key] = counts.get(key, 0) + 1

        result = {"rRMSE": result.get("rRMSE"), "error": result.get("error")}
        # Before the answer: the client deletes its payload as soon as it
        # has read one, so afterwards there may be nothing left to keep.
        if on_scored is not None and not result.get("error"):
            try:
                on_scored(req, result, req_dir / str(req.get("pred", "")))
            except Exception:
                logger.exception("post-score hook failed")
        _write_atomic(out, json.dumps(result))

        entry = {
            "id": jf.stem,
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **{k: req.get(k) for k in LEDGER_KEYS},
            "queue": me,
            **result,
        }
        _append_line(ledger, json.dumps(entry))


def _still_held(fd: int, path: Path) -> bool:
    """Is `fd` still the lock file at `path` — not one deleted with the queue?"""
    try:
        return os.fstat(fd).st_ino == os.lstat(path).st_ino
    except OSError:
        return False


def _is_real_run(root: Path, run: Path) -> bool:
    """A run of the library, as opposed to a smoke's or an anchor's scratch copy."""
    return Path(run).resolve().parent == (Path(root) / "runs").resolve()


def _hold_lock(path: Path) -> int | None:
    """Take the daemon lock for this process's lifetime, if nobody holds it.

    A second daemon on the same queue is not refused — the queue is
    idempotent — but only the first one's lock says "someone is serving".
    """
    try:
        fd = oscompat.open_file(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        with suppress(OSError):
            path.unlink()
        try:
            fd = oscompat.open_file(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return None
    try:
        oscompat.lock(fd, block=False)
    except OSError:
        os.close(fd)
        return None
    return fd


class Server:
    """The score daemon `serving()` runs in this process, which the driver can check on.

    `ensure()` is asked between segments: a thread that has died is started
    again, and only one that cannot be is reported — so the run stops rather
    than spending GPU time on evaluations nobody scores.
    """

    def __init__(self, root: Path, run: Path, private_dir: Path | None = None,
                 dispatch: Dispatch | None = None):
        self.root, self.run, self.private_dir, self.dispatch = root, run, private_dir, dispatch
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.restarts = 0

    def _target(self) -> None:
        try:
            serve(self.root, self.run, private_dir=self.private_dir, stop=self.stop,
                  dispatch=self.dispatch)
        except BaseException:
            logger.exception("the score service for %s stopped", self.run)
            raise

    def start(self) -> None:
        self.thread = threading.Thread(target=self._target, name="pinnforge-scored", daemon=True)
        self.thread.start()
        # The client's first request must not race the daemon's first heartbeat.
        deadline = time.time() + 10
        while not scorer_alive(self.run) and self.thread.is_alive() and time.time() < deadline:
            time.sleep(0.05)

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def ensure(self) -> bool:
        """True if the daemon is serving — restarting it first if it had died."""
        if self.alive():
            return True
        if self.stop.is_set():
            return False
        logger.error("the score service thread for %s has died; restarting it", self.run)
        self.restarts += 1
        self.start()
        return self.alive()

    def close(self, timeout: float = 5.0) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    # `serving()` used to yield the bare stop event; keep its two methods.
    def set(self) -> None:
        self.stop.set()

    def is_set(self) -> bool:
        return self.stop.is_set()


class Standalone:
    """A `pinnforge scored` in another process: checked on, never restarted from here.

    That daemon is usually another user's, with the answers only it can
    read; one started here instead could not score anything.
    """

    def __init__(self, run: Path):
        self.run = run

    def ensure(self) -> bool:
        if standalone_serving(self.run) or scorer_alive(self.run):
            return True
        logger.error("the standalone score daemon for %s is gone", self.run)
        return False


@contextmanager
def serving(root: Path, run: Path, private_dir: Path | None = None,
            dispatch: Dispatch | None = None):
    """Guarantee a score daemon for the duration of the block.

    Every path that runs `eval.py` needs one, and that includes measuring the
    b00 anchor — which happens before the first block is dispatched. Owning
    the daemon here rather than inside the block loop is what keeps the
    anchor from filing a request nobody answers.

    Yields None if a standalone `pinnforge scored` is already serving this
    run; that is the deployment where the reference is another user's to
    read. Otherwise the configured private directory, if any, is the one the
    in-process daemon scores against, and the `Server` running it is yielded.
    `dispatch`, when given, is what the in-process daemon checks a request's
    block against.
    """
    if standalone_serving(run):
        logger.info("score daemon already serving %s", run)
        yield None
        return
    if private_dir is None:
        from pinnforge import private

        private_dir = private.configured(Path(root))
    server = Server(root, run, private_dir, dispatch)
    server.start()
    logger.info("serving scores in-process for %s", run)
    try:
        yield server
    finally:
        server.close()


def _replay_counts(ledger: Path) -> dict:
    """Rebuild per-block scored counts so the cap survives a restart."""
    counts: dict = {}
    if not ledger.is_file() or ledger.is_symlink():
        return counts
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and not r.get("error"):
            # the daemon's own record of which run it served; older ledgers
            # only have the requester's word for it
            key = (r.get("queue") or r.get("run"), str(r.get("block") or ""))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _collect(*dirs: Path) -> None:
    cutoff = time.time() - STALE_S
    for d in dirs:
        for f in d.iterdir():
            try:
                if f.lstat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
