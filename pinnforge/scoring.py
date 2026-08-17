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

The client removes its own request files once answered; the daemon skips
requests that already have a response, so a re-scan is idempotent and no
locking is needed.

Reference resolution: `<private>/<task>/<reference>` when a private
directory is configured (`FORGE_PRIVATE_DIR`, or `pinnforge scored
--private`; chown it to a dedicated user to make the isolation an OS fact),
otherwise the in-repo `tasks/<task>/`. Results are appended to a ledger for
offline cross-checks against `evals.jsonl`, and scored requests are capped
per block so the service cannot be farmed as a free oracle.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUEUE = ".scoreq"
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


# ───────────────────────── client (eval.py side) ─────────────────────────


def submit_score(run: Path, meta: dict[str, Any], pred_path: Path, wait_s: float = WAIT_S) -> dict:
    """File a score request and block for the result.

    `meta` carries task/run/block/candidate/seed/smoke; `pred_path` is an
    `.npz` of the predicted fields. Returns the result dict — `rRMSE`/`MSE`
    on success, `error` set otherwise. A timeout reports the scorer as
    unavailable rather than raising, so a block still gets a logged record.
    """
    _, req, res = queue_dirs(run)
    rid = uuid.uuid4().hex
    payload, request, answer = req / f"{rid}.npz", req / f"{rid}.json", res / f"{rid}.json"
    payload.write_bytes(Path(pred_path).read_bytes())
    request.write_text(json.dumps({**meta, "pred": payload.name}), encoding="utf-8")
    deadline = time.time() + wait_s
    try:
        while time.time() < deadline:
            if answer.is_file():
                return json.loads(answer.read_text(encoding="utf-8"))
            time.sleep(POLL_S)
        return {
            "rRMSE": None,
            "MSE": None,
            "error": "scorer unavailable — start `pinnforge scored` "
            "(`pinnforge run start` serves it in-process)",
        }
    finally:
        for f in (payload, request):
            with suppress(OSError):
                f.unlink()


# ───────────────────────── daemon ─────────────────────────


def _task_dir(root: Path, private_dir: Path | None, task: str) -> Path:
    """Where the answers live.

    A `--private` deployment wins: point it at a directory owned by this
    service's user and the isolation stops depending on anyone's restraint.
    Otherwise the in-repo `tasks/<task>/private/`, which the installer never
    copies into a run.
    """
    if private_dir:
        p = Path(private_dir) / task
        if p.is_dir():
            return p
    return Path(root) / "tasks" / task / "private"


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
    spec = importlib.util.spec_from_file_location(f"pinnforge_score_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    fn = getattr(module, "score_predictions", None)
    if not callable(fn):
        raise AttributeError(f"tasks/{task}/eval.py must define score_predictions()")
    return fn


def score_request(root: Path, private_dir: Path | None, req_dir: Path, req: dict) -> dict:
    """Compare one request's predictions against the reference. No agent code runs."""
    import numpy as np

    task, pred = str(req.get("task", "")), str(req.get("pred", ""))
    if not _NAME_RE.fullmatch(task) or Path(pred).name != pred:
        return {"error": f"malformed request (task={task!r}, pred={pred!r})"}

    score_predictions = _load_metric(root, task)
    ref_dir = _task_dir(root, private_dir, task)
    with np.load(req_dir / pred, allow_pickle=False) as data:
        fields = {k: np.asarray(data[k], dtype=np.float64) for k in data.files}
    return score_predictions(fields, ref_dir)


def serve(
    root: Path,
    run: Path,
    private_dir: Path | None = None,
    once: bool = False,
    max_scores: int = MAX_SCORES,
    stop: Any = None,
) -> None:
    """Answer queued requests until stopped: heartbeat, score, ledger, GC.

    Run standalone via `pinnforge scored` — as a dedicated user when the
    reference is private — or let `pinnforge run start` serve it in a thread
    when no standalone daemon is alive.
    """
    root, run = Path(root), Path(run)
    q, req_dir, res_dir = queue_dirs(run)
    private_dir = Path(private_dir) if private_dir else None
    ledger = (private_dir if private_dir else q) / "scores.jsonl"
    counts = _replay_counts(ledger)

    while True:
        (q / "scored.alive").touch()
        for jf in sorted(req_dir.glob("*.json")):
            out = res_dir / jf.name
            if out.exists():
                continue
            try:
                req = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue  # half-written; the next scan will pick it up

            key = (req.get("run"), req.get("block"))
            if counts.get(key, 0) >= max_scores:
                result = {"error": f"score cap reached ({max_scores}) for {key[0]}/{key[1]}"}
            else:
                try:
                    result = score_request(root, private_dir, req_dir, req)
                except Exception as e:  # one bad request must not kill the daemon
                    result = {"error": f"scoring failed: {type(e).__name__}: {e}"}
                if not result.get("error"):
                    counts[key] = counts.get(key, 0) + 1

            result = {"rRMSE": None, "MSE": None, "error": None, **result}
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(json.dumps(result), encoding="utf-8")
            os.replace(tmp, out)  # atomic: a client never reads a partial answer

            entry = {
                "id": jf.stem,
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **{k: req.get(k) for k in ("task", "run", "block", "candidate", "seed", "smoke")},
                **result,
            }
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(entry) + "\n")

        _collect(req_dir, res_dir)
        if once or (stop is not None and stop.is_set()):
            return
        time.sleep(POLL_S)


@contextmanager
def serving(root: Path, run: Path, private_dir: Path | None = None):
    """Guarantee a score daemon for the duration of the block.

    Every path that runs `eval.py` needs one, and that includes measuring the
    b00 anchor — which happens before the first block is dispatched. Owning
    the daemon here rather than inside the block loop is what keeps the
    anchor from filing a request nobody answers.

    Yields immediately if a standalone `pinnforge scored` is already alive;
    that is the deployment where the reference is another user's to read.
    """
    if scorer_alive(run):
        logger.info("score daemon already serving %s", run)
        yield None
        return
    stop = threading.Event()
    thread = threading.Thread(
        target=serve,
        kwargs={"root": root, "run": run, "private_dir": private_dir, "stop": stop},
        name="pinnforge-scored",
        daemon=True,
    )
    thread.start()
    # The client's first request must not race the daemon's first heartbeat.
    deadline = time.time() + 10
    while not scorer_alive(run) and time.time() < deadline:
        time.sleep(0.05)
    logger.info("serving scores in-process for %s", run)
    try:
        yield stop
    finally:
        stop.set()
        thread.join(timeout=5)


def _replay_counts(ledger: Path) -> dict:
    """Rebuild per-block scored counts so the cap survives a restart."""
    counts: dict = {}
    if not ledger.is_file():
        return counts
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("error"):
            key = (r.get("run"), r.get("block"))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _collect(*dirs: Path) -> None:
    cutoff = time.time() - STALE_S
    for d in dirs:
        for f in d.iterdir():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
