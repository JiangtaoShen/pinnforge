"""The score service — the process boundary around the answers.

The point of these is not that rRMSE is computed correctly (the tasks own
that) but that it is computed *somewhere else*: in a process that never
executes candidate code, reading a reference the block cannot open.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from pinnforge import paths, scoring
from pinnforge.task import registry

TASK_SRC = """
import numpy as np
from pathlib import Path

def score_predictions(fields, ref_dir):
    gt = np.load(Path(ref_dir) / "ref_values.npy", allow_pickle=False).reshape(-1)
    u = np.asarray(fields["u"], dtype=np.float64).reshape(-1)
    if u.shape != gt.shape:
        return {"error": f"shape {u.shape} != {gt.shape}"}
    diff = u - gt
    return {"rRMSE": float(np.linalg.norm(diff) / np.linalg.norm(gt)),
            "MSE": float(np.mean(diff ** 2))}
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A minimal project whose one task ships a private reference."""
    (tmp_path / "pinnforge").mkdir()
    task = tmp_path / "tasks" / "toy"
    task.mkdir(parents=True)
    (task / "eval.py").write_text(TASK_SRC, encoding="utf-8")
    (task / "private").mkdir()
    np.save(task / "private" / "ref_values.npy", np.arange(10, dtype=np.float64) + 1.0)
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    return tmp_path



def test_scoring_reads_the_reference_the_client_never_touches(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "abc.npz", u=np.arange(10, dtype=np.float64) + 1.0)
    result = scoring.score_request(project, None, req, {"task": "toy", "pred": "abc.npz"})
    assert result["rRMSE"] == pytest.approx(0.0)


def test_wrong_shape_is_an_error_not_a_crash(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "abc.npz", u=np.zeros(3))
    result = scoring.score_request(project, None, req, {"task": "toy", "pred": "abc.npz"})
    assert result == {"error": scoring.UNSCORABLE}, "the metric's own words stay inside"
    note = project / "tasks" / "toy" / "private" / "logs" / "score-errors.jsonl"
    assert "(3,) != (10,)" in note.read_text(), "the operator still gets them"


def test_an_rrmse_is_all_the_service_ever_returns(project):
    """An MSE beside the rRMSE gives the reference's norm away: n·MSE / rRMSE² = |ref|²."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "abc.npz", u=np.arange(10, dtype=np.float64))
    result = scoring.score_request(project, None, req, {"task": "toy", "pred": "abc.npz"})
    assert set(result) == {"rRMSE"}


def test_a_metric_that_raises_says_nothing_about_the_reference(project):
    task = project / "tasks" / "toy"
    (task / "eval.py").write_text(TASK_SRC.replace(
        "    diff = u - gt\n",
        "    raise ValueError(f'reference peaks at {gt.max()}')\n"), encoding="utf-8")
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "abc.npz", u=np.zeros(10))
    result = scoring.score_request(project, None, req, {"task": "toy", "pred": "abc.npz"})
    assert result == {"error": scoring.UNSCORABLE}
    assert "10.0" in (task / "private" / "logs" / "score-errors.jsonl").read_text()


def test_a_malformed_submission_is_explained_from_the_public_grid(project):
    task = project / "tasks" / "toy"
    (task / "eval.py").write_text(
        'from pathlib import Path\nGRID_CSV = Path(__file__).parent / "eval_grid.csv"\n'
        'FIELDS = ("u",)\n' + TASK_SRC, encoding="utf-8")
    np.savetxt(task / "eval_grid.csv", np.arange(10.0), header="x", comments="")
    run = project / "runs" / "toy_1"
    (run / "task").mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    metric = scoring.Metric(project, run, "toy")
    try:
        for fields, says in (({"v": np.zeros(10)}, "lack ['u']"),
                             ({"u": np.zeros(4)}, "4 values; the task's grid has 10"),
                             ({"u": np.full(10, np.nan)}, "non-finite")):
            np.savez(req / "abc.npz", **fields)
            result = scoring.score_request(project, None, req, {"task": "toy", "pred": "abc.npz"},
                                           metric=metric)
            assert set(result) == {"error"} and says in result["error"], result
    finally:
        metric.close()


def test_private_dir_wins_over_the_library(project, tmp_path):
    """Chowning this directory is what makes the isolation an OS fact."""
    private = tmp_path / "private"
    (private / "toy").mkdir(parents=True)
    np.save(private / "toy" / "ref_values.npy", np.full(10, 2.0))
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "abc.npz", u=np.full(10, 2.0))
    result = scoring.score_request(project, private, req, {"task": "toy", "pred": "abc.npz"})
    assert result["rRMSE"] == pytest.approx(0.0)  # matches the private copy, not the library


def test_malformed_request_is_rejected(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    for bad in ({"task": "../etc", "pred": "a.npz"}, {"task": "toy", "pred": "../x.npz"}):
        assert "malformed" in scoring.score_request(project, None, req, bad)["error"]


def test_round_trip_through_the_queue(project):
    """Client files a request, daemon answers it, client reads the verdict."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    pred = run / "pred.npz"
    np.savez(pred, u=np.arange(10, dtype=np.float64) + 1.0)

    result: dict = {}

    def client():
        result.update(
            scoring.submit_score(
                run, {"task": "toy", "run": "toy_1", "block": "b01"}, pred, wait_s=20
            )
        )

    t = threading.Thread(target=client)
    t.start()
    for _ in range(200):  # serve until the client has been answered
        scoring.serve(project, run, once=True)
        if not t.is_alive():
            break
    t.join(timeout=10)
    assert result.get("rRMSE") == pytest.approx(0.0)
    assert result.get("error") is None


def test_score_cap_stops_the_service_being_farmed(project):
    """A block must not be able to use scoring as a free oracle."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)
    for i in range(3):
        np.savez(req / f"r{i}.npz", u=np.zeros(10))
        (req / f"r{i}.json").write_text(
            json.dumps({"task": "toy", "run": "toy_1", "block": "b01", "pred": f"r{i}.npz"}),
            encoding="utf-8",
        )
    scoring.serve(project, run, once=True, max_scores=2)
    verdicts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(res.glob("*.json"))]
    assert sum(1 for v in verdicts if v.get("error") and "cap" in v["error"]) == 1


def test_heartbeat_reports_a_live_daemon(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    assert not scoring.scorer_alive(run)
    scoring.serve(project, run, once=True)
    assert scoring.scorer_alive(run)


def test_reference_is_not_copied_into_a_run(project):
    """The firewall in one assertion: the answers are not in the run."""
    task = registry.describe(paths.tasks_dir() / "toy")
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    registry.install(task, run)
    assert (run / "task" / "eval.py").is_file()
    assert not (run / "task" / "private").exists()
    assert not (run / "task" / "ref_values.npy").exists()


@pytest.mark.parametrize("name", ["ldc", "ks", "cyl", "naca"])
def test_library_tasks_expose_the_service_hook(name):
    """Every task must hand its metric to the service, not compute it inline."""
    src = (paths.tasks_dir() / name / "eval.py").read_text(encoding="utf-8")
    assert "def score_predictions(" in src
    # the queue client comes from the shared machinery, vendored into the run
    assert "_forgekit" in src or "submit_score" in src
    grid = paths.tasks_dir() / name / "eval_grid.csv"
    assert grid.is_file(), "the public grid is what the agent predicts on"


def test_daemon_is_up_before_the_anchor_is_measured(project, monkeypatch):
    """Measuring b00 runs eval.py, which files a score request like any other.

    The daemon used to start inside the block loop, so the anchor filed a
    request nobody answered and every run died before its first block.
    """
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    seen = {}
    with scoring.serving(project, run):
        seen["alive"] = scoring.scorer_alive(run)
    assert seen["alive"], "the anchor measurement must find a live scorer"


def test_serving_yields_to_a_standalone_daemon(project):
    """A `pinnforge scored` running as another user must not be duplicated."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    halt = threading.Event()
    daemon = threading.Thread(
        target=scoring.serve, kwargs={"root": project, "run": run, "stop": halt}
    )
    daemon.start()
    try:
        for _ in range(100):
            if scoring.scorer_alive(run):
                break
            threading.Event().wait(0.05)
        with scoring.serving(project, run) as stop:
            assert stop is None, "should defer to the daemon already serving"
    finally:
        halt.set()
        daemon.join(timeout=10)


def test_a_dead_daemons_heartbeat_is_not_a_daemon(project):
    """A run resumed seconds after a crash must not defer to its dead scorer.

    The heartbeat outlives its writer by up to ten seconds; deferring to it
    left every evaluation of the resumed run filed into a queue nobody served.
    """
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    scoring.serve(project, run, once=True)  # exits, leaving a fresh heartbeat
    assert scoring.scorer_alive(run)
    with scoring.serving(project, run) as stop:
        assert stop is not None, "nobody holds the lock, so this process must serve"


def test_a_planted_symlink_is_not_followed(project, tmp_path):
    """The queue is the block's to write; the daemon must not write through it."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    np.savez(req / "r0.npz", u=np.arange(10, dtype=np.float64) + 1.0)
    (req / "r0.json").write_text(
        json.dumps({"task": "toy", "run": "toy_1", "block": "b01", "pred": "r0.npz"}),
        encoding="utf-8",
    )
    (res / "r0.json").symlink_to(victim)  # a planted answer file
    (req / "r1.json").write_text(
        json.dumps({"task": "toy", "run": "toy_1", "block": "b01", "pred": "link.npz"}),
        encoding="utf-8",
    )
    (req / "link.npz").symlink_to(project / "tasks" / "toy" / "private" / "ref_values.npy")
    scoring.serve(project, run, once=True)
    assert victim.read_text(encoding="utf-8") == "untouched"
    answer = json.loads((res / "r1.json").read_text(encoding="utf-8"))
    assert answer["rRMSE"] is None and answer["error"], "a linked payload is not read"


def test_the_cap_is_counted_outside_the_queue(project):
    """Deleting the queue must not reset the per-block cap."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, _ = scoring.queue_dirs(run)
    np.savez(req / "r0.npz", u=np.zeros(10))
    (req / "r0.json").write_text(
        json.dumps({"task": "toy", "run": "toy_1", "block": "b01", "pred": "r0.npz"}),
        encoding="utf-8",
    )
    scoring.serve(project, run, once=True)
    ledger = scoring.ledger_path(project, run, None)
    assert ledger.is_file() and scoring.QUEUE not in ledger.parts
    import shutil

    shutil.rmtree(run / scoring.QUEUE)
    assert scoring._replay_counts(ledger) == {("toy_1", "b01"): 1}


def test_a_library_edited_mid_run_is_not_the_metric(project):
    """A block can reach `../../tasks/<t>/eval.py`; editing it must change nothing."""
    run = project / "runs" / "toy_1"
    (run / "task").mkdir(parents=True)
    lib = project / "tasks" / "toy" / "eval.py"
    (run / "task" / "eval.py").write_bytes(lib.read_bytes())
    metric = scoring.Metric(project, run, "toy")
    try:
        lib.write_text(
            "def score_predictions(fields, ref_dir):\n    return {'rRMSE': 0.0, 'MSE': 0.0}\n",
            encoding="utf-8",
        )
        _, req, _ = scoring.queue_dirs(run)
        np.savez(req / "a.npz", u=np.zeros(10))
        out = scoring.score_request(
            project, None, req, {"task": "toy", "pred": "a.npz"}, metric=metric
        )
        assert out["rRMSE"] == pytest.approx(1.0), "scored by the snapshot, not the edit"
    finally:
        metric.close()
    with pytest.raises(scoring.MetricDrift):
        scoring.Metric(project, run, "toy")  # the library no longer matches the run


def test_a_configured_private_dir_without_the_task_is_an_error(project, tmp_path):
    """Falling back to the library copy is how an isolation boundary quietly ends."""
    from pinnforge import private

    empty = tmp_path / "private-store"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        private.reference_dir(project, "toy", empty)


def test_client_gives_up_before_the_eval_wall():
    """`WAIT_S` must be smaller than every task's `EVAL_WALL_S`.

    `eval.py` runs the scoring worker under `EVAL_WALL_S`. If the client
    waits longer than that, the worker is killed at the wall before
    `submit_score` can return its "scorer unavailable" verdict, and the
    record blames the candidate ("hung or runaway process") for a missing
    daemon. Ordering the two the other way is what makes the diagnostic
    reachable.
    """
    import re

    walls = {}
    for task_dir in sorted(paths.tasks_dir().iterdir()):
        ev = task_dir / "eval.py"
        if not ev.is_file():
            continue
        m = re.search(r"^EVAL_WALL_S\s*=\s*([0-9.]+)", ev.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            walls[task_dir.name] = float(m.group(1))
    assert walls, "no task declares EVAL_WALL_S"
    tightest = min(walls.values())
    assert tightest > scoring.WAIT_S, (
        f"WAIT_S={scoring.WAIT_S} must be below the tightest EVAL_WALL_S "
        f"({tightest} in {min(walls, key=walls.get)}); walls={walls}"
    )


def test_a_finished_exchange_leaves_no_queue_files(tmp_path, monkeypatch):
    """Every run used to keep the results of its last hour.

    `_collect` only sweeps files already an hour old and the daemon stops with
    the run, so the tail of every run's scoring was never cleaned: 8, 16, 37
    and 19 orphaned `res/*.json` in the four runs this was found on. The client
    deleted its request and left the answer.
    """
    import numpy as np

    from pinnforge import scoring

    run = tmp_path / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)

    pred = tmp_path / "pred.npz"
    np.savez(pred, u=np.zeros(4))

    # answer the request the moment it lands, the way the daemon does
    def fake_sleep(_):
        for jf in req.glob("*.json"):
            (res / jf.name).write_text(
                '{"rRMSE": 0.5, "MSE": 0.25, "error": null}', encoding="utf-8"
            )

    monkeypatch.setattr(scoring.time, "sleep", fake_sleep)
    out = scoring.submit_score(run, {"task": "toy", "run": "toy_1", "block": "b01"}, pred, wait_s=5)

    assert out["rRMSE"] == 0.5
    assert list(req.iterdir()) == [], "the request and its payload must go"
    assert list(res.iterdir()) == [], "and so must the answer the client consumed"


# ─────────────── a daemon that keeps going, and knows who is asking ───────────────


def _file(req, rid, block, run_id="toy_1", u=None):
    np.savez(req / f"{rid}.npz", u=np.arange(10, dtype=np.float64) + 1.0 if u is None else u)
    (req / f"{rid}.json").write_text(
        json.dumps({"task": "toy", "run": run_id, "block": block, "pred": f"{rid}.npz"}),
        encoding="utf-8",
    )


def test_a_deleted_queue_does_not_end_the_service(project):
    """A pass that raised used to kill the thread; every eval then read 'scorer unavailable'."""
    import shutil

    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    pred = run / "pred.npz"
    np.savez(pred, u=np.arange(10, dtype=np.float64) + 1.0)
    with scoring.serving(project, run) as server:
        shutil.rmtree(run / scoring.QUEUE)
        threading.Event().wait(1.0)  # a pass or two over a queue that is gone
        assert server.alive()
        out = scoring.submit_score(run, {"task": "toy", "run": "toy_1", "block": "b01"}, pred,
                                   wait_s=20)
    assert out.get("rRMSE") == pytest.approx(0.0), out


def test_a_dead_score_thread_is_restarted(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    with scoring.serving(project, run) as server:
        gone = threading.Thread(target=lambda: None)
        gone.start()
        gone.join()
        real, server.thread = server.thread, gone  # as if the thread had died
        try:
            assert server.ensure(), "a dead thread is started again"
            assert server.restarts == 1 and server.alive()
        finally:
            server.stop.set()
            real.join(timeout=10)


def test_requests_from_a_block_that_is_not_running_are_refused(project):
    """A block could name a new block per request and get a fresh cap each time."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)
    _file(req, "mine", "b02")
    _file(req, "forged", "b07")
    scoring.serve(project, run, once=True, dispatch=scoring.Dispatch("b02"))
    mine = json.loads((res / "mine.json").read_text(encoding="utf-8"))
    forged = json.loads((res / "forged.json").read_text(encoding="utf-8"))
    assert mine["rRMSE"] == pytest.approx(0.0)
    assert forged == {"rRMSE": None, "error": scoring.NOT_DISPATCHED}, "nothing else is said"

    _file(req, "between", "b02")
    scoring.serve(project, run, once=True, dispatch=scoring.Dispatch(None))
    between = json.loads((res / "between.json").read_text(encoding="utf-8"))
    assert between["error"] == scoring.NOT_DISPATCHED, "nothing is dispatched between segments"


def test_the_anchor_is_answered_while_it_is_measured(project):
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)
    _file(req, "a", "b00")
    scoring.serve(project, run, once=True, dispatch=scoring.Dispatch("b00"))
    assert json.loads((res / "a.json").read_text(encoding="utf-8"))["rRMSE"] == pytest.approx(0.0)


def test_a_per_run_cap_bounds_a_daemon_that_cannot_see_the_dispatch(project):
    """`pinnforge scored` runs apart from the driver; the run as a whole is what it can cap."""
    run = project / "runs" / "toy_1"
    run.mkdir(parents=True)
    _, req, res = scoring.queue_dirs(run)
    for i in range(3):
        _file(req, f"r{i}", f"b{i + 1:02d}", run_id=f"forged_{i}")  # a new identity each time
    scoring.serve(project, run, once=True, max_run_scores=2)
    verdicts = [json.loads((res / f"r{i}.json").read_text(encoding="utf-8")) for i in range(3)]
    assert [v["error"] is None for v in verdicts] == [True, True, False]
    assert "score cap reached (2) for toy_1" in verdicts[2]["error"]

    # the count is the daemon's, kept outside the queue, and survives a restart
    _file(req, "r3", "b09")
    scoring.serve(project, run, once=True, max_run_scores=2)
    assert "cap" in json.loads((res / "r3.json").read_text(encoding="utf-8"))["error"]
    counts = scoring._replay_counts(scoring.ledger_path(project, run, None))
    assert sum(counts.values()) == 2 and {r for r, _ in counts} == {"toy_1"}, (
        "keyed on the run the daemon serves, not on what the requests said"
    )
