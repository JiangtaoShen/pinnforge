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
    assert "shape" in result["error"]


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


@pytest.mark.parametrize("name", ["ldc", "ks", "naca"])
def test_library_tasks_expose_the_service_hook(name):
    """Every task must hand its metric to the service, not compute it inline."""
    src = (paths.tasks_dir() / name / "eval.py").read_text(encoding="utf-8")
    assert "def score_predictions(" in src
    assert "submit_score" in src
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
    scoring.serve(project, run, once=True)  # leaves a fresh heartbeat
    with scoring.serving(project, run) as stop:
        assert stop is None, "should defer to the daemon already serving"


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
