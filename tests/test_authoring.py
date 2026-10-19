"""Task authoring: the staged pipeline, and the reference checks it rests on."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from pinnforge.task import authoring, reference

EVAL = '''
from pathlib import Path
HERE = Path(__file__).resolve().parent
GRID_CSV = HERE / "eval_grid.csv"
REF_VALUES = "ref_values.npy"
INPUT_COLUMNS = 1
FIELDS = ("u",)
REF_COLUMNS = ("u", "v")

def score_predictions(fields, ref_dir):
    import numpy as np
    gt = np.load(Path(ref_dir) / REF_VALUES)[:, 0]
    u = np.asarray(fields["u"], dtype=np.float64).reshape(-1)
    d = u - gt
    return {"rRMSE": float(np.linalg.norm(d) / np.linalg.norm(gt)), "MSE": float(np.mean(d**2))}
'''

SOLVE = '''
import argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--level", type=int); ap.add_argument("--out")
a = ap.parse_args()
x = np.linspace(0, 1, 101)
h = 0.1 / 2 ** a.level
u = np.sin(np.pi * x) + h ** {order} * np.cos(3 * x)
np.save(a.out, np.stack([u, 2 * u], axis=1))
'''


def _task(tmp_path, order=2.0, levels=3, write_ref=True):
    task = tmp_path / "tasks" / "t"
    work = task / "private" / "reference"
    (work / "levels").mkdir(parents=True)
    (task / "eval.py").write_text(EVAL, encoding="utf-8")
    x = np.linspace(0, 1, 101)
    np.savetxt(task / "eval_grid.csv", x, header="x", comments="")
    (work / "solve.py").write_text(SOLVE.format(order=order), encoding="utf-8")
    import subprocess

    for k in range(1, levels + 1):
        subprocess.run([sys.executable, str(work / "solve.py"), "--level", str(k), "--out",
                        str(work / "levels" / f"L{k}.npy")], check=True)
    (work / "levels.json").write_text(json.dumps({"method": "test", "ratio": 2}), encoding="utf-8")
    if write_ref:
        (task / "private" / "ref_values.npy").write_bytes(
            (work / "levels" / f"L{levels}.npy").read_bytes())
    return task


def test_a_converging_reference_passes_with_its_order_measured(tmp_path):
    task = _task(tmp_path, order=2.0, levels=4)
    v = reference.check(task, task / "private", tol=1e-3, python=sys.executable)
    assert v["verdict"] == "pass", v["reasons"]
    assert v["order"] == pytest.approx(2.0, abs=0.05)
    assert v["estimate"] == pytest.approx(v["differences"][-1] / 3, rel=0.05)
    assert v["reproducible"] is True


def test_a_reference_that_is_not_accurate_enough_fails(tmp_path):
    task = _task(tmp_path, order=1.0, levels=3)
    v = reference.check(task, task / "private", tol=1e-6, python=sys.executable)
    assert v["verdict"] == "fail"
    assert any("exceeds the tolerance" in r for r in v["reasons"])


def test_identical_levels_are_not_a_convergence_study(tmp_path):
    task = _task(tmp_path, levels=3)
    lv = task / "private" / "reference" / "levels"
    (lv / "L3.npy").write_bytes((lv / "L2.npy").read_bytes())
    v = reference.check(task, task / "private", tol=1e-3, rerun=False)
    assert v["verdict"] == "fail"
    assert any("identical" in r for r in v["reasons"])


def test_two_levels_are_not_enough(tmp_path):
    task = _task(tmp_path, levels=2)
    v = reference.check(task, task / "private", rerun=False)
    assert v["verdict"] == "fail" and "need 3" in v["reasons"][0]


def test_levels_that_the_program_does_not_reproduce_fail(tmp_path):
    task = _task(tmp_path, levels=3)
    (task / "private" / "reference" / "solve.py").write_text(
        SOLVE.format(order=2.0).replace("np.cos(3 * x)", "np.cos(4 * x)"), encoding="utf-8")
    v = reference.check(task, task / "private", python=sys.executable)
    assert v["reproducible"] is False and v["verdict"] == "fail"


def test_the_verdict_publishes_only_what_the_framework_computed(tmp_path):
    task = _task(tmp_path, order=2.0, levels=4)
    v = reference.check(task, task / "private", tol=1e-3, python=sys.executable)
    public = tmp_path / "reference.json"
    reference.record(v, task / "private", public)
    shown = json.loads(public.read_text(encoding="utf-8"))
    assert set(shown) <= set(reference.PUBLIC_KEYS)
    assert "method" not in shown and "levels" not in shown, "the agent's account stays private"
    assert shown["estimate"] == v["estimate"] and shown["n_levels"] == 4
    kept = json.loads((task / "private" / "reference" / reference.VERDICT)
                      .read_text(encoding="utf-8"))
    assert kept["method"] == "test"


def test_a_failing_solver_s_output_stays_private(tmp_path):
    task = _task(tmp_path, levels=3)
    (task / "private" / "reference" / "solve.py").write_text(
        "import sys; sys.exit('u at the centre is 0.7071067')", encoding="utf-8")
    v = reference.check(task, task / "private", python=sys.executable)
    assert v["reproducible"] is False
    assert not any("0.7071067" in r for r in v["reasons"]), v["reasons"]
    assert "0.7071067" in v["details"]
    assert "0.7071067" not in json.dumps(reference.public(v))


def _crafted(tmp_path, arrays, ratio=2, ref=None):
    """A task whose refinement levels are given outright (no solver; no re-run)."""
    task = tmp_path / "tasks" / "t"
    work = task / "private" / "reference"
    (work / "levels").mkdir(parents=True)
    (task / "eval.py").write_text(EVAL, encoding="utf-8")
    np.savetxt(task / "eval_grid.csv", np.linspace(0, 1, len(arrays[0])), header="x",
               comments="")
    for k, a in enumerate(arrays, start=1):
        np.save(work / "levels" / f"L{k}.npy", np.asarray(a, dtype=np.float64))
    meta = {"method": "crafted"} if ratio is None else {"method": "crafted", "ratio": ratio}
    (work / "levels.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(task / "private" / "ref_values.npy", arrays[-1] if ref is None else ref)
    return task


def _field(n=101):
    x = np.linspace(0, 1, n)
    base = np.sin(np.pi * x) + 0.5
    return x, base, np.cos(5 * x)


def _levels_with(diffs, n=101):
    """Levels whose successive rRMSE differences are (close to) `diffs`."""
    _, base, g = _field(n)
    levels = [base]
    for d in reversed(diffs):
        prev = levels[0]
        levels.insert(0, prev + d * np.linalg.norm(prev) / np.linalg.norm(g) * g)
    return [np.stack([a, 2 * a], axis=1) for a in levels]


def test_an_order_below_one_is_not_taken_as_accurate(tmp_path):
    # order 0.5: each difference only 1/sqrt(2) of the last; the finest level is
    # still ~2.4 times its last difference away from the limit
    arrays = _levels_with([1.17e-3, 8.27e-4])
    task = _crafted(tmp_path, arrays)
    v = reference.check(task, task / "private", tol=1e-3, rerun=False)
    assert v["order"] == pytest.approx(0.5, abs=0.02)
    assert v["estimate"] > 1.9e-3
    assert v["verdict"] == "fail" and any("exceeds the tolerance" in r for r in v["reasons"])


@pytest.mark.parametrize("ratio", [1, 0.5, 100, "two", None, float("nan"), True, "empty"])
def test_the_refinement_ratio_must_be_a_usable_number(tmp_path, ratio):
    task = _crafted(tmp_path, _levels_with([1e-3, 2.5e-4]), ratio=ratio)
    if ratio == "empty":
        (task / "private" / "reference" / "levels.json").write_text("{}", encoding="utf-8")
    v = reference.check(task, task / "private", rerun=False)
    assert v["verdict"] == "fail"
    assert any("ratio" in r for r in v["reasons"]), v["reasons"]
    if ratio == "two":
        assert "two" not in json.dumps(reference.public(v)), "what the agent wrote is not echoed"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_metric_that_gives_nan_is_a_reason_not_a_crash(tmp_path):
    zero = np.zeros((101, 2))
    task = _crafted(tmp_path, [zero, zero + 0.0, zero * 1.0])
    v = reference.check(task, task / "private", rerun=False)
    assert v["verdict"] == "fail"
    assert any("non-finite" in r for r in v["reasons"]), v["reasons"]


def test_a_reference_file_of_the_wrong_shape_is_a_reason_not_a_crash(tmp_path):
    arrays = _levels_with([1e-3, 2.5e-4])
    task = _crafted(tmp_path, arrays, ref=np.zeros((7, 3)))
    v = reference.check(task, task / "private", rerun=False)
    assert any("ref_values.npy has shape" in r for r in v["reasons"]), v["reasons"]


def test_the_level_count_reason_does_not_say_where_the_levels_are(tmp_path):
    task = _task(tmp_path, levels=2)
    v = reference.check(task, task / "private", rerun=False)
    assert str(task) not in v["reasons"][0] and "need 3" in v["reasons"][0]


def test_a_reference_converged_to_a_roundoff_floor_passes(tmp_path):
    # a spectral solver: the finest levels differ by noise far below the tolerance
    arrays = _levels_with([1e-4, 3e-9, 8e-9])
    task = _crafted(tmp_path, arrays)
    v = reference.check(task, task / "private", tol=1e-3, rerun=False)
    assert v["verdict"] == "pass", v["reasons"]
    assert v["estimate"] == pytest.approx(max(v["differences"][-2:]))


def test_a_reference_that_stalls_above_the_floor_still_fails(tmp_path):
    arrays = _levels_with([1e-3, 1e-4, 2e-4])
    task = _crafted(tmp_path, arrays)
    v = reference.check(task, task / "private", tol=1e-3, rerun=False)
    assert v["verdict"] == "fail" and any("not converging" in r for r in v["reasons"])


def test_the_ingested_grid_is_pinned_and_put_back(tmp_path):
    task = tmp_path / "tasks" / "t"
    home = task / "private"
    home.mkdir(parents=True)
    grid = task / "eval_grid.csv"
    grid.write_text("x,y\n0,0\n1,0\n0,1\n", encoding="utf-8")
    reference.pin_grid(task, home)
    assert reference.grid_drift(task, home) == ""
    grid.write_text("x,y\n0,0\n0,1\n1,0\n", encoding="utf-8")  # sorted: misaligned
    assert "no longer the grid" in reference.grid_drift(task, home)
    assert "put back" in reference.grid_drift(task, home, restore=True)
    assert grid.read_text(encoding="utf-8") == "x,y\n0,0\n1,0\n0,1\n"
    assert reference.grid_drift(task, home) == ""


# ─────────────────────────────── the pipeline ───────────────────────────────


def _pipe(tmp_path, mode_reference=""):
    (tmp_path / "tasks").mkdir(exist_ok=True)
    (tmp_path / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    p = authoring.Pipeline(tmp_path, "heat", out=lambda *_: None)
    p.begin(authoring.Inputs(describe="heat equation", reference=mode_reference))
    return p


def test_stages_run_in_order_and_a_task_is_released_only_at_the_end(tmp_path):
    p = _pipe(tmp_path)
    ran = []

    def step(name):
        def f():
            ran.append(name)
            assert not authoring.is_ready(p.task_dir)
            return True, ""
        return f

    assert p.run({s: step(s) for s in authoring.STAGES}) == 0
    assert ran == list(authoring.STAGES)
    assert authoring.is_ready(p.task_dir)


def test_operator_data_fixes_the_grid_before_the_author_writes(tmp_path):
    p = _pipe(tmp_path, mode_reference="/data/ldc.mat")
    assert p.plan().index("reference") < p.plan().index("author")


def test_a_failed_stage_resumes_where_it_stopped(tmp_path):
    p = _pipe(tmp_path)
    calls = []

    def ok(name):
        return lambda: (calls.append(name), (True, ""))[1]

    steps = {s: ok(s) for s in authoring.STAGES}
    steps["smoke"] = lambda: (calls.append("smoke"), (False, "nan"))[1]
    assert p.run(steps) == 1
    assert authoring.status(p.task_dir)["stages"]["smoke"]["status"] == "failed"
    calls.clear()
    steps["smoke"] = ok("smoke")
    resumed = authoring.Pipeline(tmp_path, "heat", out=lambda *_: None)
    assert resumed.run(steps) == 0
    assert calls == ["smoke", "anchor", "finalize"], "done stages are not redone"


def test_one_author_at_a_time(tmp_path):
    p = _pipe(tmp_path)
    p.lock()
    try:
        with pytest.raises(RuntimeError, match="already being authored"):
            authoring.Pipeline(tmp_path, "heat").lock()
    finally:
        p.unlock()


def test_task_names_must_be_serviceable(tmp_path):
    with pytest.raises(ValueError):
        authoring.Pipeline(tmp_path, "my task!")


def test_a_legacy_task_without_a_record_is_ready(tmp_path):
    (tmp_path / "old").mkdir()
    assert authoring.is_ready(tmp_path / "old")


def test_redo_sets_a_stage_and_everything_after_it_back(tmp_path):
    p = _pipe(tmp_path)
    assert p.run({s: (lambda: (True, "")) for s in authoring.STAGES}) == 0
    again = p.redo("reference")
    assert again == ["reference", "reference_check", "validate", "smoke", "anchor", "finalize"]
    st = authoring.status(p.task_dir)
    assert st["stages"]["author"]["status"] == "done"
    assert all(st["stages"][s]["status"] == "pending" for s in again)
    assert not st["finalized"]
    with pytest.raises(ValueError):
        p.redo("nonsense")


def test_a_crashed_stage_says_only_that_it_crashed(tmp_path):
    p = _pipe(tmp_path)
    secret = "/home/op/secret/cavity.mat"

    def boom():
        raise FileNotFoundError(f"no reference data at {secret}")

    steps = {s: (lambda: (True, "")) for s in authoring.STAGES}
    steps["reference"] = boom
    assert p.run(steps) == 1
    text = authoring.state_path(p.task_dir).read_text(encoding="utf-8")
    assert secret not in text and "FileNotFoundError" in text
    kept = (p.task_dir / "private" / "logs" / authoring.ERROR_LOG).read_text(encoding="utf-8")
    assert secret in kept and "Traceback" in kept


def test_before_runs_under_the_lock(tmp_path):
    p = _pipe(tmp_path)
    seen = []

    def before():
        with pytest.raises(RuntimeError):
            authoring.Pipeline(tmp_path, "heat").lock()
        seen.append(True)

    p.run({s: (lambda: (True, "")) for s in authoring.STAGES}, before=before)
    assert seen == [True]


def test_the_demo_template_states_the_source_not_the_solution():
    from pinnforge.task import examples

    text = (examples.template_dir() / "problem.md").read_text(encoding="utf-8")
    assert "tenth" not in text, "a tenth of the amplitude is u's, not f's"
    assert "12.8" in text and "6.4 times" in text
