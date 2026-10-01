"""Nothing of the reference reaches the task package, the author agent, or a public record.

The package is what a block is handed. It states the problem and nothing
about its solution; the author agent that writes it never sees the answers;
whatever an agent that did see them wrote stays in `private/`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pinnforge.cli import task_new
from pinnforge.task import authoring, contract, examples, reference


def _demo(tmp_path):
    return examples.install_demo(tmp_path / "tasks")


def _failures(task) -> dict[str, str]:
    return {name: detail for name, _, detail in contract.validate(task).failures}


def _ref_and_grid(task):
    ref = np.load(task / "private" / "ref_values.npy")
    grid = np.loadtxt(task / "eval_grid.csv", delimiter=",", skiprows=1)
    return ref.reshape(len(ref), -1), grid


def test_a_clean_package_passes(tmp_path):
    report = contract.validate(_demo(tmp_path))
    assert report.ok, report.render()


@pytest.mark.parametrize("how", ["copy", "rescaled", "coarse"])
def test_the_reference_in_public_data_is_caught(tmp_path, how):
    task = _demo(tmp_path)
    ref, grid = _ref_and_grid(task)
    u = ref[:, 0]
    leak = {"copy": u, "rescaled": 3.0 * u - 1.0,
            "coarse": u * (1 + 3e-3 * np.sin(40 * grid[:, 0]))}[how]
    np.savetxt(task / "hints.csv", np.column_stack([grid, leak]), delimiter=",",
               header="x,y,w", comments="")
    failed = _failures(task)
    assert "no public data file holds the reference" in failed
    assert "hints.csv:w" in failed["no public data file holds the reference"]


def test_coordinates_and_problem_data_are_not_answers(tmp_path):
    task = _demo(tmp_path)
    ref, grid = _ref_and_grid(task)
    # coordinates repeated in another file, and data of another length (an
    # initial or boundary condition is problem data, whatever its values)
    np.savetxt(task / "points.csv", grid, delimiter=",", header="x,y", comments="")
    np.savetxt(task / "edge.csv", np.column_stack([grid[:50], ref[:50, 0]]), delimiter=",",
               header="x,y,u", comments="")
    assert "no public data file holds the reference" not in _failures(task)


def test_printed_reference_values_are_caught(tmp_path):
    task = _demo(tmp_path)
    ref, _ = _ref_and_grid(task)
    table = "\n".join(f"{v:.6g}" for v in ref[::3, 0])
    (task / "notes.txt").write_text(table, encoding="utf-8")
    assert "no public text prints reference values" in _failures(task)


def test_words_about_the_solution_and_paths_to_the_answers_are_caught(tmp_path):
    task = _demo(tmp_path)
    problem = task / "problem.md"
    problem.write_text(problem.read_text().replace(
        "## Scoring\n", "## Scoring\n\nThe exact solution is smooth.\n"), encoding="utf-8")
    base = task / "baseline.py"
    base.write_text(base.read_text() + '\nHINT = "private/ref_values.npy"\nX = "answers.csv"\n',
                    encoding="utf-8")
    failed = _failures(task)
    assert "exact solution" in failed["problem.md says nothing about the solution"]
    assert "baseline.py never names the answers" in failed
    assert "answers.csv" in failed["baseline.py reads only the task's public files"]


LEAK_DATA = "no public data file holds the reference"
LEAK_TEXT = "no public text prints reference values"


def _u(task):
    ref, grid = _ref_and_grid(task)
    return ref[:, 0], grid


@pytest.mark.parametrize("how", ["numpy print", "json", "savetxt", "symmetric %.6g", "repr"])
def test_printed_values_are_caught_however_printed(tmp_path, how):
    task = _demo(tmp_path)
    u, _ = _u(task)
    field = u.reshape(101, 101)
    text = {
        "numpy print": str(np.round(u[1000:1400], 8)),
        "json": json.dumps(u[::7].tolist()),
        "savetxt": None,
        # a quarter of a field symmetric in x and y: every value repeats elsewhere
        "symmetric %.6g": "\n".join(f"{v:.6g}" for v in field[1:50, 1:50].ravel()),
        "repr": repr(list(u[2000:2300])),
    }[how]
    if text is None:
        np.savetxt(task / "notes.md", u[5::9])
    else:
        (task / "notes.md").write_text(text, encoding="utf-8")
    assert LEAK_TEXT in _failures(task), how


def test_problem_constants_and_hyperparameters_are_not_printed_values(tmp_path):
    task = _demo(tmp_path)
    problem = task / "problem.md"
    problem.write_text(problem.read_text().replace(
        "## Domain\n", "Constants: $\\nu = 0.001234$, $\\pi \\approx 3.14159$, $L = 2.7183$, "
        "$c = 0.70711$.\n\n## Domain\n"), encoding="utf-8")
    base = task / "baseline.py"
    base.write_text(base.read_text() + "\nHP = dict(lr=1.234e-3, width=128, depth=4, "
                    "decay=0.9995, beta=0.12345, eps=1.0e-8, warmup=1000)\n", encoding="utf-8")
    assert LEAK_TEXT not in _failures(task)


def test_the_authoring_records_are_not_the_package(tmp_path):
    task = _demo(tmp_path)
    u, _ = _u(task)
    auth = task / ".authoring"
    auth.mkdir()
    (auth / "author.log").write_text("\n".join(f"step {i} loss {v:.6g}" for i, v in
                                               enumerate(u[::3])), encoding="utf-8")
    assert LEAK_TEXT not in _failures(task), "training-loss lines in a transcript are not a leak"


@pytest.mark.parametrize("how", ["headerless csv", "txt", "json", "string column", "2-D field",
                                 "subsampled", "offset subsample", "npz"])
def test_data_files_holding_the_reference_are_caught(tmp_path, how):
    task = _demo(tmp_path)
    u, grid = _u(task)
    if how == "headerless csv":
        np.savetxt(task / "hints.csv", np.column_stack([grid, u]), delimiter=",")
    elif how == "txt":
        np.savetxt(task / "u.txt", u)
    elif how == "json":
        (task / "field.json").write_text(json.dumps({"meta": "x", "u": u.tolist()}))
    elif how == "string column":
        rows = "\n".join(f"p{i},{x:.6f},{y:.6f},{v:.8f}" for i, ((x, y), v) in
                         enumerate(zip(grid, u, strict=True)))
        (task / "points.csv").write_text("label,x,y,w\n" + rows + "\n")
    elif how == "2-D field":
        np.save(task / "field.npy", u.reshape(101, 101))
    elif how == "subsampled":
        np.save(task / "coarse.npy", u[::4])
    elif how == "offset subsample":
        np.savetxt(task / "coarse.dat", u[1::3])
    else:
        np.savez(task / "bundle.npz", a=np.arange(5.0), b=u.reshape(101, 101).T)
    assert LEAK_DATA in _failures(task), how


def test_the_grid_s_coordinates_are_set_aside_before_any_file_is_read(tmp_path):
    task = _demo(tmp_path)
    _, grid = _u(task)
    # a field affine in x: the coordinate itself, repeated in a file that sorts
    # before eval_grid.csv, is not the answers
    np.save(task / "private" / "ref_values.npy", (2.0 * grid[:, 0] + 1.0).reshape(-1, 1))
    np.savetxt(task / "collocation.csv", grid, delimiter=",", header="x,y", comments="")
    assert LEAK_DATA not in _failures(task)


def test_masks_and_flags_are_not_compared(tmp_path):
    task = _demo(tmp_path)
    _, grid = _u(task)
    step = (grid[:, 0] > 0.5).astype(np.float64)
    np.save(task / "private" / "ref_values.npy", step.reshape(-1, 1))
    np.savetxt(task / "mask.csv", np.column_stack([grid, step]), delimiter=",",
               header="x,y,inside", comments="")
    assert LEAK_DATA not in _failures(task)


def test_sentences_that_deny_a_solution_and_prose_about_privacy_pass(tmp_path):
    task = _demo(tmp_path)
    problem = task / "problem.md"
    problem.write_text(problem.read_text().replace(
        "## Domain\n", "No closed-form solution is known. There is no exact solution "
        "for these data either; the analytical solution is unknown.\n\n## Domain\n"),
        encoding="utf-8")
    base = task / "baseline.py"
    base.write_text(base.read_text() + '\n"""Kept private to this module."""\n'
                    'SUFFIX = ".csv"\nPATTERN = "*.csv"\nNOTE = "results go to out.csv later"\n',
                    encoding="utf-8")
    failed = _failures(task)
    assert "problem.md says nothing about the solution" not in failed
    assert "baseline.py never names the answers" not in failed
    assert "baseline.py reads only the task's public files" not in failed


@pytest.mark.parametrize("literal", ['"private"', '"../private/x.npy"', '"private\\\\ref.npy"'])
def test_private_as_a_path_is_still_caught(tmp_path, literal):
    task = _demo(tmp_path)
    base = task / "baseline.py"
    base.write_text(base.read_text() + f"\nWHERE = {literal}\n", encoding="utf-8")
    assert "baseline.py never names the answers" in _failures(task)


def test_the_scoring_truth_is_the_answers_not_a_directory(tmp_path):
    task = _demo(tmp_path)
    (task / "private" / "ref_values.npy").unlink()
    (task / "private" / "logs").mkdir()
    (task / "private" / "logs" / "pipeline.log").write_text("x")
    assert "ships the scoring truth under private/" in _failures(task)


# ─────────────────────────────── the pipeline ───────────────────────────────


def _steps(tmp_path, reference_location=""):
    (tmp_path / "tasks").mkdir(exist_ok=True)
    (tmp_path / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    pipe = authoring.Pipeline(tmp_path, "heat", out=lambda *_: None)
    if reference_location:
        reference.remember_source(pipe.task_dir / "private", reference_location)
    pipe.begin(authoring.Inputs(describe="heat equation",
                                reference="operator" if reference_location else "",
                                agent={"runtime": "claude_code", "model": "m"}))
    return task_new.Steps(pipe, tmp_path)


def test_the_intake_and_the_record_never_say_where_the_answers_are(tmp_path):
    data = tmp_path / "secret_solution.npz"
    data.write_bytes(b"x")
    steps = _steps(tmp_path, str(data))
    assert steps.intake()[0]
    for f in steps.auth_dir.iterdir():
        assert "secret_solution" not in f.read_text(), f.name
    assert "supplied by the operator." in (steps.auth_dir / "intake.md").read_text()
    assert reference.source_location(steps.private_home, "operator") == str(data)


def test_private_is_out_of_the_tree_while_the_author_works(tmp_path):
    steps = _steps(tmp_path, str(tmp_path / "data.npz"))
    answers = steps.private_home / "ref_values.npy"
    answers.write_bytes(b"answers")
    with steps._sealed():
        assert not steps.private_home.exists(), "the author's directory has no private/"
    assert answers.read_bytes() == b"answers", "and the answers come back after"

    # a crash mid-author leaves them sealed away; the next session puts them back
    # (under the pipeline's lock, before any stage)
    held = steps._sealed_home() / "private"
    held.parent.mkdir(parents=True, exist_ok=True)
    steps.private_home.rename(held)
    task_new.Steps(steps.pipe, tmp_path).recover()
    assert answers.read_bytes() == b"answers"


def _transcript(path, *calls):
    lines = [json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "input": args}]}}) for name, args in calls]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_audit_flags_what_the_author_reached_for(tmp_path):
    steps = _steps(tmp_path, "/data/runs/cavity_solution.mat")
    log = steps.auth_dir / "author.log"
    _transcript(log,
                ("Bash", {"command": "ls -la && cat eval.py"}),
                ("Write", {"file_path": "baseline.py", "content": "# answers live in private/"}),
                ("Read", {"file_path": "../ldc/eval_grid.csv"}),
                ("Read", {"file_path": "/ws/tasks/heat/.authoring/intake.md"}))
    assert steps._audit(log) == [], "listing, reading public files and writing are fine"

    for call in (("Bash", {"command": "python -c 'np.load(\"../ldc/private/ref_values.npy\")'"}),
                 ("Read", {"file_path": "/data/runs/cavity_solution.mat"}),
                 ("Bash", {"command": "ls ~/pf/.pinnforge/sealed/heat"}),
                 ("Read", {"file_path": "/ws/tasks/laplace/.authoring/author.log"})):
        _transcript(log, call)
        assert steps._audit(log), call


def test_an_author_that_reached_for_the_answers_is_not_trusted(tmp_path, monkeypatch):
    steps = _steps(tmp_path)
    steps.intake()

    def author(name, **kw):
        _transcript(kw["log_path"], ("Bash", {"command": "cat ../ks/private/ref_values.npy"}))
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author)
    ok, note = steps.author()
    assert ok is False and "reached for the answers" in note
    assert not (steps.auth_dir / "author.log").exists(), "its transcript leaves the public record"
    assert (steps.private_home / "logs" / "author.untrusted.log").is_file()


def test_the_reference_agent_may_not_touch_the_package(tmp_path):
    steps = _steps(tmp_path)
    steps.pipe.state["stages"]["author"]["status"] = "done"
    (steps.task_dir / "problem.md").write_text("# HEAT — heat\n", encoding="utf-8")
    before = steps._public()
    assert steps._guard(before, set()) == ""
    (steps.task_dir / "problem.md").write_text("# HEAT — heat\nhint: u peaks at 0.98\n")
    (steps.task_dir / "u_peak_0.98.txt").write_text("0.98\n")
    why = steps._guard(before, set())
    assert "2 file(s)" in why
    assert "problem.md" not in why and "0.98" not in why, "the note names nothing it wrote"
    # what it wrote left the package for the private logs, and the author writes it again
    assert not (steps.task_dir / "problem.md").exists()
    assert not (steps.task_dir / "u_peak_0.98.txt").exists()
    kept = steps.private_home / "logs" / "reference-guard-1"
    assert "0.98" in (kept / "problem.md").read_text()
    assert steps.pipe.state["stages"]["author"]["status"] == "pending"
    before = steps._public()
    (steps.task_dir / "eval_grid.csv").write_text("x\n0\n")
    assert steps._guard(before, {"eval_grid.csv"}) == ""


def test_the_guard_covers_the_authoring_records_and_the_anchor_cache(tmp_path):
    steps = _steps(tmp_path)
    steps.intake()
    before = steps._public()
    # the framework's own files may change while an agent runs
    steps.pipe._save()
    (steps.auth_dir / "detach.json").write_text("{}")
    (steps.auth_dir / "lock").write_text("1")
    assert steps._guard(before, set()) == ""
    for rel in (".authoring/notes.md", ".anchor/b00.json"):
        before = steps._public()
        f = steps.task_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("u(0.5) = 0.7071\n")
        assert steps._guard(before, set()), rel
        assert not f.exists()
