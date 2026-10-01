"""view.json: the task decides what the fixed field window draws; nothing in it runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pinnforge.task import contract, examples, registry, view
from pinnforge.ui import viz

SPEED = {"title": "speed |V|", "show": {"magnitude": ["u", "v"]}, "colors": "sequential"}


@pytest.mark.parametrize(
    "coords, spec",
    [
        (["x", "y"], {"axes": {"x": "x", "y": "y"}, "panels": [SPEED, {"show": "v"}]}),
        (["t", "x"], {"axes": {"x": "x", "y": "t"}, "panels": [{"show": "u"}]}),
        (["t", "x"], {"axes": {"x": "x"}, "time": "t", "panels": [{"show": {"abs": "u"}}]}),
        (["t", "x", "y"], {"axes": {"x": "x", "y": "y"}, "time": "t", "panels": [SPEED]}),
        (
            ["x", "y", "z"],
            {
                "axes": {"x": "x", "y": "y"},
                "slice": {"axis": "z", "at": 0.25},
                "panels": [{"show": "u", "colors": "diverging"}],
            },
        ),
    ],
)
def test_views_of_every_shape_are_accepted(coords, spec):
    shown, problems = view.check(spec, coords, ["u", "v"])
    assert problems == [] and shown is not None
    assert all({"op", "of"} <= set(p["show"]) for p in shown["panels"])
    assert all(p["title"] for p in shown["panels"]), (
        "an untitled panel is named after what it shows"
    )


@pytest.mark.parametrize(
    "spec, says",
    [
        ({"axes": {"x": "x"}, "panels": [{"show": "u"}]}, "on no axis"),
        ({"axes": {"x": "x", "y": "x"}, "panels": [{"show": "u"}]}, "used twice"),
        ({"axes": {"x": "q", "y": "y"}, "panels": [{"show": "u"}]}, "not a grid coordinate"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": "p"}]}, "not an output head"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": {"magnitude": ["u"]}}]}, "magnitude"),
        ({"axes": {"x": "x", "y": "y"}, "panels": []}, "1-4 panels"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": "u"}] * 5}, "1-4 panels"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": "u", "colors": "rainbow"}]}, "colors"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": "u", "title": "x" * 41}]}, "title"),
        ({"axes": {"x": "x", "y": "y"}, "panels": [{"show": "u", "script": "alert(1)"}]}, "keys"),
        ({"axes": {"x": "x", "y": "y"}, "code": "1", "panels": [{"show": "u"}]}, "unknown keys"),
        ([], "JSON object"),
    ],
)
def test_anything_else_is_refused(spec, says):
    shown, problems = view.check(spec, ["x", "y"], ["u", "v"])
    assert shown is None and any(says in p for p in problems), problems


def _task(tmp_path, spec=None):
    task = examples.install_demo(tmp_path / "tasks")
    if spec is not None:
        (task / view.VIEW_FILE).write_text(json.dumps(spec), encoding="utf-8")
    return task


def test_the_demo_template_ships_a_valid_view(tmp_path):
    task = _task(tmp_path)
    assert (task / view.VIEW_FILE).is_file(), "authors adapt the template, view included"
    names = {n for n, ok, _ in contract.validate(task).checks if ok}
    assert "view.json draws the task's own coordinates and output heads" in names


@pytest.mark.parametrize("name", ["ldc", "ks", "cyl", "naca"])
def test_the_benchmark_tasks_have_valid_views(name):
    task = Path(__file__).resolve().parents[1] / "tasks" / name
    shown, problems = view.check(view.load(task / view.VIEW_FILE), *view.task_axes(task))
    assert shown is not None, problems


def test_a_bad_view_fails_validation_and_says_why(tmp_path):
    task = _task(tmp_path, {"axes": {"x": "x"}, "panels": [{"show": "p"}]})
    failed = {n: d for n, _, d in contract.validate(task).failures}
    assert (
        "not an output head"
        in failed["view.json draws the task's own coordinates and output heads"]
    )
    task2 = _task(
        tmp_path / "b",
        {"axes": {"x": "x", "y": "y"}, "panels": [{"title": "exact solution", "show": "u"}]},
    )
    assert "view.json says nothing about the solution" in {
        n for n, _, _ in contract.validate(task2).failures
    }


def test_the_view_is_not_part_of_the_anchor_key(tmp_path):
    task = _task(tmp_path)
    before = registry.fingerprint(task)
    (task / view.VIEW_FILE).write_text(
        json.dumps({"axes": {"x": "y", "y": "x"}, "panels": [{"show": "u", "colors": "diverging"}]})
    )
    assert registry.fingerprint(task) == before, "what the page draws cannot move a score"


def test_the_dashboard_serves_only_a_valid_view_of_what_is_archived(tmp_path, monkeypatch):
    from pinnforge.ui import api

    run = tmp_path / "runs" / "demo_1"
    (run / "task").mkdir(parents=True)
    monkeypatch.setattr(api, "ROOT_REAL", str(tmp_path.resolve()))
    grid = {"coords": ["x", "y"], "fields": ["u", "v"]}
    (run / "task" / view.VIEW_FILE).write_text(
        json.dumps({"axes": {"x": "x", "y": "y"}, "panels": [SPEED]})
    )
    shown = api._view(run, grid)
    assert shown["panels"][0]["show"] == {"op": "magnitude", "of": ["u", "v"]}
    # a block rewrote it: anything not a valid view is dropped for the default
    (run / "task" / view.VIEW_FILE).write_text(
        '{"axes": {"x": "x"}, "panels": [{"show": "<img>"}]}'
    )
    assert api._view(run, grid) is None
    (run / "task" / view.VIEW_FILE).write_text("x" * (view.MAX_BYTES + 10))
    assert api._view(run, grid) is None


def test_a_time_dependent_grid_is_sampled_slice_by_slice():
    t, x, y = np.meshgrid(
        np.linspace(0, 1, 31), np.linspace(0, 1.7, 171), np.linspace(0, 0.41, 42), indexing="ij"
    )
    grid = np.column_stack([t.ravel(), x.ravel(), y.ravel()])
    hole = (grid[:, 1] - 0.2) ** 2 + (grid[:, 2] - 0.2) ** 2 < 0.05**2  # a cylinder
    grid = grid[~hole]
    rows = viz.sample_rows(grid, 3, step=0, limit=40_000)
    kept = grid[rows]
    slices = np.unique(kept[:, 0])
    assert len(rows) <= 40_000 and len(slices) == viz.MAX_STEPS
    first = kept[kept[:, 0] == slices[0]][:, 1:]
    last = kept[kept[:, 0] == slices[-1]][:, 1:]
    assert np.array_equal(first, last), "every slice is the same set of points"
    nx, ny = len(np.unique(first[:, 0])), len(np.unique(first[:, 1]))
    assert nx * ny <= 2 * len(first), "still a lattice (with its hole), so it draws as an image"


def test_the_slider_coordinate_comes_from_the_view_or_the_default(tmp_path):
    (tmp_path / view.VIEW_FILE).write_text(
        json.dumps(
            {
                "axes": {"x": "x", "y": "z"},
                "slice": {"axis": "y", "at": 0.5},
                "panels": [{"show": "u"}],
            }
        )
    )
    assert viz.step_column(tmp_path, ["x", "y", "z"], ["u"]) == 1
    (tmp_path / view.VIEW_FILE).unlink()
    assert viz.step_column(tmp_path, ["t", "x", "y"], ["u"]) == 0, "a time coordinate slides"
    assert viz.step_column(tmp_path, ["x", "y", "z"], ["u"]) == 2, "else the last one is sliced"
    assert viz.step_column(tmp_path, ["x", "t"], ["u"]) is None, "two coordinates are both axes"
