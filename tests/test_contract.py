"""The task contract, and the paradigm it enforces."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pinnforge import paths
from pinnforge.task import contract, registry

LIBRARY = ("ldc", "ks", "naca")


def _clone(name: str, dest: Path) -> None:
    """Copy a library task, private data and all, so it can be mutated."""
    shutil.copytree(paths.tasks_dir() / name, dest, dirs_exist_ok=True)


@pytest.mark.parametrize("name", LIBRARY)
def test_library_tasks_are_valid(name):
    report = contract.validate(paths.tasks_dir() / name)
    assert report.ok, report.render()


@pytest.mark.parametrize("name", LIBRARY)
def test_library_tasks_declare_budgets(name):
    task_dir = paths.tasks_dir() / name
    assert contract.train_time(task_dir) > 0
    assert contract.wall_budget(task_dir) > 0


def test_missing_files_fail_fast(tmp_path):
    (tmp_path / "problem.md").write_text("# X — y\n", encoding="utf-8")
    report = contract.validate(tmp_path)
    assert not report.ok
    names = [n for n, ok, _ in report.failures]
    assert "baseline.py present" in names
    assert "eval.py present" in names


def test_extra_section_is_rejected(tmp_path):
    _clone("ldc", tmp_path)
    problem = tmp_path / "problem.md"
    problem.write_text(
        problem.read_text(encoding="utf-8") + "\n## Data files\n\nextra.\n", encoding="utf-8"
    )
    report = contract.validate(tmp_path)
    assert not report.ok
    assert any("extra top-level" in n for n, ok, _ in report.failures)


def test_subsections_are_rejected(tmp_path):
    _clone("ldc", tmp_path)
    problem = tmp_path / "problem.md"
    text = problem.read_text(encoding="utf-8").replace(
        "## Scoring", "## Scoring\n\n### Reference accuracy\n\nnotes."
    )
    problem.write_text(text, encoding="utf-8")
    report = contract.validate(tmp_path)
    assert any("'###'" in n for n, ok, _ in report.failures)


def test_fingerprint_covers_what_decides_the_anchor():
    fp = registry.fingerprint(paths.tasks_dir() / "ldc")
    assert fp["schema"] == registry.FINGERPRINT_SCHEMA
    # ldc evaluates through the shared machinery, so that is part of its key
    assert set(fp["files"]) == {"baseline.py", "eval.py", "_forgekit.py"}
    assert "eval_grid.csv" in fp["data"]


def test_rewording_problem_md_does_not_stale_the_anchor(tmp_path):
    """Prose a block reads cannot move a number measured without a block."""
    _clone("ldc", tmp_path)
    before = registry.fingerprint(tmp_path)
    problem = tmp_path / "problem.md"
    problem.write_text(problem.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert registry.fingerprint(tmp_path) == before


def test_changed_public_data_stales_the_anchor(tmp_path):
    _clone("ldc", tmp_path)
    before = registry.fingerprint(tmp_path)
    grid = tmp_path / "eval_grid.csv"
    grid.write_text(grid.read_text(encoding="utf-8") + "0,0\n", encoding="utf-8")
    assert registry.fingerprint(tmp_path) != before


def test_a_v1_fingerprint_is_still_recognised(tmp_path):
    """An anchor cached before schema 2 stays valid while its files are unchanged."""
    _clone("ldc", tmp_path)
    assert registry.same_package(registry.fingerprint_v1(tmp_path), tmp_path)


def test_fingerprint_changes_with_baseline(tmp_path):
    _clone("ldc", tmp_path)
    before = registry.fingerprint(tmp_path)
    (tmp_path / "baseline.py").write_text(
        (tmp_path / "baseline.py").read_text(encoding="utf-8") + "\n# tweak\n", encoding="utf-8"
    )
    assert registry.fingerprint(tmp_path) != before


def test_inline_scoring_is_rejected(tmp_path):
    """A task that scores inside eval.py cannot honour the firewall."""
    _clone("ldc", tmp_path)
    ev = tmp_path / "eval.py"
    ev.write_text(
        ev.read_text(encoding="utf-8")
        .replace("def score_predictions(", "def _old_inline_score(")
        .replace("_forgekit", "_gone")
        .replace("submit_score", "_gone"),
        encoding="utf-8",
    )
    names = [n for n, ok, _ in contract.validate(tmp_path).failures]
    assert any("score_predictions" in n for n in names)
    assert any("submits predictions" in n for n in names)


def test_missing_public_grid_is_rejected(tmp_path):
    """Without a grid the agent has nowhere to predict."""
    _clone("ldc", tmp_path)
    (tmp_path / "eval_grid.csv").unlink()
    names = [n for n, ok, _ in contract.validate(tmp_path).failures]
    assert any("eval_grid.csv" in n for n in names)


@pytest.mark.parametrize("body, says", [
    ("x,y\nnp.float64(0.0),np.float64(0.0)\n0.5,0.5\n", "plain decimal numbers"),
    ("0.0,0.0\n0.5,0.5\n", "header line"),
    ("x,y\n0.0,0.0,1.0\n0.5,0.5,1.0\n", "3 values for 2 header names"),
    ("x,y\n0.0,nan\n0.5,0.5\n", "non-finite"),
    ("x,y\n0.0,0.0\n", "fewer than two rows"),
])
def test_an_unreadable_grid_is_named(tmp_path, body, says):
    grid = tmp_path / "eval_grid.csv"
    grid.write_text(body, encoding="utf-8")
    assert says in contract.grid_problem(grid)


def test_a_plain_grid_passes(tmp_path):
    grid = tmp_path / "eval_grid.csv"
    grid.write_text("x,y\n0,0\n0.5,1e-3\n1,1\n", encoding="utf-8")
    assert contract.grid_problem(grid) is None
