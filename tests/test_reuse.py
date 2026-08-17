"""Re-running a task must cost nothing it has already paid for.

A run freezes the task it was given and keeps the b00 it measured. So a
library that lost its cached anchor, or its copy of the package, can take
both back from the run library instead of spending a GPU or an authoring
session — and, more to the point, gets back the *same* definition, which is
what keeps the new runs comparable with the old.
"""

from __future__ import annotations

import json

from pinnforge import paths
from pinnforge.run import anchor
from pinnforge.task import registry

CONTRACT = ("problem.md", "baseline.py", "eval.py")


def _seed_task(root, name="toy", body="x = 1\n"):
    task = root / "tasks" / name
    (task / "private").mkdir(parents=True)
    for f in CONTRACT:
        (task / f).write_text(body, encoding="utf-8")
    (task / "grid.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (task / "private" / "answers.npy").write_bytes(b"secret")
    return task


def _seed_run(root, name="toy", n=1, score=0.5, fingerprint_from=None):
    run = root / "runs" / f"{name}_{n}"
    (run / "task").mkdir(parents=True)
    src = fingerprint_from or (root / "tasks" / name)
    for f in sorted(src.iterdir()):
        if f.is_file():
            (run / "task" / f.name).write_bytes(f.read_bytes())
    b00 = run / "blocks" / "b00"
    b00.mkdir(parents=True)
    (b00 / "b00_v01.py").write_text("x = 1\n", encoding="utf-8")
    (b00 / "b00_v01.pkl").write_bytes(b"params")
    (b00 / ".budget").write_text("42.0\n", encoding="utf-8")
    (b00 / "evals.jsonl").write_text(
        json.dumps({"block": "b00", "smoke": False, "diag": False,
                    "wall_s": 42.0, "rRMSE": score}) + "\n",
        encoding="utf-8",
    )
    (run / "blocks" / "kb2").mkdir(parents=True)
    (run / "blocks" / "kb2" / "b00.md").write_text("# b00\n", encoding="utf-8")
    return run


def _project(tmp_path, monkeypatch):
    (tmp_path / "pinnforge").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    return tmp_path


def test_anchor_is_harvested_from_a_matching_run(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    _seed_run(root, score=0.37)

    result = anchor.harvest("toy", root)
    assert result is not None
    assert result.rRMSE == 0.37
    assert "toy_1" in result.detail

    cache = anchor.cache_dir("toy", root)
    for f in (*anchor.ANCHOR_FILES, "b00.md", "meta.json"):
        assert (cache / f).is_file(), f
    assert json.loads((cache / "meta.json").read_text())["rRMSE"] == 0.37


def test_a_run_of_different_contract_files_is_not_harvested(tmp_path, monkeypatch):
    """The same guard as `restore`: a cheaper source, not a different one."""
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    run = _seed_run(root)
    (run / "task" / "baseline.py").write_text("x = 2  # changed\n", encoding="utf-8")

    assert anchor.harvest("toy", root) is None
    assert not anchor.cache_dir("toy", root).exists()


def test_the_newest_matching_run_wins(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    old = _seed_run(root, n=1, score=0.9)
    new = _seed_run(root, n=2, score=0.4)
    import os

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    assert anchor.harvest("toy", root).rRMSE == 0.4


def test_no_run_means_no_harvest(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    assert anchor.harvest("toy", root) is None


def test_definition_is_recovered_from_a_run(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    _seed_run(root)
    original = {f: (task / f).read_bytes() for f in (*CONTRACT, "grid.csv")}
    for f in (*CONTRACT, "grid.csv"):
        (task / f).unlink()

    found = registry.recover_from_run("toy", root)
    assert found is not None
    dest, run = found
    assert run.name == "toy_1"
    for f, want in original.items():
        assert (dest / f).read_bytes() == want, f


def test_recovery_cannot_bring_back_the_answers(tmp_path, monkeypatch):
    """`private/` is never copied into a run — that is the split, not a gap."""
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    _seed_run(root)
    import shutil

    shutil.rmtree(task)

    dest, _ = registry.recover_from_run("toy", root)
    assert (dest / "problem.md").is_file()
    assert not (dest / "private").exists()


def test_runs_of_ignores_other_tasks(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root, "toy")
    _seed_task(root, "other")
    _seed_run(root, "toy")
    _seed_run(root, "other")
    assert [r.name for r in registry.runs_of("toy", root)] == ["toy_1"]


def test_paths_still_resolve_under_a_test_root(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    assert paths.tasks_dir() == root / "tasks"
    assert paths.runs_dir() == root / "runs"
