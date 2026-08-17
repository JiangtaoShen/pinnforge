"""Run layout, id allocation, and the block done-check."""

from __future__ import annotations

import json
import shutil

import pytest

from pinnforge import paths
from pinnforge.config import RunConfig
from pinnforge.orchestrator import verify_block
from pinnforge.run import layout
from pinnforge.types import EvalRecord, RunState, best_score, spent_seconds


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A throwaway project root holding a copy of one library task."""
    (tmp_path / "pinnforge").mkdir()
    (tmp_path / "tasks").mkdir()
    shutil.copytree(paths.tasks_dir() / "ldc", tmp_path / "tasks" / "ldc")
    (tmp_path / "kb1").mkdir()
    # No charter/ here on purpose: the charter must come from the package,
    # which is what a real deployment does (the override path has its own test).
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    return tmp_path


def test_run_ids_increment_per_task(project):
    assert paths.next_run_id("ldc") == "ldc_1"
    layout.create_run(RunConfig(task="ldc"))
    assert paths.next_run_id("ldc") == "ldc_2"
    layout.create_run(RunConfig(task="ldc"))
    assert paths.next_run_id("ldc") == "ldc_3"
    # a different task numbers independently
    assert paths.next_run_id("ks") == "ks_1"


def test_run_id_parsing_allows_underscores_in_task_names():
    assert paths.parse_run_id("heat_lt_12") == ("heat_lt", 12)
    with pytest.raises(ValueError):
        paths.parse_run_id("no-number")


def test_run_layout_looks_like_a_project_root(project):
    """The charter's relative paths must resolve inside a run directory."""
    run = layout.create_run(RunConfig(task="ldc"))
    assert (run / "block.md").is_file()
    assert (run / "task" / "problem.md").is_file()
    assert (run / "task" / "eval.py").is_file()
    assert (run / "kb1").is_dir()
    assert (run / "blocks" / "kb2").is_dir()
    assert (run / "run.yaml").is_file()
    assert (run / "state.json").is_file()


def test_task_is_copied_not_linked(project):
    """Editing the library mid-run must not change a running experiment."""
    run = layout.create_run(RunConfig(task="ldc"))
    assert not (run / "task").is_symlink()
    (project / "tasks" / "ldc" / "problem.md").write_text("# CHANGED — x\n", encoding="utf-8")
    assert "CHANGED" not in (run / "task" / "problem.md").read_text(encoding="utf-8")


def test_budget_comes_from_the_task_and_reaches_the_charter(project):
    """One source of truth: the task declares it, the charter quotes it.

    In the framework this replaces the number was duplicated by hand across
    the charter and several docs, and a task swap that missed one left blocks
    working to a budget nobody had set.
    """
    run = layout.create_run(RunConfig(task="ldc"))
    cfg = RunConfig.load(paths.config_path(run))
    assert cfg.sandbox.wall_budget_s == 3600.0  # ldc's eval.py declares this
    charter = (run / "block.md").read_text(encoding="utf-8")
    assert "**Budget:** 3600 s" in charter
    assert "3600 s of\nGPU-run wall time" in charter


def test_explicit_budget_overrides_the_task(project):
    cfg = RunConfig(task="ldc")
    cfg.sandbox.wall_budget_s = 5000.0
    run = layout.create_run(cfg)
    assert RunConfig.load(paths.config_path(run)).sandbox.wall_budget_s == 5000.0
    assert "**Budget:** 5000 s" in (run / "block.md").read_text(encoding="utf-8")


def test_second_run_of_same_task_refuses_to_clobber(project):
    layout.create_run(RunConfig(task="ldc"))
    with pytest.raises(FileExistsError):
        layout.create_run(RunConfig(task="ldc"), run_id="ldc_1")


def test_block_id_allocation(project):
    run = layout.create_run(RunConfig(task="ldc"))
    assert layout.next_block_id(run) == "b00"
    (run / "blocks" / "b00").mkdir()
    assert layout.next_block_id(run) == "b01"
    (run / "blocks" / "b01").mkdir()
    assert layout.next_block_id(run) == "b02"
    # kb2 is not a block
    assert "kb2" not in layout.existing_block_ids(run)


def _write_evals(run, block, records):
    d = run / "blocks" / block
    d.mkdir(parents=True, exist_ok=True)
    (d / "evals.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_verify_block_requires_all_three_conditions(project):
    cfg = RunConfig(task="ldc")
    cfg.sandbox.wall_budget_s = 1000
    run = layout.create_run(cfg)
    per_run = 100.0  # threshold becomes 900

    _write_evals(run, "b01", [{"smoke": False, "diag": False, "wall_s": 950, "rRMSE": 0.5}])
    v = verify_block(run, "b01", cfg, per_run)
    assert not v.done and not v.has_summary  # summary missing
    assert v.only_summary_missing

    (run / "blocks" / "kb2" / "b01.md").write_text("# b01 — x\n", encoding="utf-8")
    assert verify_block(run, "b01", cfg, per_run).done

    # budget short of the threshold keeps the block open
    _write_evals(run, "b02", [{"smoke": False, "diag": False, "wall_s": 100, "rRMSE": 0.4}])
    (run / "blocks" / "kb2" / "b02.md").write_text("# b02 — x\n", encoding="utf-8")
    assert not verify_block(run, "b02", cfg, per_run).done

    # a spent budget with only smoke/diag runs is not a finished block
    _write_evals(
        run,
        "b03",
        [
            {"smoke": True, "diag": False, "wall_s": 0, "rRMSE": 0.9},
            {"smoke": False, "diag": True, "wall_s": 950, "rRMSE": None},
        ],
    )
    (run / "blocks" / "kb2" / "b03.md").write_text("# b03 — x\n", encoding="utf-8")
    v3 = verify_block(run, "b03", cfg, per_run)
    assert not v3.done and v3.scored_evals == 0


def test_budget_file_wins_when_larger(project):
    cfg = RunConfig(task="ldc")
    cfg.sandbox.wall_budget_s = 1000
    run = layout.create_run(cfg)
    _write_evals(run, "b01", [{"smoke": False, "diag": False, "wall_s": 10, "rRMSE": 0.5}])
    (run / "blocks" / "b01" / ".budget").write_text("980.0", encoding="utf-8")
    assert verify_block(run, "b01", cfg, 100.0).spent == 980.0


def test_state_round_trips(project):
    run = layout.create_run(RunConfig(task="ldc", blocks=3))
    state = RunState.load(paths.state_path(run))
    assert state.task == "ldc" and state.run_id == run.name
    state.status = "running"
    state.save(paths.state_path(run))
    assert RunState.load(paths.state_path(run)).status == "running"


def test_eval_record_accounting():
    recs = [
        EvalRecord.parse('{"smoke": true, "diag": false, "wall_s": 0, "rRMSE": 0.9}'),
        EvalRecord.parse('{"smoke": false, "diag": true, "wall_s": 60, "rRMSE": null}'),
        EvalRecord.parse('{"smoke": false, "diag": false, "wall_s": 150, "rRMSE": 0.3}'),
        EvalRecord.parse('{"smoke": false, "diag": false, "train_s": 150}'),
    ]
    assert spent_seconds(recs) == 360.0  # smoke is free, crashes still charged
    assert best_score(recs) == 0.3
    assert EvalRecord.parse("not json") is None


def test_running_outside_a_project_says_so(tmp_path, monkeypatch):
    """A missing project must not look like an empty task library."""
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(paths.ProjectNotFound, match="no PINNForge project"):
        paths.project_root()


def test_a_crash_does_not_hide_gpu_time_from_the_summary(tmp_path):
    """`.budget` is charged as a run starts; its record is written when it ends.

    A block killed mid-evaluation therefore leaves the counter ahead of the
    log, and summing records alone under-reports exactly the run that was
    interrupted. `run status` and `run_summary.md` both have to read the
    counter, or the same run gets two different costs depending on which
    command you ask.
    """
    import json

    from pinnforge.run import ledger
    from pinnforge.types import EvalRecord, charged_seconds

    b01 = tmp_path / "blocks" / "b01"
    b01.mkdir(parents=True)
    (b01 / "evals.jsonl").write_text(
        json.dumps({"block": "b01", "smoke": False, "diag": False,
                    "wall_s": 300.0, "rRMSE": 0.5}) + "\n",
        encoding="utf-8",
    )
    (b01 / ".budget").write_text("600.0\n", encoding="utf-8")  # one record lost

    recs = EvalRecord.read_all(b01 / "evals.jsonl")
    assert charged_seconds(recs, b01 / ".budget") == 600.0
    assert ledger.block_rows(tmp_path)[0]["wall_s"] == 600.0


def test_records_ahead_of_the_counter_still_count(tmp_path):
    """The counter can lag a record that just landed — take the larger."""
    import json

    from pinnforge.types import EvalRecord, charged_seconds

    b = tmp_path / "b"
    b.mkdir()
    (b / "evals.jsonl").write_text(
        json.dumps({"block": "b01", "smoke": False, "diag": False,
                    "wall_s": 900.0, "rRMSE": 0.1}) + "\n",
        encoding="utf-8",
    )
    (b / ".budget").write_text("600.0\n", encoding="utf-8")
    assert charged_seconds(EvalRecord.read_all(b / "evals.jsonl"), b / ".budget") == 900.0


def test_a_missing_counter_falls_back_to_the_records(tmp_path):
    import json

    from pinnforge.types import EvalRecord, charged_seconds

    b = tmp_path / "b"
    b.mkdir()
    (b / "evals.jsonl").write_text(
        json.dumps({"block": "b01", "smoke": False, "diag": False,
                    "wall_s": 120.0, "rRMSE": 0.2}) + "\n",
        encoding="utf-8",
    )
    assert charged_seconds(EvalRecord.read_all(b / "evals.jsonl"), b / ".budget") == 120.0


def test_the_run_directory_carries_what_the_charter_names(tmp_path, monkeypatch):
    """Every path the charter tells a block to use has to resolve from cwd.

    The charter offers `uv add` for stack extensions "if the pins stay
    intact". A run directory without the pins makes that sentence a dead end
    — the command the block was told to use fails on a manifest that is not
    there.
    """
    import shutil

    from pinnforge import paths as _paths
    from pinnforge.config import RunConfig
    from pinnforge.run import layout

    root = tmp_path
    (root / "pinnforge").mkdir()
    (root / "runs").mkdir()
    (root / "tasks").mkdir()
    shutil.copytree(_paths.tasks_dir() / "ldc", root / "tasks" / "ldc")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "uv.lock").write_text("# lock\n", encoding="utf-8")
    monkeypatch.setenv("PINNFORGE_ROOT", str(root))

    run = layout.create_run(RunConfig(task="ldc"))
    for named in ("task/eval.py", "task/problem.md", "task/baseline.py",
                  "kb1", "pyproject.toml", "uv.lock"):
        assert (run / named).exists(), named
    assert (run / "pyproject.toml").is_symlink(), "one environment, one manifest"
