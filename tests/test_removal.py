"""Deleting runs and tasks: all they own, nothing while in use, never through a link."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from pinnforge import removal
from pinnforge.cli import delete_cmd
from pinnforge.run import lifecycle


@pytest.fixture()
def ws(tmp_path, monkeypatch, plant):
    root = tmp_path / "ws"
    (root / "pinnforge.yaml").parent.mkdir(parents=True)
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    task = root / "tasks" / "toy"
    (task / "private").mkdir(parents=True)
    (task / "problem.md").write_text("# TOY\n", encoding="utf-8")
    (task / "private" / "ref_values.npy").write_bytes(b"answers")
    (root / ".venv" / "Lib").mkdir(parents=True)
    (root / ".venv" / "Lib" / "keep.txt").write_text("the workspace's own", encoding="utf-8")
    for rid in ("toy_1", "toy_2"):
        run = root / "runs" / rid
        (run / "blocks" / "b01").mkdir(parents=True)
        (run / "run.yaml").write_text("task: toy\nblocks: 1\n", encoding="utf-8")
        (run / "blocks" / "b01" / "evals.jsonl").write_text("{}\n", encoding="utf-8")
        plant(run / ".venv", root / ".venv")  # what a run really links back to
        (root / ".pinnforge" / "scores").mkdir(parents=True, exist_ok=True)
        (root / ".pinnforge" / "scores" / f"{rid}.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("PINNFORGE_ROOT", str(root))
    monkeypatch.delenv("FORGE_PRIVATE_DIR", raising=False)
    return root


def test_a_run_goes_with_its_ledger_and_its_links_are_not_followed(ws):
    plan = removal.run_plan(ws, "toy_1")
    assert plan.blockers == [] and ws / ".pinnforge" / "scores" / "toy_1.jsonl" in plan.paths
    assert removal.carry_out(plan) == []
    assert not (ws / "runs" / "toy_1").exists()
    assert not (ws / ".pinnforge" / "scores" / "toy_1.jsonl").exists()
    assert (ws / ".venv" / "Lib" / "keep.txt").read_text(encoding="utf-8") == "the workspace's own"
    assert (ws / "runs" / "toy_2").is_dir(), "only the run asked for"


def test_a_live_run_is_not_deleted(ws):
    lock = lifecycle.RunLock(ws / "runs" / "toy_1").acquire()
    try:
        plan = removal.run_plan(ws, "toy_1")
        assert plan.blockers and "pinnforge run stop toy_1" in plan.blockers[0]
        with pytest.raises(RuntimeError):
            removal.carry_out(plan)
        assert (ws / "runs" / "toy_1").is_dir()
    finally:
        lock.release()


def test_a_shared_ledger_loses_only_the_runs_lines(ws, tmp_path, monkeypatch):
    shared = tmp_path / "private"
    shared.mkdir()
    (shared / "scores.jsonl").write_text(
        "".join(json.dumps({"queue": q, "n": i}) + "\n" for i, q in
                enumerate(["toy_1", "toy_2", "toy_1"])), encoding="utf-8")
    monkeypatch.setenv("FORGE_PRIVATE_DIR", str(shared))
    plan = removal.run_plan(ws, "toy_1")
    assert plan.ledger_lines == (shared / "scores.jsonl", "toy_1")
    assert removal.carry_out(plan) == []
    left = [json.loads(ln)["queue"] for ln in (shared / "scores.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert left == ["toy_2"]


@pytest.mark.parametrize("name", ["../tasks", "toy_1/..", "", "a" * 65])
def test_only_a_plain_name_is_deleted(ws, name):
    with pytest.raises(ValueError):
        removal.run_plan(ws, name)


def test_a_linked_run_folder_is_refused(ws, plant, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    plant(ws / "runs" / "toy_9", elsewhere)
    with pytest.raises(ValueError, match="link"):
        removal.run_plan(ws, "toy_9")
    assert elsewhere.is_dir()


def test_a_task_goes_with_its_answers_and_names_its_runs(ws):
    (ws / ".pinnforge" / "sealed" / "toy" / "private").mkdir(parents=True)
    plan = removal.task_plan(ws, "toy")
    assert plan.blockers == [] and "toy_1, toy_2" in plan.notes[0] and plan.runs == []
    assert removal.carry_out(plan) == []
    assert not (ws / "tasks" / "toy").exists()
    assert not (ws / ".pinnforge" / "sealed" / "toy").exists()
    assert (ws / "runs" / "toy_1").is_dir(), "its runs stay unless asked for"


def test_a_task_can_take_its_runs_with_it(ws):
    plan = removal.task_plan(ws, "toy", with_runs=True)
    assert [r.what for r in plan.runs] == ["run toy_1", "run toy_2"]
    assert removal.carry_out(plan) == []
    assert not (ws / "runs" / "toy_1").exists() and not (ws / "runs" / "toy_2").exists()
    assert (ws / ".venv" / "Lib" / "keep.txt").is_file()


def test_a_task_a_live_run_scores_against_is_not_deleted(ws):
    lock = lifecycle.RunLock(ws / "runs" / "toy_2").acquire()
    try:
        plan = removal.task_plan(ws, "toy")
        assert any("toy_2" in b for b in plan.blockers)
        assert delete_cmd.cmd_task_delete(Namespace(name="toy", with_runs=False, yes=True)) == 1
        assert (ws / "tasks" / "toy" / "private" / "ref_values.npy").is_file()
    finally:
        lock.release()


def test_the_command_asks_first(ws, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert delete_cmd.cmd_run_delete(Namespace(run_id="toy_1", yes=False)) == 1
    assert (ws / "runs" / "toy_1").is_dir() and "nothing deleted" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert delete_cmd.cmd_run_delete(Namespace(run_id="toy_1", yes=False)) == 0
    assert not (ws / "runs" / "toy_1").exists()
    assert delete_cmd.cmd_run_delete(Namespace(run_id="toy_7", yes=True)) == 2
