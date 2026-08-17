"""Record integrity — what a block may change, and what it may not.

The framework states an invariant ("each block's non-smoke records stay
untouched once written") that nothing used to enforce or even observe. These
tests pin both halves: legitimate block work passes untouched, and every way
of corrupting the record is caught.
"""

from __future__ import annotations

import json
import shutil

import pytest

from pinnforge import integrity, paths
from pinnforge.config import RunConfig
from pinnforge.run import layout


@pytest.fixture()
def run(tmp_path, monkeypatch):
    (tmp_path / "pinnforge").mkdir()
    (tmp_path / "tasks").mkdir()
    shutil.copytree(paths.tasks_dir() / "ldc", tmp_path / "tasks" / "ldc")
    kb1 = tmp_path / "kb1"
    kb1.mkdir()
    (kb1 / "001_2019_A_paper.md").write_text("# note\n", encoding="utf-8")
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))

    r = layout.create_run(RunConfig(task="ldc"))
    # one finished block, so there is a record worth protecting
    b01 = r / "blocks" / "b01"
    b01.mkdir(parents=True)
    (b01 / "evals.jsonl").write_text(
        json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.7}) + "\n",
        encoding="utf-8",
    )
    (b01 / "b01_first.py").write_text("# candidate\n", encoding="utf-8")
    (r / "blocks" / "kb2" / "b01.md").write_text("# b01 — done\n", encoding="utf-8")
    return r


def _snap(run, block="b02"):
    return integrity.snapshot(run, block, paths.project_root())


def test_a_wellbehaved_block_leaves_no_violations(run):
    """Everything the charter permits must pass: own workspace, own summary."""
    man = _snap(run)
    b02 = run / "blocks" / "b02"
    b02.mkdir(parents=True)
    (b02 / "b02_try.py").write_text("# mine\n", encoding="utf-8")
    (b02 / "evals.jsonl").write_text(
        json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.6}) + "\n",
        encoding="utf-8",
    )
    (b02 / "b02_try.pkl").write_bytes(b"params")
    (run / "blocks" / "kb2" / "b02.md").write_text("# b02 — mine\n", encoding="utf-8")
    assert integrity.verify(run, man) == []


def test_editing_another_blocks_ledger_is_caught(run):
    """The silent corruption this exists to find."""
    man = _snap(run)
    ledger = run / "blocks" / "b01" / "evals.jsonl"
    ledger.write_text(
        json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.01}) + "\n",
        encoding="utf-8",
    )
    v = integrity.verify(run, man)
    assert [x.kind for x in v] == ["modified"]
    assert "b01/evals.jsonl" in v[0].path


def test_deleting_another_blocks_work_is_caught(run):
    man = _snap(run)
    (run / "blocks" / "b01" / "b01_first.py").unlink()
    assert [x.kind for x in integrity.verify(run, man)] == ["removed"]


def test_overwriting_another_blocks_summary_is_caught(run):
    man = _snap(run)
    (run / "blocks" / "kb2" / "b01.md").write_text("# b01 — rewritten\n", encoding="utf-8")
    v = integrity.verify(run, man)
    assert any("kb2/b01.md" in x.path and x.kind == "modified" for x in v)


def test_editing_the_task_package_is_caught(run):
    """A block that softens its own problem statement would be undetectable."""
    man = _snap(run)
    p = run / "task" / "problem.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nextra\n", encoding="utf-8")
    assert any("problem.md" in x.path for x in integrity.verify(run, man))


def test_editing_the_charter_is_caught(run):
    man = _snap(run)
    (run / "block.md").write_text("# relaxed rules\n", encoding="utf-8")
    assert any("block.md" in x.path for x in integrity.verify(run, man))


def test_editing_the_corpus_is_caught(run):
    man = _snap(run)
    (paths.kb1_dir() / "001_2019_A_paper.md").write_text("# altered\n", encoding="utf-8")
    assert any("001_2019" in x.path for x in integrity.verify(run, man))


def test_own_ledger_may_grow_but_not_change(run):
    """`evals.jsonl` is append-only: new lines fine, old lines frozen."""
    ledger = run / "blocks" / "b02" / "evals.jsonl"
    ledger.parent.mkdir(parents=True)
    first = json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.8}) + "\n"
    ledger.write_text(first, encoding="utf-8")

    man = _snap(run)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.5}) + "\n")
    assert integrity.verify(run, man) == [], "appending is how a block works"

    man2 = _snap(run)
    ledger.write_text(first.replace("0.8", "0.05"), encoding="utf-8")
    v = integrity.verify(run, man2)
    assert v and v[0].kind in {"modified", "truncated"}


def test_own_ledger_truncation_is_caught(run):
    ledger = run / "blocks" / "b02" / "evals.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"smoke": False, "diag": False, "wall_s": 150, "rRMSE": 0.8}) + "\n",
        encoding="utf-8",
    )
    man = _snap(run)
    ledger.write_text("", encoding="utf-8")
    assert [x.kind for x in integrity.verify(run, man)] == ["truncated"]


def test_the_verdict_is_written_where_a_reader_can_check_it(run):
    man = _snap(run)
    integrity.save(run, man, segment=0)
    (run / "blocks" / "kb2" / "b01.md").write_text("# tampered\n", encoding="utf-8")
    v = integrity.verify(run, man)
    integrity.record_result(run, "b02", 0, v)

    report = integrity.audit(run)
    assert report["segments_checked"] == 1
    assert not report["clean"]
    assert report["violations"][0]["block"] == "b02"


def test_audit_of_a_clean_run(run):
    man = _snap(run)
    integrity.save(run, man, segment=0)
    integrity.record_result(run, "b02", 0, integrity.verify(run, man))
    report = integrity.audit(run)
    assert report["clean"] and report["segments_checked"] == 1


def test_manifest_round_trips(run):
    man = _snap(run)
    path = integrity.save(run, man, segment=2)
    back = integrity.Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert back.block == man.block and back.files == man.files
