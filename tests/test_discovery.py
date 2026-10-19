"""The discovery film: the record an agent is given, the script it may write, the film made."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pinnforge import discovery, film_check, paths
from pinnforge.config import AgentConfig, RunConfig
from pinnforge.ui import api

CANARY = "ANSWER-CANARY-90210"

SUMMARY = """# b01 — hard boundary conditions

## Thrust & rationale
Forked from b00_v01.py; following kb1 #12, exact boundary conditions.

## Evals
| # | file | change vs parent | rRMSE |
|---|------|------------------|-------|
| 1 | b01_hard.py | exact BCs | 0.126 |

## Findings
- Exact boundary conditions remove the penalty's bias.
"""

INDEX = """# KB1 INDEX

- **012_2020_A_Method.md** (2020) — A method for periodic functions
- **026_2020_When_and_why.md** (2020) — When and why PINNs fail to train
"""


def _rec(block, cand, r, ts):
    return json.dumps({"ts": ts, "block": block, "candidate": f"blocks/{block}/{cand}.py",
                       "smoke": False, "rRMSE": r, "wall_s": 55.0}) + "\n"


@pytest.fixture()
def run(tmp_path):
    root = tmp_path / "ws"
    (root / "tasks" / "toy" / "private").mkdir(parents=True)
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    (root / "tasks" / "toy" / "private" / "secret.md").write_text(CANARY, encoding="utf-8")
    run = root / "runs" / "toy_1"
    for d in ("blocks/b00", "blocks/b01", "blocks/kb2", "task", "kb1", "logs"):
        (run / d).mkdir(parents=True)
    (run / "run.yaml").write_text(
        "task: toy\nblocks: 1\nagent: {runtime: replay, model: ''}\n"
        "sandbox: {gpus: [], wall_budget_s: 240}\n", encoding="utf-8")
    (run / "state.json").write_text(json.dumps(
        {"task": "toy", "run_id": "toy_1", "status": "finished",
         "blocks": {"b01": {"status": "done"}}}), encoding="utf-8")
    (run / "blocks" / "b00" / "evals.jsonl").write_text(
        _rec("b00", "b00_v01", 1.25, "2026-10-01T00:00:00Z"), encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(
        _rec("b01", "b01_hard", 0.126, "2026-10-01T00:02:00Z")
        + _rec("b01", "b01_decay", 1.9, "2026-10-01T00:03:00Z"), encoding="utf-8")
    (run / "blocks" / "kb2" / "b01.md").write_text(SUMMARY, encoding="utf-8")
    (run / "task" / "problem.md").write_text("# TOY — Toy Poisson\n\n## Domain\n\nUnit square.\n",
                                             encoding="utf-8")
    (run / "kb1" / "INDEX.md").write_text(INDEX, encoding="utf-8")
    (run / "logs" / "b01.0.log").write_text("\n".join(json.dumps(e) for e in (
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Reading kb1/012 for periodic embeddings."}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "python task/eval.py"}}]}},
    )) + "\n", encoding="utf-8")
    api.ROOT_REAL = str(root.resolve())
    return run


def _script(**over):
    s = {"version": 1, "logline": "A toy run, told briefly.", "blocks": {
        "b00": {"headline": "b00 — the control", "thrust": "", "tags": [], "findings": [],
                "cited": [], "candidates": {}},
        "b01": {"headline": "b01 — exact boundary conditions: 1.25 → 0.126",
                "thrust": "Forked from the baseline to test exact boundary conditions.",
                "tags": ["hard BCs"], "findings": ["Exact BCs remove the penalty's bias."],
                "cited": ["T-012"], "candidates": {"b01_hard": "exact BCs"}},
    }}
    s.update(over)
    return s


def test_the_package_holds_the_record_and_nothing_of_the_answers(run):
    pkg = discovery.package(run.parent.parent, run)
    for name in ("facts.json", "draft.json", "BRIEF.md", "check.py", "example/script.json",
                 "summaries/b01.md", "activity/b01.md", "problem.md", "kb1/INDEX.md"):
        assert (pkg / name).is_file(), name
    activity = (pkg / "activity" / "b01.md").read_text(encoding="utf-8")
    assert "periodic embeddings" in activity and "python task/eval.py" in activity
    brief = (pkg / "BRIEF.md").read_text(encoding="utf-8")
    assert "`b00`, `b01`" in brief and "{" + "max_headline}" not in brief, "every field filled"
    for f in pkg.rglob("*"):
        if f.is_file():
            assert CANARY not in f.read_text(encoding="utf-8", errors="replace"), f


def test_the_draft_and_the_example_pass_the_check(run):
    pkg = discovery.package(run.parent.parent, run)
    facts = json.loads((pkg / "facts.json").read_text(encoding="utf-8"))
    assert film_check.validate(json.loads((pkg / "draft.json").read_text(encoding="utf-8")),
                               facts)[1] == []
    example = json.loads((paths.prompts_dir() / "discovery_example.json").read_text(
        encoding="utf-8"))
    papers = {c for b in example["blocks"].values() for c in b["cited"]}
    shaped = {"kb1": {"entries": [{"id": c} for c in papers]},
              "blocks": [{"id": bid, "roster": [{"id": c} for c in b["candidates"]]}
                         for bid, b in example["blocks"].items()]}
    assert film_check.validate(example, shaped)[1] == [], "the paradigm obeys its own rules"


def test_a_script_is_checked_against_the_facts(run):
    facts = json.loads((discovery.package(run.parent.parent, run) / "facts.json").read_text(
        encoding="utf-8"))
    assert film_check.validate(_script(), facts)[1] == []
    bad = _script(version=2, extra=1)
    bad["blocks"]["b01"]["cited"] = ["T-099"]
    bad["blocks"]["b01"]["candidates"] = {"b01_nope": "x"}
    bad["blocks"]["b01"]["headline"] = "x" * 200
    bad["blocks"]["b07"] = {"headline": "no such block"}
    del bad["blocks"]["b00"]
    errors = " | ".join(film_check.validate(bad, facts)[1])
    for part in ("version", "'extra'", "T-099", "b01_nope", "at most 110", "blocks.b07",
                 "blocks.b00: missing"):
        assert part in errors, part


def test_the_script_writes_the_words_and_the_facts_keep_the_numbers(run):
    facts = json.loads((discovery.package(run.parent.parent, run) / "facts.json").read_text(
        encoding="utf-8"))
    script, _ = film_check.validate(_script(), facts)
    film = discovery.lay(facts, script)
    b01 = film["blocks"][1]
    assert b01["headline"] == "b01 — exact boundary conditions: 1.25 → 0.126"
    assert [r.get("label") for r in b01["roster"]] == ["exact BCs", ""]
    assert [n["t"] for n in b01["notes"]] == ["Exact BCs remove the penalty's bias."]
    assert [c["id"] for c in b01["cited"]] == ["T-012"] and b01["theory_cum"] == 1
    assert film["logline"] == "A toy run, told briefly."
    for key in ("entry", "exit", "best", "cost", "counts"):
        assert b01[key] == facts["blocks"][1][key], key
    assert [r["r"] for r in b01["roster"]] == [r["r"] for r in facts["blocks"][1]["roster"]]


def test_a_failed_script_gives_way_to_the_draft(run):
    root = run.parent.parent
    pkg = discovery.package(root, run)
    (pkg / "script.json").write_text("{not json", encoding="utf-8")
    film = discovery.render(root, run)
    assert film.scripted_by == "draft" and "not valid JSON" in film.problems[0]
    html = film.path.read_text(encoding="utf-8")
    assert film.path == run / "discovery.html" and "window.__PF_FILM__ = true" in html
    assert CANARY not in html and not list(run.glob(".discovery.html*"))


def test_the_checker_travels_with_the_package(run):
    pkg = discovery.package(run.parent.parent, run)
    (pkg / "script.json").write_text((pkg / "draft.json").read_text(encoding="utf-8"),
                                     encoding="utf-8")
    out = subprocess.run([sys.executable, "-I", "check.py"], cwd=pkg, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0 and "ok: the script is ready" in out.stdout


def test_the_scripted_agent_makes_a_film_end_to_end(run):
    """The demo's agent hands back the draft; the film is made from it all the same."""
    film = discovery.make(run.parent.parent, run, AgentConfig(runtime="replay"), timeout_s=120)
    assert film.scripted_by == "agent" and film.problems == []
    assert (run / "discovery" / "script.json").is_file() and film.path.is_file()
    assert "film-toy_1" not in [p.name for p in (run / "blocks").iterdir()]


def test_a_run_can_leave_its_film_to_the_command(tmp_path):
    cfg = RunConfig(task="toy", film=False)
    cfg.save(tmp_path / "run.yaml")
    assert RunConfig.load(tmp_path / "run.yaml").film is False
    RunConfig(task="toy").save(tmp_path / "run.yaml")
    assert "film" not in (tmp_path / "run.yaml").read_text(encoding="utf-8")
    assert RunConfig.load(tmp_path / "run.yaml").film is True
