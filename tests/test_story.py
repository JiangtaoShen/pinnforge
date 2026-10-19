"""The story a run is told as, for the discovery replay — and only what the dashboard shows."""

from __future__ import annotations

import json

import pytest

from pinnforge.ui import api, story

CANARY = "ANSWER-CANARY-31337"

SUMMARY = """# b01 — Fourier features fix the stiff source term

## Thrust & rationale
Forked from b00_v01.py: the baseline's soft boundary penalty loses the tug-of-war
with a large residual. Following kb1 #12 and `026_2020_When_and_why_PINNs.md`,
exact boundary conditions and a learning-rate decay.

## Evals
| # | file | change vs parent | rRMSE | steps | note |
|---|------|------------------|-------|-------|------|
| 1 | b01_hard.py | exact BCs | 0.126 | 4063 | |
| 2 | b01_decay.py | lr decay 0.5 per 10k | 1.955 | 4145 | worse |
| 3 | b01_ff.py | a much longer change than any tag should ever carry | 0.0068 | 4100 | |

## Findings
- Exact boundary conditions remove the penalty's bias: **0.126** at once.
- The decay alone *hurts*; it needs the hard BCs first.
"""

PROBLEM = """# TOY — Multi-scale Poisson (toy)

## Equation

2-D Poisson equation for $u(x, y)$:

$$
-(u_{xx} + u_{yy}) = f(x, y)
$$

## Domain

Unit square: $(x, y) \\in [0, 1] \\times [0, 1]$.

## Boundary conditions

Homogeneous Dirichlet on all four edges.

## Time budget

**60 s of wall clock** (`TRAIN_TIME = 60.0`).
"""

INDEX = """# KB1 INDEX — 3 PINN papers

- **001_2019_Adaptive_activation.md** (2019) — Adaptive activation functions
  TL;DR line.
- **012_2020_A_Method_for_Periodic.md** (2020) — A Method for Representing Periodic Functions
  TL;DR line.
- **026_2020_When_and_why_PINNs.md** (2020) — When and why PINNs fail to train
  TL;DR line.
"""


def _rec(block, cand, r, ts, smoke=False):
    return json.dumps({"ts": ts, "block": block, "candidate": f"blocks/{block}/{cand}.py",
                       "smoke": smoke, "rRMSE": r, "wall_s": 55.0}) + "\n"


@pytest.fixture()
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "tasks" / "toy" / "private").mkdir(parents=True)
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    (root / "tasks" / "toy" / "private" / "secret.md").write_text(CANARY, encoding="utf-8")
    run = root / "runs" / "toy_1"
    for d in ("blocks/b00", "blocks/b01", "blocks/kb2", "task", "kb1", "logs"):
        (run / d).mkdir(parents=True)
    (run / "run.yaml").write_text(
        "task: toy\nblocks: 1\nagent: {runtime: claude_code, model: m}\n"
        "sandbox: {gpus: [], wall_budget_s: 240}\n", encoding="utf-8")
    (run / "state.json").write_text(json.dumps(
        {"task": "toy", "run_id": "toy_1", "status": "finished",
         "blocks": {"b01": {"status": "done"}}}), encoding="utf-8")
    (run / "blocks" / "b00" / "evals.jsonl").write_text(
        _rec("b00", "b00_v01", 1.25, "2026-10-01T00:00:00Z"), encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(
        _rec("b01", "b01_hard", None, "2026-10-01T00:01:00Z", smoke=True)
        + _rec("b01", "b01_hard", 0.126, "2026-10-01T00:02:00Z")
        + _rec("b01", "b01_decay", 1.955, "2026-10-01T00:03:00Z")
        + _rec("b01", "b01_ff", 0.0068, "2026-10-01T00:04:00Z"), encoding="utf-8")
    (run / "blocks" / "kb2" / "b01.md").write_text(SUMMARY, encoding="utf-8")
    (run / "task" / "problem.md").write_text(PROBLEM, encoding="utf-8")
    (run / "kb1" / "INDEX.md").write_text(INDEX, encoding="utf-8")
    api.ROOT_REAL = str(root.resolve())
    return root


def test_a_block_summary_becomes_its_beat(ws):
    s = story.build(ws, "toy_1")
    b01 = s["blocks"][1]
    assert b01["headline"] == "b01 — Fourier features fix the stiff source term"
    assert b01["thrust"].startswith("Forked from b00_v01.py") and "`" not in b01["thrust"]
    assert b01["fork"] == "b00_v01"
    assert b01["tags"] == ["exact BCs", "lr decay 0.5 per 10k"], "a tag is short or left out"
    assert b01["findings"] == [
        "Exact boundary conditions remove the penalty's bias: 0.126 at once.",
        "The decay alone hurts; it needs the hard BCs first.",
    ]
    assert [n["id"] for n in b01["notes"]] == ["P-b01-01", "P-b01-02"]
    assert [c["id"] for c in b01["cited"]] == ["T-026", "T-012"]


def test_each_candidate_is_told_by_what_it_did_to_the_champion(ws):
    s = story.build(ws, "toy_1")
    b00, b01 = s["blocks"]
    assert b00["entry"] is None and b00["exit"] == 1.25, "the control sets the first champion"
    assert b01["entry"] == 1.25 and b01["exit"] == 0.0068
    roster = {r["id"]: r["st"] for r in b01["roster"]}
    assert roster == {"b01_hard": "improved", "b01_decay": "regressed", "b01_ff": "winner"}
    assert b01["outcomes"] == {"improved": 1, "regressed": 1, "winner": 1}
    assert b01["counts"] == {"cand": 3, "full": 3, "smoke": 1}
    assert s["baseline"] == 1.25 and s["final"] == 0.0068
    assert s["factor"] == pytest.approx(1.25 / 0.0068)


def test_the_problem_statement_fills_the_pde_panel(ws):
    s = story.build(ws, "toy_1")
    assert s["name"] == "Multi-scale Poisson (toy)"
    assert s["equations"] == [{"name": "2-D Poisson equation for $u(x, y)$",
                               "tex": "-(u_{xx} + u_{yy}) = f(x, y)"}]
    assert [k for k, _ in s["facts"]] == ["Domain", "BC"]
    assert s["train_s"] == 60.0 and s["cpu"] is True
    assert [e["id"] for e in s["kb1"]["entries"]] == ["T-001", "T-012", "T-026"]


def test_the_training_wall_is_the_one_eval_py_enforces(ws):
    (ws / "runs" / "toy_1" / "task" / "eval.py").write_text(
        'TRAIN_TIME_LITERAL = "TRAIN_TIME = 150.0"\n', encoding="utf-8")
    assert api.run_detail(ws, "toy_1")["config"]["train_s"] == 150.0
    assert story.build(ws, "toy_1")["train_s"] == 150.0, "eval.py over the prose in problem.md"


def test_the_story_carries_nothing_of_the_answers(ws):
    """Built from what the dashboard serves: the private directory never enters it."""
    assert CANARY not in json.dumps(story.build(ws, "toy_1"))


def test_the_tallies_count_notes_and_distinct_papers(ws):
    blocks = story.build(ws, "toy_1")["blocks"]
    assert [b["notes_cum"] for b in blocks] == [0, 2]
    assert [b["theory_cum"] for b in blocks] == [0, 2]
    blocks[1]["cited"].append(blocks[1]["cited"][0])  # the same paper twice counts once
    story.tally(blocks)
    assert blocks[1]["theory_cum"] == 2


def test_a_block_without_a_summary_is_told_in_few_words(ws):
    (ws / "runs" / "toy_1" / "blocks" / "kb2" / "b01.md").unlink()
    b01 = story.build(ws, "toy_1")["blocks"][1]
    assert b01["headline"] == "b01" and b01["thrust"] == "" and b01["notes"] == []
    assert b01["roster"], "the scores still tell what happened"
