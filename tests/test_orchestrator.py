"""The block loop, driven by a stub harness.

These exercise the orchestrator end to end without a model or a GPU: a fake
runtime writes exactly the artefacts a real block would, and the loop has to
reach the same verdicts it would in production — accept a finished block,
continue an unfinished one, repair a missing summary, and stop when the
environment is gone.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from pinnforge import integrity, paths
from pinnforge.config import RunConfig
from pinnforge.orchestrator import EnvironmentFailure, Orchestrator, verify_block
from pinnforge.run import layout, ledger
from pinnforge.runtime.base import AgentHandle
from pinnforge.types import RunState


class StubRuntime:
    """A harness that does whatever the test scripted, then 'exits'."""

    name = "stub"
    default_model = "stub-model"
    default_command = "stub"

    def __init__(self, script):
        self.script = script
        self.calls: list[dict] = []

    def start(self, *, block_id, cwd, prompt, model, log_path, env, **kw):
        self.calls.append(
            {
                "block_id": block_id,
                "prompt": prompt,
                "model": model,
                "cwd": cwd,
                "env": env,
                "resume": kw.get("resume_session_id"),
            }
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('{"session_id":"sid-1"}\n', encoding="utf-8")
        self.script(Path(cwd), block_id, len(self.calls))
        return AgentHandle(block_id=block_id, process=None, cwd=Path(cwd), log_path=log_path)

    def extract_session_id(self, log_path):
        return "sid-1"


@pytest.fixture()
def project(tmp_path, monkeypatch):
    (tmp_path / "pinnforge").mkdir()
    (tmp_path / "tasks").mkdir()
    shutil.copytree(paths.tasks_dir() / "ldc", tmp_path / "tasks" / "ldc")
    (tmp_path / "kb1").mkdir()
    # No charter/ here on purpose: the charter must come from the package,
    # which is what a real deployment does (the override path has its own test).
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    monkeypatch.setattr("pinnforge.orchestrator.gpu_healthy", lambda gpus: True)
    return tmp_path


def _finish(run: Path, block_id: str, wall: float = 4000.0, score: float = 0.5) -> None:
    """Write what a completed block leaves behind."""
    d = run / "blocks" / block_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{block_id}_v1.py").write_text("# candidate\n", encoding="utf-8")
    (d / "evals.jsonl").write_text(
        json.dumps(
            {
                "block": block_id,
                "candidate": f"blocks/{block_id}/{block_id}_v1.py",
                "smoke": False,
                "diag": False,
                "wall_s": wall,
                "rRMSE": score,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (d / ".budget").write_text(str(wall), encoding="utf-8")
    (run / "blocks" / "kb2" / f"{block_id}.md").write_text(
        f"# {block_id} — stub\n", encoding="utf-8"
    )


def _anchor(run: Path) -> None:
    """Install a b00 control node, the way `anchor.ensure` would.

    Real runs always have one, and it is measured rather than dispatched —
    so the first block an agent ever sees is b01.
    """
    _finish(run, "b00", wall=146.0, score=0.94)


def _orch(run: Path, runtime) -> Orchestrator:
    cfg = RunConfig.load(paths.config_path(run))
    state = RunState.load(paths.state_path(run))
    orch = Orchestrator(run, cfg, state)
    orch.runtime = runtime
    orch.model = runtime.default_model
    return orch


def test_a_compliant_block_is_accepted_first_time(project):
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    orch = _orch(run, rt)

    done = orch.run_blocks(1)

    assert done == ["b01"], "b00 is the anchor; the first dispatched block is b01"
    assert len(rt.calls) == 1, "a finished block must not be dispatched twice"
    assert orch.state.blocks["b01"].status == "done"
    assert orch.state.status == "finished"


def test_block_is_started_in_the_run_root_with_the_budget_in_env(project):
    """The charter's relative paths only work from here, and eval.py reads the budget."""
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    _orch(run, rt).run_blocks(1)

    call = rt.calls[0]
    assert call["cwd"] == run
    assert call["env"]["FORGE_WALL_BUDGET"] == "3600"  # ldc's declared budget
    assert str(run) in call["prompt"] and "block.md" in call["prompt"]


def test_unfinished_block_is_resumed_in_its_own_session(project):
    """First segment leaves the budget unspent; the loop continues the session."""

    def script(r, b, n):
        if n == 1:
            d = r / "blocks" / b
            d.mkdir(parents=True, exist_ok=True)
            (d / "evals.jsonl").write_text(
                json.dumps({"smoke": False, "diag": False, "wall_s": 100, "rRMSE": 0.9}) + "\n",
                encoding="utf-8",
            )
        else:
            _finish(r, b)

    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(script)
    orch = _orch(run, rt)
    assert orch.run_blocks(1) == ["b01"]

    assert len(rt.calls) == 2
    assert rt.calls[1]["resume"] == "sid-1"
    assert "Resume block b01" in rt.calls[1]["prompt"]
    assert "wall-seconds spent" in rt.calls[1]["prompt"]


def test_missing_summary_triggers_the_repair_prompt(project):
    """Budget spent and evals logged, but no summary: repair, and forbid evals."""

    def script(r, b, n):
        if n == 1:
            d = r / "blocks" / b
            d.mkdir(parents=True, exist_ok=True)
            (d / "evals.jsonl").write_text(
                json.dumps({"smoke": False, "diag": False, "wall_s": 3500, "rRMSE": 0.4}) + "\n",
                encoding="utf-8",
            )
            (d / ".budget").write_text("3500", encoding="utf-8")
        else:
            (r / "blocks" / "kb2" / f"{b}.md").write_text(f"# {b} — x\n", encoding="utf-8")

    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(script)
    assert _orch(run, rt).run_blocks(1) == ["b01"]
    assert "Do not run any evaluations." in rt.calls[1]["prompt"]


def test_second_block_gets_a_fresh_id(project):
    run = layout.create_run(RunConfig(task="ldc", blocks=2))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b, score=0.5 - 0.1 * n))
    done = _orch(run, rt).run_blocks(2)
    assert done == ["b01", "b02"]
    assert [c["block_id"] for c in rt.calls] == ["b01", "b02"]


def test_ledger_and_summary_are_written(project):
    run = layout.create_run(RunConfig(task="ldc", blocks=2))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b, score=0.5 - 0.1 * n))
    _orch(run, rt).run_blocks(2)

    usage = ledger.read_usage(run)
    assert set(usage) == {"b01", "b02"}, "the anchor is measured, so it has no ledger line"
    assert usage["b01"]["model"] == "stub-model"

    summary = paths.run_summary_path(run).read_text(encoding="utf-8")
    for b in ("b00", "b01", "b02"):
        assert b in summary, "the summary covers the anchor too"
    best, block = ledger.overall_best(run)
    assert best == pytest.approx(0.3) and block == "b02"


def test_dead_gpu_stops_the_run_instead_of_retrying(project, monkeypatch):
    monkeypatch.setattr("pinnforge.orchestrator.gpu_healthy", lambda gpus: False)
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    with pytest.raises(EnvironmentFailure):
        _orch(run, rt).run_blocks(1)
    assert rt.calls == [], "no dispatch should happen once the GPUs are gone"


def test_hopeless_block_is_abandoned_not_looped_forever(project):
    """Two dispatches, not six: the second bought nothing the first had not.

    MAX_SEGMENTS is the backstop for a block that keeps inching forward and
    never arrives. An agent that produces the same evidence twice has decided
    it is finished, and the remaining segments would be the same exchange
    again.
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: None)  # writes nothing, ever
    orch = _orch(run, rt)
    assert orch.run_blocks(1) == []
    assert orch.state.blocks["b01"].status == "failed", "nothing usable was produced"
    assert len(rt.calls) == 2


def test_a_block_that_stops_just_short_is_short_not_failed(project):
    """`failed` has to mean "produced nothing", or the summary libels the run.

    ldc_4's b02 was recorded as failed while holding the best score in the run,
    a written summary and 23 scored evaluations — it had simply stopped 104 s
    shy of the budget threshold, and four further dispatches in 42 seconds
    could not move it.
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    # spends almost all of the 3450 s threshold, writes a summary, then insists
    rt = StubRuntime(lambda r, b, n: _finish(r, b, wall=3346.0, score=0.29))
    orch = _orch(run, rt)

    assert orch.run_blocks(1) == [], "the budget still decides `done`"
    bs = orch.state.blocks["b01"]
    assert bs.status == "short", f"a usable block must not read as failed: {bs.status}"
    assert "short of the budget" in bs.exit_reason
    assert len(rt.calls) == 2, "and it must not be re-dispatched four more times"


def test_interrupted_run_resumes_from_the_checkpoint(project):
    """A killed orchestrator leaves enough on disk to carry on."""
    run = layout.create_run(RunConfig(task="ldc", blocks=2))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    _orch(run, rt).run_blocks(1)  # only the first of two blocks

    # a fresh process reads the checkpoint and the on-disk evidence
    run2, cfg2, state2 = layout.load_run(run.name)
    assert state2.blocks["b01"].status == "done"
    assert verify_block(run2, "b01", cfg2, 150.0).done

    rt2 = StubRuntime(lambda r, b, n: _finish(r, b, score=0.2))
    orch2 = _orch(run2, rt2)
    assert orch2.run_blocks(1) == ["b02"]
    assert [c["block_id"] for c in rt2.calls] == ["b02"], "must not redo a finished block"


def test_crashed_workspace_is_readopted_with_recovery_instructions(project):
    """An id with a half-finished workspace is continued, not abandoned."""
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    d = run / "blocks" / "b01"
    d.mkdir(parents=True)
    (d / "evals.jsonl").write_text(
        json.dumps({"smoke": False, "diag": False, "wall_s": 200, "rRMSE": 0.8}) + "\n",
        encoding="utf-8",
    )
    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    _orch(run, rt).run_blocks(1)
    assert rt.calls[0]["block_id"] == "b01", "the crashed id is re-adopted, not skipped"
    assert "already exists from a crashed run" in rt.calls[0]["prompt"]


# ─────────────────── the anchor is not a block ───────────────────


def _anchor_without_summary(run: Path) -> None:
    """A restored anchor as most tasks actually have one.

    `anchor.save` copies a `b00.md` only when the run it came from had one, so
    a cached anchor normally carries the candidate, its params, the records and
    the spent budget — and no summary at all. Of the four library tasks only
    `ldc` has one, which is why the fixture above is the exception rather than
    the rule.
    """
    _finish(run, "b00", wall=146.0, score=0.94)
    (run / "blocks" / "kb2" / "b00.md").unlink()


def test_an_anchor_without_a_summary_is_still_complete(project):
    """The control node is judged on its measurement, not on a kb2 write-up.

    Requiring a summary made every such anchor look like a crashed block, and
    the loop then spent the run's first block sending an agent — on the full
    charter, with a full GPU budget — to "finish" the very node every other
    block is measured against.
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor_without_summary(run)
    cfg = RunConfig.load(paths.config_path(run))

    v = verify_block(run, "b00", cfg, 150.0)
    assert v.done, "a measured anchor is complete without a summary"
    assert not v.has_summary
    assert v.scored_evals == 1


def test_an_anchor_without_a_summary_is_not_dispatched(project):
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor_without_summary(run)

    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    done = _orch(run, rt).run_blocks(1)

    assert [c["block_id"] for c in rt.calls] == ["b01"], "b00 must never be dispatched"
    assert done == ["b01"], "the run's blocks must not be spent on the anchor"


def test_b00_is_never_selected_even_when_it_looks_unfinished(project):
    """Defence in depth: whatever any verdict says, the anchor is not a block."""
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    paths.block_dir(run, "b00").mkdir(parents=True)  # worst case: an empty anchor

    rt = StubRuntime(lambda r, b, n: _finish(r, b))
    _orch(run, rt).run_blocks(1)

    assert [c["block_id"] for c in rt.calls] == ["b01"]


# ─────────────── an interrupted segment still books its cost ───────────────


def _partial(run: Path, block_id: str) -> None:
    """What a block that was working when it got interrupted leaves behind.

    Records and spent budget, no summary — the shape b03 was in when the run
    this test came from was stopped.
    """
    d = run / "blocks" / block_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "evals.jsonl").write_text(
        json.dumps(
            {
                "block": block_id,
                "candidate": f"blocks/{block_id}/{block_id}_v1.py",
                "smoke": False,
                "diag": False,
                "wall_s": 900.0,
                "rRMSE": 0.088,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (d / ".budget").write_text("900", encoding="utf-8")


class UsageReportingStub(StubRuntime):
    """A harness that reports accounting, the way a real one does."""

    def extract_usage(self, log_path):
        from pinnforge.runtime.base import AgentUsage

        return AgentUsage(tokens=284_403, tool_uses=94, cost_usd=10.88, model="stub-resolved")


def test_an_interrupted_segment_still_books_what_it_cost(project, monkeypatch):
    """Ctrl-C used to throw away the block's whole model spend.

    Found on a real run: the block had been going an hour and had 2832 GPU
    seconds and 15 evaluations to show for it, but `run_usage.jsonl` had no
    line for it at all and the summary printed a dash where the money went.
    The GPU seconds survived because eval.py writes those itself; everything
    the orchestrator was responsible for booking was lost, because it was
    booked only after the wait returned normally.
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    orch = _orch(run, UsageReportingStub(lambda r, b, n: _partial(r, b)))

    def interrupted(self, handle, block_id):
        raise KeyboardInterrupt

    monkeypatch.setattr(Orchestrator, "_await", interrupted)
    with pytest.raises(KeyboardInterrupt):
        orch.run_blocks(1)

    lines = [
        json.loads(ln)
        for ln in paths.ledger_path(run).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert [r["block"] for r in lines] == ["b01"], "the interrupted segment must reach the ledger"
    assert lines[0]["tokens"] == 284_403
    assert lines[0]["cost_usd"] == 10.88
    assert lines[0]["model_resolved"] == "stub-resolved"
    # duration is not asserted above zero: this interrupt fires instantly, so
    # the segment really did last under a millisecond. What matters is that the
    # line exists at all — it did not, before.
    assert "duration_ms" in lines[0]

    _, _, state = layout.load_run(run.name)
    bs = state.blocks["b01"]
    assert bs.status == "interrupted", f"a stale 'running' misreports a dead block: {bs.status}"
    assert "interrupted" in bs.exit_reason
    assert bs.model_resolved == "stub-resolved"


def test_the_summary_shows_an_interrupted_block_rather_than_a_dash(project, monkeypatch):
    """The row was there, but every column the orchestrator fills was a dash."""
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    orch = _orch(run, UsageReportingStub(lambda r, b, n: _partial(r, b)))
    monkeypatch.setattr(
        Orchestrator,
        "_await",
        lambda self, handle, block_id: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        orch.run_blocks(1)

    row = next(r for r in ledger.block_rows(run) if r["block"] == "b01")
    assert row["tokens"] == 284_403, "the summary showed a dash here"
    assert row["model"] == "stub-resolved"
    assert row["segments"] == 1
    assert row["wall_s"] == 900.0, "GPU seconds always survived; they come from eval.py"


def test_an_interrupted_dispatch_is_still_verified(project, monkeypatch):
    """A manifest with no verdict is a hole in what `run audit` reports.

    `run audit` counts verdicts, so an unverified dispatch does not show up as
    unchecked — it shows up as not having happened, and the run reads as fully
    audited when part of it never was. Seen on naca_1: three dispatches, a
    b03.0.json manifest, no b03.0.result.json, and audit reporting "2 dispatch
    segment(s) verified".
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    orch = _orch(run, UsageReportingStub(lambda r, b, n: _partial(r, b)))
    monkeypatch.setattr(
        Orchestrator,
        "_await",
        lambda self, handle, block_id: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        orch.run_blocks(1)

    manifests = sorted(p.name for p in (run / ".integrity").glob("b01.*.json"))
    assert "b01.0.json" in manifests
    assert "b01.0.result.json" in manifests, "the interrupted dispatch was never verified"
    report = integrity.audit(run)
    assert report["segments_checked"] == 1
    assert report["clean"]


def test_a_violation_is_not_overwritten_by_complete(project):
    """A block that altered another's record must not read as complete.

    The verdict on disk was always right — `run audit` reads that — but
    state.json's exit_reason was set to "complete" straight after, so anything
    reading the checkpoint saw a clean finish.
    """
    run = layout.create_run(RunConfig(task="ldc", blocks=2))
    _anchor(run)
    _finish(run, "b01", score=0.4)

    def tamper_then_finish(r, b, n):
        # rewrite an earlier block's ledger, which the charter forbids
        (r / "blocks" / "b01" / "evals.jsonl").write_text("{}\n", encoding="utf-8")
        _finish(r, b)

    orch = _orch(run, StubRuntime(tamper_then_finish))
    orch.run_blocks(1)

    assert integrity.audit(run)["clean"] is False
    bs = orch.state.blocks["b02"]
    assert bs.status == "done"
    assert "integrity" in bs.exit_reason, f"the violation was hidden: {bs.exit_reason!r}"


def test_a_readopted_block_is_not_reported_stalled_immediately(project, monkeypatch):
    """Silence is measured from this dispatch, not from the workspace's mtimes.

    A re-adopted block carries the mtimes of the segment that stopped hours
    ago, so measuring from those made every resume cry wolf on its first poll:
    ldc_4's b02 was reported "quiet for 82 min" sixty seconds after it was
    dispatched, while it was in fact working and went on to finish.
    """
    import logging
    import time as _time

    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    d = paths.block_dir(run, "b01")
    d.mkdir(parents=True, exist_ok=True)
    stale = d / "old.py"
    stale.write_text("# from hours ago\n", encoding="utf-8")
    long_ago = _time.time() - 6 * 3600
    os.utime(stale, (long_ago, long_ago))
    os.utime(d, (long_ago, long_ago))

    cfg = RunConfig.load(paths.config_path(run))
    orch = Orchestrator(run, cfg, RunState.load(paths.state_path(run)))
    orch.cfg.poll_seconds = 0  # one immediate poll, then the handle reports exit

    class OneShot:
        log_path = run / "logs" / "x.log"
        _n = 0

        @property
        def alive(self):
            return OneShot._n < 2

        def wait(self, timeout=None):
            OneShot._n += 1
            return None if OneShot._n < 2 else 0

        def close(self):
            pass

    with pytest.MonkeyPatch.context():
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logging.getLogger("pinnforge.orchestrator").addHandler(handler)
        try:
            orch._await(OneShot(), "b01")
        finally:
            logging.getLogger("pinnforge.orchestrator").removeHandler(handler)

    stalls = [r for r in records if "quiet for" in r.getMessage()]
    assert not stalls, f"a freshly dispatched block must not read as stalled: {stalls}"
