"""The block loop, driven by a stub harness.

These exercise the orchestrator end to end without a model or a GPU: a fake
runtime writes exactly the artefacts a real block would, and the loop has to
reach the same verdicts it would in production — accept a finished block,
continue an unfinished one, repair a missing summary, and stop when the
environment is gone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pinnforge import paths
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
    run = layout.create_run(RunConfig(task="ldc", blocks=1))
    _anchor(run)
    rt = StubRuntime(lambda r, b, n: None)  # writes nothing, ever
    orch = _orch(run, rt)
    assert orch.run_blocks(1) == []
    assert orch.state.blocks["b01"].status == "failed"
    assert len(rt.calls) == 6  # MAX_SEGMENTS


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
