"""`pinnforge run` as a driver: how a run is started, resumed, audited and stopped.

No agent and no GPU: the anchor measurement and the block loop are stubbed,
so what is under test is what the command decides around them.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from typing import ClassVar

import pytest

from pinnforge import gpus, integrity, paths
from pinnforge.cli import run_cmd
from pinnforge.config import RunConfig
from pinnforge.run import anchor, layout
from pinnforge.types import RunState

TWO = [gpus.Gpu(0, "GPU-a", "NVIDIA RTX A6000", 49140),
       gpus.Gpu(1, "GPU-b", "NVIDIA RTX A6000", 49140)]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A CPU workspace holding the demo task, on a pretend two-card machine."""
    from pinnforge.task import examples

    (tmp_path / "pinnforge.yaml").write_text("sandbox:\n  gpus: []\n", encoding="utf-8")
    examples.install_demo(tmp_path / "tasks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gpus, "inventory", lambda timeout=20.0: list(TWO))
    return tmp_path


def _start_args(**kw):
    ns = {"task": "demo", "blocks": 1, "agent": None, "runtime": "replay", "model": "",
          "gpus": "", "budget": "", "no_anchor": False, "force": False, "share": False,
          "set": [], "foreground": False, "detach": True, "no_ui": True}
    ns.update(kw)
    return argparse.Namespace(**ns)


# ─────────────────────────────── --gpus ───────────────────────────────


@pytest.mark.parametrize("typed, pool", [("0 1", [0, 1]), ("1", [1]), ("0,1", [0, 1]),
                                         ("all", [0, 1]), ("none", [])])
def test_gpus_takes_the_words_setup_gpus_takes(ws, typed, pool):
    cfg = run_cmd._build_config(_start_args(gpus=typed), ws)
    assert cfg.sandbox.gpus == pool


def test_a_card_the_machine_lacks_is_refused_before_anything_is_built(ws, capsys):
    assert run_cmd.cmd_start(_start_args(gpus="7")) == 2
    assert "no GPU 7" in capsys.readouterr().out
    assert not (ws / "runs").exists()


def test_the_cards_source_names_a_set_override(ws):
    args = _start_args(gpus="0", set=["blocks=2", "sandbox.gpus=[1]"])
    assert run_cmd._gpu_source(args, ws) == "--set sandbox.gpus=[1]", "--set is applied last"
    assert run_cmd._gpu_source(_start_args(gpus="0"), ws) == "--gpus 0"
    assert run_cmd._gpu_source(_start_args(), ws) == "sandbox.gpus: [] in pinnforge.yaml"


def test_a_setting_that_will_not_render_still_refuses_without_an_orphan(ws, monkeypatch, capsys):
    """A refusal after create_run left a run directory behind for nothing."""
    (ws / "pinnforge.yaml").write_text("sandbox:\n  gpus: [0, 3]\n", encoding="utf-8")

    def render(spec):
        raise TypeError("cannot render that")

    monkeypatch.setattr(gpus, "render", render)
    assert run_cmd.cmd_start(_start_args()) == 2
    out = capsys.readouterr().out
    assert "GPU 3 is not on this machine" in out and "in pinnforge.yaml" in out
    assert not (ws / "runs").exists()


# ─────────────────────────── the anchor on resume ───────────────────────────


class _NoBlocks:
    """The block loop, stubbed: these tests are about what happens before it."""

    built: ClassVar[list] = []

    def __init__(self, run, cfg, state):
        _NoBlocks.built.append(self)

    def run_blocks(self, n):
        return []


@pytest.fixture
def run(ws, monkeypatch):
    run = layout.create_run(RunConfig(task="demo", blocks=1))
    _NoBlocks.built = []
    monkeypatch.setattr(run_cmd, "Orchestrator", _NoBlocks)
    return run


def _ensure_stub(monkeypatch, result, seen):
    def ensure(run, task, root=None, gpu=0, hw=None, rebuild=False):
        b00 = paths.block_dir(run, "b00")
        seen.append({"b00_left": sorted(p.name for p in b00.iterdir()) if b00.is_dir() else []})
        return result

    monkeypatch.setattr(anchor, "ensure", ensure)


def _partial_anchor(run):
    """What an anchor measurement interrupted half-way leaves: a charge, a copy, no score."""
    b00 = paths.block_dir(run, "b00")
    b00.mkdir(parents=True)
    (b00 / "b00_v01.py").write_text("x = 1\n", encoding="utf-8")
    (b00 / ".budget").write_text("60.0", encoding="utf-8")


def _drive(run, *flags):
    _, cfg, _ = layout.load_run(run.name)
    return run_cmd._execute(run_cmd.build_drive_args([run.name, *flags]), run, cfg)


def test_a_resume_installs_the_anchor_its_start_never_finished(run, monkeypatch, capsys):
    seen = []
    _ensure_stub(monkeypatch, anchor.AnchorResult(None, False, "eval failed"), seen)
    _partial_anchor(run)
    assert _drive(run, "--resume") == 1, "the same refusal as start"
    assert seen == [{"b00_left": []}], "measured afresh, not on top of the half-measured one"
    assert "refusing to run blocks without an anchor" in capsys.readouterr().out
    assert _NoBlocks.built == []


def test_a_forced_start_stays_forced_on_resume(run, monkeypatch):
    seen = []
    _ensure_stub(monkeypatch, anchor.AnchorResult(None, False, "eval failed"), seen)
    assert _drive(run, "--force") == 0
    assert RunState.load(paths.state_path(run)).force, "the flag is remembered"
    assert _drive(run, "--resume") == 0, "and the resume is forced too"
    assert len(seen) == 2, "the anchor was tried again on resume"


def test_a_start_without_an_anchor_resumes_without_one(run, monkeypatch):
    seen = []
    _ensure_stub(monkeypatch, anchor.AnchorResult(0.5, False, "measured"), seen)
    assert _drive(run, "--no-anchor") == 0
    assert RunState.load(paths.state_path(run)).no_anchor
    assert _drive(run, "--resume") == 0
    assert seen == []


def test_a_scored_anchor_is_not_measured_again_on_resume(run, monkeypatch):
    seen = []
    _ensure_stub(monkeypatch, anchor.AnchorResult(0.5, False, "measured"), seen)
    b00 = paths.block_dir(run, "b00")
    b00.mkdir(parents=True)
    (b00 / "evals.jsonl").write_text(json.dumps(
        {"block": "b00", "smoke": False, "diag": False, "wall_s": 60.0, "rRMSE": 0.5}) + "\n",
        encoding="utf-8")
    assert _drive(run, "--resume") == 0
    assert seen == []


def test_the_scorer_answers_for_b00_while_the_anchor_is_measured(run, monkeypatch):
    from pinnforge import scoring

    dispatch, during = scoring.Dispatch(), []

    def ensure(run, task, root=None, gpu=0, hw=None, rebuild=False):
        during.append(dispatch.block)
        return anchor.AnchorResult(0.5, False, "measured")

    monkeypatch.setattr(anchor, "ensure", ensure)
    _, cfg, _ = layout.load_run(run.name)
    run_cmd._execute(run_cmd.build_drive_args([run.name]), run, cfg, dispatch=dispatch)
    assert during == ["b00"] and dispatch.block is None


# ─────────────────────── a detached driver's log ───────────────────────


def test_a_drivers_stderr_reaches_its_log(tmp_path, monkeypatch):
    """Only stdout was tee'd: a detached driver's traceback went nowhere anyone looked."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    log = tmp_path / "orchestrator.log"
    run_cmd._log_crashes(log)
    print("Traceback (most recent call last): boom", file=sys.stderr)
    sys.stderr.flush()
    assert "boom" in log.read_text(encoding="utf-8")


def test_a_log_that_is_also_the_terminal_gets_each_line_once(tmp_path):
    log = tmp_path / "orchestrator.log"
    with log.open("a", encoding="utf-8") as terminal:  # detached without tmux
        tee = run_cmd._Tee(terminal, log)
        tee.write("one line\n")
        tee.flush()
    assert log.read_text(encoding="utf-8") == "one line\n"


# ─────────────────────────── audit and stop ───────────────────────────


def test_audit_reports_a_dispatch_that_was_never_verified(run, capsys):
    man = integrity.snapshot(run, "b01", paths.project_root())
    integrity.save(run, man)
    assert run_cmd.cmd_audit(argparse.Namespace(run_id=run.name)) == 1
    out = capsys.readouterr().out
    assert "never verified" in out and "block b01 segment 0" in out


def test_stop_reports_a_refusal_and_leaves_the_state_alone(run, monkeypatch, capsys):
    from pinnforge.run import lifecycle

    state = RunState.load(paths.state_path(run))
    state.status = "running"
    state.save(paths.state_path(run))
    monkeypatch.setattr(lifecycle, "stop", lambda run, force=False: "refused: driven from b")
    assert run_cmd.cmd_stop(argparse.Namespace(run_id=run.name, force=False)) == 1
    assert "refused" in capsys.readouterr().out
    assert RunState.load(paths.state_path(run)).status == "running", (
        "a run driven from another host is still running"
    )


# ─────────────────────────── task anchor ───────────────────────────


def _anchor_args(**kw):
    ns = {"name": "demo", "measure": False, "rebuild": False, "rekey": False, "gpu": None}
    ns.update(kw)
    return argparse.Namespace(**ns)


def test_rebuild_measures_rather_than_harvesting_an_old_run(ws, monkeypatch):
    from pinnforge.cli import task_cmd

    asked = []

    def ensure(run, task, root=None, gpu=0, hw=None, rebuild=False):
        asked.append(rebuild)
        return anchor.AnchorResult(0.5, False, "measured on CPU")

    monkeypatch.setattr(anchor, "ensure", ensure)
    assert task_cmd.cmd_anchor(_anchor_args(rebuild=True)) == 0
    assert asked == [True]


@pytest.mark.parametrize("gpu, stamped", [(None, None),
                                          (0, {"platform": "gpu", "gpu": "NVIDIA RTX A6000"})])
def test_rekey_stamps_only_a_gpus_class(ws, monkeypatch, gpu, stamped):
    from pinnforge.cli import task_cmd

    seen = []

    def rekey(run, task, root=None, hw=None):
        seen.append(hw)
        return anchor.AnchorResult(0.5, True, "rekeyed")

    monkeypatch.setattr(anchor, "rekey", rekey)
    assert task_cmd.cmd_anchor(_anchor_args(rekey=True, gpu=gpu)) == 0
    assert seen == [stamped]


def test_an_older_run_started_without_an_anchor_resumes_without_one(run, monkeypatch):
    """Its state predates the recorded flags; blocks with no b00 at all say what it was."""
    seen = []
    _ensure_stub(monkeypatch, anchor.AnchorResult(0.5, False, "measured"), seen)
    raw = json.loads(paths.state_path(run).read_text(encoding="utf-8"))
    for key in ("no_anchor", "force"):
        raw.pop(key, None)
    paths.state_path(run).write_text(json.dumps(raw), encoding="utf-8")
    paths.block_dir(run, "b01").mkdir(parents=True)
    assert _drive(run, "--resume") == 0
    assert seen == [], "no GPU measurement its start had skipped"
