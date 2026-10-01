"""`pinnforge task new`, stage by stage, with the agents replaced by stand-ins.

What an author leaves is accepted only if it finished, wrote the package in
this session and reached for nothing; whatever an agent that saw the answers
wrote stays in the private directory, wherever that is configured to be; a
resumed authoring takes the flags it is given; and no agent outlives the
session that started it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pinnforge import agents, private
from pinnforge.cli import task_new
from pinnforge.config import AgentConfig
from pinnforge.task import authoring, examples, reference


@pytest.fixture(autouse=True)
def _no_external_private(monkeypatch):
    monkeypatch.delenv("FORGE_PRIVATE_DIR", raising=False)


@pytest.fixture
def demo_src(tmp_path):
    return examples.install_demo(tmp_path / "src")


def _steps(tmp_path, reference_location="", name="heat"):
    (tmp_path / "tasks").mkdir(exist_ok=True)
    (tmp_path / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    pipe = authoring.Pipeline(tmp_path, name, out=lambda *_: None)
    if reference_location:
        reference.remember_source(private.locate(pipe.task_dir), reference_location)
    pipe.begin(authoring.Inputs(describe="poisson", reference="operator" if reference_location
                                else "", agent={"runtime": "claude_code", "model": "m"}))
    steps = task_new.Steps(pipe, tmp_path)
    steps.intake()
    return steps


def _as_ingested(steps, src: Path) -> None:
    """The state an ingest stage leaves: the grid written and pinned, its shape recorded."""
    shutil.copy(src / "eval_grid.csv", steps.task_dir / "eval_grid.csv")
    shutil.copy(src / "private" / "ref_values.npy", steps.private_home / "ref_values.npy")
    reference.pin_grid(steps.task_dir, steps.private_home)
    steps.pipe.state["reference"] = {"grid_columns": ["x", "y"], "input_columns": 2,
                                     "ref_columns": ["u"], "rows": 10201}
    steps.pipe._save()


def _write_package(task_dir: Path, src: Path, grid: bool = True) -> None:
    names = ["problem.md", "baseline.py", "eval.py"] + (["eval_grid.csv"] if grid else [])
    for n in names:
        shutil.copy(src / n, task_dir / n)  # a fresh mtime: written now


def _transcript(path: Path, *calls) -> None:
    path.write_text("\n".join(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": n, "input": a}]}}) for n, a in calls) + "\n",
        encoding="utf-8")


def _author(src, *, calls=(), rc=0, timed_out=False, write=True, also=None):
    def author_task(name, **kw):
        task_dir = kw["root"] / "tasks" / name
        if write:
            _write_package(task_dir, src, grid=True)
        _transcript(kw["log_path"], *calls)
        if also:
            also(task_dir, kw)
        return SimpleNamespace(duration_s=1.0, returncode=rc, timed_out=timed_out)
    return author_task


# ─────────────────────────────── the author stage ───────────────────────────────


def test_a_clean_author_is_accepted(tmp_path, monkeypatch, demo_src):
    steps = _steps(tmp_path)
    monkeypatch.setattr(task_new.agents, "author_task", _author(demo_src))
    ok, note = steps.author()
    assert ok, note


def test_an_untrusted_package_leaves_the_task(tmp_path, monkeypatch, demo_src):
    steps = _steps(tmp_path)

    def notes(task_dir, kw):
        (task_dir / ".authoring" / "notes.md").write_text("I looked around\n")

    monkeypatch.setattr(task_new.agents, "author_task", _author(
        demo_src, calls=[("Bash", {"command": "cat ../ks/private/ref_values.npy"})], also=notes))
    ok, note = steps.author()
    assert ok is False and "reached for the answers" in note and "a private directory" in note
    assert "../ks/private" not in note, "the public note names the category only"
    left = {p.name for p in steps.task_dir.iterdir()}
    assert left <= {"private", ".authoring"}, left
    kept = steps.private_home / "logs" / "untrusted-package-1"
    assert (kept / "problem.md").is_file() and (kept / ".authoring" / "notes.md").is_file()
    assert (steps.private_home / "logs" / "author.untrusted.log").is_file()
    assert "../ks/private" in (steps.private_home / "logs" / task_new.AUDIT_LOG).read_text()

    # a resume whose harness then fails at once has nothing left to accept
    monkeypatch.setattr(task_new.agents, "author_task", _author(demo_src, rc=1, write=False))
    ok, note = steps.author()
    assert ok is False and "did not finish" in note


@pytest.mark.parametrize("how", ["exit 1", "timed out"])
def test_an_author_that_did_not_finish_is_not_accepted(tmp_path, monkeypatch, demo_src, how):
    steps = _steps(tmp_path)
    monkeypatch.setattr(task_new.agents, "author_task", _author(
        demo_src, rc=1 if how == "exit 1" else None, timed_out=how == "timed out"))
    ok, note = steps.author()
    assert ok is False and "did not finish" in note


def test_a_package_from_an_earlier_session_is_not_accepted(tmp_path, monkeypatch, demo_src):
    steps = _steps(tmp_path)
    _write_package(steps.task_dir, demo_src)
    old = time.time() - 3600
    for f in steps.task_dir.iterdir():
        if f.is_file():
            os.utime(f, (old, old))
    monkeypatch.setattr(task_new.agents, "author_task", _author(demo_src, write=False))
    ok, note = steps.author()
    assert ok is False and "did not write" in note and "problem.md" in note


def test_stray_private_directories_are_quarantined_in_the_private_logs(tmp_path, monkeypatch,
                                                                     demo_src):
    steps = _steps(tmp_path, reference_location=str(tmp_path / "data.npz"))
    _as_ingested(steps, demo_src)
    answers = steps.private_home / "ref_values.npy"
    np.save(answers, np.arange(3.0))

    def stray(task_dir, kw):
        (task_dir / "private").mkdir()
        np.save(task_dir / "private" / "ref_values.npy", np.zeros(3))

    def author_task(name, **kw):
        task_dir = kw["root"] / "tasks" / name
        _write_package(task_dir, demo_src, grid=False)
        stray(task_dir, kw)
        _transcript(kw["log_path"])
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author_task)
    for n in (1, 2, 3):
        ok, note = steps.author()
        assert ok is False and "created private/" in note
        # the answers are the task's own, untouched; the stray is a log
        assert np.array_equal(np.load(answers), np.arange(3.0))
        assert (steps.private_home / "logs" / f"stray-private-{n}" / "ref_values.npy").is_file()
    assert not (steps.auth_dir / "private.by-author").exists()
    assert not list(steps.auth_dir.glob("private*"))


def test_a_stray_in_build_mode_is_not_taken_for_the_answers(tmp_path, monkeypatch, demo_src):
    steps = _steps(tmp_path)

    def stray(task_dir, kw):
        (task_dir / "private").mkdir()
        np.save(task_dir / "private" / "ref_values.npy", np.zeros(3))

    monkeypatch.setattr(task_new.agents, "author_task", _author(demo_src, also=stray))
    steps.author()
    assert not (steps.private_home / "ref_values.npy").exists()


def test_the_author_never_sees_the_operator_s_path_in_a_note(tmp_path, monkeypatch, demo_src):
    data = "/data/secret/cavity_solution.mat"
    steps = _steps(tmp_path, reference_location=data)
    _as_ingested(steps, demo_src)

    def author_task(name, **kw):
        _write_package(kw["root"] / "tasks" / name, demo_src, grid=False)
        _transcript(kw["log_path"], ("Read", {"file_path": data}))
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author_task)
    ok, note = steps.author()
    assert ok is False and "the operator's data" in note
    assert data not in note and data not in authoring.state_path(steps.task_dir).read_text()


# ─────────────────────────────── an external private directory ───────────────────────────────


def test_an_external_private_directory_is_where_every_stage_looks(tmp_path, monkeypatch,
                                                                  demo_src):
    srv = tmp_path / "srv-private"
    monkeypatch.setenv("FORGE_PRIVATE_DIR", str(srv))
    steps = _steps(tmp_path)
    assert steps.private_home == srv / "heat" and not steps.home_in_tree
    seen = {}

    def author_task(name, **kw):
        seen["record"] = kw["record"]
        _write_package(kw["root"] / "tasks" / name, demo_src)
        _transcript(kw["log_path"])
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author_task)
    ok, note = steps.author()
    assert ok, note
    assert str(seen["record"]).startswith(str(srv)), "nothing is created in the task directory"
    assert not (steps.task_dir / "private").exists()

    captured = {}

    def drive(prompt, **kw):
        captured.update(kw, prompt=prompt)
        return agents.AgentRun(True, kw["log_path"], 0.0, 0)

    monkeypatch.setattr(agents, "_drive", drive)
    agents.build_reference("heat", agent=AgentConfig(runtime="claude_code"), python=Path("py"),
                           uv="uv", gpu="", tol=1e-3, budget_s=60, log_path=tmp_path / "l",
                           root=tmp_path)
    assert captured["cwd"] == srv / "heat" / "reference"
    assert f"{srv / 'heat'}/ref_values.npy" in captured["prompt"]
    assert "../ref_values.npy" not in captured["prompt"]


# ─────────────────────────────── the reference stages ───────────────────────────────


def test_a_failed_build_check_says_how_to_rebuild(tmp_path, demo_src):
    steps = _steps(tmp_path)
    _write_package(steps.task_dir, demo_src)
    ok, note = steps.reference_check()
    assert ok is False and "--resume --redo reference" in note


def _ingest(tmp_path, monkeypatch, *, header="x,y", info=None, extra=None, also=None):
    data = tmp_path / "op" / "cavity.npz"
    data.parent.mkdir()
    data.write_bytes(b"operator data")
    steps = _steps(tmp_path, reference_location=str(data))
    t = np.linspace(0, 1, 21)
    x, y = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    u = np.sin(np.pi * x) * np.cos(2 * y) + 0.3 * x * y

    def ingest_reference(name, **kw):
        cols = [x, y] + ([extra(u)] if extra else [])
        np.savetxt(steps.task_dir / "eval_grid.csv", np.column_stack(cols), delimiter=",",
                   header=header, comments="")
        np.save(steps.private_home / "ref_values.npy", u.reshape(-1, 1))
        meta = info if info is not None else {
            "grid_columns": ["x", "y; the solution peaks at 0.98"], "input_columns": 2,
            "ref_columns": ["u"], "rows": 99999, "source": "cavity run 7, Re=1000",
            "notes": "u max 0.98"}
        (steps.private_home / "reference" / "ingest.json").write_text(json.dumps(meta))
        if also:
            also(steps)
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.workspace, "uv_binary", lambda: "uv")
    monkeypatch.setattr(task_new.reference, "solver_env", lambda *a, **k: Path(sys.executable))
    monkeypatch.setattr(task_new.agents, "ingest_reference", ingest_reference)
    return steps


def test_an_ingest_publishes_only_the_shape_the_files_show(tmp_path, monkeypatch):
    steps = _ingest(tmp_path, monkeypatch)
    ok, note = steps.reference()
    assert ok, note
    shown = steps.pipe.state["reference"]
    assert shown["grid_columns"] == ["x", "y"] and shown["rows"] == 441
    assert shown["input_columns"] == 2 and shown["ref_columns"] == ["u"]
    text = authoring.state_path(steps.task_dir).read_text()
    assert "0.98" not in text and "cavity" not in text and "Re=1000" not in text
    assert reference.grid_drift(steps.task_dir, steps.private_home) == ""


@pytest.mark.parametrize("case", ["header", "input_columns", "ref_columns"])
def test_an_ingest_that_cannot_be_read_off_the_files_fails(tmp_path, monkeypatch, case):
    good = {"input_columns": 2, "ref_columns": ["u"]}
    kw = {
        "header": {"header": "x,u_at_0_70710678"},
        "input_columns": {"info": {**good, "input_columns": "2"}},
        "ref_columns": {"info": {**good, "ref_columns": ["u", "v"]}},
    }[case]
    steps = _ingest(tmp_path, monkeypatch, **kw)
    ok, note = steps.reference()
    assert ok is False and "70710678" not in note
    assert "reference" not in steps.pipe.state or not steps.pipe.state["reference"]


def test_a_grid_that_carries_the_answers_never_reaches_the_author(tmp_path, monkeypatch):
    steps = _ingest(tmp_path, monkeypatch, header="x,y,w", extra=lambda u: 2.0 * u + 1.0)
    ok, note = steps.reference()
    assert ok is False and "holds the reference" in note
    assert not (steps.task_dir / "eval_grid.csv").exists()
    assert list((steps.private_home / "logs").glob("leaky-eval_grid-*"))


def test_an_ingest_agent_that_writes_into_the_package_fails(tmp_path, monkeypatch):
    def writes(steps):
        (steps.task_dir / "problem.md").write_text("u peaks at 0.98\n")
        (steps.task_dir / ".authoring" / "hint.md").write_text("0.98\n")

    steps = _ingest(tmp_path, monkeypatch, also=writes)
    ok, note = steps.reference()
    assert ok is False and "2 file(s)" in note
    assert not (steps.task_dir / "problem.md").exists()
    assert not (steps.task_dir / ".authoring" / "hint.md").exists()


def test_the_author_may_not_touch_the_ingested_grid(tmp_path, monkeypatch, demo_src):
    steps = _ingest(tmp_path, monkeypatch)
    assert steps.reference()[0]
    pinned = (steps.task_dir / "eval_grid.csv").read_bytes()

    def resort(task_dir, kw):
        lines = (task_dir / "eval_grid.csv").read_text().splitlines()
        (task_dir / "eval_grid.csv").write_text("\n".join([lines[0], *sorted(lines[1:])]) + "\n")

    def author_task(name, **kw):
        task_dir = kw["root"] / "tasks" / name
        _write_package(task_dir, demo_src, grid=False)
        resort(task_dir, kw)
        _transcript(kw["log_path"])
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author_task)
    ok, note = steps.author()
    assert ok is False and "eval_grid.csv" in note
    assert (steps.task_dir / "eval_grid.csv").read_bytes() == pinned, "the ingested grid is back"


def test_the_author_is_told_only_the_sanitized_shape(tmp_path, monkeypatch, demo_src):
    steps = _ingest(tmp_path, monkeypatch)
    assert steps.reference()[0]
    seen = {}

    def author_task(name, **kw):
        seen["mode"] = kw["mode"]
        _write_package(kw["root"] / "tasks" / name, demo_src, grid=False)
        _transcript(kw["log_path"])
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "author_task", author_task)
    steps.author()
    assert "441 rows" in seen["mode"] and "`x`, `y`" in seen["mode"]
    assert "0.98" not in seen["mode"] and "99999" not in seen["mode"]


# ─────────────────────────────── resuming ───────────────────────────────


def _parse(*argv):
    ap = argparse.ArgumentParser()
    task_new.register(ap.add_subparsers())
    return ap.parse_args(["new", *argv])


def test_flags_given_to_resume_override_the_recorded_inputs(tmp_path):
    steps = _steps(tmp_path)
    before = dict(steps.pipe.state["inputs"])
    args = _parse("heat", "--resume", "--ref-tol", "1e-4", "--accept-reference", "checked by hand",
                  "--model", "m2", "--no-load", "--author-budget", "30")
    changed = task_new._merge_flags(steps.pipe, args, tmp_path)
    assert changed
    inputs = authoring.status(steps.task_dir)["inputs"]
    assert inputs["ref_tol"] == 1e-4 and inputs["accept_reference"] == "checked by hand"
    assert inputs["agent"]["model"] == "m2" and inputs["agent"]["runtime"] == "claude_code"
    assert inputs["no_anchor"] is True and inputs["author_budget_s"] == 1800
    assert inputs["ref_budget_s"] == before["ref_budget_s"], "a flag left out changes nothing"
    assert task_new._merge_flags(steps.pipe, _parse("heat", "--resume"), tmp_path) == []


def test_resume_with_redo_resets_the_stages(tmp_path, monkeypatch):
    steps = _steps(tmp_path)
    for s in authoring.STAGES:
        steps.pipe.state["stages"][s] = {"status": "done"}
    steps.pipe.state["stages"]["reference_check"] = {"status": "failed"}
    steps.pipe._save()
    monkeypatch.setattr(task_new.paths, "project_root", lambda *a, **k: tmp_path)
    ran = []
    monkeypatch.setattr(authoring.Pipeline, "run",
                        lambda self, mapping, before=None: ran.append(self.state) or 0)
    monkeypatch.setattr(task_new.lifecycle, "install_signal_handlers", lambda **k: None)
    assert task_new.cmd_new(_parse("heat", "--resume", "--redo", "reference")) == 0
    st = authoring.status(steps.task_dir)
    assert st["stages"]["reference"]["status"] == "pending"
    assert st["stages"]["author"]["status"] == "done"


def test_the_pipeline_re_executes_without_the_reference_path(tmp_path, monkeypatch):
    (tmp_path / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "tasks").mkdir()
    data = tmp_path / "secret" / "cavity.mat"
    monkeypatch.setattr(task_new.paths, "project_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(task_new.bindings, "resolve",
                        lambda *a, **k: AgentConfig(runtime="claude_code", model="m"))
    monkeypatch.setattr(task_new.gpus, "workspace_pool", lambda *a, **k: [])
    argv = ["pinnforge", "task", "new", "heat", "--describe", "poisson", "--reference", str(data)]
    monkeypatch.setattr(sys, "argv", argv)
    execs = []

    class Exec(Exception):
        pass

    def execv(path, cmd):
        execs.append(cmd)
        raise Exec

    monkeypatch.setattr(task_new.os, "execv", execv)
    with pytest.raises(Exec):
        task_new.cmd_new(_parse(*argv[3:]))
    cmd = execs[0]
    assert "--resume" in cmd and "--reference" not in cmd and str(data) not in " ".join(cmd)
    assert reference.source_location(tmp_path / "tasks" / "heat" / "private") == str(data)


# ─────────────────────────────── no agent outlives its session ───────────────────────────────


class _Runtime:
    """A harness stand-in: a shell that leaves a child behind in its own group."""

    name = "fake"

    def __init__(self, interrupt=False):
        self.interrupt = interrupt

    def start(self, *, block_id, cwd, prompt, model, log_path, env, **kw):
        from pinnforge.runtime.base import spawn

        handle = spawn(["sh", "-c", "sleep 30 & sleep 30"], cwd=cwd, env=env, log_path=log_path,
                       block_id=block_id)
        if self.interrupt:
            def wait(timeout=None):
                raise KeyboardInterrupt
            handle.wait = wait
        return handle


@pytest.mark.parametrize("how", ["interrupted", "timed out"])
def test_the_agent_s_group_is_gone_when_the_session_ends(tmp_path, monkeypatch, how):
    rt = _Runtime(interrupt=how == "interrupted")
    monkeypatch.setattr(agents.runtime_registry, "get_runtime", lambda name: rt)

    # A harness that goes down at Ctrl-C but leaves its children running —
    # the common case; what is left of the group is `_drive`'s to end.
    from pinnforge.runtime.base import AgentHandle

    def cli_only(self, grace=15.0):
        if self.alive:
            self.process.kill()
            self.process.wait()

    monkeypatch.setattr(AgentHandle, "interrupt", cli_only)
    monkeypatch.setattr(agents.runtime_registry, "resolve_model", lambda r, m: m)
    record = tmp_path / "logs" / "agent.json"
    seen = {}
    real_write = Path.write_text

    def spy(self, text, *a, **k):
        if self == record:
            seen.update(json.loads(text))
        return real_write(self, text, *a, **k)

    monkeypatch.setattr(Path, "write_text", spy)
    kwargs = {"cwd": tmp_path, "agent": AgentConfig(runtime="fake", model="m"),
              "log_path": tmp_path / "a.log", "label": "t", "record": record}
    if how == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            agents._drive("p", **kwargs)
    else:
        run = agents._drive("p", timeout_s=0.5, **kwargs)
        assert run.timed_out and not run.ok
    assert seen.get("pgid")
    assert not agents.group_alive(seen["pgid"]), "nothing of the agent is left running"
    assert not record.exists() and agents.live_pgid() is None


def test_resume_is_refused_while_an_earlier_agent_lives(tmp_path):
    steps = _steps(tmp_path)
    held = steps._sealed_home() / "private"
    held.mkdir(parents=True)
    (held / "ref_values.npy").write_bytes(b"answers")
    proc = subprocess.Popen(["sleep", "30"], cwd=steps.task_dir, start_new_session=True)
    try:
        rec = held / "logs" / task_new.AGENT_RECORD
        rec.parent.mkdir()
        rec.write_text(json.dumps({"pgid": proc.pid, "cwd": str(steps.task_dir)}))
        with pytest.raises(RuntimeError, match="still running"):
            steps.recover()
        assert (held / "ref_values.npy").is_file(), "the answers stay sealed while it lives"
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    steps.recover()
    assert (steps.task_dir / "private" / "ref_values.npy").read_bytes() == b"answers"
    assert not (steps.task_dir / "private" / "logs" / task_new.AGENT_RECORD).exists()


def test_the_pipeline_installs_signal_handlers_and_logs_stderr(tmp_path, monkeypatch):
    steps = _steps(tmp_path)
    for s in authoring.STAGES:
        steps.pipe.state["stages"][s] = {"status": "done"}
    steps.pipe._save()
    monkeypatch.setattr(task_new.paths, "project_root", lambda *a, **k: tmp_path)
    installed = []
    monkeypatch.setattr(task_new.lifecycle, "install_signal_handlers",
                        lambda **k: installed.append(k))
    log = tmp_path / "pipeline.log"
    out, err = sys.stdout, sys.stderr
    try:
        assert task_new.cmd_new(_parse("heat", "--resume", "--log", str(log))) == 0
        print("a traceback would land here", file=sys.stderr)
        sys.stderr.flush()
    finally:
        sys.stdout, sys.stderr = out, err
    assert installed and installed[0]["escalate"] is agents.kill_live
    assert "a traceback would land here" in log.read_text()


def test_framework_records_an_agent_changed_are_put_back(tmp_path, monkeypatch):
    def scribble(steps):
        (steps.auth_dir / "intake.md").write_text("the answer is 0.98\n")

    steps = _ingest(tmp_path, monkeypatch, also=scribble)
    ok, note = steps.reference()
    assert ok, note
    assert "0.98" not in (steps.auth_dir / "intake.md").read_text()
    assert (steps.auth_dir / "intake.md").read_text() == steps._intake_text()


def test_a_builder_whose_harness_keeps_its_session_warns_in_the_note(tmp_path, monkeypatch,
                                                                      demo_src):
    steps = _steps(tmp_path)
    _write_package(steps.task_dir, demo_src)
    steps.agent = AgentConfig(runtime="opencode", model="deepseek/x")
    monkeypatch.setattr(task_new.workspace, "uv_binary", lambda: "uv")
    monkeypatch.setattr(task_new.reference, "solver_env", lambda *a, **k: Path(sys.executable))
    monkeypatch.setattr(task_new.gpus, "workspace_pool", lambda *a, **k: [])

    def build_reference(name, **kw):
        np.save(steps.private_home / "ref_values.npy", np.zeros((10201, 1)))
        kw["log_path"].write_text("message=created id=ses_0123456789\n")
        return SimpleNamespace(duration_s=1.0, returncode=0, timed_out=False)

    monkeypatch.setattr(task_new.agents, "build_reference", build_reference)
    ok, note = steps.reference()
    assert ok and "WARNING" in note and "OpenCode" in note and "ses_0123456789" in note
