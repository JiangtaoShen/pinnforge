"""Pi as a block runtime: its argv, its accounting, its transcript, its leftovers.

`data/pi-1.3.0-json.jsonl` is what `pi --mode json` 1.3.0 wrote for a session
of four tool calls against a scripted OpenAI-compatible endpoint, with the
paths and the system prompt shortened: the shapes the adapter, the dashboard
and the author audit all read.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pinnforge import oscompat
from pinnforge.config import AgentConfig
from pinnforge.runtime import registry
from pinnforge.runtime.base import AgentRuntime, clean_env, spawn
from pinnforge.runtime.builtin import pi as pi_rt
from pinnforge.task import audit
from pinnforge.ui import feeds

FIXTURE = Path(__file__).parent / "data" / "pi-1.3.0-json.jsonl"
LINES = FIXTURE.read_text(encoding="utf-8").splitlines()
GOOD_REPLY = next(line for line in reversed(LINES)
                  if line.startswith('{"type":"message_end"') and '"role":"assistant"' in line)
LINUX = sys.platform.startswith("linux")
posix_only = pytest.mark.skipif(oscompat.WINDOWS, reason="POSIX signals")
linux_only = pytest.mark.skipif(not LINUX, reason="a subreaper is a Linux facility")

FAILED_REPLY = json.dumps({"type": "message_end", "message": {
    "role": "assistant", "content": [], "provider": "anthropic", "model": "claude-opus-4-8",
    "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
              "cost": {"total": 0}},
    "stopReason": "error", "errorMessage": "401 invalid x-api-key"}})


def _start(monkeypatch, tmp_path, **kw):
    seen = {}

    def fake_spawn(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)

    monkeypatch.setattr(pi_rt, "spawn", fake_spawn)
    kw.setdefault("model", "anthropic/claude-opus-4-8")
    kw.setdefault("env", {"PINNFORGE_ROLE": "block"})
    pi_rt.PiRuntime().start(block_id="b01", cwd=tmp_path, prompt="PROMPT",
                            log_path=tmp_path / "logs" / "b01.0.log", **kw)
    cmd = seen["cmd"]
    assert cmd[:4] == [sys.executable, "-P", "-m", "pinnforge.runtime.reaper"], cmd
    end = cmd.index("--")
    seen["drop"] = cmd[4:end][1::2]
    assert cmd[4:end][::2] == ["--drop-type"] * len(seen["drop"])
    return cmd[end + 1 :], seen


# ─────────────────────────────── starting it ───────────────────────────────


def test_pi_is_a_runtime_by_its_names():
    assert "pi" in registry.known_runtimes()
    assert isinstance(registry.get_runtime("pi"), AgentRuntime)
    for alias in ("pi", "pi-agent", "pi-coding-agent"):
        assert registry.canonical(alias) == "pi"


def test_a_block_is_started_headless_with_a_fixed_surface(monkeypatch, tmp_path):
    cmd, seen = _start(monkeypatch, tmp_path)
    assert cmd[0] == "pi"
    assert cmd[cmd.index("--mode") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "anthropic/claude-opus-4-8"
    sessions = Path(cmd[cmd.index("--session-dir") + 1])
    assert sessions == tmp_path / "logs" / pi_rt.SESSIONS_DIR and sessions.is_dir(), (
        "beside the log, never in ~/.pi/agent/sessions")
    for flag in ("--no-extensions", "--no-skills", "--no-context-files", "--no-mcp",
                 "--no-approve", "--no-prompt-templates"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--tools") + 1] == "read,bash,edit,write"
    assert cmd[-2:] == ["--", "PROMPT"], "the prompt last, after the end of options"
    assert "--session" not in cmd
    assert seen["stdin"] is subprocess.DEVNULL, "Pi reads a piped stdin to its end first"
    assert seen["handle"] is pi_rt.PiHandle
    assert seen["env"]["PI_SKIP_VERSION_CHECK"] == "1"
    assert seen["cwd"] == tmp_path
    assert set(seen["drop"]) == set(pi_rt.UNLOGGED) == {
        "message_update", "tool_execution_update", "message_start", "turn_end", "agent_end",
        "tool_execution_end"}, "deltas and repeats stay out"
    kept = [json.loads(line)["type"] for line in LINES]
    assert {"session", "message_end", "tool_execution_start",
            "agent_settled"} <= set(kept) - set(seen["drop"]), "what the readers use stays"


def test_a_sandbox_prefix_wraps_pi_and_not_the_reaper(monkeypatch, tmp_path):
    """The reaper is the framework's, on the host; a sandbox may not have its Python."""
    cmd, seen = _start(monkeypatch, tmp_path, command_prefix=["bwrap", "--ro-bind", "/", "/"])
    assert cmd[:4] == ["bwrap", "--ro-bind", "/", "/"] and cmd[4] == "pi"
    assert seen.get("command_prefix") is None


def test_resume_thinking_and_extra_arguments(monkeypatch, tmp_path):
    cmd, _ = _start(monkeypatch, tmp_path, resume_session_id="01a159a0-e2e8",
                    runtime_options={"thinking": "high", "extra_args": ["--verbose"]},
                    command="/opt/pi/bin/pi")
    assert cmd[0] == "/opt/pi/bin/pi"
    end = cmd.index("--")
    assert cmd[cmd.index("--session") + 1] == "01a159a0-e2e8" and cmd.index("--session") < end
    assert cmd[cmd.index("--thinking") + 1] == "high"
    assert cmd.index("--verbose") < end


@pytest.mark.parametrize("role", ["author", "reference", "film", "kb1", "ping"])
def test_only_a_block_keeps_a_session(monkeypatch, tmp_path, role):
    """None of the others is resumed; a session file beside an author's log that
    reached for the answers stayed public when the log was moved private."""
    cmd, _ = _start(monkeypatch, tmp_path, env={"PINNFORGE_ROLE": role})
    assert "--no-session" in cmd and "--session-dir" not in cmd
    assert not (tmp_path / "logs" / pi_rt.SESSIONS_DIR).exists()
    cmd, _ = _start(monkeypatch, tmp_path, env={"PINNFORGE_ROLE": role},
                    resume_session_id="01a159a0")
    assert "--session-dir" in cmd and "--no-session" not in cmd, "a resume needs its store"


@pytest.mark.parametrize("model", ["", "claude-opus", "sonnet:high"])
def test_a_model_without_its_provider_is_refused(monkeypatch, tmp_path, model):
    with pytest.raises(ValueError, match="provider/id"):
        _start(monkeypatch, tmp_path, model=model)
    with pytest.raises(ValueError, match="states no default model"):
        registry.resolve_model(registry.get_runtime("pi"), "")


def test_a_block_does_not_inherit_the_operators_pi_session(monkeypatch):
    for key in ("PI_SESSION_FILE", "PI_SESSION_ID", "PI_CODING_AGENT", "AI_AGENT"):
        monkeypatch.setenv(key, "parent")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/keep/this")
    env = clean_env()
    assert not {"PI_SESSION_FILE", "PI_SESSION_ID", "PI_CODING_AGENT", "AI_AGENT"} & set(env)
    assert env["PI_CODING_AGENT_DIR"] == "/keep/this", "where Pi keeps its logins is config"


# ─────────────────────────────── reading its log ───────────────────────────────


def test_the_session_id_comes_from_the_header():
    assert registry.get_runtime("pi").extract_session_id(FIXTURE) == (
        "01a159a0-e2e8-733d-b28c-2b2cce8f9e81")


def test_usage_is_summed_over_replies_and_names_the_model_that_answered():
    u = registry.get_runtime("pi").extract_usage(FIXTURE)
    assert (u.turns, u.tool_uses) == (5, 4), "turn_end and agent_end repeat replies"
    assert (u.tokens, u.cache_read, u.output_tokens) == (5 * 230, 5 * 1000, 5 * 30)
    assert u.cost_usd == pytest.approx(5 * 0.00135)
    assert u.model == "mock/mock-coder-1-20261001", "responseModel, under its provider"


def test_a_failed_last_reply_is_a_failure(tmp_path):
    assert pi_rt.failure(FIXTURE) == ""
    log = tmp_path / "b01.0.log"
    log.write_text("\n".join([*LINES, FAILED_REPLY]) + "\n", encoding="utf-8")
    assert pi_rt.failure(log) == "401 invalid x-api-key"
    assert registry.get_runtime("pi").extract_error(log) == "401 invalid x-api-key"


@pytest.mark.parametrize("reply, rc", [(GOOD_REPLY, 0), (FAILED_REPLY, 1)], ids=["ok", "error"])
def test_exit_0_with_a_failed_reply_is_reported_as_a_failure(tmp_path, reply, rc):
    """JSON mode exits 0 whatever the reply; Pi's own print mode exits 1 on these."""
    code = f"import sys; sys.stdout.write({reply!r} + '\\n')"
    h = spawn([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ),
              log_path=tmp_path / "x.log", block_id="b01", handle=pi_rt.PiHandle)
    assert h.wait(timeout=30) == rc
    assert h.returncode == rc
    h.close()


@posix_only
def test_an_interrupted_pi_gets_sigterm(tmp_path):
    """Pi's JSON mode stops the commands its tools run on SIGTERM; SIGINT just ends it."""
    ready, said = tmp_path / "ready", tmp_path / "said"
    code = (
        "import signal, sys, time, pathlib\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        f"def term(*a): pathlib.Path({str(said)!r}).write_text('term'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, term)\n"
        f"pathlib.Path({str(ready)!r}).write_text('1')\n"
        "while True: time.sleep(0.05)\n"
    )
    h = spawn([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ),
              log_path=tmp_path / "x.log", block_id="b01", handle=pi_rt.PiHandle)
    deadline = time.time() + 20
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    t = time.time()
    h.interrupt(grace=10)
    assert said.read_text() == "term" and time.time() - t < 5


@pytest.mark.parametrize("pinned, problem", [
    ("mock/mock-coder-1", False),
    ("mock/MOCK-Coder-1", False),
    ("mock/mock-coder-1:high", False),
    ("mock/mock-coder", True),  # a pattern Pi matched, as an alias would be
    ("mock/coder-1", True),
])
def test_a_pin_pi_only_matched_loosely_is_a_problem(pinned, problem):
    said = registry.get_runtime("pi").pin_problem(pinned, FIXTURE)
    assert bool(said) is problem, said
    if problem:
        assert "mock/mock-coder-1" in said


def test_a_colon_in_the_id_is_not_a_thinking_level(tmp_path):
    reply = json.loads(GOOD_REPLY)
    reply["message"].update(provider="ollama", model="qwen2.5-coder:7b")
    log = tmp_path / "x.log"
    log.write_text(json.dumps(reply) + "\n", encoding="utf-8")
    rt = registry.get_runtime("pi")
    assert rt.pin_problem("ollama/qwen2.5-coder:7b", log) == ""
    assert rt.pin_problem("ollama/qwen2.5-coder", log)


# ─────────────────────────────── the dashboard ───────────────────────────────


def test_the_dashboard_reads_a_pi_session():
    events = feeds.parse("pi", LINES, "/tmp/run")
    texts = [(e["kind"], e["text"]) for e in events]
    assert texts[0] == ("info", "session started")
    assert ("tool", "bash: echo hi > note.txt && ls -la") in texts
    assert ("tool", "read: note.txt") in texts
    assert ("text", "All done.") in texts
    assert sum(k == "tool" for k, _ in texts) == 4, "a tool once, not once per repeat"
    assert texts[-1] == ("info", "session finished")
    assert sum(t == "session finished" for _, t in texts) == 1
    assert feeds.parse("pi", ['{"type":"agent_settled","aborted":true}']) == [
        {"kind": "info", "text": "session aborted"}]
    failed = feeds.parse("pi", [FAILED_REPLY])
    assert failed == [{"kind": "info", "text": "reply failed · 401 invalid x-api-key"}]


# ─────────────────────────────── the author audit ───────────────────────────────


def test_the_audit_reads_every_pi_tool_call_once_with_its_output():
    calls = audit.tool_calls("pi", LINES)
    assert [c.tool for c in calls] == ["bash", "read", "find", "grep"]
    bash, read, find, grep = calls
    assert "echo hi > note.txt" in bash.shell and "note.txt" in bash.result
    assert read.paths == ["note.txt"] and "hi" in read.result
    assert find.globs == [(".", "**/*.txt")]
    assert grep.search and grep.paths == ["."] and grep.globs == [(".", "*.txt")]


# ─────────────────────────────── the ping ───────────────────────────────


def _fake_pi(tmp_path: Path, reply: str) -> str:
    """A `pi` that answers with one reply, reading stdin to its end as Pi does."""
    script = tmp_path / "fake-pi"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdin.read()\n"
        f"print({LINES[0]!r})\n"
        f"print({reply!r})\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


@posix_only
def test_ping_goes_through_the_adapter_and_reads_the_reply(tmp_path):
    from pinnforge.cli import agents_cmd

    good = json.loads(GOOD_REPLY)
    assert good["message"]["responseModel"] == "mock-coder-1-20261001"
    ok, detail, ran = agents_cmd.ping(AgentConfig(
        runtime="pi", model="mock/mock-coder-1",
        command=_fake_pi(tmp_path, GOOD_REPLY)), timeout=60)
    assert ok and ran == "mock/mock-coder-1-20261001", detail
    ok, detail, ran = agents_cmd.ping(AgentConfig(
        runtime="pi", model="anthropic/claude-opus-4-8",
        command=_fake_pi(tmp_path, FAILED_REPLY)), timeout=60)
    assert not ok and "401 invalid x-api-key" in detail, detail
    ok, detail, _ = agents_cmd.ping(AgentConfig(
        runtime="pi", model="mock/mock-coder",
        command=_fake_pi(tmp_path, GOOD_REPLY)), timeout=60)
    assert not ok and "pin that id exactly" in detail, detail


# ─────────────────────────────── what its tools leave running ───────────────────────────────


def _gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.time() + within
    while time.time() < deadline:
        stat = Path(f"/proc/{pid}/stat")
        try:
            state = stat.read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


def _reaper(*argv):
    return [sys.executable, "-m", "pinnforge.runtime.reaper", "--", *argv]


@linux_only
def test_what_the_cli_leaves_in_a_session_of_its_own_is_stopped(tmp_path):
    pidfile = tmp_path / "pid"
    out = subprocess.run(_reaper("sh", "-c", f"setsid sleep 60 & echo $! > {pidfile}; exit 3"),
                         capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 3, "the CLI's own status"
    assert _gone(int(pidfile.read_text())), "a session the CLI split off went with it"
    assert "left running" in out.stderr


@linux_only
def test_a_stop_reaches_the_cli_as_sigterm_and_takes_its_tools_along(tmp_path):
    pidfile = tmp_path / "pid"
    script = (f"trap 'exit 7' TERM; setsid sleep 60 & echo $! > {pidfile}; "
              "while :; do sleep 0.1; done")
    p = subprocess.Popen(_reaper("sh", "-c", script), start_new_session=True,
                         stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 20
    while not (pidfile.exists() and pidfile.read_text().strip()) and time.time() < deadline:
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGINT)  # any stop signal: the CLI is asked with SIGTERM
    assert p.wait(timeout=30) == 7
    assert _gone(int(pidfile.read_text()))
    p.stderr.close()


def test_a_log_without_the_deltas_reads_the_same(tmp_path):
    """What the adapter keeps out of the log is nothing the readers needed."""
    rt = registry.get_runtime("pi")
    log = tmp_path / "b01.0.log"
    log.write_text("\n".join(line for line in LINES
                             if json.loads(line)["type"] not in pi_rt.UNLOGGED) + "\n",
                   encoding="utf-8")
    assert rt.extract_usage(log) == rt.extract_usage(FIXTURE)
    assert rt.extract_session_id(log) == rt.extract_session_id(FIXTURE)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert feeds.parse("pi", lines) == feeds.parse("pi", LINES)
    calls = audit.tool_calls("pi", lines)
    assert [(c.describe(), c.result) for c in calls] == [
        (c.describe(), c.result) for c in audit.tool_calls("pi", LINES)]
    assert "note.txt" in calls[0].result, "a tool's output: the reply the model was given"
    assert log.stat().st_size < FIXTURE.stat().st_size / 2


def test_the_reaper_drops_the_lines_it_is_told_to(tmp_path):
    code = ("import json\n"
            "for t in ('session', 'message_update', 'message_end', 'message_update'):\n"
            "    print(json.dumps({'type': t}, separators=(',', ':')), flush=True)\n"
            "print('plain text')\n")
    out = subprocess.run(_reaper_drop(["message_update"], sys.executable, "-c", code),
                         capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ['{"type":"session"}', '{"type":"message_end"}',
                                       "plain text"]


def _reaper_drop(types, *argv):
    return [sys.executable, "-m", "pinnforge.runtime.reaper",
            *[a for t in types for a in ("--drop-type", t)], "--", *argv]


_STOP_ORDER = """
import os, signal, subprocess, sys, time
marks = sys.argv[1]
def mark(name):
    with open(marks, "a") as f:
        f.write(f"{name} {time.monotonic()}\\n")
if len(sys.argv) > 2:   # the tool: a session of its own, as Pi's bash tool makes
    def term(*a):
        time.sleep(0.5)  # an evaluation settling its budget
        mark("tool"); sys.exit(0)
    signal.signal(signal.SIGTERM, term)
    open(marks + ".ready", "w").close()
    while True: time.sleep(0.05)
def term(*a):
    mark("cli"); sys.exit(0)
signal.signal(signal.SIGTERM, term)
subprocess.Popen([sys.executable, __file__, marks, "tool"], start_new_session=True)
while True: time.sleep(0.05)
"""


@linux_only
def test_a_stopped_agent_s_tools_are_stopped_before_the_cli(tmp_path):
    """SIGTERM reaches an evaluation before Pi, which would SIGKILL it unsettled."""
    script, marks = tmp_path / "cli.py", tmp_path / "marks"
    script.write_text(_STOP_ORDER, encoding="utf-8")
    p = subprocess.Popen(_reaper(sys.executable, str(script), str(marks)),
                         start_new_session=True, stdin=subprocess.DEVNULL)
    deadline = time.time() + 20
    while not Path(str(marks) + ".ready").exists() and time.time() < deadline:
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGTERM)  # what AgentHandle.stop sends the agent's group
    assert p.wait(timeout=30) == 0
    order = [line.split()[0] for line in marks.read_text().splitlines()]
    assert order == ["tool", "cli"], order


@linux_only
def test_the_cli_does_not_outlive_a_killed_reaper(tmp_path):
    said, ready = tmp_path / "said", tmp_path / "ready"
    code = ("import pathlib, signal, sys, time\n"
            f"def term(*a): pathlib.Path({str(said)!r}).write_text('term'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, term)\n"
            f"pathlib.Path({str(ready)!r}).write_text('1')\n"
            "while True: time.sleep(0.05)\n")
    p = subprocess.Popen(_reaper(sys.executable, "-c", code), start_new_session=True,
                         stdin=subprocess.DEVNULL)
    deadline = time.time() + 20
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    deadline = time.time() + 10
    while not said.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert said.read_text() == "term"


@posix_only
def test_without_a_subreaper_the_reaper_relays(tmp_path):
    """macOS, or a kernel that refuses: a pass-through that still filters and relays."""
    ready, said = tmp_path / "ready", tmp_path / "said"
    cli = ("import pathlib, signal, sys, time\n"
           f"def term(*a): pathlib.Path({str(said)!r}).write_text('term'); sys.exit(5)\n"
           "signal.signal(signal.SIGTERM, term)\n"
           "print('{\"type\":\"noise\"}'); print('kept', flush=True)\n"
           f"pathlib.Path({str(ready)!r}).write_text('1')\n"
           "while True: time.sleep(0.05)\n")
    runner = ("import sys, pinnforge.runtime.reaper as r\n"
              "r._become_subreaper = lambda: False\n"
              "sys.exit(r.main(sys.argv[1:]))\n")
    p = subprocess.Popen([sys.executable, "-c", runner, "--drop-type", "noise", "--",
                          sys.executable, "-c", cli], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    deadline = time.time() + 20
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 5 and said.read_text() == "term"
    assert out.splitlines() == ["kept"]


@linux_only
def test_a_cli_that_will_not_stop_is_killed_in_time(tmp_path):
    ready = tmp_path / "ready"
    code = ("import pathlib, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"pathlib.Path({str(ready)!r}).write_text('1')\n"
            "while True: time.sleep(0.05)\n")
    p = subprocess.Popen(_reaper(sys.executable, "-c", code), start_new_session=True,
                         stdin=subprocess.DEVNULL)
    deadline = time.time() + 20
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    t = time.time()
    os.killpg(p.pid, signal.SIGTERM)
    p.wait(timeout=30)
    took = time.time() - t
    assert took < 15, f"{took:.1f} s: past the 15 s a PiHandle gives the reaper"
    assert p.returncode == 128 + signal.SIGKILL


@linux_only
def test_what_the_cli_left_running_has_time_to_settle(tmp_path):
    """An evaluation left in the background settles its budget on SIGTERM; it takes seconds."""
    marks, ready = tmp_path / "settled", tmp_path / "ready"
    tool = ("import pathlib, signal, sys, time\n"
            f"def term(*a): time.sleep(3); pathlib.Path({str(marks)!r}).write_text('1'); "
            "sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, term)\n"
            f"pathlib.Path({str(ready)!r}).write_text('1')\n"
            "while True: time.sleep(0.05)\n")
    cli = (f"import subprocess, sys, time, pathlib\n"
           f"subprocess.Popen([sys.executable, '-c', {tool!r}], start_new_session=True)\n"
           f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.05)\n")
    out = subprocess.run(_reaper(sys.executable, "-c", cli), capture_output=True, text=True,
                         timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    assert marks.exists(), "SIGKILLed before it could settle"


@posix_only
def test_the_agent_s_directory_cannot_supply_the_reaper(tmp_path):
    """`-m` would look in the working directory first, and it is the agent's."""
    planted = tmp_path / "pinnforge" / "runtime"
    planted.mkdir(parents=True)
    (tmp_path / "pinnforge" / "__init__.py").write_text("")
    (planted / "__init__.py").write_text("")
    marker = tmp_path / "planted-ran"
    (planted / "reaper.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    h = pi_rt.PiRuntime().start(
        block_id="b01", cwd=tmp_path, prompt="P", model="mock/mock-coder-1",
        log_path=tmp_path / "logs" / "b01.0.log", env={"PINNFORGE_ROLE": "block"},
        command=_fake_pi(tmp_path, GOOD_REPLY))
    assert h.wait(timeout=60) == 0
    h.close()
    assert not marker.exists()
    assert "All done." in (tmp_path / "logs" / "b01.0.log").read_text()


def test_the_reaper_needs_a_command(capsys):
    from pinnforge.runtime import reaper

    assert reaper.main(["--"]) == 2
    assert reaper.main(["--drop-type"]) == 2
    assert reaper.main(["--bogus", "--", "x"]) == 2
    assert "usage" in capsys.readouterr().err
