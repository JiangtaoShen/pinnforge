"""Runtime registry and adapter argv construction.

The adapters are the only place a harness's identity leaks into the system,
so these tests check that each one is put into headless, unattended mode and
that the pieces the orchestrator depends on — model pin, resume token, log
capture — are actually passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pinnforge.runtime import registry
from pinnforge.runtime.base import AgentRuntime, clean_env


def test_all_builtins_satisfy_the_protocol():
    for name in registry.known_runtimes():
        assert isinstance(registry.get_runtime(name), AgentRuntime)


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("claude", "claude_code"),
        ("claude-code", "claude_code"),
        ("openai", "codex"),
        ("cursor", "cursor_agent"),
        ("open-code", "opencode"),
        ("claude_code", "claude_code"),
    ],
)
def test_aliases(alias, expected):
    assert registry.canonical(alias) == expected
    assert registry.get_runtime(alias).name == expected


def test_unknown_runtime_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        registry.get_runtime("nope")
    assert "claude_code" in str(e.value)


def test_custom_entrypoint_must_satisfy_the_protocol():
    with pytest.raises(ValueError, match="does not satisfy"):
        registry.get_runtime("pathlib:PurePath")
    with pytest.raises(ValueError, match="cannot import"):
        registry.get_runtime("no.such.module:Thing")


def _argv(monkeypatch, runtime, **kw):
    """Capture the argv an adapter would exec, without starting anything."""
    seen = {}

    def fake_spawn(cmd, *, cwd, env, log_path, block_id, command_prefix=None):
        seen["cmd"] = cmd
        seen["env"] = env
        seen["cwd"] = cwd
        seen["prefix"] = command_prefix
        return

    for mod in (
        "pinnforge.runtime.builtin.claude_code",
        "pinnforge.runtime.builtin.codex",
        "pinnforge.runtime.builtin.cursor_agent",
        "pinnforge.runtime.builtin.opencode",
    ):
        monkeypatch.setattr(mod + ".spawn", fake_spawn, raising=True)

    kw.setdefault("block_id", "b01")
    kw.setdefault("cwd", Path("/tmp/run"))
    kw.setdefault("prompt", "PROMPT")
    kw.setdefault("model", "")
    kw.setdefault("log_path", Path("/tmp/run/logs/b01.0.log"))
    kw.setdefault("env", {})
    runtime.start(**kw)
    return seen


@pytest.mark.parametrize("name", ["claude_code", "codex", "cursor_agent", "opencode"])
def test_prompt_and_model_reach_the_cli(monkeypatch, name):
    rt = registry.get_runtime(name)
    seen = _argv(monkeypatch, rt, model="M")
    assert "PROMPT" in seen["cmd"], seen["cmd"]
    assert "M" in seen["cmd"], seen["cmd"]
    assert seen["cwd"] == Path("/tmp/run")


@pytest.mark.parametrize("name", ["claude_code", "codex", "cursor_agent"])
def test_default_model_used_when_unpinned(monkeypatch, name):
    rt = registry.get_runtime(name)
    seen = _argv(monkeypatch, rt)
    assert rt.default_model in seen["cmd"]


def test_opencode_refuses_to_start_without_a_model(monkeypatch):
    """It fronts many providers, so any default this repo picked would be a guess.

    A guess that runs is worse than one that does not: the ledger records it as
    though it were chosen, and the run reads as pinned when nobody pinned it.
    """
    rt = registry.get_runtime("opencode")
    assert rt.default_model == ""
    with pytest.raises(ValueError, match="no default model"):
        _argv(monkeypatch, rt)

    seen = _argv(monkeypatch, rt, model="anthropic/claude-opus-4-8")
    assert "anthropic/claude-opus-4-8" in seen["cmd"]


def test_an_unpinned_run_is_refused_before_anything_is_built():
    """The error has to arrive before a run directory and a measured anchor do."""
    rt = registry.get_runtime("opencode")
    with pytest.raises(ValueError, match="states no default model"):
        registry.resolve_model(rt, "")
    assert registry.resolve_model(rt, "anthropic/claude-opus-4-8") == (
        "anthropic/claude-opus-4-8"
    )
    # a harness that does name one is unaffected
    assert registry.resolve_model(registry.get_runtime("claude_code")) == "claude-opus-4-8"


@pytest.mark.parametrize(
    "name,flag",
    [
        ("claude_code", "--dangerously-skip-permissions"),
        ("codex", "--dangerously-bypass-approvals-and-sandbox"),
        ("cursor_agent", "--force"),
    ],
)
def test_unattended_mode_is_requested(monkeypatch, name, flag):
    """No human approves tools mid-run; confinement is the sandbox's job."""
    seen = _argv(monkeypatch, registry.get_runtime(name))
    assert flag in seen["cmd"]


@pytest.mark.parametrize("name", ["claude_code", "codex", "cursor_agent", "opencode"])
def test_resume_token_is_passed_through(monkeypatch, name):
    # pinned explicitly: opencode names no default and refuses to start without one
    seen = _argv(monkeypatch, registry.get_runtime(name), model="M", resume_session_id="SID")
    assert "SID" in seen["cmd"], seen["cmd"]


def test_sandbox_prefix_wraps_the_command(monkeypatch):
    seen = _argv(
        monkeypatch,
        registry.get_runtime("claude_code"),
        command_prefix=["firejail", "--"],
    )
    assert seen["prefix"] == ["firejail", "--"]
    assert seen["env"]["IS_SANDBOX"] == "1"


def test_command_override(monkeypatch):
    seen = _argv(monkeypatch, registry.get_runtime("claude_code"), command="/opt/claude")
    assert seen["cmd"][0] == "/opt/claude"


def test_clean_env_drops_inherited_session_markers(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/else")
    env = clean_env({"FORGE_WALL_BUDGET": "7200"})
    assert "CLAUDE_CODE_SESSION_ID" not in env
    assert "VIRTUAL_ENV" not in env
    assert env["FORGE_WALL_BUDGET"] == "7200"


def test_session_id_extraction(tmp_path):
    log = tmp_path / "b01.0.log"
    log.write_text(
        '\n'.join(
            [
                "not json",
                '{"type":"system","session_id":"abc-123"}',
                '{"type":"assistant","session_id":"later"}',
            ]
        ),
        encoding="utf-8",
    )
    assert registry.get_runtime("claude_code").extract_session_id(log) == "abc-123"


def test_session_id_absent_is_not_an_error(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("", encoding="utf-8")
    assert registry.get_runtime("codex").extract_session_id(log) is None
    assert registry.get_runtime("opencode").extract_session_id(tmp_path / "nope.log") is None


def test_claude_code_reads_its_own_accounting(tmp_path):
    """The numbers were always in the log; the ledger just was not reading them.

    A block's wall time says nothing about how much model went into it, and
    the previous framework carried tokens and tool calls in its summary. The
    stream-json a headless session writes closes with one `result` object
    holding the whole session's usage.
    """
    import json

    from pinnforge.runtime.builtin.claude_code import ClaudeCodeRuntime

    log = tmp_path / "b01.0.log"
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {}},
            {"type": "text", "text": "thinking"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {}},
        ]}},
        {
            "type": "result",
            "num_turns": 9,
            "total_cost_usd": 1.25,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 900,
                "cache_read_input_tokens": 50_000,
                "output_tokens": 2_000,
            },
        },
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    u = ClaudeCodeRuntime().extract_usage(log)
    assert u.tokens == 3_000, "input + cache-creation + output, cache reads excluded"
    assert u.cache_read == 50_000
    assert u.output_tokens == 2_000
    assert u.turns == 9
    assert u.tool_uses == 2
    assert u.cost_usd == 1.25
    assert bool(u)


def test_usage_of_a_missing_or_unreadable_log_is_empty_not_fatal(tmp_path):
    from pinnforge.runtime.base import AgentUsage
    from pinnforge.runtime.builtin.claude_code import ClaudeCodeRuntime

    assert ClaudeCodeRuntime().extract_usage(tmp_path / "nope.log") == AgentUsage()
    junk = tmp_path / "junk.log"
    junk.write_text("not json\n{\n", encoding="utf-8")
    assert not ClaudeCodeRuntime().extract_usage(junk)


def test_every_builtin_reports_usage_or_says_it_cannot():
    """A harness that reports nothing must still answer the question."""
    from pinnforge.runtime import registry
    from pinnforge.runtime.base import AgentUsage

    for name in registry.known_runtimes():
        rt = registry.get_runtime(name)
        assert isinstance(rt.extract_usage(Path("/nonexistent")), AgentUsage), name


def test_the_default_runtime_pins_an_exact_id_not_an_alias():
    """A pin that moves is not a pin.

    `--model opus` resolved to `claude-opus-4-8` for a year and to
    `claude-opus-5` after. A run defaulting to the alias records "opus" in its
    ledger either way, so two runs a generation apart look identical and their
    scores get compared as though one framework beat another. The default
    runtime is the one this repo actually ships blocks on, so its default is
    held to an exact id.
    """
    from pinnforge.config import AgentConfig
    from pinnforge.runtime import registry

    default_runtime = AgentConfig().runtime
    model = registry.get_runtime(default_runtime).default_model
    assert model.lower() not in {"opus", "sonnet", "haiku", "auto", "default", "latest"}
    assert model == "claude-opus-4-8", "blocks default to the model the archive was measured on"


def test_every_runtime_states_some_default():
    """`auto` is what one CLI calls "unpinned" — legitimate for that harness,
    and the reason the README tells the user to pass `--model` anyway. What is
    not acceptable is a run whose model nobody recorded: an empty id reaches the
    ledger as an empty string and the run cannot be read against any other. A
    harness may therefore state a default *or* refuse to start without one, but
    it may not quietly run unpinned."""
    from pinnforge.runtime import registry

    for name in registry.known_runtimes():
        rt = registry.get_runtime(name)
        if rt.default_model.strip():
            continue
        with pytest.raises(ValueError, match="states no default model"):
            registry.resolve_model(rt, "")


# ───────────────────── interrupting a live agent ─────────────────────


def _spawn_tree(tmp_path, marker: Path):
    """A real agent stand-in: a process that itself spawns a worker.

    Mirrors the shape the orchestrator actually starts — the CLI spawns a
    shell, which spawns `eval.py`, which spawns the training process that
    holds the GPU lock. The grandchild writes `marker` while it lives, so a
    survivor is detectable without inspecting pids.
    """
    import subprocess
    import sys
    import time

    from pinnforge.runtime.base import spawn

    child = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',\"import time,pathlib;"
        f"pathlib.Path(r'{marker}').write_text('alive');time.sleep(60)\"]);"
        "time.sleep(60)"
    )
    handle = spawn(
        [sys.executable, "-c", child],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        log_path=tmp_path / "agent.log",
        block_id="b01",
    )
    for _ in range(200):  # wait for the grandchild to exist
        if marker.is_file():
            break
        time.sleep(0.05)
    assert handle.alive, "the agent stand-in did not start"
    assert marker.is_file(), "the worker under it did not start — the test would be vacuous"
    return handle, subprocess


def test_interrupt_takes_down_the_whole_process_tree(tmp_path):
    """Signalling only the CLI would orphan the run that holds the GPU.

    The agent is started in its own session, so nothing that reaches the
    orchestrator reaches the block. `interrupt()` addresses the group, which
    is the only thing that also gets the training worker underneath it.
    """
    import os

    marker = tmp_path / "worker.txt"
    handle, _ = _spawn_tree(tmp_path, marker)
    pgid = os.getpgid(handle.process.pid)

    handle.interrupt(grace=10.0)

    assert not handle.alive
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)  # the worker underneath is gone too, not orphaned


def test_ctrl_c_during_a_block_does_not_leave_the_agent_running(tmp_path, monkeypatch):
    """The regression this pins: Ctrl-C used to return, leaving a live block.

    `pinnforge run resume` would then dispatch a *second* agent onto the same
    workspace, both appending to one `evals.jsonl` and contending for one GPU
    lock — while the README promises an interrupt and a crash end the same way.
    """
    import os

    from pinnforge.config import RunConfig
    from pinnforge.orchestrator import Orchestrator
    from pinnforge.types import RunState

    marker = tmp_path / "worker.txt"
    handle, _ = _spawn_tree(tmp_path, marker)
    pgid = os.getpgid(handle.process.pid)

    cfg = RunConfig(task="ldc")
    orch = Orchestrator.__new__(Orchestrator)  # no run directory needed
    orch.cfg = cfg
    orch.run = tmp_path
    orch.state = RunState(task="ldc", run_id="ldc_1")
    monkeypatch.setattr(Orchestrator, "_newest_mtime", lambda self, b: 0.0)

    def interrupt_once(timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(handle, "wait", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        orch._await(handle, "b01")

    assert not handle.alive, "the block must not survive the interrupt"
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
