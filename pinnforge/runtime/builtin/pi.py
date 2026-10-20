"""Pi (`pi --mode json`, earendil-works/pi-coding-agent) as a block runtime."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, spawn

# Where a block's sessions are kept: beside its log in the run's `logs/` —
# one directory for all of them, so a block's next segment finds the session
# it resumes — and never in `~/.pi/agent/sessions`, where any later agent of
# the same user could read them. No other agent is ever resumed, so the
# others keep no session at all (`--no-session`): their log is their whole
# record, and the framework already puts it where its role demands — an
# author's that reached for the answers, or a reference agent's that saw
# them, into the private directory. A session file beside it would have been
# a copy left behind.
SESSIONS_DIR = "pi-sessions"

# A block's tool surface is part of what a block is, so it must not vary with
# whatever the operator has installed or configured: Pi's four built-in tools,
# named outright (`defaultTools` in a settings file could change them), and
# none of the operator's extensions, skills, prompt templates, MCP servers or
# AGENTS.md/CLAUDE.md files — nor a `.pi/` a block wrote into its own run.
TOOLS = "read,bash,edit,write"
FIXED = ["--tools", TOOLS, "--no-extensions", "--no-skills", "--no-prompt-templates",
         "--no-themes", "--no-mcp", "--no-context-files", "--no-approve"]

# Events kept out of a block's log (see `reaper`): token deltas and partial
# tool output, and the copies that `message_start`, `turn_end`, `agent_end`
# (the whole conversation, on one line) and `tool_execution_end` (each output
# twice over) make of what is logged anyway. Every finished message is there
# once, as `message_end` — a tool's output as the reply the model was given —
# and every tool call as `tool_execution_start`; a block's session file in
# `pi-sessions/` keeps everything.
UNLOGGED = ("message_update", "tool_execution_update", "message_start", "turn_end",
            "agent_end", "tool_execution_end")

# What Pi's own print mode exits nonzero on; JSON mode exits 0 regardless.
FAILED = {"error", "aborted", "length"}
# `provider/id:level` asks for a thinking level; any other `:` is the id's own.
THINKING = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}

_SESSION_ID = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]{5,80}")
_UPDATE = '{"type":"message_update"'  # in a log written without the reaper: never read


def _events(log_path: Path) -> Iterator[dict]:
    """The JSON objects in a Pi log, a line at a time, skipping anything else."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{") or line.startswith(_UPDATE):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _replies(log_path: Path) -> Iterator[dict]:
    """Each finished assistant message, once.

    `turn_end` and `agent_end` repeat messages a `message_end` has already
    delivered; only `message_end` is counted.
    """
    for obj in _events(log_path):
        msg = obj.get("message")
        if obj.get("type") == "message_end" and isinstance(msg, dict) \
                and msg.get("role") == "assistant":
            yield msg


def failure(log_path: Path) -> str:
    """Why the session's last reply failed, or "" if it did not."""
    last = None
    for msg in _replies(log_path):
        last = msg
    if last is None or last.get("stopReason") not in FAILED:
        return ""
    return str(last.get("errorMessage") or f"stopped: {last.get('stopReason')}")


@dataclass
class PiHandle(AgentHandle):
    """A Pi agent: stopped with SIGTERM, and judged by its last reply."""

    _judged: int | None = None

    def _judge(self, rc: int | None) -> int | None:
        if rc != 0:
            return rc
        if self._judged is None:
            self._judged = 1 if failure(self.log_path) else 0
        return self._judged

    @property
    def returncode(self) -> int | None:
        return self._judge(None if self.process is None else self.process.poll())

    def wait(self, timeout: float | None = None) -> int | None:
        return self._judge(super().wait(timeout))

    def stop(self, grace: float = 15.0) -> None:
        """As AgentHandle.stop, with the time the reaper's own stop takes (see `reaper`)."""
        super().stop(grace)

    def interrupt(self, grace: float = 15.0) -> None:
        """SIGTERM, not SIGINT.

        Pi keeps nothing extra on SIGINT — its session file is written as it
        goes — and in JSON mode SIGINT ends it without the commands its tools
        are running, each in a session of its own. SIGTERM is its orderly
        stop: those commands are killed, then it exits.
        """
        if self.alive:
            self.stop(grace)


class PiRuntime:
    name = "pi"

    @property
    def default_model(self) -> str:
        """None. Pi fronts many providers, and which ones a user has signed in
        to is theirs to say; a run names its model as `provider/id`."""
        return ""

    @property
    def default_command(self) -> str:
        return "pi"

    def start(
        self,
        *,
        block_id: str,
        cwd: Path,
        prompt: str,
        model: str,
        log_path: Path,
        env: dict[str, str],
        command: str | None = None,
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 0,
        resume_session_id: str | None = None,
        command_prefix: list[str] | None = None,
    ) -> AgentHandle:
        model = (model or "").strip()
        if "/" not in model:
            raise ValueError(
                "runtime 'pi' needs its model as provider/id (e.g. --model "
                "anthropic/claude-opus-4-8): a bare name is matched fuzzily against "
                "every provider Pi knows, so it pins nothing"
            )
        opts = runtime_options or {}
        cmd = [command or self.default_command, "--mode", "json", "--model", model, *FIXED]
        if resume_session_id or env.get("PINNFORGE_ROLE", "block") == "block":
            sessions = log_path.parent / SESSIONS_DIR
            sessions.mkdir(parents=True, exist_ok=True)
            cmd += ["--session-dir", str(sessions)]
        else:
            cmd += ["--no-session"]
        if opts.get("thinking"):
            cmd += ["--thinking", str(opts["thinking"])]
        if resume_session_id:
            cmd += ["--session", resume_session_id]
        cmd += [str(a) for a in opts.get("extra_args") or []]
        cmd += ["--", prompt]  # a prompt that starts with "-" is still the prompt
        # Pi's bash tool puts each command in a session of its own, out of the
        # group the framework stops; the reaper brings them back (Linux), and
        # keeps the log to what is read. It runs outside any sandbox prefix,
        # which wraps Pi alone, and with `-P`: the working directory is the
        # agent's, and must not be where `pinnforge.runtime.reaper` is found.
        drops = [arg for t in UNLOGGED for arg in ("--drop-type", t)]
        cmd = [sys.executable, "-P", "-m", "pinnforge.runtime.reaper", *drops, "--",
               *(command_prefix or []), *cmd]
        child = clean_env(env)
        child.setdefault("PI_SKIP_VERSION_CHECK", "1")  # no request to pi.dev per block
        return spawn(
            cmd,
            cwd=cwd,
            env=child,
            log_path=log_path,
            block_id=block_id,
            # Pi reads stdin to its end when it is not a terminal, before it
            # starts; under a harness that leaves a pipe open, that is forever.
            stdin=subprocess.DEVNULL,
            handle=PiHandle,
        )

    def extract_session_id(self, log_path: Path) -> str | None:
        """The id in the session header Pi writes first, which `--session` resumes."""
        for obj in _events(log_path):
            sid = obj.get("id")
            if obj.get("type") == "session" and isinstance(sid, str) \
                    and _SESSION_ID.fullmatch(sid):
                return sid
        return None

    def extract_usage(self, log_path: Path) -> AgentUsage:
        """Summed over the session's replies; each `message_end` carries its own.

        Pi reports `input` without the cached part, as Claude Code does, so
        `tokens` is input + cache writes + output and `cache_read` stays
        beside it. The model is the one that answered: `responseModel` when
        the provider names a concrete one, under the provider it came from.
        """
        usage = AgentUsage()
        if not log_path.is_file():
            return usage
        for msg in _replies(log_path):
            usage.turns += 1
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "toolCall":
                    usage.tool_uses += 1
            _add(usage, msg.get("usage"))
            model = msg.get("responseModel") or msg.get("model")
            if model:
                provider = msg.get("provider")
                usage.model = f"{provider}/{model}" if provider else str(model)
        for obj in _events(log_path):
            if obj.get("type") == "compaction_end" and isinstance(obj.get("result"), dict):
                _add(usage, obj["result"].get("usage"))  # summarising is model spend too
        return usage

    def extract_error(self, log_path: Path) -> str:
        return failure(log_path)

    def pin_problem(self, model: str, log_path: Path) -> str:
        """What is wrong with `model` as a pin, judged by the session it ran; "" if nothing.

        Pi matches `--model` fuzzily within the provider: `anthropic/claude-opus`
        runs whichever Opus it finds, as an alias would. A reply names the model
        Pi chose, so a pin that is not that id — case and a thinking level
        aside — is a pattern, and the ping says which id to pin instead.
        """
        asked, _, level = model.strip().rpartition(":")
        if not asked or level not in THINKING:
            asked = model.strip()
        for msg in _replies(log_path):
            took = f"{msg.get('provider')}/{msg.get('model')}"
            if took.lower() != asked.lower():
                return (f"Pi took {model} to mean {took}; pin that id exactly "
                        "(`pi --list-models` lists them)")
            return ""
        return ""


def _add(usage: AgentUsage, u) -> None:
    if not isinstance(u, dict):
        return

    def n(key: str) -> int:
        v = u.get(key)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    usage.output_tokens += n("output")
    usage.cache_read += n("cacheRead")
    usage.tokens += n("input") + n("cacheWrite") + n("output")
    cost = u.get("cost")
    total = cost.get("total") if isinstance(cost, dict) else None
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        usage.cost_usd += float(total)

