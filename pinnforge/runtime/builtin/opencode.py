"""OpenCode (`opencode run`) as a block runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, scan_jsonl_for, spawn


class OpenCodeRuntime:
    name = "opencode"

    @property
    def default_model(self) -> str:
        """None. OpenCode is a front-end to many providers, so there is no id
        this adapter could pick that would be right for someone else's setup —
        the previous `openai/gpt-5` was a guess, and a guess that runs is worse
        than one that does not, because the ledger records it as though it were
        chosen. A run on this harness names its model or does not start.
        """
        return ""

    @property
    def default_command(self) -> str:
        return "opencode"

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
        if not (model or "").strip():
            raise ValueError(
                "runtime 'opencode' has no default model; name one with "
                "`--model <provider/id>` (e.g. --model anthropic/claude-opus-4-8) "
                "or `--set agent.model=<provider/id>`"
            )
        opts = runtime_options or {}
        cmd = [command or self.default_command, "run"]
        if resume_session_id:
            cmd += ["--session", resume_session_id]
        cmd += ["--model", model, "--print-logs"]
        cmd += [str(a) for a in opts.get("extra_args") or []]
        cmd += [prompt]
        return spawn(
            cmd,
            cwd=cwd,
            env=clean_env(env),
            log_path=log_path,
            block_id=block_id,
            command_prefix=command_prefix,
        )

    def extract_session_id(self, log_path: Path) -> str | None:
        return scan_jsonl_for(log_path, ("sessionID", "sessionId", "session_id"))

    def extract_usage(self, log_path: Path) -> AgentUsage:
        """Not taught to read this harness's accounting yet — a dash, not a lie."""
        return AgentUsage()
