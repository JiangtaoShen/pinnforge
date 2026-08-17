"""OpenCode (`opencode run`) as a block runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, scan_jsonl_for, spawn


class OpenCodeRuntime:
    name = "opencode"

    @property
    def default_model(self) -> str:
        return "openai/gpt-5"

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
        opts = runtime_options or {}
        cmd = [command or self.default_command, "run"]
        if resume_session_id:
            cmd += ["--session", resume_session_id]
        cmd += ["--model", model or self.default_model, "--print-logs"]
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
