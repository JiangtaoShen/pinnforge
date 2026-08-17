"""Cursor Agent (`cursor-agent`) as a block runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, scan_jsonl_for, spawn


class CursorAgentRuntime:
    name = "cursor_agent"

    @property
    def default_model(self) -> str:
        return "auto"

    @property
    def default_command(self) -> str:
        return "cursor-agent"

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
        cmd = [
            command or self.default_command,
            "-p",
            prompt,
            "--model",
            model or self.default_model,
            "--force",  # run tools without interactive confirmation
            "--output-format",
            "stream-json",
        ]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        cmd += [str(a) for a in opts.get("extra_args") or []]
        return spawn(
            cmd,
            cwd=cwd,
            env=clean_env(env),
            log_path=log_path,
            block_id=block_id,
            command_prefix=command_prefix,
        )

    def extract_session_id(self, log_path: Path) -> str | None:
        return scan_jsonl_for(log_path, ("chatId", "chat_id", "session_id", "sessionId"))

    def extract_usage(self, log_path: Path) -> AgentUsage:
        """Not taught to read this harness's accounting yet — a dash, not a lie."""
        return AgentUsage()
