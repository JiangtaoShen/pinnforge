"""Claude Code (`claude`) as a block runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, scan_jsonl_for, spawn


class ClaudeCodeRuntime:
    name = "claude_code"

    @property
    def default_model(self) -> str:
        """An exact id, never an alias.

        `opus` is a moving target: it meant `claude-opus-4-8` for a year and
        `claude-opus-5` after, so a run pinned to the alias is not pinned at
        all — two runs a generation apart both record "opus" and stop being
        comparable without anyone noticing. Blocks default to the model the
        previous framework's runs were measured on, so a new run can be read
        against the archive; pass `--model` to use anything else.
        """
        return "claude-opus-4-8"

    @property
    def default_command(self) -> str:
        return "claude"

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
            # A block edits files and runs GPU evals unattended; there is no
            # user to approve tools. Confinement is the sandbox's job.
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if max_turns > 0:
            cmd += ["--max-turns", str(max_turns)]
        for extra in opts.get("add_dirs") or []:
            cmd += ["--add-dir", str(extra)]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        cmd += [str(a) for a in opts.get("extra_args") or []]

        child = clean_env(env)
        # `--dangerously-skip-permissions` is refused as root unless the CLI
        # is told it is contained; under a sandbox prefix that is true.
        if command_prefix:
            child["IS_SANDBOX"] = "1"
        return spawn(
            cmd,
            cwd=cwd,
            env=child,
            log_path=log_path,
            block_id=block_id,
            command_prefix=command_prefix,
        )

    def extract_session_id(self, log_path: Path) -> str | None:
        return scan_jsonl_for(log_path, ("session_id", "sessionId"))

    def extract_usage(self, log_path: Path) -> AgentUsage:
        """Read the accounting out of the stream-json the block already wrote.

        Claude Code closes a headless session with one `type: "result"` object
        carrying the whole session's usage; tool calls are counted from the
        transcript because that number is what a reader compares against the
        eval count to see how much of a block was thinking and how much was
        running things.
        """
        import json

        usage = AgentUsage()
        if not log_path.is_file():
            return usage
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            for block in (obj.get("message", {}) or {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    usage.tool_uses += 1
            if obj.get("type") == "system" and obj.get("model"):
                usage.model = str(obj["model"])
            if obj.get("type") != "result":
                continue
            u = obj.get("usage") or {}
            usage.output_tokens = int(u.get("output_tokens") or 0)
            usage.cache_read = int(u.get("cache_read_input_tokens") or 0)
            usage.tokens = (
                int(u.get("input_tokens") or 0)
                + int(u.get("cache_creation_input_tokens") or 0)
                + usage.output_tokens
            )
            usage.turns = int(obj.get("num_turns") or 0)
            usage.cost_usd = float(obj.get("total_cost_usd") or 0.0)
        return usage
