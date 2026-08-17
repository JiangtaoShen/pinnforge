"""Agent runtime protocol — one coding-agent CLI, driven headlessly.

A *runtime* is an adapter around an external CLI (`claude`, `codex`,
`cursor-agent`, `opencode`, …). The orchestrator never talks to a model; it
starts one of these as a subprocess, waits, and reads the filesystem to
decide what happened.

Every adapter must deliver the **same block**: identical charter, identical
prompt, identical working directory, identical environment. Only the argv
that puts the CLI into headless mode differs. Anything else that varies
between harnesses would make runs incomparable, which is the one thing this
framework cannot afford.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class AgentHandle:
    """A live (or finished) agent subprocess."""

    block_id: str
    process: subprocess.Popen | None
    cwd: Path
    log_path: Path
    session_id: str | None = None
    _log_file: Any = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def wait(self, timeout: float | None = None) -> int | None:
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self) -> None:
        """Release the log file and any pipes. Safe to call more than once."""
        if self.process:
            for pipe in (self.process.stdout, self.process.stderr):
                if pipe:
                    with suppress(OSError, ValueError):
                        pipe.close()
        if self._log_file is not None:
            with suppress(OSError, ValueError):
                self._log_file.close()
            self._log_file = None

    def _pgid(self) -> int | None:
        """The group id, captured before the child is reaped — after that it is gone."""
        if self.process is None:
            return None
        try:
            return os.getpgid(self.process.pid)
        except (ProcessLookupError, PermissionError):
            return None

    @staticmethod
    def _signal_group(pgid: int | None, sig: int, fallback) -> None:
        """Signal the whole group, falling back to the CLI alone if that fails."""
        if pgid is None:
            fallback()
            return
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            fallback()

    def _group_gone(self, pgid: int | None, timeout: float) -> bool:
        """Has every process in the group exited — not just the CLI?

        The CLI is the one process there is a handle to wait on, and it is not
        the one that matters: underneath it sit a shell, `eval.py`, and the
        training worker holding the GPU lock. They do not exit together, so
        reaping the CLI proves nothing about the worker — and a worker that
        outlives its block is the orphan this signalling exists to prevent.
        """
        if pgid is None:
            return True
        deadline = time.time() + timeout
        while True:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.05)

    def stop(self, grace: float = 10.0) -> None:
        """Terminate the whole process group, and confirm it is actually gone.

        Agents spawn shells, which spawn `eval.py`, which spawns a GPU worker.
        Signalling only the CLI would orphan a training run holding a GPU
        lock, so the group is signalled — and then waited on as a group.
        """
        pgid = self._pgid()
        if self.alive and self.process is not None:
            self._signal_group(pgid, signal.SIGTERM, self.process.terminate)
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                logger.warning("block %s ignored SIGTERM; killing group", self.block_id)
                self._signal_group(pgid, signal.SIGKILL, self.process.kill)
                self.process.wait(timeout=5)
        if not self._group_gone(pgid, grace):
            logger.warning(
                "block %s: the CLI exited but its process group did not; killing it",
                self.block_id,
            )
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)  # type: ignore[arg-type]
            if not self._group_gone(pgid, 5.0):
                logger.error(
                    "block %s: process group %s survived SIGKILL — a run may still "
                    "hold a GPU lock",
                    self.block_id,
                    pgid,
                )
        self.close()

    def interrupt(self, grace: float = 15.0) -> None:
        """SIGINT the group, the way Ctrl+C would.

        Harnesses that checkpoint on interrupt get the chance to save a
        resumable session; `stop()` is the fallback when they do not — and
        also when the CLI goes down but leaves its worker running, which is
        the common case rather than the exotic one, since the worker is
        several processes deeper and unwinding a training step takes longer
        than exiting a CLI.
        """
        if not self.alive or self.process is None:
            return
        pgid = self._pgid()
        self._signal_group(pgid, signal.SIGINT, lambda: self.process.send_signal(signal.SIGINT))
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.stop()
            return
        if not self._group_gone(pgid, grace):
            self.stop()
            return
        self.close()


@dataclass
class AgentUsage:
    """What a block cost, as its harness reports it.

    `tokens` counts what the model actually had to process — input plus
    cache-creation plus output — and leaves `cache_read` beside it rather than
    folded in, because a cached read is two orders of magnitude cheaper and
    burying it makes a long block look like a runaway one.

    `model` is the id the harness actually ran, not the alias it was asked
    for. `--model opus` resolved to `claude-opus-4-8` for a year and to
    `claude-opus-5` after; a ledger that stores the alias cannot tell two runs
    apart across that boundary, and a comparison between them silently stops
    being a comparison of anything else.

    Every field defaults to zero: a harness that reports nothing is not an
    error, it is a dash in the summary.
    """

    tokens: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    turns: int = 0
    tool_uses: int = 0
    cost_usd: float = 0.0
    model: str = ""  # what the alias actually resolved to, e.g. claude-opus-5

    def __bool__(self) -> bool:
        return bool(self.tokens or self.turns or self.tool_uses or self.cost_usd or self.model)


@runtime_checkable
class AgentRuntime(Protocol):
    """What every harness adapter implements."""

    name: str

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
    ) -> AgentHandle: ...

    def extract_session_id(self, log_path: Path) -> str | None:
        """Session token this harness can be resumed with, or None."""
        ...

    def extract_usage(self, log_path: Path) -> AgentUsage:
        """What the block cost this harness, read back from its own log.

        Every adapter may return an empty `AgentUsage`; the summary prints a
        dash for whatever a harness does not report, rather than making one
        harness's accounting a requirement for the others.
        """
        ...

    @property
    def default_model(self) -> str: ...

    @property
    def default_command(self) -> str: ...


# ─────────────────────────── shared helpers ───────────────────────────


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A child environment free of the parent harness's own markers.

    A block must not inherit variables that make a nested agent think it is
    resuming its parent's session, nor a `VIRTUAL_ENV` that shadows the
    project venv the charter tells it to use.
    """
    env = dict(os.environ)
    for key in (
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDECODE",
        "CLAUDE_SESSION_ID",
        "ANTHROPIC_MODEL",
        "CODEX_SESSION_ID",
        "VIRTUAL_ENV",
        "PYTHONHOME",
    ):
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def spawn(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    block_id: str,
    command_prefix: list[str] | None = None,
) -> AgentHandle:
    """Start the CLI in its own process group with stdout teed to `log_path`."""
    if command_prefix:
        cmd = [*command_prefix, *cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    logger.info("block %s: %s", block_id, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group; see AgentHandle.stop()
        text=True,
    )
    return AgentHandle(
        block_id=block_id, process=proc, cwd=cwd, log_path=log_path, _log_file=log_file
    )


def scan_jsonl_for(log_path: Path, keys: tuple[str, ...]) -> str | None:
    """First value found under any of `keys` in a JSONL log.

    Harnesses emit their session id under different names and at different
    depths; this covers the common shapes without each adapter reimplementing
    a parser.
    """
    import json

    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in keys:
            val = obj.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict):
                for sub in ("id", "session_id"):
                    if isinstance(val.get(sub), str) and val[sub]:
                        return val[sub]
    return None
