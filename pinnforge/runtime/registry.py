"""Runtime registry — config string to adapter.

`agent.runtime` also accepts a `module.path:ClassName` entrypoint, so a
harness this repo has never heard of can drive blocks without patching the
framework. The protocol check happens at resolve time, not mid-run.
"""

from __future__ import annotations

import importlib
import shutil

from pinnforge.runtime.base import AgentRuntime
from pinnforge.runtime.builtin.claude_code import ClaudeCodeRuntime
from pinnforge.runtime.builtin.codex import CodexRuntime
from pinnforge.runtime.builtin.cursor_agent import CursorAgentRuntime
from pinnforge.runtime.builtin.opencode import OpenCodeRuntime

_RUNTIMES: dict[str, type] = {
    "claude_code": ClaudeCodeRuntime,
    "codex": CodexRuntime,
    "cursor_agent": CursorAgentRuntime,
    "opencode": OpenCodeRuntime,
}

_ALIASES: dict[str, str] = {
    "claude": "claude_code",
    "claude-code": "claude_code",
    "openai": "codex",
    "openai-codex": "codex",
    "cursor": "cursor_agent",
    "cursor-agent": "cursor_agent",
    "open-code": "opencode",
}


def _is_entrypoint(name: str) -> bool:
    return ":" in name


def _load_entrypoint(spec: str) -> AgentRuntime:
    mod_path, _, cls_name = spec.partition(":")
    if not mod_path or not cls_name:
        raise ValueError(f"custom runtime must be 'module.path:ClassName', got {spec!r}")
    try:
        module = importlib.import_module(mod_path)
    except ImportError as e:
        raise ValueError(f"cannot import custom runtime module {mod_path!r}: {e}") from e
    try:
        cls = getattr(module, cls_name)
    except AttributeError as e:
        raise ValueError(f"{mod_path!r} has no {cls_name!r}") from e
    try:
        inst = cls()
    except Exception as e:
        raise ValueError(f"custom runtime {spec} needs a no-argument constructor: {e}") from e
    if not isinstance(inst, AgentRuntime):
        raise ValueError(
            f"custom runtime {spec} does not satisfy the AgentRuntime protocol "
            "(see pinnforge/runtime/base.py)"
        )
    return inst


def canonical(name: str) -> str:
    key = (name or "").strip()
    return key if _is_entrypoint(key) else _ALIASES.get(key, key)


def get_runtime(name: str) -> AgentRuntime:
    key = canonical(name)
    if _is_entrypoint(key):
        return _load_entrypoint(key)
    if key not in _RUNTIMES:
        known = ", ".join(sorted(_RUNTIMES))
        raise KeyError(
            f"unknown runtime {name!r}. Available: {known}. "
            "For a custom harness set agent.runtime = 'module.path:ClassName'."
        )
    return _RUNTIMES[key]()


def resolve_model(runtime: AgentRuntime, requested: str = "") -> str:
    """The id a run will actually pin, or an error naming what to pass.

    A harness may decline to name a default — one that fronts many providers
    cannot pick one that is right for someone else's account. That is not the
    same as running unpinned: an empty id reaches the ledger as an empty
    string, and a run whose model nobody recorded cannot be read against any
    other. So the choice is deferred to the caller, and refused if the caller
    does not make it. Raised before a run directory or an anchor is built, so
    the correction costs nothing.
    """
    model = (requested or runtime.default_model or "").strip()
    if not model:
        raise ValueError(
            f"runtime {runtime.name!r} states no default model, so a run has to name "
            f"one: pinnforge run start --model <id>, or --set agent.model=<id>"
        )
    return model


def known_runtimes() -> list[str]:
    return sorted(_RUNTIMES)


def detect_available() -> list[dict[str, object]]:
    """Which harness CLIs are actually on PATH — backs `pinnforge agents doctor`."""
    out = []
    for name in known_runtimes():
        rt = get_runtime(name)
        binary = rt.default_command
        out.append(
            {
                "runtime": name,
                "command": binary,
                "path": shutil.which(binary),
                "installed": shutil.which(binary) is not None,
                "default_model": rt.default_model,
            }
        )
    return out
