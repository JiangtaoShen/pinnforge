"""Agent bindings — which coding-agent CLI drives the blocks, chosen once.

A binding is a name for a runtime plus an exact model id (and, if needed,
the CLI binary and its extra options). `pinnforge setup` creates them after
proving each one answers; runs then say `--agent opus` instead of repeating
`--runtime claude_code --model claude-opus-4-8` and hoping it is spelled the
same way every time. They live per machine, in
`~/.config/pinnforge/agents.yaml`:

    default: opus
    bindings:
      opus:     {runtime: claude_code, model: claude-opus-4-8}
      deepseek: {runtime: opencode, model: deepseek/deepseek-chat}

Which agent a command uses, highest precedence first:

1. `--agent NAME` — a binding;
2. `--runtime` (with `--model`) — spelled out on the command line;
3. `agent:` in the workspace's or the user's settings file — a binding name
   or a `{runtime, model}` mapping;
4. the default binding;
5. the built-in default, `claude_code` with its own default model.

`--model` alone keeps whichever runtime the rest of the chain chose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pinnforge import settings
from pinnforge.config import AgentConfig
from pinnforge.runtime import registry as runtime_registry


def path() -> Path:
    return settings.user_dir() / "agents.yaml"


def load() -> dict[str, Any]:
    p = path()
    data = settings.read_yaml(p) if p.is_file() else None  # a broken file: ValueError naming it
    data = data if isinstance(data, dict) else {}
    data.setdefault("bindings", {})
    data.setdefault("default", None)
    return data


def save(data: dict[str, Any]) -> Path:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# PINNForge agent bindings — written by `pinnforge setup`.\n"
        "# A binding pins a coding-agent CLI to an exact model id.\n"
    )
    p.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
    return p


def names() -> list[str]:
    return sorted(load()["bindings"])


def _agent(d: dict[str, Any]) -> AgentConfig:
    cfg = AgentConfig.from_dict(d)
    cfg.runtime = runtime_registry.canonical(cfg.runtime)
    return cfg


def get(name: str) -> AgentConfig:
    data = load()
    if name not in data["bindings"]:
        known = ", ".join(sorted(data["bindings"])) or "none yet — run `pinnforge setup`"
        raise KeyError(f"no agent binding named {name!r}. Known: {known}")
    return _agent(data["bindings"][name])


def put(name: str, agent: AgentConfig, make_default: bool = False) -> Path:
    data = load()
    entry = {"runtime": agent.runtime, "model": agent.model}
    if agent.command:
        entry["command"] = agent.command
    if agent.runtime_options:
        entry["runtime_options"] = agent.runtime_options
    data["bindings"][name] = entry
    if make_default or not data.get("default"):
        data["default"] = name
    return save(data)


def remove(name: str) -> Path:
    data = load()
    data["bindings"].pop(name, None)
    if data.get("default") == name:
        data["default"] = next(iter(sorted(data["bindings"])), None)
    return save(data)


def default_name() -> str | None:
    return load().get("default") or None


def resolve(
    root: Path | None,
    *,
    agent: str | None = None,
    runtime: str | None = None,
    model: str | None = None,
) -> AgentConfig:
    """The agent a command should use, by the precedence in the module docstring."""
    if agent:
        cfg = get(agent)
    elif runtime:
        cfg = AgentConfig(runtime=runtime_registry.canonical(runtime), model="")
    else:
        chosen = settings.merged(root).get("agent") if root is not None else settings.user().get(
            "agent"
        )
        if isinstance(chosen, str) and chosen:
            cfg = get(chosen)
        elif isinstance(chosen, dict) and chosen.get("runtime"):
            cfg = _agent(chosen)
        elif (name := default_name()) is not None:
            cfg = get(name)
        else:
            cfg = AgentConfig()
    if model:
        cfg.model = model
    return cfg
