"""Run configuration — YAML on disk, dataclasses in memory.

One `RunConfig` fully determines a run: which task, which harness, how much
GPU wall each block gets, and what sandbox the block's processes see. It is
frozen into the run directory at `pinnforge run start`, so resuming a run
never silently picks up edited defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# A block's budget when neither the operator nor the task names one. The
# pool is found, not assumed: `auto` resolves to the GPUs actually visible
# (see pinnforge/gpus.py), so a one-card workstation gets a one-card run.
DEFAULT_WALL_BUDGET = 7200.0
DEFAULT_GPUS = "auto"
DEFAULT_STALL_MINUTES = 30
DEFAULT_POLL_SECONDS = 60


@dataclass
class SandboxConfig:
    """Execution environment handed to every block process.

    `gpus` is the pool a block may use (`--gpu 0` / `--gpu 1`); `eval.py`'s
    per-GPU file lock still serialises runs on the same id. `command_prefix`
    wraps the agent CLI — set it to a container/jail launcher to confine the
    whole process tree, since everything the agent spawns inherits it.
    """

    gpus: list[int] = field(default_factory=lambda: _pool(DEFAULT_GPUS))
    env: dict[str, str] = field(default_factory=dict)
    command_prefix: list[str] = field(default_factory=list)
    # Passed to the block as FORGE_WALL_BUDGET; eval.py enforces it. None
    # until a run is created: then it is whatever the operator asked for, or
    # the task's declared budget. A sentinel rather than the default value,
    # because an operator who explicitly asks for 7200 s must get 7200 s.
    wall_budget_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SandboxConfig:
        d = d or {}
        budget = d.get("wall_budget_s")
        return cls(
            gpus=_pool(d.get("gpus", DEFAULT_GPUS)),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            command_prefix=[str(c) for c in (d.get("command_prefix") or [])],
            wall_budget_s=None if budget is None else float(budget),
        )


def _pool(spec) -> list[int]:
    from pinnforge import gpus

    return gpus.resolve(spec)


@dataclass
class AgentConfig:
    """Which coding-agent CLI runs the blocks, and how it is pinned.

    `model` is pinned deliberately: without it a harness silently inherits
    whatever the ambient session uses and runs stop being comparable. An
    empty string means "use the runtime's documented default", which the
    registry resolves and the ledger records.
    """

    runtime: str = "claude_code"
    model: str = ""
    command: str = ""  # override the CLI binary path
    runtime_options: dict[str, Any] = field(default_factory=dict)
    # 0 = no cap; the orchestrator restarts on exit and resumes the session
    max_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AgentConfig:
        d = d or {}
        return cls(
            runtime=str(d.get("runtime", "claude_code")),
            model=str(d.get("model", "")),
            command=str(d.get("command", "")),
            runtime_options=dict(d.get("runtime_options") or {}),
            max_turns=int(d.get("max_turns", 0)),
        )


@dataclass
class RunConfig:
    """Everything one run needs. Serialised to `runs/<id>/run.yaml`."""

    task: str
    blocks: int = 1
    agent: AgentConfig = field(default_factory=AgentConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    # A block that writes nothing for this long is treated as stalled and
    # nudged; it is not killed, because a long GPU eval is legitimately quiet.
    stall_minutes: int = DEFAULT_STALL_MINUTES
    poll_seconds: int = DEFAULT_POLL_SECONDS
    # The hardware class the pool resolved to when the run was created — what
    # its anchor and its wall-clock-bounded evaluations were measured on.
    hw: dict[str, Any] = field(default_factory=dict)
    # Once the run is over, its agent writes the discovery film's script
    # (`pinnforge/discovery.py`); `film: false` leaves that to `pinnforge run film`.
    film: bool = True

    def to_dict(self) -> dict[str, Any]:
        out = {
            "task": self.task,
            "blocks": self.blocks,
            "agent": self.agent.to_dict(),
            "sandbox": self.sandbox.to_dict(),
            "stall_minutes": self.stall_minutes,
            "poll_seconds": self.poll_seconds,
        }
        if self.hw:
            out["hw"] = self.hw
        if not self.film:
            out["film"] = False
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        return cls(
            task=str(d["task"]),
            blocks=int(d.get("blocks", 1)),
            agent=AgentConfig.from_dict(d.get("agent")),
            sandbox=SandboxConfig.from_dict(d.get("sandbox")),
            stall_minutes=int(d.get("stall_minutes", DEFAULT_STALL_MINUTES)),
            poll_seconds=int(d.get("poll_seconds", DEFAULT_POLL_SECONDS)),
            hw=dict(d.get("hw") or {}),
            film=d.get("film", True) is not False,
        )

    def save(self, path: Path) -> None:
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def apply_overrides(d: dict[str, Any], dotlist: list[str]) -> dict[str, Any]:
    """Apply ``a.b=c`` overrides onto a nested dict, in place.

    Values are parsed as YAML so ``sandbox.gpus=[0]`` and ``blocks=3`` arrive
    as a list and an int rather than strings.
    """
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, _, raw = item.partition("=")
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {key!r}: {p!r} is a scalar")
        node[parts[-1]] = yaml.safe_load(raw)
    return d
