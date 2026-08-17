"""Core types shared across the orchestrator, runtimes and CLI.

Everything here is plain data. The orchestrator is a deterministic process,
so its state has to survive being written to disk and read back after an
interruption — no live objects, no LLM in the loop.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ──────────────────────────── evaluations ────────────────────────────


@dataclass(frozen=True)
class EvalRecord:
    """One line of ``blocks/bNN/evals.jsonl``, written by ``task/eval.py`` alone.

    The task contract fixes the field names; ``rRMSE`` in particular is the
    primary metric every summary keys on. Unknown fields are kept in ``raw``
    so a task may log extras without the orchestrator dropping them.
    """

    block: str
    candidate: str
    smoke: bool
    diag: bool
    wall_s: float
    rRMSE: float | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def scored(self) -> bool:
        """A real scoring run: not a free CPU smoke, not a diagnostic probe."""
        return not self.smoke and not self.diag

    @classmethod
    def parse(cls, line: str) -> EvalRecord | None:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(r, dict):
            return None
        return cls(
            block=r.get("block", ""),
            candidate=r.get("candidate", ""),
            smoke=bool(r.get("smoke", False)),
            diag=bool(r.get("diag", False)),
            # a crashed run still logs its wall time; fall back to train_s
            wall_s=float(r.get("wall_s") or r.get("train_s") or 0.0),
            rRMSE=r.get("rRMSE"),
            raw=r,
        )

    @classmethod
    def read_all(cls, path: Path) -> list[EvalRecord]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and (rec := cls.parse(line)) is not None:
                out.append(rec)
        return out


def best_score(records: list[EvalRecord]) -> float | None:
    """Lowest finite rRMSE across scoring runs (lower is better)."""
    finite = [r.rRMSE for r in records if r.scored and r.rRMSE is not None]
    return min(finite) if finite else None


def spent_seconds(records: list[EvalRecord]) -> float:
    """GPU wall charged to the block — crashes and diagnostics included.

    Mirrors ``eval.py``'s own accounting: every non-smoke invocation is
    charged its actual wall time whether or not it produced a score.
    """
    return sum(r.wall_s for r in records if not r.smoke)


def charged_seconds(records: list[EvalRecord], budget_file: Path | None) -> float:
    """What the block actually spent — the authoritative figure.

    ``.budget`` is ``eval.py``'s own counter, written under a lock as a run is
    charged, so it survives a kill that takes the record with it. Summing
    records alone under-reports by exactly the runs that were interrupted —
    which is the case anyone reading the number most needs to be right about.

    Take the larger of the two: the counter can lag a record that has just
    landed, and a run that died before the counter synced leaves the records
    ahead instead.
    """
    spent = spent_seconds(records)
    if budget_file is not None and budget_file.is_file():
        with suppress(ValueError):
            spent = max(spent, float(budget_file.read_text(encoding="utf-8").strip() or 0))
    return spent


# ──────────────────────────── block state ────────────────────────────


@dataclass
class BlockState:
    """Orchestrator's view of one block. Persisted in the run's ``state.json``."""

    block_id: str
    status: str = "pending"  # pending | running | done | failed
    runtime: str = ""
    model: str = ""  # the alias asked for, e.g. "opus"
    model_resolved: str = ""  # what it became, e.g. "claude-opus-5"
    session_id: str | None = None
    attempts: int = 0
    # usage is summed across every dispatch segment (a resumed block reports
    # per-segment numbers, exactly like the notifications it replaces)
    duration_ms: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    exit_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlockState:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ──────────────────────────── run state ────────────────────────────


@dataclass
class RunState:
    """Checkpoint for one run directory. Written after every state change.

    This is what makes an interrupted run resumable without an LLM: the
    orchestrator reconstructs exactly where it stopped from this file plus
    the on-disk evidence (``evals.jsonl``, ``.budget``, ``kb2/bNN.md``).
    """

    task: str
    run_id: str
    status: str = "created"  # created | running | stopped | finished
    created_at: str = ""
    updated_at: str = ""
    blocks: dict[str, BlockState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunState:
        st = cls(
            task=d.get("task", ""),
            run_id=d.get("run_id", ""),
            status=d.get("status", "created"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
        st.blocks = {k: BlockState.from_dict(v) for k, v in (d.get("blocks") or {}).items()}
        return st

    def save(self, path: Path) -> None:
        self.updated_at = utcnow()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)  # atomic — a kill mid-write must not lose the run

    @classmethod
    def load(cls, path: Path) -> RunState:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
