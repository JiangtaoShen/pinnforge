"""Usage ledger and run summary.

`run_usage.jsonl` is append-only: one line per block dispatch segment, so a
block that paused and resumed contributes several lines and its totals are
the sum. `run_summary.md` is regenerated from the ledger joined with every
`evals.jsonl`, so it always reflects every block recorded so far — including
b00, which has no ledger line because it is measured, not dispatched.
"""

from __future__ import annotations

import json
from pathlib import Path

from pinnforge import paths
from pinnforge.runtime.base import AgentUsage
from pinnforge.types import EvalRecord, best_score, charged_seconds, utcnow


def append_usage(
    run: Path,
    block_id: str,
    runtime: str,
    model: str,
    duration_ms: int,
    usage: AgentUsage | None = None,
) -> None:
    """One line per dispatch — what it cost as well as how long it took.

    A block's wall time says nothing about how much model went into it. The
    previous framework carried tokens and tool calls in this ledger and its
    summary; the numbers are in every harness's own log, so there is no reason
    for the ledger to be poorer.
    """
    u = usage or AgentUsage()
    line = json.dumps(
        {
            "block": block_id,
            "runtime": runtime,
            "model": model,
            "model_resolved": u.model or model,
            "duration_ms": int(duration_ms),
            "tokens": u.tokens,
            "cache_read": u.cache_read,
            "output_tokens": u.output_tokens,
            "turns": u.turns,
            "tool_uses": u.tool_uses,
            "cost_usd": round(u.cost_usd, 4),
            "completed_at": utcnow(),
        }
    )
    path = paths.ledger_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_usage(run: Path) -> dict[str, dict]:
    """Per-block totals. Segments are summed; the last model/runtime wins."""
    out: dict[str, dict] = {}
    path = paths.ledger_path(run)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cur = out.setdefault(
            r["block"],
            {"duration_ms": 0, "segments": 0, "model": "", "runtime": "",
             "tokens": 0, "tool_uses": 0, "cost_usd": 0.0},
        )
        cur["duration_ms"] += int(r.get("duration_ms") or 0)
        cur["segments"] += 1
        for k in ("tokens", "tool_uses"):
            cur[k] += int(r.get(k) or 0)
        cur["cost_usd"] += float(r.get("cost_usd") or 0.0)
        # The resolved id is what a later reader needs; the alias is what was
        # typed, and two runs a model generation apart can share it.
        cur["model"] = r.get("model_resolved") or r.get("model") or cur["model"]
        cur["runtime"] = r.get("runtime") or cur["runtime"]
    return out


def _run_line(run: Path, rows: list[dict]) -> str:
    """Run-level constants belong here, not repeated down a column.

    `runtime` is the same on every row of a run — a column of it carries no
    information at all. The GPU pool and per-block budget are the two numbers
    a reader needs to know what the wall seconds mean.
    """
    # b00 carries a placeholder, not a harness: it is measured, not dispatched.
    harnesses = sorted({r["runtime"] for r in rows if r.get("runtime") not in ("", "—", None)})
    parts = [f"harness `{'`, `'.join(harnesses)}`"] if harnesses else []
    try:
        from pinnforge.config import RunConfig

        cfg = RunConfig.load(paths.config_path(run))
        parts.append(f"gpus {cfg.sandbox.gpus}")
        parts.append(f"budget {cfg.sandbox.wall_budget_s:.0f} s/block")
    except (OSError, ValueError, KeyError):
        pass
    return " · ".join(parts)


def _num(n: int) -> str:
    """A dash, not a zero: a harness that reports nothing has not spent nothing."""
    return f"{n:,}" if n else "—"


def _fmt_dur(ms: int) -> str:
    if not ms:
        return "—"
    s = round(ms / 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def block_rows(run: Path) -> list[dict]:
    usage = read_usage(run)
    rows = []
    for ev in sorted(paths.blocks_dir(run).glob("b*/evals.jsonl")):
        block = ev.parent.name
        recs = EvalRecord.read_all(ev)
        scored = [r for r in recs if r.scored]
        u = usage.get(block, {})
        rows.append(
            {
                "block": block,
                "runtime": u.get("runtime", "—"),
                "model": u.get("model", "—"),
                "duration_ms": u.get("duration_ms", 0),
                "segments": u.get("segments", 0),
                "tokens": u.get("tokens", 0),
                "tool_uses": u.get("tool_uses", 0),
                "cost_usd": u.get("cost_usd", 0.0),
                "wall_s": charged_seconds(recs, ev.parent / ".budget"),
                "evals": len(scored),
                "diags": sum(1 for r in recs if r.diag and not r.smoke),
                "best": best_score(recs),
            }
        )
    return rows


def write_run_summary(run: Path) -> Path:
    rows = block_rows(run)
    tot_ms = sum(r["duration_ms"] for r in rows)
    tot_wall = sum(r["wall_s"] for r in rows)
    tot_tok = sum(r["tokens"] for r in rows)
    tot_tools = sum(r["tool_uses"] for r in rows)
    lines = [
        "# PINNForge — run summary",
        "",
        f"_Generated {utcnow()} · {len(rows)} block(s) · run `{run.name}`_",
        "",
        _run_line(run, rows),
        "",
        (
            "| Block | Model | Duration | Tokens | Tools | "
            "Wall s | Evals | Diags | Best rRMSE |"
        ),
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        best = format(r["best"], ".5g") if r["best"] is not None else "—"
        lines.append(
            f"| {r['block']} | {r['model'] or '—'} | "
            f"{_fmt_dur(r['duration_ms'])} | {_num(r['tokens'])} | "
            f"{_num(r['tool_uses'])} | "
            f"{r['wall_s']:.0f} | {r['evals']} | {r['diags']} | {best} |"
        )
    lines += [
        (
            f"| **total** | | **{_fmt_dur(tot_ms)}** | **{_num(tot_tok)}** | "
            f"**{_num(tot_tools)}** | **{tot_wall:.0f}** | | | |"
        ),
        "",
    ]
    out = paths.run_summary_path(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def overall_best(run: Path) -> tuple[float | None, str | None]:
    best: tuple[float | None, str | None] = (None, None)
    for r in block_rows(run):
        if r["best"] is not None and (best[0] is None or r["best"] < best[0]):
            best = (r["best"], r["block"])
    return best
