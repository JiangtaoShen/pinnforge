"""A scripted stand-in for a coding agent — the zero-token demo, and the e2e test.

`pinnforge demo` runs the whole machine — workspace, score service, anchor,
blocks, integrity checks, summaries, the dashboard — with this in the
driver's seat instead of a model. It is not clever: for the packaged demo
task it writes candidates from a fixed playbook (Fourier features, hard
boundary conditions, a learning-rate decay), evaluates them through
`task/eval.py` exactly as a block would, keeps going until the block's GPU
budget is spent, and writes the block summary. It speaks Claude Code's
stream-JSON on stdout, so the dashboard's activity feed reads it like any
other agent.

It needs no network, no key and no model, so it is also how the framework
is tested end to end.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pinnforge import paths
from pinnforge.runtime.base import AgentHandle, AgentUsage, clean_env, spawn
from pinnforge.runtime.builtin.claude_code import ClaudeCodeRuntime


class ReplayRuntime:
    name = "replay"

    @property
    def default_model(self) -> str:
        return "replay-1"

    @property
    def default_command(self) -> str:
        return sys.executable

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
        cmd = [command or self.default_command, "-m", "pinnforge.runtime.builtin.replay",
               "--block", block_id]
        return spawn(cmd, cwd=cwd, env=clean_env(env), log_path=log_path, block_id=block_id,
                     command_prefix=command_prefix)

    def extract_session_id(self, log_path: Path) -> str | None:
        return None

    def extract_usage(self, log_path: Path) -> AgentUsage:
        return ClaudeCodeRuntime().extract_usage(log_path)


# ─────────────────────────────── the playbook ───────────────────────────────

NET = '''class PINN(nn.Module):
    """{doc}"""
    hidden_width: int = {width}
    n_hidden_layers: int = 4

    @nn.compact
    def __call__(self, xy):
        {embed}
        for _ in range(self.n_hidden_layers):
            h = nn.tanh(nn.Dense(self.hidden_width)(h))
        out = nn.Dense(OUTPUT_DIM)(h)
        {hard}return out

'''

PLAIN = "h = xy"
FOURIER = ('B = self.param("B", lambda k, s: random.normal(k, s) * {sigma}, (INPUT_DIM, 32))\n'
           "        z = 2.0 * PI * xy @ jax.lax.stop_gradient(B)\n"
           "        h = jnp.concatenate([xy, jnp.sin(z), jnp.cos(z)], axis=-1)")
HARD = ("x, y = xy[:, 0:1], xy[:, 1:2]\n"
        "        out = out * x * (1.0 - x) * y * (1.0 - y) * 16.0  # u = 0 on the walls, exactly\n"
        "        ")

BEST = {"hard": True, "decay": 10000}
PLAYBOOK = {
    "b01": [
        ("hard", "exact boundary conditions (u = x(1-x)y(1-y)·N)", {"hard": True}),
        ("decay", "learning rate halved every 10k steps", {"decay": 10000}),
        ("hard_decay", "both", BEST),
        ("ff1_hard_decay", "+ Fourier features, sigma 1", {**BEST, "sigma": 1.0}),
    ],
    "b02": [
        ("hard_decay_w96", "best recipe, width 96", {**BEST, "width": 96}),
        ("hard_decay_slow", "decay every 15k steps", {"hard": True, "decay": 15000}),
        ("hard_decay_bs1k", "interior batch 1024", {**BEST, "batch": 1024}),
        ("ff1_hard_decay_w96", "Fourier sigma 1, width 96", {**BEST, "sigma": 1.0, "width": 96}),
    ],
}


def candidate(base: str, sigma: float | None = None, width: int = 64, hard: bool = False,
              decay: int = 0, batch: int = 0) -> str:
    """The baseline with the playbook's changes — the frozen header untouched."""
    start = base.index("class PINN(nn.Module):")
    end = base.index("# ══════════ Data")
    parts = [f"Fourier features (sigma {sigma}) + " if sigma else "", "tanh MLP",
             ", exact BC" if hard else ""]
    net = NET.format(doc="".join(parts), width=width,
                     embed=FOURIER.format(sigma=sigma) if sigma else PLAIN,
                     hard=HARD if hard else "")
    src = base[:start] + net + base[end:]
    if hard:
        src = src.replace("WEIGHT_BC = 1.0", "WEIGHT_BC = 0.0")
    if decay:
        src = src.replace(
            "optimizer = optax.adam(learning_rate=LR)",
            f"optimizer = optax.adam(learning_rate=optax.exponential_decay(LR, {decay}, 0.5))",
        )
    if batch:
        src = src.replace("BS_INT    = 512", f"BS_INT    = {batch}")
    return src


# ─────────────────────────────── the "agent" ───────────────────────────────


def _say(kind: str, **payload) -> None:
    print(json.dumps({"type": kind, **payload}), flush=True)


def _text(text: str) -> None:
    _say("assistant", message={"content": [{"type": "text", "text": text}]})


def _tool(name: str, **inp) -> None:
    _say("assistant", message={"content": [{"type": "tool_use", "name": name, "input": inp}]})


def _spent(block_dir: Path) -> float:
    try:
        return float((block_dir / ".budget").read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0.0


def _train_time(run: Path) -> float:
    m = re.search(r'TRAIN_TIME_LITERAL\s*=\s*"TRAIN_TIME = ([\d.]+)"',
                  (run / "task" / "eval.py").read_text(encoding="utf-8"))
    return float(m.group(1)) if m else 60.0


def agent(block: str) -> int:
    run = Path.cwd()
    started = time.time()
    _say("system", subtype="init", model="replay-1", session_id=None)
    import yaml

    cfg = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8")) or {}
    pool = [str(g) for g in (cfg.get("sandbox") or {}).get("gpus") or []] or ["0"]
    budget = float(os.environ.get("FORGE_WALL_BUDGET") or 600)
    per_run = _train_time(run)
    base = (run / "task" / "baseline.py").read_text(encoding="utf-8")
    block_dir = run / "blocks" / block
    block_dir.mkdir(parents=True, exist_ok=True)

    _tool("Read", file_path=str(run / "block.md"))
    _text(f"Block {block}: the baseline enforces u = 0 with a soft penalty that competes "
          "with a residual dominated by the large source term, and trains at a fixed learning "
          "rate. Plan: exact boundary conditions, a learning-rate decay, then capacity.")
    plan = PLAYBOOK.get(block) or [
        (f"hard_decay_w{64 + 16 * i}", f"best recipe, width {64 + 16 * i}",
         {**BEST, "width": 64 + 16 * i})
        for i in range(1, 9)
    ]
    results = []
    python = str(paths.venv_bin(run / ".venv"))
    queue = list(plan)
    turns = 0
    # at least one batch: a block is complete only with a real evaluation, and
    # eval.py itself refuses only once the budget is spent outright
    while queue and (not results or _spent(block_dir) < budget - per_run):
        batch = [queue.pop(0) for _ in range(min(len(pool), len(queue)))]
        procs = []
        for (slug, what, spec), gpu in zip(batch, pool, strict=False):
            path = block_dir / f"{block}_{slug}.py"
            path.write_text(candidate(base, **spec), encoding="utf-8", newline="\n")
            _tool("Write", file_path=path.relative_to(run).as_posix())
            cmd = [python, "task/eval.py", path.relative_to(run).as_posix(), "--gpu", gpu]
            _tool("Bash", command=" ".join(cmd[1:]))
            procs.append((slug, what, subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE, text=True,
                                                       encoding="utf-8", errors="replace")))
        for slug, what, proc in procs:
            out, _ = proc.communicate()
            lines = [ln for ln in out.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else {}
            turns += 1
            score = rec.get("rRMSE")
            results.append((slug, what, score, rec.get("step_count")))
            shown = f"{score:.4g}" if isinstance(score, float) else f"failed: {rec.get('error')}"
            _text(f"{block}_{slug} ({what}): rRMSE {shown}")
    scored = [r for r in results if isinstance(r[2], float)]
    best = min(scored, key=lambda r: r[2]) if scored else None
    kb2 = run / "blocks" / "kb2" / f"{block}.md"
    lines = [f"# {block} — scripted replay of the demo playbook", "",
             "## Thrust & rationale", "",
             "The source's fine sin(8πx)sin(8πy) term makes the residual loss large and stiff, "
             "so the soft boundary penalty loses the tug-of-war and a fixed-rate Adam stalls. "
             "Exact boundary conditions remove the penalty; a decaying learning rate lets the "
             "optimiser settle; extra width and low-frequency features add capacity.", "",
             "## Evals", "", "| # | file | change | rRMSE | steps |", "|---|---|---|---|---|"]
    for i, (slug, what, score, steps) in enumerate(results, 1):
        s = f"{score:.4g}" if isinstance(score, float) else "failed"
        lines.append(f"| {i} | {block}_{slug}.py | {what} | {s} | {steps or '—'} |")
    lines += ["", "## Findings", ""]
    if best:
        lines.append(f"Best: **{block}_{best[0]}.py, rRMSE {best[2]:.4g}** ({best[1]}).")
    kb2.parent.mkdir(parents=True, exist_ok=True)
    kb2.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _tool("Write", file_path=str(kb2.relative_to(run)))
    _text(f"best: {best[2]:.4g} blocks/{block}/{block}_{best[0]}.py" if best else "no score")
    _say("result", num_turns=turns, total_cost_usd=0.0, duration_ms=int(1000 * (time.time()
                                                                              - started)),
         usage={"input_tokens": 0, "output_tokens": 0})
    return 0


def film() -> int:
    """A discovery-film session (`PINNFORGE_ROLE=film`): the script is the draft, unedited.

    A scripted agent has no judgment to add to what the summaries say, so the
    demo's film is told in the words the draft extracted from them.
    """
    here = Path.cwd()
    started = time.time()
    _say("system", subtype="init", model="replay-1", session_id=None)
    _tool("Read", file_path="draft.json")
    text = (here / "draft.json").read_text(encoding="utf-8")
    (here / "script.json").write_text(text, encoding="utf-8", newline="\n")
    _tool("Write", file_path="script.json")
    _text("script.json: the draft as it stands — a scripted agent has nothing to add")
    _say("result", num_turns=2, total_cost_usd=0.0, duration_ms=int(1000 * (time.time() - started)),
         usage={"input_tokens": 0, "output_tokens": 0})
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    block = ap.parse_args().block
    raise SystemExit(film() if os.environ.get("PINNFORGE_ROLE") == "film" else agent(block))
