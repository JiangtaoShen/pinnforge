"""Rendering the block charter, and the prompt that points a block at it.

The charter is the block's whole world: resources, direction, rules, budget,
required output. Its text is carried over verbatim from the framework this
replaces, with the task-dependent constants — the GPU-wall budget and the
GPU pool — turned into placeholders. They used to be duplicated by hand
across the charter and the framework docs, and a task swap that missed one
left a charter quietly telling blocks the wrong budget.

The charter no longer names the scoring truth either. It used to forbid
reading a file, which meant the rule went stale whenever a task swap
renamed it; now the reference is not in the run at all, and the sentence
describes an arrangement rather than asking for restraint.

Rendering happens once per run, into `<run>/block.md`. Because the block's
working directory *is* the run root, every path the charter mentions
(`task/`, `kb1/`, `blocks/bNN/`, `.venv/bin/python task/eval.py`) resolves
exactly as it always has — which is what keeps a block's behaviour identical
across harnesses.
"""

from __future__ import annotations

from pathlib import Path

from pinnforge import paths
from pinnforge.config import RunConfig

# The dispatch prompt. Held constant across every runtime — a block that got
# different instructions on a different harness would not be the same block.
BLOCK_PROMPT = (
    "You are PINNForge block {block_id}. Read {root}/block.md and execute the full "
    "block protocol autonomously until done. Your block id is {block_id}; your "
    "workspace is {root}/blocks/{block_id}/. Reminder: run every evaluation in "
    "the foreground (block.md §4)."
)

CRASHED_SUFFIX = (
    " Workspace blocks/{block_id}/ already exists from a crashed run. Inspect its "
    "evals.jsonl, keep what is useful, spend only the remaining budget, then write "
    "(or update) blocks/kb2/{block_id}.md."
)

RESUME_PROMPT = (
    "Resume block {block_id}: {spent:.0f}/{budget:.0f} wall-seconds spent, summary "
    "{summary_state}. Run evals in the foreground only (block.md §4), spend the "
    "remaining budget, then write blocks/kb2/{block_id}.md."
)

SUMMARY_REPAIR_PROMPT = (
    "Block {block_id} finished without writing blocks/kb2/{block_id}.md. Read "
    "block.md §5, read blocks/{block_id}/ (code + evals.jsonl), and write the "
    "summary faithfully. Do not run any evaluations."
)


def render_charter(cfg: RunConfig, root: Path | None = None) -> str:
    template = paths.charter_template(root).read_text(encoding="utf-8")
    gpus = cfg.sandbox.gpus or [0]
    flags = " / ".join(f"`--gpu {g}`" for g in gpus)
    count = {1: "one", 2: "two"}.get(len(gpus), str(len(gpus)))
    if len(gpus) == 2:
        subject = "both GPUs"
    elif len(gpus) > 2:
        subject = f"all {len(gpus)} GPUs"
    else:
        subject = "the GPU"
    if len(gpus) > 1:
        joined = " & ".join(f"… --gpu {g}" for g in gpus)
        parallel = f"{subject} only within one blocking command\n  (`{joined} & wait`)"
    else:
        parallel = f"{subject} used by one blocking command at a time"

    return (
        template.replace("{{WALL_BUDGET}}", f"{cfg.sandbox.wall_budget_s:.0f}")
        .replace("{{FIRST_GPU}}", str(gpus[0]))
        .replace("{{GPU_COUNT}}", count)
        .replace("{{GPU_FLAGS}}", flags)
        .replace("{{GPU_PARALLEL}}", parallel)
    )


def write_charter(run: Path, cfg: RunConfig, root: Path | None = None) -> Path:
    dest = run / "block.md"
    dest.write_text(render_charter(cfg, root), encoding="utf-8")
    return dest


def block_prompt(block_id: str, run: Path, crashed: bool = False) -> str:
    prompt = BLOCK_PROMPT.format(block_id=block_id, root=str(run))
    if crashed:
        prompt += CRASHED_SUFFIX.format(block_id=block_id)
    return prompt


def resume_prompt(block_id: str, spent: float, budget: float, has_summary: bool) -> str:
    return RESUME_PROMPT.format(
        block_id=block_id,
        spent=spent,
        budget=budget,
        summary_state="present" if has_summary else "missing",
    )


def summary_repair_prompt(block_id: str) -> str:
    return SUMMARY_REPAIR_PROMPT.format(block_id=block_id)
