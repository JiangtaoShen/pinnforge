# PINNForge — block-serial autonomous PINN design

Goal: drive rRMSE down on the current task in `task/problem.md` by
running **blocks** in series. A block = one autonomous subagent
(**Opus 4.8**) + 7200 s of GPU-run wall time + one written summary.
Knowledge compounds through the summaries, not through orchestrator
state.

The framework (workflow, `block.md`, `kb1/`) is task-agnostic; it
touches the task only through the **task contract** below. Swapping
`task/` swaps the problem.

## The one workflow

    /pinnforge <n>        # run n blocks, serially

The orchestrator is mechanical (see `.claude/commands/pinnforge.md`); all
science happens inside block subagents following `block.md` (the block
charter: resources, direction, rules, budget, required output).

## Maintenance skills

Project skills in `.claude/skills/` (say the phrase):

- **"archive the run data"** (`archive-data`) — pack `blocks/` (incl.
  `kb2/`) + `kb1/` + `task/` + `run_summary.md` into
  `/home/jiangtao/pinnforge_data/`.
- **"reset the framework"** (`reset-framework`) — archive if unbacked,
  then reset `blocks/` (incl. `blocks/kb2/`) to the b00 state.
- **"swap the task to `<pkg>`"** (`swap-task`) — install a new `task/`
  package (per the Task contract) and rebuild the b00 baseline node.

## Layout

| Path | Role | Mutability |
|---|---|---|
| `task/` | **what to solve**: `problem.md` + `baseline.py` + `eval.py` + reference data (see Task contract) | frozen |
| `kb1/` | **what is known**: fixed corpus of paper notes + `INDEX.md` | fixed |
| `blocks/kb2/` | accumulated block summaries: `b00.md, b01.md, …` | append-only |
| `blocks/bNN/` | **the work**: per-block workspace (`bNN_*.py`, `bNN_*.pkl`, `evals.jsonl`) | owned by block NN |
| `blocks/run_usage.jsonl` | append-only per-block usage ledger (model, duration, tokens, tool-uses) | appended by `/pinnforge` |
| `blocks/run_summary.md` | regenerated each run: per-block model + time + tokens + best rRMSE, with totals | written by `/pinnforge` |
| `block.md` | block charter (subagent instructions) | edit deliberately |

## Quick status

    .venv/bin/python -c "
    import json, pathlib
    for f in sorted(pathlib.Path('blocks').glob('b*/evals.jsonl')):
        rs=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        real=[r['rRMSE'] for r in rs if not r.get('smoke') and r.get('rRMSE')]
        print(f.parent.name, min(real) if real else 'no finite score')"

## Invariants

- **Block subagents run on Opus 4.8** — `/pinnforge` pins `model="opus"`
  (exact id `claude-opus-4-8`) on every block and repair dispatch, so
  results stay reproducible regardless of the session's model. Never
  omit the pin.
- Blocks run strictly serially; block N may read (never write) blocks 1…N-1.
- Every GPU run — full evals and `--diag` probes alike — goes through
  `task/eval.py` (wall-time budget, GPU locks, logging, params — see
  Task contract); CPU smoke runs are free. A block is done only with
  its budget spent (≥1 full eval logged) and its summary written
  (b00 exempt); operational discipline (foreground evals, stall
  watchdog, resume) lives in block.md §4 and pinnforge.md.
- Workspace `.py` files are named `bNN_<slug>.py` (free slug, no
  implied order) and must be referenced by `evals.jsonl` records; a
  full eval freezes a candidate, diag scripts stay editable.
  Candidates keep `task/baseline.py`'s frozen header and
  `train`/`predict_fn` contract. The JAX core stack stays frozen (pins
  in `pyproject.toml`); extensions may be added via `uv add`.
- `blocks/kb2/bNN.md` is written by block NN alone, at block end (a
  resumed block may revise its own summary); nothing else edits it.
- `evals.jsonl` is written by `task/eval.py` alone; each block's
  non-smoke records stay untouched once written.
- After every run `/pinnforge` regenerates `blocks/run_summary.md` from the
  append-only ledger `blocks/run_usage.jsonl` joined with each
  `evals.jsonl`; usage numbers come from block completion notifications
  (mechanics: pinnforge.md step 4).
- b00 is the setup control block (baseline measured as-is), not a
  subagent product.

## Task contract

The framework touches `task/` only through this interface; any task
package that honors it plugs in without framework changes:

- `task/problem.md` — human/agent-readable task definition, scoring
  protocol, environment notes (GPUs, time budget).
- `task/baseline.py` — the root candidate. It defines the frozen
  header and the module-level `train(rng, eval_callback=None) ->
  (params, step_count)` / `predict_fn(params, X) -> dict` contract
  that every descendant keeps.
- `task/eval.py` — the single evaluation tool, CLI:
  `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`.
  It must: enforce the per-block budget (`FORGE_WALL_BUDGET`, default
  7200 wall-seconds across all GPU runs, concurrency-safe; CPU
  `--smoke` free; `--diag` runs an arbitrary block-owned script on
  GPU, metered the same way); append one JSON record per run to
  `blocks/bNN/evals.jsonl` with at least `smoke: bool`, `diag: bool`
  (diag runs only), `wall_s: float` and `rRMSE: float|null` (**the
  primary metric, lower is better — the orchestrator and summaries key
  on this exact field name**); maintain the spent-seconds counter
  `blocks/bNN/.budget`; save trained params next to the candidate;
  exit 0 whenever a record was written (failed runs included).
- Any reference data eval.py needs lives inside `task/`.

For a new problem: swap the contents of `task/` (honoring the
contract), reset `blocks/` (incl. `blocks/kb2/`) to its b00 state;
`kb1/` and the workflow stay.

## Repo policy

The public repo carries the framework plus minimal examples: the
current `task/` package, `blocks/b00/` and `blocks/kb2/b00.md` (the
initial nodes), and `kb1/INDEX.md` + a few sample notes.
Everything else under `blocks/` (incl. `blocks/kb2/`) and `kb1/` is
local-only (gitignored) — as is any future task's data.
