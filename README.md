# PINNForge

Knowledge-centric multi-agent system for autonomous PINN design.

A **block** is one autonomous coding agent, a fixed GPU-wall budget, and one
written summary. Blocks run in series; knowledge compounds through the
summaries, not through orchestrator state. The orchestrator is a plain Python
process: whether a block is finished is a question the filesystem answers, so
no model is asked to decide it.

## Install

```bash
uv sync
pinnforge info                            # resolved paths, tasks, runs
pinnforge agents doctor                   # is a coding-agent CLI installed
```

## Running blocks

```bash
pinnforge task list                       # what can be solved
pinnforge run start -t ks -n 5            # five blocks on the KS task
pinnforge run status ks_1                 # per-block wall, evals, summary, best
pinnforge run resume ks_1                 # after an interruption
```

Reading the results:

```bash
pinnforge run list                        # every run, with its best score
pinnforge run log ks_1 -n 20              # scored evaluations, best first
pinnforge run show ks_1 b03               # one block's records and summary
pinnforge run audit ks_1                  # did any block alter another's record
```

`run start` freezes the task, installs the b00 anchor, then dispatches blocks
one at a time. Every state change is checkpointed to `runs/<id>/state.json`, so
Ctrl-C, a reboot and a dead harness all end the same way: `run resume` re-reads
the checkpoint and the on-disk evidence and carries on. A block counts as
finished only when its summary exists, its budget is spent, and at least one
real evaluation is logged. An interrupt takes the block's whole process group
with it, so a resume never lands a second agent on a workspace that is still
live.

## Layout

| Path | Role |
|---|---|
| `tasks/<name>/` | task library: the three contract files, public data, and the answers under `private/` |
| `runs/<task>_<n>/` | one run: frozen task copy, `blocks/bNN/`, `blocks/kb2/`, ledger, logs |
| `pinnforge/kb1/` | corpus of paper notes (`INDEX.md` is the map); a `kb1/` at the project root overrides it |
| `pinnforge/prompts/block.md.template` | the block charter; `charter/block.md.template` at the project root overrides it |

A run directory is self-contained and is the working directory blocks start in,
so `task/eval.py`, `kb1/INDEX.md` and `blocks/kb2/bNN.md` resolve for the
charter without it knowing anything about the multi-task layout above it. Its
`task/` is a **copy**: editing the library mid-run cannot change a running
experiment. `tests/test_charter.py` pins the charter against drift.

## Harnesses and models

Blocks run on whichever coding-agent CLI is installed, each driven headlessly
with the same prompt, working directory and environment.

| `agent.runtime` | CLI | default model |
|---|---|---|
| `claude_code` (default) | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | `openai/gpt-5` |

```bash
pinnforge run start -t ks --runtime codex --model gpt-5.4
pinnforge run start -t ldc --model claude-opus-5
```

**Pin an exact id, not an alias.** `opus` meant `claude-opus-4-8` for a year and
`claude-opus-5` after, so two runs a generation apart both record "opus" and
quietly stop being comparable. Whatever was asked for, the id the harness
*actually ran* is read back out of its log into `run_usage.jsonl`, `state.json`
and the summary. For a harness this repo has never heard of, point
`agent.runtime` at `module.path:ClassName`. Anything implementing the
`AgentRuntime` protocol in `pinnforge/runtime/base.py` works.

## Sandbox and GPUs

```yaml
sandbox:
  gpus: [0, 1]              # the pool a block may use; eval.py locks per id
  wall_budget_s: 7200       # GPU wall per block, enforced by eval.py
  env: {}                   # extra environment for block processes
  command_prefix: []        # wraps the agent CLI, e.g. ["firejail", "--"]
```

```bash
pinnforge run start -t ks --gpus 0 --budget 3600
pinnforge run start -t ks --set sandbox.command_prefix='["firejail","--"]'
```

`command_prefix` applies to the whole process tree, so everything the agent
spawns inherits it: shells, `eval.py`, the training worker.

## Scoring, and the record

rRMSE is not computed inside `eval.py`. `eval.py` trains and predicts on a
public grid, then files the predictions on a queue; a separate **score service**
compares them against the reference and files the score back. It never executes
agent code, so the reference can live where a block cannot read it. That is a
boundary rather than a rule. `run start` serves the queue in-process; to make the
isolation an OS fact, keep the references outside the repo and serve them as
another user:

```bash
pinnforge scored ks_1 --private /srv/pinnforge-private   # or FORGE_PRIVATE_DIR
```

`tasks/*/private/` is `.gitignore`d for the same reason: a clone can train, but
cannot score until the references are supplied.

Every dispatch is bracketed by a hash of what the block must leave alone:
earlier evaluations and summaries, the task package, the charter, the corpus.
The verdict lands in `runs/<id>/.integrity/` for `run audit` to read back.
This detects rather than prevents, deliberately: a jail would have to enumerate
every path a block legitimately writes, and missing one turns a working block
into a failing one.

## Adding a task

```bash
pinnforge task new burgers --source ~/papers/burgers --prompt "Inviscid Burgers, IC -sin(pi x)"
```

One command, because a task is finished when it *loads*, not when its files
exist. `task new` drives an authoring agent, then walks the same three gates you
can also run by hand:

```bash
pinnforge task validate burgers            # 29 contract checks
pinnforge task smoke burgers               # free CPU dress rehearsal, end to end
pinnforge task anchor burgers --measure    # the b00 score every block must beat
```

The smoke passes only on a record carrying a score and no error. `eval.py`
exits 0 whenever a record was *written*, failed runs included, so the exit
status proves nothing.

`task anchor` on its own only reports: the cached score, and whether the
contract files still fingerprint to what it was measured against. **Judge
staleness by that fingerprint, never by the score.** The same task re-measured
at the same seed moves anywhere from 0.4% to 17%, because training stops on a
wall clock and the run is not bit-deterministic.

Nothing already paid for is paid for twice. `task anchor` and `run start`
harvest a b00 from the newest run whose `task/` fingerprints to today's package
instead of measuring a new one, and `task new` on a name the run library already
knows recovers the definition from that run rather than authoring a different
one. Recovery returns the public half only: `private/` never enters a run, so it
cannot come back from one.

## Growing kb1

```bash
pinnforge kb1 add --source ~/papers/new --prompt "distil these three"
pinnforge kb1 check
```

Nodes are `NNN_YYYY_Title.md` with four sections (TL;DR, Problem, Method,
Results), where Method carries runnable JAX, because a block will copy it.
`kb1 check` validates shape *and* size: the 130 hand-written nodes run 3.6–6.3
KB, and one far outside that band has either padded past what a block can afford
to read or dropped the Method that made it worth opening.

## The task contract

The framework touches a task only through this interface, so any package
honouring it plugs in without framework changes.

* **`problem.md`**: the task definition, and the only prose a block reads.
  Seven fixed sections in a fixed order, so an agent on its second task already
  knows where to look.
* **`baseline.py`**: the root candidate. Defines the frozen `PDE CONSTANTS`
  header and the `train(rng, eval_callback=None) -> (params, step_count)` /
  `predict_fn(params, X) -> dict` contract every descendant keeps.
* **`eval.py`**: the single evaluation tool,
  `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`.
  Enforces `FORGE_WALL_BUDGET` across GPU runs (CPU `--smoke` is free), appends
  one JSON record per run to `blocks/bNN/evals.jsonl` carrying `smoke`, `diag`,
  `wall_s` and `rRMSE` (the primary metric, lower is better), maintains
  `.budget`, saves params beside the candidate, and exits 0 whenever a record
  was written.
* **Public data** (geometry, level sets, initial conditions, the prediction
  grid) sits in the task directory and is copied into every run. The scoring
  truth sits in `private/`, which is never copied: the installer takes files
  only, so secrecy is a matter of where a file sits, not what it is named.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check .
```

`tasks/` is excluded from lint on purpose: its `baseline.py` and `eval.py` are
md5-fingerprinted by the anchor cache, so a whitespace-only fix would mark every
cached b00 stale and cost a GPU re-measurement. They are held to `pinnforge task
validate` instead.
