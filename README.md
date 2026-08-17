# PINNForge

Knowledge-centric multi-agent system for autonomous PINN design.

A **block** is one autonomous coding agent, a fixed GPU-wall budget, and one
written summary. Blocks run in series; knowledge compounds through the
summaries, not through orchestrator state. The orchestrator itself is a
plain Python process — whether a block is finished is a question the
filesystem answers, so no model is asked to decide it.

```bash
pinnforge task list                       # what can be solved
pinnforge run start -t ks -n 5            # five blocks on the KS task
pinnforge run status ks_1                 # where it got to
pinnforge run resume ks_1                 # after an interruption
```

## Layout

| Path | Role |
|---|---|
| `tasks/<name>/` | task library — one directory per problem: the three contract files, any public data the physics needs (geometry, level sets, initial conditions), and the answers under `private/` |
| `runs/<task>_<n>/` | one run: its frozen task copy, `blocks/bNN/`, `blocks/kb2/`, ledger, logs |
| `pinnforge/kb1/` | the fixed corpus of paper notes (`INDEX.md` is the map) — ships with the framework; a `kb1/` at the project root overrides it |
| `pinnforge/prompts/block.md.template` | the block charter — rendered per run; drop a `charter/block.md.template` at the project root to override it |
| `pinnforge/` | the framework |

Several tasks coexist and each can have several runs: `runs/ks_1`, `runs/ks_2`,
`runs/naca_1`. A run directory is self-contained — its `task/` is a **copy**,
so editing the library mid-run cannot change a running experiment.

### Why a run directory looks like a project root

Blocks are started with the run directory as their working directory. That
makes every path in the charter — `task/eval.py`, `kb1/INDEX.md`,
`blocks/bNN/`, `blocks/kb2/bNN.md` — resolve exactly as it did in the
single-task layout the charter was written for. The charter needs no
rewriting, and a block behaves identically no matter which harness is
driving it.

## The block charter

The charter template is the previous framework's `block.md`, carried over
verbatim except for the constants substituted per run — the GPU-wall budget
and the GPU pool. They used to be duplicated by hand across the charter and
the framework docs; a task swap that missed one left the charter quoting a
budget nobody had set.

Two passages differ from the original by intent:

* the charter no longer forbids reading the scoring truth, because the truth
  is not in the run at all — a rule that named a file went stale the moment a
  task renamed it;
* `no run_in_background` became `no background jobs` — the former is a Claude
  Code tool name and means nothing to another harness.

`tests/test_charter.py` pins this: it asserts nothing else drifts.

## Harnesses

Blocks run on whichever coding-agent CLI is installed. Each is driven
headlessly with the same prompt, the same working directory and the same
environment; only the argv that selects non-interactive mode differs.

| `agent.runtime` | CLI | default model |
|---|---|---|
| `claude_code` (default) | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | `openai/gpt-5` |

```bash
pinnforge agents doctor           # is the CLI installed and answering
pinnforge run start -t ks --runtime codex --model gpt-5.4
```

A harness this repo has never heard of works too: set
`agent.runtime = "module.path:ClassName"` for anything implementing the
`AgentRuntime` protocol in `pinnforge/runtime/base.py`.

## Choosing the model

Blocks run on one model, named per run:

```bash
pinnforge run start -t ldc                              # claude-opus-4-8, the default
pinnforge run start -t ldc --model claude-opus-5
pinnforge run start -t ldc --set agent.model=claude-opus-5
```

**Pin an exact id, not an alias.** `opus` meant `claude-opus-4-8` for a year
and `claude-opus-5` after; a run pinned to the alias is not pinned at all, and
two runs a generation apart both record "opus" while their scores get compared
as though the difference were the framework. The default is an exact id for
that reason, and it is the model the archived runs were measured on, so a new
run can be read against them.

Whatever was asked for, the id the harness *actually ran* is read back out of
its own log and recorded — in `run_usage.jsonl` as `model_resolved`, in
`state.json`, and in the summary's Model column. `run status` shows both when
they differ:

```
agent    claude_code / opus -> claude-opus-5
```

## Sandbox and GPUs

```yaml
sandbox:
  gpus: [0, 1]              # the pool a block may use; eval.py locks per id
  wall_budget_s: 7200       # GPU wall per block, enforced by eval.py
  env: {}                   # extra environment for block processes
  command_prefix: []        # wraps the agent CLI, e.g. ["firejail", "--"]
```

`command_prefix` applies to the whole process tree, so everything the agent
spawns — shells, `eval.py`, the training worker — inherits it.

```bash
pinnforge run start -t ks --gpus 0 --budget 3600
pinnforge run start -t ks --set sandbox.command_prefix='["firejail","--"]'
```

## Scoring, and where the answers are not

rRMSE is not computed inside `eval.py`. `eval.py` trains a candidate and
predicts on a public grid, then files the predictions on a queue; a separate
**score service** loads them, compares them against the reference with the
task's own metric, and files the score back. The service never executes agent
code, so the reference can live where the block's processes cannot read it.

That replaces a rule with a boundary. The old charter said "never read
`task/<reference>`" — a sentence that went stale the moment a task swap
renamed the file, and that nothing enforced anyway. Now there is no file to
name: the answers are not in the run at all.

`run start` serves the queue in-process, which is enough when the operator
and the blocks are the same user. To make the isolation an OS fact, keep the
references outside the repo and serve them as someone else:

```bash
pinnforge scored ks_1 --private /srv/pinnforge-private   # or FORGE_PRIVATE_DIR
```

`run start` yields to a standalone daemon whenever one is already alive.
`tasks/*/private/` is `.gitignore`d for the same reason, so a clone arrives
able to train but not to score until the references are supplied.

## Interruption and resume

Every state change is checkpointed to `runs/<id>/state.json`. Ctrl-C, a
reboot, or a dead harness all end the same way:

```bash
pinnforge run resume ks_1
```

An interrupt takes the block down with it. The agent runs in its own session,
so a terminal's Ctrl-C reaches the orchestrator and not the block — left
alone it would keep editing `blocks/bNN/`, keep charging the budget and keep
its training worker on the card, and the resume above would then put a second
agent on the same workspace. The process *group* is signalled instead, SIGINT
first so a harness that checkpoints can leave a resumable session behind.

Resume re-reads the checkpoint and the on-disk evidence, then continues. A
block is finished only when all three hold — its summary exists, its budget
is spent, and at least one real evaluation is logged — and an unfinished
block is continued in its own session when the harness supports resuming
one, or handed to a fresh agent that adopts the workspace. Because the
workspace, not the conversation, is the block's memory, that hand-off costs
nothing.

## Proving the record

A run's conclusions rest on files a block could edit but must not: earlier
blocks' evaluations and summaries, the task package, the charter, the corpus.
The charter forbids touching them and blocks have obeyed — but "we asked them
not to" is a weak thing to put in a paper, and a violation would be silent.

So every dispatch is bracketed: the protected set is hashed before the agent
starts and re-checked after it exits, and the verdict is written to
`runs/<id>/.integrity/`. The active block's own `evals.jsonl` is checked
differently — it may grow, never change, so its previous bytes must survive
as a prefix.

```bash
pinnforge run audit ks_1     # exits 1 if any segment violated the record
```

This detects rather than prevents, deliberately: a filesystem jail would have
to enumerate every path a block legitimately writes — its workspace, the
score queue, the GPU locks, the harness's own session state, whatever `uv add`
touches — and missing one turns a block that used to work into a block that
fails, which is the behavioural drift this framework cannot afford.

## Inspecting a run

```bash
pinnforge run list                  # every run, with its best score
pinnforge run status ks_1           # per-block wall, evals, summary, best
pinnforge run log ks_1 -n 20        # scored evaluations, best first
pinnforge run show ks_1 b03         # one block's records and its summary
pinnforge run stop ks_1             # mark it stopped
pinnforge info                      # resolved paths, tasks, runs
```

## Adding a task

```bash
pinnforge task new burgers --source ~/papers/burgers --prompt "Inviscid Burgers, IC -sin(pi x)"
```

One command, because a task is not finished when its files exist — it is
finished when it loads. `task new` drives an authoring agent that adapts an
existing package, then walks the same three gates by hand:

```bash
pinnforge task validate burgers            # 29 contract checks
pinnforge task smoke burgers               # free CPU dress rehearsal, end to end
pinnforge task anchor burgers --measure    # the b00 score every block must beat
```

Each gate reads the previous one's evidence, and each is a real gate: the
smoke passes only on a record that carries a score and no error — `eval.py`
exits 0 whenever a record was *written*, failed runs included, so the exit
status proves nothing. `task new --no-load` stops after validation.

`task anchor` on its own only *reports*: the cached score, and whether the
contract files still fingerprint to what it was measured against (STALE if
not). Judge staleness by that fingerprint, never by the score — the same task
re-measured at the same seed moves anywhere from 0.4% to 17%, because training
stops on a wall clock and the run is not bit-deterministic.

### Re-running a task costs nothing it already paid for

A run freezes the task it was given and keeps the b00 it measured, so the run
library is a second copy of both. Before spending anything, the framework
looks there:

* **the anchor** — `task anchor` and `run start` harvest a b00 from the newest
  run whose `task/` fingerprints to today's package, instead of measuring one.
  Only then does building it cost a GPU, and only when asked: `--measure`, or
  `--rebuild` to replace a cached one.
* **the definition** — `task new` on a name the run library already knows
  recovers `tasks/<name>/` from that run and skips the authoring agent
  entirely. Re-authoring would produce a *different* definition, and the point
  of recovering is that the new runs stay comparable with the old.

The recovered definition is the public half only: `private/` is never copied
into a run, so it cannot come back from one. That is the split working, and
the command says so rather than pretending the task is ready.

The validator is the arbiter: 29 checks covering the contract files, the
frozen header, the wall constants and — deliberately — the *shape* of
`problem.md`. The prose paradigm is the point of the corpus: every task
states the same seven sections in the same order, so an agent that has solved
one task already knows where to look in the next.

## Growing kb1

```bash
pinnforge kb1 add --source ~/papers/new --prompt "distil these three"
pinnforge kb1 check
```

Nodes are `NNN_YYYY_Title.md` with four sections — TL;DR, Problem, Method,
Results — where Method carries runnable JAX, because a block will copy it.

`kb1 check` validates the shape *and the size*: the 130 hand-written nodes run
3.6–6.3 KB (median 5.0), and a node far outside that band has either padded
past what a block can afford to read before choosing a line of attack, or
dropped the runnable Method that made it worth opening. The distiller is told
the same band, so generated nodes sit alongside the corpus rather than
standing out from it.

## The task contract

The framework touches a task only through this interface:

* **`problem.md`** — the task definition; the only prose a block reads.
* **`baseline.py`** — the root candidate. Defines the frozen `PDE CONSTANTS`
  header and the `train(rng, eval_callback=None) -> (params, step_count)` /
  `predict_fn(params, X) -> dict` contract every descendant keeps.
* **`eval.py`** — the single evaluation tool:
  `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`.
  Enforces `FORGE_WALL_BUDGET` across all GPU runs (CPU `--smoke` is free),
  appends one JSON record per run to `blocks/bNN/evals.jsonl` carrying at
  least `smoke`, `diag`, `wall_s` and `rRMSE` — the primary metric, lower is
  better — maintains `blocks/bNN/.budget`, saves params next to the
  candidate, and exits 0 whenever a record was written, failed runs included.
* Public data — geometry, level sets, initial conditions, the prediction
  grid — lives in the task directory and is copied into every run. The
  scoring truth lives in `private/`, which is never copied: the installer
  takes files only, so secrecy is a matter of where a file sits rather than
  what it is named.

Any package honouring it plugs in without framework changes.

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -q
uv run ruff check .
```
