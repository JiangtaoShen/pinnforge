# PINNForge

**English** | [中文](README_CN.md)

PINNForge designs physics-informed neural networks with autonomous coding
agents. A **block** is one agent, a fixed GPU-wall budget, and one written
summary. Blocks run in series; knowledge compounds through the summaries, not
through orchestrator state. Every candidate is scored by relative L2 error
(rRMSE) against a reference solution held by a score service the agents cannot
read. The orchestrator is a plain Python process: whether a block is finished
is a question the filesystem answers, so no model is asked to decide it.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
```

`uv tool install` puts a light `pinnforge` (PyYAML, NumPy) on PATH, from this
repository. Not `pip install pinnforge`: that name on PyPI is an unrelated
project. Linux with an NVIDIA GPU is the target; macOS and CPU-only machines
run, slowly. Windows: use WSL2.

## Five commands

```bash
pinnforge init ~/pf && cd ~/pf     # workspace, which GPUs to use, pinned JAX environment
pinnforge demo                     # a whole run in minutes with a scripted agent — no API key
pinnforge setup                    # choose and prove the coding agent that runs blocks
pinnforge task new                 # describe a PDE; no reference? an agent builds one
pinnforge run start -t <task> -n 5 # detached; opens the live dashboard
```

`pinnforge doctor` checks every prerequisite and prints the fix for each
problem. `pinnforge ui` opens the dashboard: the score of every evaluation
against GPU time with the best-so-far line and the b00 anchor, a block
timeline with live budgets, the best predicted field so far as an animation,
and the running agent's tool calls as they happen. Over SSH:
`ssh -L 8765:127.0.0.1:8765 <host>`.

## Tasks, and where the answers come from

```bash
pinnforge task new burgers --describe "u_t + u u_x = 0.01/π u_xx on [-1,1]×[0,1],
  u(x,0) = -sin(πx), u(±1,t) = 0; score u"                  # reference built and checked
pinnforge task new ldc2 --describe @spec.md --reference ~/data/ldc.mat   # your reference
pinnforge task status burgers --follow
```

`task new` is a staged pipeline, checkpointed in `tasks/<name>/.authoring/`:
an author agent writes the package (`problem.md`, `baseline.py`, `eval.py`, the
public grid) without ever seeing the answers; the reference is ingested from
your data, or computed by a dedicated **reference-builder agent** with a
classical solver at three or more resolutions in its own environment; the
framework then checks that reference itself — each level scored against the
next with the task's own metric, the order of convergence measured, the finest
level's error estimated, the coarsest levels re-run to prove they reproduce;
then validation, a free CPU smoke, and the b00 anchor. A failed stage resumes
with `--resume` (flags given then, such as `--ref-budget` or
`--accept-reference`, override the recorded ones); `--redo STAGE` re-runs a
stage and everything after it — after a failed convergence check,
`--resume --redo reference` builds the reference again. Know the exact solution? Give it as the reference — a `.py` or
`.txt` with the formula — never in the description, which the author reads.

Nothing of the reference reaches the package a block is handed — no values,
no closed form, no account of how it was made or how accurate it is. The
author's `private/` is moved out of its directory while it works, and its tool
calls are audited afterwards: a package whose author reached for any task's
answers is moved into `private/logs/` and written again. Whatever the ingest
and reference agents write — they saw the answers — stays in `private/`: the
framework moves anything they put elsewhere into `private/logs/` and fails the
stage, and moves their harness's own session transcripts (Claude Code, Codex)
there too. The authoring record and the dashboard get only the numbers the
framework computed in its check. `task new` keeps the answers wherever
`private_dir` says. `task validate` checks every package for
words about the solution, paths to the answers, copies of the reference in
public data and reference values in public text.

`pinnforge examples add ldc ks cyl naca` fetches the benchmark tasks with their
references (sha256-checked); the demo task ships in the package.

## Runs

```bash
pinnforge run start -t ldc -n 5 --agent opus    # a binding from `pinnforge setup`
pinnforge run start -t ks --gpus 1 --budget 3600
pinnforge run attach ldc_1                      # the driver's terminal (Ctrl-b d leaves it)
pinnforge run status ldc_1                      # per block: status, GPU s, evals, summary, best
pinnforge run stop ldc_1                        # unwinds like Ctrl-C
pinnforge run resume ldc_1
pinnforge run audit ldc_1                       # did any block alter another's record
```

`run start` freezes the task and the settings into `runs/<task>_<n>/`,
installs the b00 anchor (from cache, from an earlier run, or measured), then
dispatches blocks one at a time in a detached tmux session, so the run outlives
the terminal. One process drives a run (`runs/<id>/.lock`); SIGTERM and SIGHUP
take the same path Ctrl-C does — the agent's whole process group interrupted,
the segment's cost booked, the integrity verdict written — and `run stop`
signals that process instead of editing its state. A block counts as finished
only when its summary exists, its budget is spent, and at least one real
evaluation is logged.

**GPUs.** Which cards a workspace's runs use is the operator's choice:
`pinnforge init` asks when the machine has more than one (or takes `--gpus 1`,
`--gpus 0,1`, `--gpus none`), and `pinnforge setup gpus` changes it later. It is
stored as `sandbox.gpus` in `pinnforge.yaml`: `auto` — every card `nvidia-smi`
shows, narrowed by `CUDA_VISIBLE_DEVICES`, the right choice under a scheduler
that assigns cards — an exact list such as `[1]` or `[0, 2, 3]`, or `[]` for
CPU only. Left `null` (a new workspace nobody chose cards for), it is this
machine's default, `pinnforge setup gpus --user`, else `auto`. One card: a block's evaluations queue on it. Several: a block may
evaluate on all of them in parallel. `run start --gpus` overrides the choice
for one run, and refuses a card another run already holds (`--share` joins it
anyway); `demo` takes only the free ones.

## Agents

| runtime | CLI | default model |
|---|---|---|
| `claude_code` | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | none; name one |

`pinnforge setup` pins each to an **exact** model id — `opus` meant
`claude-opus-4-8` for a year and `claude-opus-5` after, so two runs a
generation apart that both say "opus" are not comparable — and pings it
through the adapter a block is started with. Whatever was asked for, the id the
harness actually ran is read back into the run's ledger. A harness this repo
has never heard of: point the runtime at `module.path:ClassName` implementing
`pinnforge/runtime/base.py`'s `AgentRuntime`.

## The corpus

Blocks read `kb1/` before choosing a line of attack: 130 paper notes shipped
with the package, each with runnable JAX in its Method section, copied into
every run. `pinnforge kb1 add --source <papers>` distils more into the
workspace's own `kb1/` (seeded from the packaged corpus); `pinnforge kb1 check`
validates shape and size.

## Talking to it

The `plugin/` directory drives PINNForge from your own agent: skills, a
**task-author** subagent that turns a conversation about a PDE into a ready
task, a **run-doctor** subagent that diagnoses a stuck run read-only, and a
session hook.

```text
/plugin marketplace add JiangtaoShen/pinnforge        # Claude Code
/plugin install pinnforge@pinnforge-marketplace
```

or just `pinnforge chat`, which opens Claude Code with the plugin loaded for
that session. Codex, Cursor and OpenCode: install it per project, in your own
project directory, never globally — blocks run those CLIs too, and only Claude
Code blocks switch the plugin off; see `plugin/README.md`. Never put an
`AGENTS.md`, `CLAUDE.md` or `.cursor/rules/` in a workspace or above it: blocks
run in `runs/<id>/` and would read it.

## Scoring, and the record

`eval.py` trains and predicts on a public grid, then files the predictions on a
queue; the score service compares them with the reference and files the score
back. The score is the only thing it ever returns: an rRMSE, or — when there is
none — a reason worked out from the submission and the public grid alone.
Whatever else a task's metric computes stays inside (an MSE beside the rRMSE
would give away the reference's norm), and so do the metric's own error
messages, which go to the task's `private/logs/`. It never executes agent code,
loads the metric once from its own snapshot of the task, and never follows a
link planted in the queue. The answers live in
`tasks/<t>/private/`, which is never copied into a run. For isolation as an OS
fact, keep them outside the workspace (`private_dir:` in `pinnforge.yaml`, or
in `~/.config/pinnforge/config.yaml` for every workspace) and
serve them as another user: `pinnforge scored <run> --private /srv/answers`.

Every dispatch is bracketed by a hash of what the block must leave alone:
earlier evaluations and summaries, the task package, the charter, the corpus.
The verdicts land in `runs/<id>/.integrity/` for `run audit`. This detects
rather than prevents, deliberately: a jail would have to enumerate every path a
block legitimately writes. In the default same-user mode a determined block
could still *read* the library's answers; the separate-user deployment above is
what closes that.

The dashboard is read-only and reachable by any local process, blocks
included, so it serves JSON built from known files only — never `private/`,
never the score queue, never a reference or an error field — checks the `Host`
header, and requires a token when it listens beyond loopback.

## The task contract

The framework touches a task only through this interface.

* **`problem.md`** — the only prose a block reads; seven fixed sections.
* **`baseline.py`** — the root candidate: a frozen `PDE CONSTANTS` header and
  `train(rng, eval_callback=None) -> (params, step_count)` /
  `predict_fn(params, X) -> dict`.
* **`eval.py`** — the task's constants, grid handling (`INPUT_COLUMNS`,
  `FIELDS`, `REF_COLUMNS`, optional `predict_rows`) and its metric
  `score_predictions(fields, ref_dir) -> {"rRMSE": x}`. The machinery — budget, per-GPU locks,
  worker subprocesses, the score queue, the record — is shared and vendored
  into each run as `task/_forgekit.py`, frozen and hashed like the rest of
  `task/`. `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`
  appends one record per run to `blocks/bNN/evals.jsonl` and exits 0 whenever
  a record was written.
* **Public data** sits in the task directory; **the answers** in `private/`.

`pinnforge task validate <name>` runs the contract checks. The b00 anchor is
cached per task, keyed on the package fingerprint, the reference and the
hardware class; judge staleness by that, never by the score (the same task
re-measured moves 0.4–17%, because training stops on a wall clock).
`task anchor --rekey` carries an anchor over without a GPU when re-scoring its
cached params reproduces the recorded score exactly.

## Workspace layout

| path | role |
|---|---|
| `pinnforge.yaml` | marks the workspace; its settings (agent, GPUs, budget, private dir, dashboard) |
| `pyproject.toml`, `.venv/` | the frozen JAX stack every `eval.py` runs in |
| `tasks/<name>/` | the task library |
| `runs/<task>_<n>/` | one run: frozen task copy, corpus copy, `blocks/bNN/`, `blocks/kb2/`, ledger, logs |
| `.pinnforge/` | locks, score ledgers, the reference agent's solver env, dashboard state |

Settings layer as: built-in defaults < `~/.config/pinnforge/config.yaml` <
`pinnforge.yaml` < flags < `--set a.b=c`. A key that is `null` or absent in a
file is unset there and the layer below decides; a new `pinnforge.yaml` leaves
`agent`, `sandbox.gpus`, `sandbox.wall_budget_s` and `private_dir` null, so
the per-machine file holds in every workspace until the workspace sets its
own. `FORGE_PRIVATE_DIR` wins over both files. A run freezes what it used into
its `run.yaml`.

## Upgrading from 0.1

The benchmark tasks' `eval.py` now delegate their machinery to the shared kit,
which changes their fingerprint: `pinnforge task anchor <task> --rekey` carries
each cached anchor over without a GPU (it re-scores the cached b00 and requires
the identical number). A run created before the upgrade has the old `eval.py`
frozen in it; resuming it against the new library is refused as metric drift
unless `PINNFORGE_ALLOW_METRIC_DRIFT=1`, and even then its old `eval.py` files
score requests the new service does not accept (they do not name the block
being run) — finish old runs with 0.1, or start new ones.

## Development

```bash
git clone https://github.com/JiangtaoShen/pinnforge && cd pinnforge
uv sync                       # dev tools + the frozen stack; the checkout is itself a workspace
uv run pytest tests/ -q       # fast suite
uv run pytest -m slow         # real evaluations (JAX, a GPU, private/): eval.py parity goldens
uv run ruff check .
```

`tasks/` and `pinnforge/examples/` are excluded from lint on purpose: their
`baseline.py` and `eval.py` are fingerprinted by the anchor cache, so a
whitespace fix would stale every cached b00. They are held to
`pinnforge task validate` instead.
