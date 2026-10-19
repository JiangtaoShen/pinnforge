<h1 align="center"><img src="pinnforge/ui/static/favicon.svg" alt="" width="44" align="absmiddle"> PINNForge</h1>

PINNForge is a knowledge-centric agent system for the autonomous discovery of
physics-informed neural network (PINN) solvers. It pairs two knowledge bases,
which decide what knowledge guides the discovery, with an agent harness, which
decides how that knowledge is used.

* The **literature base** holds notes on the methods of 130 published PINN
  papers, screened from 3,490 candidates and verified by human researchers.
  Each note comes with a JAX implementation that a round can adapt.
* The **experience base** accumulates the summaries that the rounds write: what
  each round tried, the error of every candidate it evaluated, and what worked,
  what failed and why. It is grounded in measured errors.
* In each **round** of a discovery run, a new coding agent redesigns the whole
  solver: network, loss, sampling, optimizer, training schedule, and the way
  the constraints are imposed. It decides when to draw on each knowledge base,
  evaluates its candidates within the round's GPU budget, and distils its
  findings into the experience base, so the knowledge that guides the
  discovery evolves with it.
* The **harness** runs the rounds without human intervention.
  * Its orchestrator is a plain Python program, not a language model. Whether
    a round is finished is a question the record of the run answers, so no
    model decides it.
  * The evaluation tool is the only path to full training. It trains every
    candidate for the problem's fixed training time.
  * The score service is the only component that reads the reference
    solution. It returns each candidate's relative L² error.

A run starts from round 0, the standard PINN of the task trained once: its
record is the **anchor** the run measures progress against. The run returns its
**incumbent**, the candidate with the lowest error over all full evaluations, as
the discovered solver.

On disk, round *r* lives in `runs/<id>/blocks/bNN/` (round 0 is `b00`), the
literature base in `kb1/` and the experience base in `blocks/kb2/`. Records,
logs and the command line say "block" for a round and `rRMSE` for the relative
L² error.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
```

On Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.ps1 | iex"
```

`uv tool install` puts a light `pinnforge` (PyYAML, NumPy) on PATH, from this
repository. Not `pip install pinnforge`: that name on PyPI is an unrelated
project.

* **Linux with an NVIDIA GPU** is the target.
* **macOS and CPU-only machines** run, slowly.
* **Windows** runs natively on the CPU, since JAX has no CUDA build for it. To
  train on its GPU, install inside WSL2 with `install.sh`. Claude Code on
  Windows needs Git for Windows, whose Git Bash runs its shell.

## Five commands

```bash
pinnforge init ~/pf && cd ~/pf     # workspace, which GPUs to use, pinned JAX environment
pinnforge demo                     # a whole run in minutes with a scripted agent, no API key
pinnforge setup                    # choose and prove the coding agent that runs the rounds
pinnforge task new                 # describe a PDE; no reference? an agent builds one
pinnforge run start -t <task> -n 5 # five rounds, detached; opens the live dashboard
```

`pinnforge doctor` checks every prerequisite and prints the fix for each
problem. `pinnforge ui` opens the dashboard of a run as it happens:
* the relative L² error of every evaluation against GPU time, with the
  incumbent and the anchor;
* the rounds with their budgets;
* the predictions of the incumbent as an animation;
* the coding agent's actions as they happen.

Over SSH: `ssh -L 8765:127.0.0.1:8765 <host>`.

## Tasks, and where the reference solution comes from

A problem reaches PINNForge as a **task package**, made of four parts:
* the description (`problem.md`);
* the standard PINN (`baseline.py`);
* the public files with the problem's input data, such as initial fields,
  coefficient grids, geometries and the points where predictions are made;
* the evaluation tool (`eval.py`).

The reference solution is not part of it.

```bash
pinnforge task new burgers --describe "u_t + u u_x = 0.01/π u_xx on [-1,1]×[0,1],
  u(x,0) = -sin(πx), u(±1,t) = 0; score u"                  # reference built and checked
pinnforge task new ldc2 --describe @spec.md --reference ~/data/ldc.mat   # your reference
pinnforge task status burgers --follow
```

`task new` is a staged pipeline, checkpointed in `tasks/<name>/.authoring/`:

1. **The package.** An author agent writes it without ever seeing the
   reference solution.
2. **The reference.** It is ingested from your data, or computed by a dedicated
   **reference-builder agent** with a classical solver at three or more
   resolutions, in its own environment.
3. **The check of the reference.** The framework scores each level against the
   next with the task's own metric, measures the order of convergence,
   estimates the finest level's error, and re-runs the coarsest levels to prove
   they reproduce.
4. **The rest.** Validation of the package's structure, an end-to-end rehearsal
   on the CPU, and the measurement of the standard PINN (the anchor).

A failed stage resumes with `--resume`. Flags given then, such as
`--ref-budget` or `--accept-reference`, override the recorded ones.
`--redo STAGE` re-runs a stage and everything after it: after a failed
convergence check, `--resume --redo reference` builds the reference again.

Know the exact solution? Give it as the reference, a `.py` or `.txt` with the
formula. Never put it in the description, which the author reads.

Nothing of the reference solution reaches the package a round is handed: no
values, no closed form, no account of how it was made or how accurate it is.
* **The author.** Its `private/` is moved out of its directory while it works,
  and its tool calls are audited afterwards. A package whose author reached for
  any task's reference is moved into `private/logs/` and written again.
* **The ingest and reference agents.** They saw the reference, so whatever they
  write stays in `private/`. The framework moves anything they put elsewhere
  into `private/logs/` and fails the stage. It moves their harness's own
  session transcripts (Claude Code, Codex) there too.
* **The authoring record and the dashboard** get only the numbers the framework
  computed in its check.

`task new` keeps the reference wherever `private_dir` says. `task validate`
checks every package for:
* words about the solution;
* paths to the reference;
* copies of the reference in public data;
* reference values in public text.

`pinnforge examples add ldc ks cyl naca` fetches the benchmark tasks with their
reference solutions (sha256-checked); the demo task ships in the package.

## Discovery runs

```bash
pinnforge run start -t ldc -n 5 --agent opus    # a binding from `pinnforge setup`
pinnforge run start -t ks --gpus 1 --budget 3600  # a GPU budget of 1 h per round
pinnforge run attach ldc_1                      # the driver's terminal (Ctrl-b d leaves it)
pinnforge run status ldc_1                      # per round: status, GPU s, evaluations, summary, best
pinnforge run stop ldc_1                        # unwinds like Ctrl-C
pinnforge run resume ldc_1
pinnforge run audit ldc_1                       # did any round alter another's record
pinnforge run film ldc_1                        # make its discovery film again
pinnforge run delete ldc_1                      # its folder and its score ledger, after asking
```

`run start` freezes the task package and the settings into `runs/<task>_<n>/`.
It copies the charter, the literature base and the public part of the task
package into the run. It installs round 0, restored from a cache or from an
earlier run, or measured. It then runs the rounds one after another, in a
detached tmux session, so the run outlives the terminal. On Windows, or
without tmux, it is a background process with a log.

For each round, the orchestrator:
* checks that the GPUs respond;
* starts the coding agent in the run directory with a fixed prompt (the round
  number, and the instruction to read the charter and carry out the round);
* waits for it to exit.

A round is finished when:
* its summary exists;
* less than one training time of its budget remains;
* at least one full evaluation is scored.

If not, the orchestrator resumes the same session with what is left to do, at
most six times.

One process drives a run (`runs/<id>/.lock`). SIGTERM and SIGHUP take the same
path Ctrl-C does: the agent's whole process group is interrupted, the
segment's cost booked and the integrity verdict written. `run stop` signals
that process instead of editing its state.

**The charter.** The charter (`block.md` in the run) is the same for every
problem apart from the budget and the GPUs. It describes the resources of the
round and sets the direction: choose the parent candidate and say why, reflect
after each evaluation against both knowledge bases, read the notes that bear
on the problem and cite them, and treat diagnostics as worth their GPU time.
It states the rules, the budget, the template of the summary and the condition
for finishing.

**Evaluations.** A round evaluates its candidates in three ways:
* **Full evaluation.** Trains a candidate with a seed on one GPU for the
  problem's training time, and returns its error.
* **Smoke test.** Trains it briefly on the CPU, at no GPU cost, and returns the
  error of the field it produces.
* **Diagnostic.** Runs any program of the agent on one GPU, at most for the
  training time, and returns its output without an error.

Full evaluations and diagnostics are charged their wall time against the
round's budget, failed ones included. The training time counts from the start
of the training process, so importing libraries and compiling count towards
it.

**The experience base.** Every round ends with a summary written to a fixed
template:
* **The direction.** The parent the round started from, and the reasons for
  its line of attack.
* **The evaluations.** A table of every evaluation, with the change of its
  candidate from that candidate's parent, its error and its number of steps.
* **The findings.** What works, what fails and why, and what later rounds
  should drop or pursue.

The direction and the findings cite literature notes by their three-digit
identifiers. Each round starts a new agent, which inherits nothing but the
files of the run, so the experience base is the memory of the discovery.

**The discovery film.** The dashboard is the live view of a run. Once the run
is over, its own agent makes a film of it:
1. The run's complete record goes into `runs/<id>/discovery/`: the facts, the
   round summaries, every agent transcript and the literature base's index.
   It is read through the dashboard's API, so it holds nothing of the reference
   solution.
2. The agent writes the script: each round's headline, line of attack and
   findings, the notes it used, and a label for each candidate.
3. A fixed player renders the script over the facts into
   `runs/<id>/discovery.html`, one self-contained file.

Every film has the same form, and every number in it comes from the record,
never from the agent. A script that fails its check gives way to a draft
extracted from the summaries. `film: false` in the run's settings leaves the
film to `pinnforge run film <id>`, and `--render-only` renders an edited script
again.

**GPUs.** Which cards a workspace's runs use is the operator's choice.
`pinnforge init` asks when the machine has more than one (or takes `--gpus 1`,
`--gpus 0,1`, `--gpus none`), and `pinnforge setup gpus` changes it later. It is
stored as `sandbox.gpus` in `pinnforge.yaml`, as one of:
* `auto`: every card `nvidia-smi` shows, narrowed by `CUDA_VISIBLE_DEVICES`.
  The right choice under a scheduler that assigns cards.
* an exact list, such as `[1]` or `[0, 2, 3]`;
* `[]` for CPU only.

Left `null` (a new workspace nobody chose cards for), it is this machine's
default, `pinnforge setup gpus --user`, else `auto`.

With one card, a round's evaluations queue on it. With several, a round may
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

The harness drives each of these with the same prompt and records the model
that each round uses.

`pinnforge setup` pins each to an **exact** model id. `opus` meant
`claude-opus-4-8` for a year and `claude-opus-5` after, so two runs a
generation apart that both say "opus" are not comparable. Setup also pings the
model through the adapter a round is started with. Whatever was asked for, the
id the harness actually ran is read back into the run's ledger.

A harness this repo has never heard of: point the runtime at
`module.path:ClassName`, implementing `pinnforge/runtime/base.py`'s
`AgentRuntime`.

## The literature base

The literature base covers methods for forward PINN problems trained without
solution data. It was built in five steps:
1. 22 seed papers from 2019 to 2026 were chosen.
2. A citation-graph pipeline expanded them to 3,490 candidates.
3. Rules on year, citations and keywords, and then LLM judges, screened these
   to 143.
4. Deduplication, expert checks and a full-text review left 130 papers.
5. An agent distilled each paper into a note, which human researchers verified.

A note has a header (title, authors, year, venue, code repository) and four
sections:
* the mechanism and the claim;
* the failure of PINN training that the method addresses;
* the method, with a JAX implementation that a round can adapt;
* the reported results.

The 130 notes ship with the package, and every run gets its own copy in
`kb1/`. Rounds read them before they choose what to build.
`pinnforge kb1 add --source <papers>` distils more into the workspace's own
`kb1/`, seeded from the packaged notes. `pinnforge kb1 check` validates their
shape and size.

## Talking to it

The `plugin/` directory drives PINNForge from your own agent. It holds:
* skills;
* a **task-author** subagent that turns a conversation about a PDE into a
  ready task;
* a **run-doctor** subagent that diagnoses a stuck run, read-only;
* a session hook.

```text
/plugin marketplace add JiangtaoShen/pinnforge        # Claude Code
/plugin install pinnforge@pinnforge-marketplace
```

Or just `pinnforge chat`, which opens Claude Code with the plugin loaded for
that session.

Codex, Cursor and OpenCode: install the plugin per project, in your own project
directory, never globally. Rounds run those CLIs too, and only Claude Code
rounds switch the plugin off; see `plugin/README.md`.

Never put an `AGENTS.md`, `CLAUDE.md` or `.cursor/rules/` in a workspace or
above it: rounds run in `runs/<id>/` and would read it.

## Scoring, and the record

**Scoring.** `eval.py` trains a candidate and predicts on the public points,
then files the predictions on a queue. The score service compares them with the
reference solution and files the score back.

The score is the only thing it ever returns: the relative L² error, or, when
there is none, a reason worked out from the submission and the public points
alone. Whatever else a task's metric computes stays inside (an MSE beside the
error would give away the reference's norm), and so do the metric's own error
messages, which go to the task's `private/logs/`.

The score service:
* never executes code of the agent;
* takes the metric from the task library, not from the copy in the run, which
  the agent can edit, and loads it once from its own snapshot of the task;
* scores at most 256 requests per round, so that it cannot serve as an
  unlimited oracle;
* keeps its own ledger, against which the records of the run can be checked;
* never follows a link planted in the queue.

**Where the reference lives.** The reference solution lives in
`tasks/<t>/private/`, which is never copied into a run. For isolation as an OS
fact, keep it outside the workspace (`private_dir:` in `pinnforge.yaml`, or in
`~/.config/pinnforge/config.yaml` for every workspace) and serve it as another
user: `pinnforge scored <run> --private /srv/answers`.

**The record.** The record of a run is protected by detection. Before each
round, the orchestrator hashes every file that the round must leave unchanged:
* earlier rounds' programs, evaluation records and summaries;
* the task package;
* the charter;
* the literature base.

It checks the hashes after the agent exits. The ledger of the current round
may only grow. The verdicts land in `runs/<id>/.integrity/` for `run audit`.

This detects rather than prevents, deliberately: a jail would have to
enumerate every path a round legitimately writes. In the default same-user
mode a determined agent could still *read* the library's reference solutions;
the separate-user deployment above is what closes that.

**The dashboard.** It is read-only and reachable by any local process, the
rounds' agents included. So it serves JSON built from known files only:
never `private/`, never the score queue, never a reference solution or an
error field. It checks the `Host` header, and requires a token when it listens
beyond loopback. Deleting a run or a task is done with `pinnforge run delete`
or `pinnforge task delete`, never through it.

## The task contract

The framework touches a task only through this interface.

* **`problem.md`**, the description: the only prose a round reads, in seven
  fixed sections (Equation, Domain, Boundary conditions, Initial condition,
  Scoring, Environment, Time budget).
* **`baseline.py`**, the standard PINN, the root candidate.
  * A frozen `PDE CONSTANTS` header, which fixes the problem's constants and
    its training time.
  * `train(rng, eval_callback=None) -> (params, step_count)`.
  * `predict_fn(params, X) -> dict`.

  A candidate keeps this interface and header, solves the problem as a PINN
  (the derivatives in its residuals taken by automatic differentiation of its
  networks), and reads no data beyond the public files.
* **`eval.py`**, the evaluation tool: the task's constants, grid handling
  (`INPUT_COLUMNS`, `FIELDS`, `REF_COLUMNS`, optional `predict_rows`) and its
  metric, `score_predictions(fields, ref_dir) -> {"rRMSE": x}`.

  The machinery (budget, per-GPU locks, worker subprocesses, the score queue,
  the record) is shared, and vendored into each run as `task/_forgekit.py`,
  frozen and hashed like the rest of `task/`.

  `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`
  appends one record per evaluation to `blocks/bNN/evals.jsonl`, and exits 0
  whenever a record was written.
* **`view.json`**: what the dashboard's fixed field window draws for this task.
  * Which grid coordinates go on the screen axes (`(x, t)` as a space-time
    map), and which on a time slider or a slice.
  * One to four panels of output heads, or quantities derived from them
    (`{"magnitude": ["u", "v"]}`).

  The task author writes it, `task validate` checks it against the task's
  coordinates and heads, and it is not part of the anchor's key. It is
  declarative and shows predictions only; without one, the dashboard picks a
  default.
* **The public files** sit in the task directory; **the reference solution**
  in `private/`.

`pinnforge task validate <name>` runs the contract checks.

**The anchor cache.** The anchor (round 0) is cached per task, keyed on:
* the package fingerprint;
* the reference solution;
* the hardware class.

Judge staleness by that, never by the score: the same task re-measured moves
0.4–17%, because training stops on a wall clock. `task anchor --rekey`
carries an anchor over without a GPU when re-scoring its cached parameters
reproduces the recorded score exactly.

`pinnforge task delete <name>` removes a task package with its reference
solution and its anchor, after asking (`--with-runs` takes its runs too).

## Workspace layout

| path | role |
|---|---|
| `pinnforge.yaml` | marks the workspace; its settings (agent, GPUs, budget, private dir, dashboard) |
| `pyproject.toml`, `.venv/` | the frozen JAX stack every `eval.py` runs in |
| `tasks/<name>/` | the task library |
| `runs/<task>_<n>/` | one discovery run: the frozen task package, the literature base (`kb1/`), the rounds (`blocks/bNN/`), the experience base (`blocks/kb2/`), the ledger, the logs, the discovery film |
| `.pinnforge/` | locks, score ledgers, the reference agent's solver environment, dashboard state |

Settings layer as: built-in defaults < `~/.config/pinnforge/config.yaml` <
`pinnforge.yaml` < flags < `--set a.b=c`.

A key that is `null` or absent in a file is unset there, and the layer below
decides. A new `pinnforge.yaml` leaves `agent`, `sandbox.gpus`,
`sandbox.wall_budget_s` and `private_dir` null, so the per-machine file holds
in every workspace until the workspace sets its own. `FORGE_PRIVATE_DIR` wins
over both files.

A run freezes what it used into its `run.yaml`.

## Upgrading from 0.1

The benchmark tasks' `eval.py` now delegate their machinery to the shared kit,
which changes their fingerprint. `pinnforge task anchor <task> --rekey` carries
each cached anchor over without a GPU: it re-scores the cached round 0 and
requires the identical number.

A run created before the upgrade has the old `eval.py` frozen in it.
* Resuming it against the new library is refused as metric drift, unless
  `PINNFORGE_ALLOW_METRIC_DRIFT=1`.
* Even then, its old `eval.py` files score requests that the new service does
  not accept, because they do not name the round being run.

Finish old runs with 0.1, or start new ones.

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
whitespace fix would stale every cached round 0. They are held to
`pinnforge task validate` instead.
