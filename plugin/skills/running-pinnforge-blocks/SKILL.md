---
name: running-pinnforge-blocks
description: Run and manage PINNForge experiments from the operator side — launch blocks with `pinnforge run start` (task, count, model, GPU pool, budget, dotlist overrides), monitor with `run status` / `run log` / `run show` / `run list`, carry on after an interruption with `run resume`, verify nobody altered the record with `run audit`, and stop with `run stop`. Use whenever the user wants to start a PINNForge run, check on blocks, read scores, resume or stop a run, or diagnose blocks that keep being resumed, get marked short or failed, or never beat the anchor.
---

# Running PINNForge blocks

Five verbs: **start → status → log/show → resume → stop**, plus `audit` for the
record. Prefer `pinnforge run <cmd> --help` over guessing flags.

**Prereq:** a task that passes `pinnforge task validate <name>` and has an
anchor. No task yet → `creating-a-pinnforge-task`. Harness not proven →
`setting-up-harnesses`.

## 1. Launch

```bash
pinnforge run start -t ldc -n 5                                  # five blocks
pinnforge run start -t ks --runtime opencode --model deepseek/deepseek-chat -n 2
pinnforge run start -t ks --gpus 1 --budget 3600                 # pool and per-block wall
pinnforge run start -t ks --set sandbox.command_prefix='["firejail","--"]'
```

`run start` freezes the task into `runs/<task>_<n>/task/`, installs the b00
anchor (restored from cache or harvested from an earlier run, GPU only as a last
resort), serves the score queue in-process, then dispatches blocks one at a time.

Long runs belong in a persistent session, because the orchestrator holds the
score daemon and the block's process group:

```bash
tmux new-session -d -s forge -c /path/to/pinnforge
tmux send-keys -t forge 'pinnforge run start -t ldc -n 10 2>&1 | tee -a logs/ldc.log' Enter
```

## 2. Watch

```bash
pinnforge run list                  # every run, with its best score
pinnforge run status ldc_5          # per-block wall, evals, summary, best
pinnforge run log ldc_5 -n 20       # scored evaluations, best first
pinnforge run show ldc_5 b03        # one block's records and its summary
```

`run status` is the one to read. Its `status` column is the block's own verdict:

| status | meaning |
|---|---|
| `done` | summary written, budget spent, at least one scored eval |
| `short` | usable work (summary + scored evals) but the agent stopped before the budget was spent |
| `interrupted` | the run was stopped while this block was dispatched |
| `failed` | produced nothing usable |

A block counts as finished only when all three hold: its summary exists, its
budget is spent, and at least one real evaluation is logged.

## 3. Resume

```bash
pinnforge run resume ldc_5          # after Ctrl-C, a reboot, or a dead harness
```

Resume re-reads `state.json` plus the on-disk evidence and carries on. An
unfinished block is continued in its own session when the harness gives a resume
token, or handed to a fresh agent that adopts the workspace. Because the
workspace is the block's memory, that hand-off costs nothing.

Ctrl-C takes the block's whole process group down with it, so a resume never
lands a second agent on a workspace that is still live.

## 4. Audit the record

```bash
pinnforge run audit ldc_5           # exits 1 if any segment violated the record
```

Every dispatch is bracketed by a hash of what the block must leave alone:
earlier evaluations and summaries, the task package, the charter, the corpus.
Verdicts land in `runs/<id>/.integrity/`. Run this before believing any number
from a run.

## 5. Stop

```bash
pinnforge run stop ldc_5            # mark it stopped, reconcile dispatched blocks
```

`run stop` records state. It does not kill anything: live processes are in their
own group, so interrupt the CLI that owns them instead.

## Diagnosing a run that is not going anywhere

| symptom | read this | usual cause |
|---|---|---|
| Every block marked `short` | `run status`, the kb2 summaries | the agent thinks it is done; the budget may be larger than the task can absorb |
| A block resumed many times | the orchestrator log | before the no-progress check, this was a spin; now it stops after one barren segment |
| No block beats b00 | `run log`, `task anchor` | the anchor may sit on a trivial floor (a task where the zero field is an exact solution) |
| Scores move by <1% between blocks | `task anchor` output | re-measurement noise on some tasks reaches 17%; sub-noise gains are not gains |
| `scorer unavailable` in records | is `run start` still alive? | the score daemon lives in the orchestrator process |

**Never edit a live run directory** to "fix" a block. It is hashed before every
dispatch, and an edit becomes an integrity violation on the next verdict.
