---
name: running-pinnforge-blocks
description: Run and manage PINNForge experiments — `pinnforge run start` (task, round count, agent binding, GPU pool, budget, overrides; detached by default), watching with the live dashboard (`pinnforge ui`), `run status` / `log` / `show` / `list` / `attach`, stopping cleanly with `run stop`, continuing with `run resume`, and checking the record with `run audit`. Use whenever the user wants to start a run, check on rounds, read scores, stop or resume a run, run on one GPU or several, or diagnose rounds that keep being resumed, end short or failed, or never beat the anchor.
---

# Running PINNForge rounds

## Start

```bash
pinnforge run start -t ldc -n 5                    # default agent binding, every visible GPU
pinnforge run start -t ks --agent deepseek -n 2    # another binding from `pinnforge setup`
pinnforge run start -t ks --gpus 1 --budget 3600   # one card, 3600 GPU-s per round
pinnforge run start -t ks --foreground             # drive it in this terminal instead
pinnforge run start -t ks --set sandbox.command_prefix='["firejail","--"]'
```

`run start` freezes the task and settings into `runs/<task>_<n>/`, installs the
b00 anchor (from cache, from an earlier run, or measured), then dispatches
rounds one at a time — **in a detached tmux session** (or a background process
with a log), so the run outlives the terminal. It starts the dashboard and
prints its URL.

**GPUs.** The workspace's cards are the operator's choice, made at `init` and
stored as `sandbox.gpus` in `pinnforge.yaml`: `auto` (every card `nvidia-smi`
shows, narrowed by `CUDA_VISIBLE_DEVICES`), an exact list like `[1]` or
`[0, 2]`, `[]` for CPU only, or `null` (nobody chose: the machine default from
`pinnforge setup gpus --user`, else `auto`). Change it with
`pinnforge setup gpus 0,1` (or `all` / `none`); `--gpus` overrides it for one run. `run start` prints where its
cards came from and refuses a card the machine lacks. One card: evaluations
queue on it. Several: a round may run evaluations on all of them in parallel.
No card: CPU mode, fine for the demo. If another run already holds a card,
`run start` refuses; `--share` starts anyway (evaluations then queue on the
shared card's lock). `demo` picks only free cards.

## Watch

```bash
pinnforge ui                     # live: score chart, round timeline, predicted field, agent activity
pinnforge run list               # every run, whether it is live, its best score
pinnforge run status ldc_5       # per round: status, GPU seconds, evals, summary, best
pinnforge run log ldc_5 -n 20    # scored evaluations, best first
pinnforge run show ldc_5 b03     # one round's records and summary
pinnforge run attach ldc_5       # the driver's terminal (Ctrl-b d leaves it running)
```

Over SSH: `ssh -L 8765:127.0.0.1:8765 <host>` and open the URL locally.

| round status | meaning |
|---|---|
| `done` | summary written, budget spent, at least one scored eval |
| `short` | usable work, but the agent stopped before spending its budget |
| `interrupted` | the run was stopped while this round ran |
| `failed` | produced nothing usable |

## Stop and resume

```bash
pinnforge run stop ldc_5            # unwinds like Ctrl-C: agent group stopped, cost booked, integrity verdict written
pinnforge run stop ldc_5 --force    # if it will not stop within 10 s
pinnforge run resume ldc_5          # continue; a crashed round's workspace is re-adopted
```

One process drives a run at a time (`runs/<id>/.lock`); `resume` refuses while
the driver is still alive, and stops an agent a crashed driver left behind
before starting a new one. A run whose b00 anchor was never finished gets it
installed on resume. A round closed as `short` or `failed` is not reopened. An
eval stopped mid-training is charged only the seconds it actually ran.

## Audit

```bash
pinnforge run audit ldc_5
```

Every dispatch is bracketed by hashes of what the round must leave alone
(other rounds' records and summaries, the task, the charter, the corpus). A
violation means that round's results cannot be trusted as they stand. A
segment listed as never verified (a hash taken before the dispatch, no
verdict after — a driver killed outright) is not clean either.

## Diagnosing a run that is not going anywhere

Delegate to the `pinnforge-run-doctor` subagent when available. Otherwise:

| symptom | check | fix |
|---|---|---|
| status `running`, driver `not running` | `run status` | `run resume <id>` |
| every eval `scorer unavailable` | `logs/orchestrator.log` | a second driver or a dead `pinnforge scored`; stop it and resume |
| `scoring unavailable: MetricDrift` | orchestrator.log | the library task changed after the run began — restore it, or start a new run |
| rounds `failed` instantly | the round's `logs/*.log` | harness credentials or model id — `pinnforge doctor --ping` |
| evals queue for minutes | `nvidia-smi` | another run holds that GPU |
| never beats b00 | `run log` | read the kb2 summaries; the task may need a different budget or TRAIN_TIME |
