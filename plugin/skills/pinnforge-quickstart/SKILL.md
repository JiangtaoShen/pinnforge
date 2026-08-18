---
name: pinnforge-quickstart
description: The fast path from zero to a running PINNForge experiment — what PINNForge is and when to reach for it, installing the CLI, picking a coding-agent harness, and the task/run layout. Use this whenever the user asks "what is pinnforge", "should I use pinnforge for this", wants to install or set it up, hits a "command not found" for pinnforge, or says "use pinnforge to solve this PDE" and needs the end-to-end path from install to a launched run. Hands off to `setting-up-harnesses` (which CLI drives the blocks), `creating-a-pinnforge-task` (task packages) and `running-pinnforge-blocks` (operating a run) for depth.
---

# PINNForge quickstart

**PINNForge** designs physics-informed neural networks by running autonomous
coding agents against a fixed GPU budget. A **block** is one agent, one budget,
and one written summary. Blocks run in series; knowledge compounds through the
summaries, not through orchestrator state. The orchestrator is a plain Python
process, so whether a block is finished is a question the filesystem answers and
no model is asked to decide it.

## When to reach for PINNForge

**Good fit**
- A PDE with a **reference solution**, so a candidate can be scored (rRMSE, lower is better).
- The work is **iterative search** over PINN design: architecture, loss weighting, sampling, optimiser schedule.
- You have GPU time to spend and want it spent by an agent that reads the literature first.

**Not a fit**
- No reference to score against. The whole loop is built on a number.
- A single forward solve. Use a classical solver.

## How a run is shaped

```
you provide:   tasks/<name>/  =  problem.md + baseline.py + eval.py + data
                                 (+ private/ holding the answers)
pinnforge:     measures b00 (the baseline as-is) as the control node
each block:    read kb1 + earlier summaries -> write candidates -> eval.py
               -> score service -> repeat until the budget is spent
               -> write blocks/kb2/bNN.md
```

Everything a block does is inside `runs/<task>_<n>/`. That directory is
self-contained: a frozen copy of the task, the rendered charter, per-block
workspaces, the ledger and the logs.

## Get running

### 1. Install

```bash
git clone git@github.com:JiangtaoShen/pinnforge.git && cd pinnforge
uv sync
pinnforge info          # resolved paths, tasks, runs
```

### 2. Pick a harness and prove it answers

Blocks run on whichever coding-agent CLI is installed, driven headlessly. Each
is installed and authenticated separately.

```bash
pinnforge agents list
pinnforge agents doctor --ping --model claude-opus-4-8
```

`doctor` alone checks the CLI is on PATH. `--ping` spends one exchange proving
the credentials, the provider config and the model id work too. Skip it and
those fail at the first block instead, an hour in. Details and the per-harness
troubleshooting matrix are in the `setting-up-harnesses` skill.

### 3. Check the task is loadable

```bash
pinnforge task list
pinnforge task validate ldc     # 29 contract checks
pinnforge task anchor ldc       # the b00 score every block must beat
```

No task yet? That is the `creating-a-pinnforge-task` skill.

### 4. Launch

```bash
pinnforge run start -t ldc -n 5
```

Then `running-pinnforge-blocks` covers status, resume, reading scores, and the
integrity audit.

## Two rules that matter more than they look

**Pin an exact model id, never an alias.** `opus` meant one model for a year and
another after. Two runs a generation apart both record "opus" and quietly stop
being comparable.

**Never write into a live `runs/<id>/`.** It is a block's workspace, and the
orchestrator hashes everything outside that workspace before each dispatch. An
edit shows up as an integrity violation and taints the run.
