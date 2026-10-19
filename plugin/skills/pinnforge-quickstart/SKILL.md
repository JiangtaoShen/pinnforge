---
name: pinnforge-quickstart
description: The fast path from zero to a running PINNForge experiment — what PINNForge is and when to reach for it, installing the CLI, creating a workspace, the zero-token demo, choosing the agent, creating a task from a described PDE, and launching and watching a run. Use whenever the user asks "what is pinnforge", "should I use pinnforge for this", wants to install or set it up, hits "command not found" for pinnforge, or says "use pinnforge to solve this PDE" and needs the end-to-end path. Hands off to `setting-up-harnesses` (the agent that runs blocks), `creating-a-pinnforge-task` (tasks and references) and `running-pinnforge-blocks` (operating a run).
---

# PINNForge quickstart

**PINNForge** designs physics-informed neural networks by running autonomous
coding agents against a fixed GPU budget. A **block** is one agent, one budget
and one written summary; blocks run in series and later blocks build on
earlier summaries. Every candidate is scored by relative L2 error (rRMSE)
against a reference solution, by a score service the agents cannot read.

## When to reach for it

**Good fit:** a PDE where a better PINN design is worth GPU hours — you want
architecture, sampling, loss weighting and optimiser choices searched by an
agent that reads the literature first (it ships 130 paper notes).

**Not a fit:** a single forward solve (use a classical solver), or a problem
with no computable reference — scoring needs one. It does not have to exist
yet: PINNForge builds one with a classical solver and checks it converged.

## Get running

```bash
curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
pinnforge init ~/pf && cd ~/pf     # workspace, which GPUs to use, pinned JAX environment
pinnforge demo                     # a full run in minutes, scripted agent, no API key
pinnforge setup                    # choose and prove the coding agent that runs blocks
pinnforge doctor                   # every prerequisite, with the fix for anything missing
```

On Windows, install from PowerShell with `powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.ps1 | iex"` and make the workspace with `pinnforge init $HOME\pf; cd $HOME\pf`. Runs there train on the CPU; to use the GPU, install inside WSL2 instead.

On a machine with several GPUs, `init` lists them with what is running on each
and asks which this workspace may use — one, several, all, or none. That is the
operator's call: ask the user rather than choosing, and pass it as
`--gpus 1` / `--gpus 0,1` / `--gpus all` / `--gpus none` when running `init`
for them. `pinnforge setup gpus` changes it later.

`pinnforge demo` measures the anchor, runs two blocks and opens the dashboard,
so the operator sees the whole loop before spending anything.

## A task from a described PDE

The user's words are enough. Delegate to the `pinnforge-task-author` subagent
when available; otherwise:

```bash
pinnforge task new heat2d --describe "2-D heat equation u_t = 0.01 Δu on [0,1]², t ∈ [0,1],
  u = 0 on the boundary, u(x,y,0) = sin(πx) sin(πy); score u"      # no reference: one is built
pinnforge task new ldc2 --describe @spec.md --reference ~/data/ldc_re1000.mat   # user's data
pinnforge task status heat2d --follow
```

`creating-a-pinnforge-task` has the details (stages, the reference check,
resuming).

## Run and watch

```bash
pinnforge run start -t heat2d -n 3     # detaches: survives the terminal; opens the dashboard
pinnforge ui                           # the live dashboard: scores, fields, agent activity
pinnforge run status heat2d_1          # or attach / log / stop — see running-pinnforge-blocks
```

## Two rules that matter more than they look

**Pin an exact model id, never an alias** — `setup` does this. `opus` meant one
model for a year and another after; two runs a generation apart both record
"opus" and quietly stop being comparable.

**Never write into a live `runs/<id>/`.** It is a block's workspace, and the
orchestrator hashes everything outside that workspace around each dispatch; an
edit shows up as an integrity violation and taints the run.
