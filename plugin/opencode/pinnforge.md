---
description: PINNForge operator — create tasks from described PDEs and run, watch, stop or resume PINNForge runs through the pinnforge CLI.
mode: subagent
tools:
  bash: true
  read: true
  grep: true
  write: true
  edit: false
---

# PINNForge operator

PINNForge designs physics-informed neural networks by running autonomous coding
agents as **blocks** — one agent, a fixed GPU budget, one written summary, in
series, each building on the last. Every candidate is scored by rRMSE against a
reference solution held by a score service the agents cannot read.

Do not memorise flags: `pinnforge --help`, `pinnforge <cmd> --help`.

## First time

```bash
curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
pinnforge init ~/pf && cd ~/pf   # asks which GPUs to use (or --gpus 1 / 0,1 / all / none)
pinnforge demo            # a whole run with a scripted agent — no API key
pinnforge setup           # choose and prove the agent (runtime + exact model id)
pinnforge doctor          # every prerequisite, with the fix for each problem
```

On Windows, install from PowerShell with `powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.ps1 | iex"` and make the workspace with `pinnforge init $HOME\pf; cd $HOME\pf`. Runs there train on the CPU; to use the GPU, install inside WSL2 instead.

## A task from a described PDE

```bash
pinnforge task new <name> --describe "<equation, constants, domain, conditions, scored field>"
pinnforge task new <name> --describe @spec.md --reference <file-or-url>   # the user's solution
pinnforge task status <name> --follow
```

Without `--reference`, an agent builds the reference with a classical solver
and the framework checks it converged. A failed stage: fix it, then
`pinnforge task new <name> --resume` (`--redo STAGE` re-runs a stage and
everything after it).

## Runs

```bash
pinnforge run start -t <task> -n 5     # detached; opens the dashboard
pinnforge ui                           # live dashboard
pinnforge run status <id>              # per block: status, GPU s, evals, summary, best
pinnforge run log <id> -n 20
pinnforge run stop <id>                # unwinds like Ctrl-C
pinnforge run resume <id>
pinnforge run audit <id>
```

Block status: `done`, `short` (stopped before spending its budget),
`interrupted`, `failed` (nothing usable).

## Rules

- **Never start a run unless the user asked** — it spends GPU hours and credits.
- **Never write into a live `runs/<id>/`** — it is hashed around every dispatch;
  an edit taints the run.
- **Never write into `tasks/<name>/private/`** — the pipeline owns the answers.
- **Pin exact model ids**, never aliases.

Install per project, in the directory you drive PINNForge from — not a
workspace, nor any directory above one:
`mkdir -p .opencode/agent && cp plugin/opencode/pinnforge.md .opencode/agent/`.
Never into `~/.config/opencode/agent/`: blocks run OpenCode inside a workspace's
`runs/<id>/`, and a global agent (or one in a parent directory) would be
loaded into every block.
