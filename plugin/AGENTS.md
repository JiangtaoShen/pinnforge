# PINNForge

Operator-facing instructions for driving PINNForge from a coding-agent CLI.

> **Where this file may live.** Inside `plugin/`, or in the project you drive
> PINNForge *from*. **Never at the root of a PINNForge checkout.** Blocks run
> with `runs/<id>/` as their working directory, and several CLIs walk up the
> tree for `AGENTS.md`. A copy at the project root is an ancestor of every run
> directory, so every block reads it and stops being the same block the other
> harnesses run. This was measured, not assumed.

PINNForge designs physics-informed neural networks by running autonomous coding
agents as **blocks**: one agent, a fixed GPU-wall budget, one written summary,
in series, with knowledge compounding through the summaries.

Do not memorise flags. Run `pinnforge --help` or `pinnforge <cmd> --help`.

## Running blocks

```bash
pinnforge run start -t ldc -n 5        # launch five blocks
pinnforge run status ldc_5             # per-block wall, evals, summary, best
pinnforge run log ldc_5 -n 20          # scored evaluations, best first
pinnforge run show ldc_5 b03           # one block's records and summary
pinnforge run resume ldc_5             # after an interruption
pinnforge run audit ldc_5              # did any block alter another's record
pinnforge run stop ldc_5
```

Block status: `done` (summary + budget spent + a scored eval), `short` (usable
work, agent stopped before the budget was spent), `interrupted`, `failed`
(nothing usable).

## Authoring a task

```bash
pinnforge task list
pinnforge task validate <name>         # 29 contract checks
pinnforge task smoke <name>            # free CPU dress rehearsal, end to end
pinnforge task anchor <name>           # the b00 score every block must beat
pinnforge task new <name> --source <dir> --prompt "..."
```

## Harnesses

```bash
pinnforge agents list
pinnforge agents doctor --ping --model <id>
```

`--ping` spends one exchange proving credentials, provider config and model id
work, instead of finding out at the first block. Pin an exact model id, never an
alias.

## Rules

**Never write into a live `runs/<id>/`.** It is a block's workspace and is
hashed before every dispatch; an edit becomes an integrity violation and taints
the run.

**Judge anchor staleness by the fingerprint, never by the score.** The same task
re-measured at the same seed moves 0.4% to 17%.
