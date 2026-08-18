---
name: creating-a-pinnforge-task
description: Turn a PDE and a reference solution into a PINNForge task package — the three contract files (problem.md, baseline.py, eval.py), the public/private data split, and the three gates that decide whether the task actually loads (validate, smoke, anchor). Use whenever the user wants to add a new PDE case to PINNForge, asks why `task validate` fails, needs the b00 anchor measured or rebuilt, or asks what a task package has to contain.
---

# Creating a PINNForge task

A task is not finished when its files exist. It is finished when it **loads**.

```bash
pinnforge task new burgers --source ~/papers/burgers --prompt "Inviscid Burgers, IC -sin(pi x)"
```

One command, because `task new` drives an authoring agent that adapts an
existing package and then walks the three gates below. `--no-load` stops after
validation. If the run library already knows the name, `task new` recovers the
definition from the newest run instead of authoring a different one, which is
what keeps new runs comparable with old ones.

## The contract

The framework touches a task only through this interface, so any package
honouring it plugs in without framework changes.

- **`problem.md`** — the task definition, and the only prose a block reads.
  Seven fixed sections in a fixed order (Equation, Domain, Boundary conditions,
  Initial condition, Scoring, Environment, Time budget), so an agent on its
  second task already knows where to look. The intro paragraph and the
  Environment body are verbatim across every task.
- **`baseline.py`** — the root candidate. Defines the frozen `PDE CONSTANTS`
  header and the `train(rng, eval_callback=None) -> (params, step_count)` /
  `predict_fn(params, X) -> dict` contract every descendant keeps.
- **`eval.py`** — the single evaluation tool:
  `eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`.
  Enforces `FORGE_WALL_BUDGET` across GPU runs (CPU `--smoke` is free), appends
  one JSON record per run to `blocks/bNN/evals.jsonl` carrying `smoke`, `diag`,
  `wall_s` and `rRMSE`, maintains `.budget`, saves params, and exits 0 whenever a
  record was written. It must also define `score_predictions()` and submit
  predictions rather than scoring inline.
- **Public data** sits in the task directory and is copied into every run.
  The **scoring truth** sits in `private/`, which is never copied: the installer
  takes files only, so secrecy is where a file sits, not what it is named.

## The three gates

```bash
pinnforge task validate burgers            # 29 contract checks
pinnforge task smoke burgers               # free CPU dress rehearsal, end to end
pinnforge task anchor burgers --measure    # the b00 score every block must beat
```

Each gate reads the previous one's evidence.

The smoke passes only on a record carrying a score and no error. `eval.py` exits
0 whenever a record was *written*, failed runs included, so the exit status
proves nothing.

## The anchor is a measurement, not a block

`task anchor` on its own only reports: the cached score, and whether the
contract files still fingerprint to what it was measured against.

**Judge staleness by that fingerprint, never by the score.** The same task
re-measured at the same seed moves anywhere from 0.4% to 17%, because training
stops on a wall clock and the run is not bit-deterministic. A score that moved
does not mean the task changed; a fingerprint that moved does.

Nothing already paid for is paid for twice: `task anchor` and `run start`
harvest a b00 from the newest run whose `task/` fingerprints to today's package
before spending a GPU. `--measure` builds one when none is cached; `--rebuild`
replaces a cached one.

## Before you trust a new task

Check what the anchor actually means. On some cases the trivial field is an
exact solution of the residual, so b00 lands near 1.0 and beating it proves
nothing; on others b00 is worse than a constant field. Compare b00 against the
trivial predictors for the task before reading any block's gain as progress.
