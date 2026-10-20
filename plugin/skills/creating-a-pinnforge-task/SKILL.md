---
name: creating-a-pinnforge-task
description: Turn a described PDE — with or without the user's own reference solution — into a PINNForge task that `run start` accepts. Covers `pinnforge task new` (interactive, `--describe`, `--reference`), the authoring stages (author agent, reference ingest or reference-builder agent, the framework's convergence check, validate, smoke, anchor), `task status` and `--resume`, what a task package contains, and the anchor cache (`--measure`, `--rebuild`, `--rekey`). Use whenever the user wants a new PDE case, has reference data to use, asks why authoring or `task validate` failed, or needs an anchor measured or carried over.
---

# Creating a PINNForge task

A task is finished when it **loads** — when `run start` accepts it — not when
its files exist. `task new` carries it all the way there.

## One command

```bash
pinnforge task new                                   # asks, question by question
pinnforge task new burgers --describe "u_t + u u_x = 0.01/π u_xx on [-1,1] × [0,1],
  u(x,0) = -sin(πx), u(±1,t) = 0; score u"
pinnforge task new cyl2 --describe @spec.md --reference https://…/cylinder.h5
pinnforge task new … --detach                        # continue in the background
pinnforge task status burgers --follow               # stages, live
```

Useful flags: `--source <paper-or-notes>` (repeatable), `--agent <binding>`,
`--ref-budget <minutes>` (default 240), `--ref-tol <x>` (default 1e-3),
`--accept-reference "<reason>"`, `--no-load` (stop before the anchor).

## The stages

| stage | what happens |
|---|---|
| intake | the description is recorded (`tasks/<n>/.authoring/intake.md`) |
| author | an agent writes `problem.md`, `baseline.py`, `eval.py` and the public grid, adapting the packaged demo task — it never sees the answers |
| reference | **user data**: fetched (sha256 recorded) and ingested by an agent into `eval_grid.csv` + `private/ref_values.npy`. **none**: a reference-builder agent solves the PDE classically at ≥3 resolutions in its own environment |
| reference_check | the framework — not the agent — scores each level against the next with the task's own metric, measures the convergence order, estimates the finest level's error, and re-runs the two coarsest levels |
| validate | the task contract |
| smoke | the scoring path end to end on the CPU, free |
| anchor | b00 — the baseline measured once; every round must beat it |
| finalize | released to `run start` |

With user data the reference stage runs *before* the author (the data fixes
the grid). A failed stage keeps its note in `task status`; fix the cause and
`pinnforge task new <name> --resume` continues from there — flags given with
`--resume` (`--ref-budget`, `--ref-tol`, `--accept-reference`, `--agent`…)
override the recorded ones, and `--redo STAGE` re-runs that stage and every
later one (after a failed convergence check: `--resume --redo reference`).
`pinnforge task reference <name> --check` re-runs the check.

Nothing of the reference reaches the package: no values, no closed form, no
word of how it was made or how accurate it is (the estimate is shown to the
operator, not written into `problem.md`). An exact solution the user knows is
supplied as the reference — `--reference solution.py` (or `.txt`) holding the
formula — and kept out of `--describe`, which the author agent reads. If the
author reaches for any `private/`, the author stage fails and its package is
moved to `logs/untrusted-package-N/` in the private home (the author must also
finish cleanly and write the package in that session). If the ingest or
reference agent writes anything outside `private/` — the package, `.authoring/`,
`.anchor/` — it is moved to `logs/reference-guard-N/` and the stage fails.
Their harness session transcripts are moved to `logs/harness-sessions/`
(OpenCode and Cursor sessions are warned about instead). `levels.json` must
declare a refinement `ratio` between 1.5 and 4; an ingested grid's header must
be plain identifiers. `task new` honours `private_dir` / `FORGE_PRIVATE_DIR`.

## What a package contains

- **`problem.md`** — the only prose a round reads; seven fixed sections.
- **`baseline.py`** — the root candidate: a frozen `PDE CONSTANTS` header and
  `train(rng, eval_callback=None) -> (params, step_count)` /
  `predict_fn(params, X) -> dict`.
- **`eval.py`** — the task's constants (`TRAIN_TIME_LITERAL`, walls, `BUDGET`),
  its grid handling (`INPUT_COLUMNS`, `FIELDS`, `REF_COLUMNS`, optional
  `predict_rows`) and its metric `score_predictions(fields, ref_dir)`. The
  machinery (budget, GPU locks, workers, score queue) is shared and vendored
  into each run as `task/_forgekit.py`.
- **`view.json`** — how the dashboard's fixed field window shows this task's
  predictions: grid coordinates on the screen axes, a time slider or a slice,
  and 1-4 panels (an output head, `{"magnitude": [...]}` or `{"abs": ...}`,
  each sequential or diverging). The author writes it; `task validate` checks
  it; it does not affect the anchor.
- **public data** (the grid, geometry, initial conditions) in the task
  directory; **the answers** in `private/`, which never enters a run.

`pinnforge task validate <name>` runs the contract checks by hand
(`--no-reference` skips the answers, for a package still being authored).

## The anchor

`pinnforge task anchor <name>` reports the cached b00 and whether it still
applies — keyed on the package fingerprint (baseline, evaluation, public data),
the reference, and the hardware class. **Judge staleness by that, never by the
score**: the same task re-measured moves 0.4–17%, because training stops on a
wall clock.

- `--measure` builds one when none is cached; `--rebuild` replaces it;
- `--rekey` carries a stale anchor over **without a GPU** when only the
  evaluation's plumbing changed: allowed only if `baseline.py` is unchanged and
  re-scoring the cached params reproduces the recorded rel. L² error exactly.

Before trusting a new task, compare b00 with a trivial predictor: on some
problems the zero field nearly satisfies the residual and beating b00 proves
little.
