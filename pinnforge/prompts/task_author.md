# Task author

You are building one PINNForge **task package** at `{dest}` from the
reference material at `{sources}`.

{instruction}

A task package is what a block agent is handed instead of a paper: it has to
be complete, self-consistent, and stated the same way every other task is,
because an agent that has solved one task should already know where to look
in the next.

## What to produce

In `{dest}`:

| file | role |
|---|---|
| `problem.md` | the task definition — the only prose a block reads |
| `baseline.py` | the root candidate: vanilla PINN, frozen header, `train` / `predict_fn` |
| `eval.py` | the single evaluation tool: budget, GPU locks, logging, and both halves of the metric |
| `eval_grid.csv` | **public** — the coordinates a candidate predicts at |
| `private/ref_values.npy` | **private** — the answers at those same rows, same order |
| any other data | **public** — geometry, level sets, initial conditions: whatever `baseline.py` reads |

A package is not a fixed set of files. Ship as many public data files as the
physics needs — a level set for a cut-out body, an initial-condition field, a
collocation grid — and name them for what they are. Only two things are
structural: the three contract files must exist, and the answers must sit
under `private/`.

Adapt `{template_task}` — an existing, valid package. Keep its structure,
wording and machinery; change only what the new problem genuinely requires.
Everything that can stay identical stays identical. Do not write these files
from scratch, and never weaken the contract to fit the source material.

## `problem.md` — shape is fixed

Exactly these sections, this order, nothing added, dropped, renamed or
reordered:

    # <ABBREV> — <full name>
    <the bold intro paragraph, verbatim from the template>
    ## Equation
    ## Domain
    ## Boundary conditions
    ## Initial condition
    ## Scoring
    ## Environment
    ## Time budget

Rules that the validator enforces:

- The intro paragraph is **byte-identical** across every task. Copy it.
- `## Environment` is byte-identical too. Copy it.
- `## Time budget` differs only in the number, which must equal `TRAIN_TIME`
  in the frozen header.
- No `###` subsections anywhere.
- `## Scoring` states the relative L2 error (rRMSE) on a named field and ends
  "Lower is better."
- Equation / Domain / Boundary conditions / Initial condition / Scoring are
  terse and factual — one formula, interval or sentence each. No citations,
  no provenance, no commentary on difficulty.

Write for a solver, not a reader. Anything a block cannot act on — how the
reference was generated, why the domain was sized that way, what you tried —
belongs in your report, not in `problem.md`. The exception is a measured
**error floor**: if the reference carries its own discretisation error, say
so and give the number, because a block that does not know it will spend its
budget fitting noise.

## `baseline.py` — vanilla, and honest about it

The baseline is a control, not an attempt. Keep the template's paradigm: a
single plain MLP, tanh, fixed-learning-rate Adam, soft boundary penalties,
strict-interior collocation, no adaptive weighting, no Fourier features, no
hard constraints. Change only the I/O dimensions, the physical constants,
the PDE residual and the dataset/BC/IC construction.

Preserve exactly: the frozen `PDE CONSTANTS` header block, the
`FORGE_EVAL_TOKEN` eval gate, the wall-deadline `train()` loop, the
`train(rng, eval_callback=None) -> (params, step_count)` and
`predict_fn(params, X) -> dict` signatures, and the absence of any RNG in
the dataset path (every descendant must see identical points).

## `eval.py` — machinery untouched, metric split in two

Start from the template's `eval.py` and keep the budget accounting, GPU
locking, worker subprocess and logging **byte-identical**. Change only the
two halves of the metric, the frozen-header literal, and the wall constants.
`TRAIN_TIME_LITERAL` and `SMOKE_TRAIN_TIME` must be the same length, or the
smoke substitution corrupts the line.

The metric is split because the process that executes candidate code must
not be the process that holds the answers:

* **`worker_eval` (agent side)** — execs the candidate, predicts on
  `eval_grid.csv`, writes the predicted fields to a `.npz`, and calls
  `_submit()`. It must never open the reference.
* **`score_predictions(fields, ref_dir)` (service side)** — loads
  `ref_values.npy` from `ref_dir`, derives the scored quantity, returns
  `{"rRMSE", "MSE"}`. Pure numpy, no candidate code, no jax.

Any masking or windowing must be computable **from the public grid alone**,
so put whatever the mask needs into `eval_grid.csv` (a level-set column, a
region flag) and apply the identical mask on both sides. If the metric is a
derived quantity — a velocity magnitude, an energy — derive it in
`score_predictions`, and have the agent side predict only the raw fields.

## Splitting the reference

Take the field you generated, and cut it column-wise:

* coordinates, plus anything a public mask needs → `eval_grid.csv`
* the values being scored → `ref_values.npy`, same rows, same order

The grid and every other public file sit in the task directory; the values go
in `private/`. The installer copies files into a run and never subdirectories,
so what is secret is decided by *where it sits*, not by what it is called — a
task whose answers are named something unexpected still cannot leak them. A
private deployment moves them further still, to a directory owned by another
user.

## Verify before you finish

Run the validator and iterate until it is clean:

    {validate_cmd}

Then prove the package runs end to end with the free CPU smoke path — this
costs no GPU budget and catches contract, shape and NaN bugs:

    {smoke_cmd}

Beyond the validator, check the things it cannot: that the reference field
actually satisfies the boundary conditions `problem.md` declares, and what a
trivial predictor (a constant field) scores under your metric. A task whose
baseline cannot beat a constant is worth knowing about before a run starts,
not after.

## Report

Finish with a short report: the equation and domain you settled on, where
the reference data came from, the measured error floor if any, what a
constant-field predictor scores, and anything about the package a future
reader would otherwise have to rediscover.
