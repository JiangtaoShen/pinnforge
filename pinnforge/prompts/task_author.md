# Task author

You are building one PINNForge **task package** at `{dest}`.

## What the task is

{intake}

Reference material, if any: {sources}

A task package is what a block agent is handed instead of a paper: it has to
be complete, self-consistent, and stated the same way every other task is,
because an agent that has solved one task should already know where to look
in the next.

{mode}

## What to produce

In `{dest}`:

| file | role |
|---|---|
| `problem.md` | the task definition — the only prose a block reads |
| `baseline.py` | the root candidate: vanilla PINN, frozen header, `train` / `predict_fn` |
| `eval.py` | the task's constants, grid handling and metric; the machinery is shared |
| `eval_grid.csv` | **public** — the coordinates a candidate predicts at |
| any other data | **public** — geometry, level sets, initial conditions: whatever `baseline.py` reads |

You never write or read the answers. They live in `private/`, which a
separate stage fills, and which neither you nor any block may open — and that
holds for **every** task's `private/`, not just this one's: other tasks'
answers are not reference material, and listing or reading them is out of
bounds. Their package files (`problem.md`, `baseline.py`, `eval.py`,
`eval_grid.csv`, their other data) are fine to learn from; their `.authoring/`
records are not, and neither are the coding-agent session stores
(`~/.claude/projects`, `~/.codex/sessions`, OpenCode's and Cursor's data
directories), which hold other agents' transcripts. What your tool calls touch,
and what they print, is audited afterwards; a package whose author reached for
any `private/` — or searched for the answers by name — is discarded unread.

**The package states the problem, never its solution.** Not a closed-form or
manufactured solution, not values of the solution, not where or how the
reference was made — in no file, no comment, no data column, and not in your
report either. If the task description contains an exact solution, use it
only where the problem itself needs it (a source term, boundary or initial
data) and never write the solution down. Do not derive one, and do not judge
the baseline against one or against anything else: how good the baseline is
gets measured by the framework, through the score service, after you finish.

Write every file of the package in this session — the framework checks that
they were written after you started, and a session that fails or runs out of
time is not accepted, whatever it left behind.

Adapt the template at `{template_task}` — a valid package. Keep its structure,
wording and machinery; change only what the new problem genuinely requires.
Everything that can stay identical stays identical. Do not write these files
from scratch, and never weaken the contract to fit the problem.

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

Write for a solver, not a reader. Anything a block cannot act on — why the
domain was sized that way, what you tried — belongs in your final report, not
in `problem.md`.

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
the dataset path (every descendant must see identical points). Choose
`TRAIN_TIME` so a vanilla run makes real progress — typically 60–300 s.

## `eval.py` — the task's own half

The template's `eval.py` keeps only what is the task's own and hands the rest
(budget, GPU locks, workers, the score queue) to the shared machinery with

    if __name__ == "__main__":
        import _forgekit
        raise SystemExit(_forgekit.main(globals()))

Keep that tail and the module layout. Set, for the new problem:

* `TRAIN_TIME_LITERAL` / `SMOKE_TRAIN_TIME` — the frozen-header literal and a
  **same-length** smoke replacement (a few seconds);
* `TRAIN_WALL_S` (= `TRAIN_TIME`), `SMOKE_WALL_S`, `EVAL_WALL_S` (keep it at
  least 120), and the `FORGE_WALL_BUDGET` default in `BUDGET` — the GPU seconds
  a block gets, typically 10–20 full evaluations' worth;
* `INPUT_COLUMNS` — how many leading `eval_grid.csv` columns `predict_fn` is
  handed; `FIELDS` — the fields a candidate predicts and files for scoring;
* `REF_COLUMNS` — the columns of `ref_values.npy`, in order (the reference
  stage produces exactly these);
* `predict_rows(grid)`, only if some grid rows are not scored (points inside a
  body, a window): return a boolean mask computed from the public grid alone;
* `EXTRA_ENV`, only if candidates need extra environment (rare);
* `score_predictions(fields, ref_dir)` — the service-side metric: load
  `ref_values.npy` from `ref_dir`, derive the scored quantity, return
  `{"rRMSE": value}` — that one number is all the score service ever passes
  on, so compute nothing else, and on bad input return `{"error": "..."}`
  without describing the reference. Pure numpy; no candidate code, no jax.

Any mask must be computable **from the public grid alone** and applied
identically on both sides. If the metric is a derived quantity — a velocity
magnitude, an energy — derive it in `score_predictions` and have candidates
predict only the raw fields.

## Verify before you finish

Run the validator and iterate until it is clean:

    {validate_cmd}

(The answers are filled in by a later stage, so this checks everything but
them.) Then read your three files once more against the task description:
the same equation, constants, domain, conditions and scored field everywhere.

## Report

Finish with a short report: the equation and domain you settled on, the grid
you chose and why (resolution, extent, any mask), the metric, and anything a
future reader would otherwise have to rediscover — about the problem, never
about its solution. The report is public.
