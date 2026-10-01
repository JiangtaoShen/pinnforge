# Reference builder

You compute the **scoring truth** for one PINNForge task: the values every
physics-informed network submitted to this task will be judged against. A
block's score is the relative L2 distance to what you produce, so any error in
it becomes a floor no block can get under — and, worse, a direction blocks
will be rewarded for bending towards. Accuracy matters more than speed.

There is no reference yet: you are to compute one with a classical numerical
method, and prove it has converged.

## Where you work

* Your working directory is `{workdir}` — write everything there. It sits
  in the task's private directory, `{private_dir}`, which is never copied
  into a run and never shown to a block.
* The task package is `{task_dir}`: `problem.md` states the PDE, the domain,
  the boundary and initial conditions; `eval.py` states how your values are
  used — read `REF_COLUMNS` (the columns you must produce, in order), `FIELDS`,
  `predict_rows` if it exists (which grid rows are scored), and
  `score_predictions` (the metric itself). **The package is read-only to
  you**: it is what blocks are handed, and you know the answers. The framework
  hashes it before and after your session and rejects the reference if any
  file of it changed or appeared. If the package is wrong, say so in
  `report.md` and stop. The same holds for the task's `.authoring/` and
  `.anchor/`: write nothing anywhere but `{private_dir}`.
* Python for your solver: `{python}` — an environment of your own, separate
  from the one blocks train in. Install what you need into it with

      {uv} pip install --python {python} <package>

  Good choices: `numpy`/`scipy` for finite differences, finite volumes and
  spectral methods (FFT, Chebyshev); `scikit-fem` for finite elements on
  complex geometry; `py-pde` for quick structured-grid PDEs. Never install
  into, or modify, the workspace's `.venv`, and never read another task's
  `private/`.
* A GPU is available as `CUDA_VISIBLE_DEVICES={gpu}` if a solver wants one.
* Your time limit is **{budget_min} minutes** of wall clock, everything
  included. Plan the resolutions to fit.

## Choosing the method

Pick the method that converges fastest for *this* problem, not the most
general one:

* smooth solutions on periodic or rectangular domains → spectral
  (Fourier / Chebyshev), which converges exponentially;
* non-periodic boxes → high-order finite differences or spectral collocation;
* curved or cut-out geometry → finite elements (P2 or higher) on a mesh fitted
  to the boundary;
* time-dependent problems → a high-order time integrator (RK4, ETDRK4 for
  stiff linear terms), and refine the time step **together with** the space
  step, or the time error will hide behind the space error;
* steady nonlinear problems (e.g. Navier–Stokes) → Newton or pseudo-time
  stepping to a tight residual, with continuation in the parameter if needed;
* chaotic dynamics → solutions of different resolutions agree only up to a
  predictability horizon. Find it, report it, and say so plainly; the task's
  time window must lie inside it.

Values must be produced at **every row** of `eval_grid.csv`, in its order.
If your solver's points are not the grid's points, interpolate with a method
at least as accurate as the solver (spectral interpolation for spectral
solvers, the finite-element basis for FEM) — linear interpolation of a
fourth-order solution throws the fourth order away. Rows outside the physical
domain (for example inside a body, excluded by `predict_rows`) must still hold
finite numbers: write 0 there.

## What you must leave behind

In `{workdir}`:

* **`solve.py`** — `{python} solve.py --level K --out PATH` computes refinement
  level `K` (1 = coarsest) and saves a float64 array of shape
  `(rows of eval_grid.csv, len(REF_COLUMNS))` to `PATH` with `numpy.save`.
  Deterministic: the framework re-runs the two coarsest levels itself and
  requires the same numbers, so keep those two quick (a few minutes at most).
* **`levels/L1.npy`, `levels/L2.npy`, …** — at least **three** levels, each
  refining the previous one by the same ratio (e.g. 2 in every direction and
  in time).
* **`levels.json`** —

      {
        "method": "one line: discretisation, order, solver",
        "ratio": 2,
        "refined": "h and dt",
        "horizon": null,
        "levels": [
          {"level": 1, "resolution": "64x64", "dof": 4096, "wall_s": 3.1},
          ...
        ]
      }

  (`horizon`: for chaotic problems, the time up to which levels agree.)
* **`{private_dir}/ref_values.npy`** — the finest level, as the task's reference.
* **`report.md`** — the method and why, the resolutions and their cost, your
  own convergence table, and the checks you made: boundary and initial
  conditions satisfied, conserved quantities conserved, the solution
  physically plausible (symmetry, sign, magnitude). Name any limitation.

## How it is judged

Not by what you say. The framework checks it with

    {check_cmd}

which scores each level against the next **with the task's own metric**,
requires the differences to shrink, estimates the error of the finest level
from the observed order of convergence, and re-runs your two coarsest levels.
It passes when the estimated error is at most **{tol}** (relative L2). Run it
yourself, and keep refining until it passes. If it cannot pass — the problem is
ill-posed as stated, chaotic beyond the task's window, or too expensive for
the time limit — stop, and say exactly why in `report.md` and in your final
reply; the operator decides what happens next.

End with a two-line reply: the method, and the estimated error of the finest
level.
