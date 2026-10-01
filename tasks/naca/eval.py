"""eval.py — PINNForge's single evaluation tool.

One full evaluation = train a candidate PINN on a GPU (TRAIN_TIME-capped)
+ score rRMSE against the reference field + append a record to the
block's ``evals.jsonl`` + save the trained params next to the candidate.

Usage (always via the project venv, from the project root):

    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py [--gpu 0] [--seed 0]
    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py --smoke
    .venv/bin/python task/eval.py blocks/bXX/<script>.py --diag [--gpu 0]

* The candidate must live under ``blocks/bNN/``. Its block is inferred
  from the path; the budget (``FORGE_WALL_BUDGET``, fixed 3600
  wall-seconds of GPU runs per block) is enforced via a file-locked
  seconds counter (``blocks/bNN/.budget``, seeded from the non-smoke
  records in ``evals.jsonl``), so concurrent runs cannot overrun it.
  Refusal → exit code 2, nothing recorded.
* Every non-smoke invocation is charged its actual wall time,
  **crash or not** — smoke-test first.
* ``--smoke``: CPU-only dress rehearsal. The frozen-header literal
  ``TRAIN_TIME = 180.0`` is patched to 10 s in a temp copy, training and
  scoring run end-to-end, the record is written with ``"smoke": true``,
  no budget charged, no params saved. Catches contract/shape/NaN bugs
  for free.
* ``--diag``: run an arbitrary block-owned diagnostic script at full
  fidelity (the eval-gate token is set, one GPU visible, GPU lock held),
  charged by the second, wall-capped at 180 s like a training run. No
  frozen-header check, no scoring, no params; the record carries
  ``"diag": true`` and ``rRMSE: null``.
* Candidate contract (see task/baseline.py): module-level
  ``train(rng, eval_callback=None) -> (params, step_count)`` and
  ``predict_fn(params, X) -> {"u","v","p"}``; the frozen PDE-CONSTANTS
  header byte-identical to baseline.py, warmup step before the clock.
* GPU discipline: a per-GPU file lock serializes training runs on the
  same GPU id (a second invocation blocks until the first finishes).
  Two concurrent evals are fine with ``--gpu 0`` and ``--gpu 1``.
* Output: one JSON record on the last stdout line (same object that is
  appended to ``evals.jsonl``). Exit 0 whenever a record was written —
  including failed runs, which score ``rRMSE: null`` / ``error`` set.

The machinery (budget, locks, workers, the score-queue client) is shared by
every task and lives in ``task/_forgekit.py``; this file holds only what is
this task's own: its constants, its grid, and its metric.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent      # task/
GRID_CSV = HERE / "eval_grid.csv"   # x, y, phi — public: where to predict
REF_VALUES = "ref_values.npy"       # u, v, p — the answers; service side only

# Scoring region (Scale-PINN protocol): fluid points cropped near the airfoil.
WIN_X, WIN_Y = (-0.2, 2.0), (-0.25, 0.25)


def _scored_grid(grid):
    """Rows that count, from the public grid alone. Both sides must agree."""
    x, y, phi = grid[:, 0], grid[:, 1], grid[:, 2]
    return (phi <= 0.0) & (x >= WIN_X[0]) & (x <= WIN_X[1]) & (y >= WIN_Y[0]) & (y <= WIN_Y[1])


INPUT_COLUMNS = 2    # predict_fn is handed (x, y)
FIELDS = ("u", "v")   # what is filed for scoring
predict_rows = _scored_grid   # predict only the rows that are scored


def score_predictions(fields, ref_dir):
    """Service side: compare predictions against the reference.

    Runs inside the score daemon, which never executes candidate code — so
    the reference may sit where the block's processes cannot read it. Takes
    the predicted fields and the directory holding ``ref_values.npy``.
    """
    import numpy as np

    grid = np.loadtxt(GRID_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    values = np.load(Path(ref_dir) / REF_VALUES, allow_pickle=False)
    mask = _scored_grid(grid)
    gt_V = np.sqrt(values[mask, 0] ** 2 + values[mask, 1] ** 2)

    u_pred = np.asarray(fields["u"], dtype=np.float64).reshape(-1)
    v_pred = np.asarray(fields["v"], dtype=np.float64).reshape(-1)
    if u_pred.shape != gt_V.shape or v_pred.shape != gt_V.shape:
        return {"error": f"prediction shape {u_pred.shape} != reference {gt_V.shape}"}
    V = np.sqrt(u_pred ** 2 + v_pred ** 2)
    if not np.all(np.isfinite(V)):
        return {"rRMSE": None, "MSE": None, "error": "non-finite predictions"}
    diff = V - gt_V
    return {"rRMSE": float(np.linalg.norm(diff) / np.linalg.norm(gt_V)),
            "MSE": float(np.mean(diff ** 2)), "n_scored": int(gt_V.size)}


BUDGET = float(os.environ.get("FORGE_WALL_BUDGET", "3600"))  # per-block GPU wall budget — fixed 3600 s, same for every task
TRAIN_TIME_LITERAL = "TRAIN_TIME = 180.0"
SMOKE_TRAIN_TIME = "TRAIN_TIME = 10.0 "  # same length keeps line layout sane
TRAIN_WALL_S = 180.0    # hard wall on any GPU run (train or --diag) —
                        # the task's whole-process budget (= TRAIN_TIME):
                        # imports, JIT compile, training and param save;
                        # the FORGE_T0 env var gives candidates the anchor
SMOKE_WALL_S = 420.0    # CPU compile of nested jacfwd can be slow
EVAL_WALL_S = 120.0


if __name__ == "__main__":
    import _forgekit

    raise SystemExit(_forgekit.main(globals()))
