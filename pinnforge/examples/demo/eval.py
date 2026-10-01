"""eval.py — PINNForge's single evaluation tool (POISSON demo).

One full evaluation = train a candidate PINN on a GPU (TRAIN_TIME-capped)
+ score rRMSE against the reference field + append a record to the
block's ``evals.jsonl`` + save the trained params next to the candidate.

Usage (always via the project venv, from the run root):

    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py [--gpu 0] [--seed 0]
    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py --smoke
    .venv/bin/python task/eval.py blocks/bXX/<script>.py --diag [--gpu 0]

* The candidate must live under ``blocks/bNN/``. Its block is inferred
  from the path; the budget (``FORGE_WALL_BUDGET``, default 600
  wall-seconds of GPU runs per block) is enforced via a file-locked
  seconds counter (``blocks/bNN/.budget``), so concurrent runs cannot
  overrun it. Refusal → exit code 2, nothing recorded.
* Every non-smoke invocation is charged its actual wall time,
  **crash or not** — smoke-test first.
* ``--smoke``: CPU-only dress rehearsal. The frozen-header literal
  ``TRAIN_TIME = 60.0`` is patched to 5 s in a temp copy, training and
  scoring run end-to-end, the record is written with ``"smoke": true``,
  no budget charged, no params saved.
* ``--diag``: run an arbitrary block-owned diagnostic script at full
  fidelity (eval-gate token set, one GPU visible, GPU lock held),
  charged by the second, wall-capped at 60 s. The record carries
  ``"diag": true`` and ``rRMSE: null``.
* Candidate contract (see task/baseline.py): module-level
  ``train(rng, eval_callback=None) -> (params, step_count)`` and
  ``predict_fn(params, X) -> {"u"}``; the frozen PDE-CONSTANTS header
  byte-identical to baseline.py.
* GPU discipline: a per-GPU file lock serializes training runs on the
  same GPU id. Two concurrent evals are fine on two different GPUs.
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
GRID_CSV = HERE / "eval_grid.csv"           # x, y — public: where to predict
REF_VALUES = "ref_values.npy"               # u — the answers; service side only

INPUT_COLUMNS = 2           # predict_fn is handed (x, y)
FIELDS = ("u",)             # what is filed for scoring
REF_COLUMNS = ("u",)        # the columns of ref_values.npy, in order

BUDGET = float(os.environ.get("FORGE_WALL_BUDGET", "600"))  # wall-seconds of GPU runs per block
TRAIN_TIME_LITERAL = "TRAIN_TIME = 60.0"
SMOKE_TRAIN_TIME = "TRAIN_TIME = 5.0 "  # same length keeps line layout sane
TRAIN_WALL_S = 60.0     # hard wall on any GPU run (train or --diag) = TRAIN_TIME
SMOKE_WALL_S = 300.0    # CPU compile of nested derivatives can be slow
EVAL_WALL_S = 120.0


def score_predictions(fields, ref_dir):
    """Service side: relative L2 on u over the whole reference grid."""
    import numpy as np

    gt = np.load(Path(ref_dir) / REF_VALUES, allow_pickle=False)[:, 0]
    u = np.asarray(fields["u"], dtype=np.float64).reshape(-1)
    if u.shape != gt.shape or not np.all(np.isfinite(u)):
        return {"error": "unscorable predictions"}
    return {"rRMSE": float(np.linalg.norm(u - gt) / np.linalg.norm(gt))}


if __name__ == "__main__":
    import _forgekit

    raise SystemExit(_forgekit.main(globals()))
