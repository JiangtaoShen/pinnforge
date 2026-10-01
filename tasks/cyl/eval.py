"""eval.py — PINNForge's single evaluation tool (CYLINDER task).

One full evaluation = train a candidate PINN on a GPU (TRAIN_TIME-capped)
+ score rRMSE against the reference field + append a record to the
block's ``evals.jsonl`` + save the trained params next to the candidate.

Usage (always via the project venv, from the project root):

    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py [--gpu 0] [--seed 0]
    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py --smoke
    .venv/bin/python task/eval.py blocks/bXX/<script>.py --diag [--gpu 0]

* The candidate must live under ``blocks/bNN/``. Its block is inferred
  from the path; the budget (``FORGE_WALL_BUDGET``, fixed 7200 wall-
  seconds of GPU runs per block) is enforced via a file-locked seconds
  counter (``blocks/bNN/.budget``, seeded from the non-smoke records in
  ``evals.jsonl``), so concurrent runs cannot overrun it.
  Refusal → exit code 2, nothing recorded.
* Every non-smoke invocation is charged its actual wall time,
  **crash or not** — smoke-test first.
* ``--smoke``: CPU-only dress rehearsal. The frozen-header literal
  ``TRAIN_TIME = 300.0`` is patched to 10 s in a temp copy, training and
  scoring run end-to-end, the record is written with ``"smoke": true``,
  no budget charged, no params saved. Catches contract/shape/NaN bugs
  for free.
* ``--diag``: run an arbitrary block-owned diagnostic script at full
  fidelity (the eval-gate token is set, one GPU visible, GPU lock held),
  charged by the second, wall-capped like a training run. No frozen-
  header check, no scoring, no params; the record carries ``"diag": true``
  and ``rRMSE: null``.
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
GRID_CSV = HERE / "eval_grid.csv"   # t, x, y — public: where to predict
REF_VALUES = "ref_values.npy"       # u, v — the answers; service side only

INPUT_COLUMNS = 3    # predict_fn is handed (t, x, y)
FIELDS = ("u", "v")   # what is filed for scoring
EXTRA_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "FORGE_TASK_DIR": str(HERE),   # candidates locate data by this
}


def score_predictions(fields, ref_dir):
    """Service side: relative L2 on the stacked velocity field [u; v].

    Runs inside the score daemon, which never executes candidate code — so
    the reference may sit where the block's processes cannot read it.
    """
    import numpy as np

    gt = np.load(Path(ref_dir) / REF_VALUES, allow_pickle=False)
    u = np.asarray(fields["u"], dtype=np.float64).reshape(-1, 1)
    v = np.asarray(fields["v"], dtype=np.float64).reshape(-1, 1)
    if u.shape != gt[:, 0:1].shape or v.shape != gt[:, 1:2].shape:
        return {"error": f"prediction shape {u.shape} != reference {gt[:, 0:1].shape}"}
    pred = np.hstack([u, v])
    if not np.all(np.isfinite(pred)):
        return {"rRMSE": None, "MSE": None, "error": "non-finite predictions"}
    diff = pred - gt
    return {"rRMSE": float(np.linalg.norm(diff) / np.linalg.norm(gt)),
            "MSE": float(np.mean(diff ** 2)), "n_scored": int(gt.shape[0])}


BUDGET = float(os.environ.get("FORGE_WALL_BUDGET", "7200"))  # per-block GPU wall budget
TRAIN_TIME_LITERAL = "TRAIN_TIME = 300.0"
SMOKE_TRAIN_TIME = "TRAIN_TIME = 10.0 "   # same length keeps line layout sane
TRAIN_WALL_S = 300.0    # hard wall on any GPU run (train or --diag) —
                        # the task's whole-process budget (= TRAIN_TIME):
                        # imports, JIT compile, training and param save;
                        # the FORGE_T0 env var gives candidates the anchor
SMOKE_WALL_S = 300.0    # CPU compile of nested jacfwd is slow
EVAL_WALL_S = 180.0


if __name__ == "__main__":
    import _forgekit

    raise SystemExit(_forgekit.main(globals()))
