"""eval.py — PINNForge's single evaluation tool.

One full evaluation = train a candidate PINN on a GPU (TRAIN_TIME-capped)
+ score rRMSE against the reference field + append a record to the
block's ``evals.jsonl`` + save the trained params next to the candidate.

Usage (always via the project venv, from the project root):

    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py [--gpu 0] [--seed 0]
    .venv/bin/python task/eval.py blocks/bXX/<candidate>.py --smoke
    .venv/bin/python task/eval.py blocks/bXX/<script>.py --diag [--gpu 0]

* The candidate must live under ``blocks/bNN/``. Its block is inferred
  from the path; the budget (``FORGE_WALL_BUDGET``, default 3000
  wall-seconds of GPU runs per block) is enforced via a file-locked
  seconds counter (``blocks/bNN/.budget``, seeded from the non-smoke
  records in ``evals.jsonl``), so concurrent runs cannot overrun it.
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
  charged by the second, wall-capped at 300 s like a training run. No
  frozen-header check, no scoring, no params; the record carries
  ``"diag": true`` and ``rRMSE: null``.
* Candidate contract (see task/baseline.py): module-level
  ``train(rng, eval_callback=None) -> (params, step_count)`` and
  ``predict_fn(params, X) -> {"u"}``; the frozen PDE-CONSTANTS
  header byte-identical to baseline.py, warmup step before the clock.
* GPU discipline: a per-GPU file lock serializes training runs on the
  same GPU id (a second invocation blocks until the first finishes).
  Two concurrent evals are fine with ``--gpu 0`` and ``--gpu 1``.
* Output: one JSON record on the last stdout line (same object that is
  appended to ``evals.jsonl``). Exit 0 whenever a record was written —
  including failed runs, which score ``rRMSE: null`` / ``error`` set.

Internal: re-invokes itself as ``--worker-train`` / ``--worker-eval``
subprocesses so crashes, OOM and hangs kill only the child. jax is
imported in workers only.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent      # task/
ROOT = HERE.parent                          # project root
REF_CSV = HERE / "ref_data.csv"

BUDGET = float(os.environ.get("FORGE_WALL_BUDGET", "3000"))  # wall-seconds of GPU runs per block
TRAIN_TIME_LITERAL = "TRAIN_TIME = 300.0"
SMOKE_TRAIN_TIME = "TRAIN_TIME = 10.0 "  # same length keeps line layout sane
TRAIN_WALL_S = 300.0    # hard wall on any GPU run (train or --diag) —
                        # the task's whole-process budget (= TRAIN_TIME):
                        # imports, JIT compile, training and param save;
                        # the FORGE_T0 env var gives candidates the anchor
SMOKE_WALL_S = 420.0    # CPU compile of nested jacfwd can be slow
EVAL_WALL_S = 120.0


# ────────────────────────── worker: train ──────────────────────────

def worker_train(code_path: str, seed: str, params_out: str) -> int:
    """Child process: exec candidate, run train(), pickle host params."""
    try:
        import jax

        code = Path(code_path).read_text(encoding="utf-8")
        import types
        mod = types.ModuleType("_forge_candidate__")
        sys.modules["_forge_candidate__"] = mod
        exec(compile(code, "<candidate>", "exec"), mod.__dict__)

        params, step_count = mod.__dict__["train"](jax.random.PRNGKey(int(seed)), None)
        np_params = jax.tree_util.tree_map(jax.device_get, params)
        with open(params_out, "wb") as f:
            pickle.dump(np_params, f)
        print(json.dumps({
            "step_count": int(step_count),
            "device_platform": jax.devices()[0].platform,
        }))
        return 0
    except BaseException:
        print(json.dumps({"error": traceback.format_exc()[-2000:]}))
        return 1


# ────────────────────────── worker: eval ───────────────────────────

_PICKLE_ALLOWED = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"),
    ("builtins", "bytes"), ("builtins", "bytearray"), ("builtins", "object"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("flax.core.frozen_dict", "FrozenDict"),
    *(("numpy", n) for n in (
        "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool_",
    )),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _PICKLE_ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def worker_eval(code_path: str, params_path: str) -> int:
    """Child process: restricted-unpickle params, exec candidate,
    predict on the reference grid, print rRMSE/MSE (KS-chaotic protocol:
    relative L2 on u over the t<=0.4 grid, domain-edge predictions
    (t=0 IC and x/t box edges) overridden by ground truth first, so the
    score is interior-dominated)."""
    try:
        import numpy as np
        import jax.numpy as jnp

        with open(params_path, "rb") as f:
            params = _RestrictedUnpickler(io.BytesIO(f.read())).load()

        code = Path(code_path).read_text(encoding="utf-8")
        import types
        mod = types.ModuleType("_forge_eval_candidate__")
        sys.modules["_forge_eval_candidate__"] = mod
        exec(compile(code, "<candidate>", "exec"), mod.__dict__)
        predict_fn = mod.__dict__["predict_fn"]

        csv = np.loadtxt(REF_CSV, delimiter=",", skiprows=1, dtype=np.float64)
        data_X, gt_u = csv[:, 0:2], csv[:, 2:3]   # columns: t, x, u
        bc_mask = (
            np.isclose(data_X[:, 0], data_X[:, 0].min())
            | np.isclose(data_X[:, 0], data_X[:, 0].max())
            | np.isclose(data_X[:, 1], data_X[:, 1].min())
            | np.isclose(data_X[:, 1], data_X[:, 1].max())
        )

        preds = predict_fn(params, jnp.asarray(data_X.astype(np.float32)))
        u_pred = np.asarray(preds["u"], dtype=np.float64).reshape(-1, 1)

        N = data_X.shape[0]
        if u_pred.shape != (N, 1):
            print(json.dumps({"error": f"predict_fn returned bad shape {u_pred.shape}"}))
            return 1
        if not np.all(np.isfinite(u_pred)):
            print(json.dumps({"rRMSE": None, "MSE": None, "error": "non-finite predictions"}))
            return 0

        u_pred[bc_mask] = gt_u[bc_mask]
        diff = u_pred - gt_u
        rRMSE = float(np.linalg.norm(diff) / np.linalg.norm(gt_u))
        MSE = float(np.mean(diff ** 2))
        print(json.dumps({"rRMSE": rRMSE, "MSE": MSE}))
        return 0
    except BaseException:
        print(json.dumps({"error": traceback.format_exc()[-2000:]}))
        return 1


# ────────────────────────── orchestrating parent ───────────────────

def _log_wall_spent(log_path: Path) -> float:
    """Wall-seconds already recorded in evals.jsonl (non-smoke records)."""
    spent = 0.0
    if log_path.is_file():
        for ln in log_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not r.get("smoke"):
                spent += r.get("wall_s") or r.get("train_s") or 0.0
    return spent


def _budget_mutate(block_dir: Path, fn) -> float:
    """File-locked read-modify-write of the block's wall-seconds counter."""
    block_dir.mkdir(parents=True, exist_ok=True)
    with open(block_dir / ".budget", "a+", encoding="utf-8") as cf:
        fcntl.flock(cf, fcntl.LOCK_EX)
        cf.seek(0)
        txt = cf.read().strip()
        try:
            spent = float(txt)
        except ValueError:
            spent = 0.0
        new = fn(spent)
        cf.seek(0)
        cf.truncate()
        cf.write(f"{new:.1f}")
        return new


def _last_json_line(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def _run_worker(mode: str, argv: list[str], env: dict, wall: float) -> tuple[dict, float]:
    t0 = time.time()
    env = {**env, "FORGE_T0": str(t0)}   # wall anchor for the candidate's deadline
    try:
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, *argv],
            capture_output=True, text=True, timeout=wall, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"killed at the {wall:.0f}s wall (hung or runaway process)"}, time.time() - t0
    payload = _last_json_line(r.stdout)
    if r.returncode != 0 and "error" not in payload:
        payload = {"error": (r.stderr[-2000:] or f"exit code {r.returncode}")}
    return payload, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="path to a candidate .py under blocks/bNN/")
    ap.add_argument("--gpu", default="0", help="GPU id to train on (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0)")
    ap.add_argument("--smoke", action="store_true", help="CPU dress rehearsal: TRAIN_TIME→10s, free")
    ap.add_argument("--diag", action="store_true",
                    help="run an arbitrary diagnostic script on GPU, charged by the second")
    args = ap.parse_args()
    if args.smoke and args.diag:
        print("error: --smoke and --diag are mutually exclusive", file=sys.stderr)
        return 2

    cand = Path(args.candidate).resolve()
    if not cand.is_file():
        print(f"error: no such file: {cand}", file=sys.stderr)
        return 2

    # block inference: candidate must live under blocks/bNN/
    try:
        rel = cand.relative_to(ROOT / "blocks")
        block = rel.parts[0]
        assert block.startswith("b") and block[1:].isdigit()
    except (ValueError, AssertionError, IndexError):
        print(f"error: candidate must live under {ROOT}/blocks/bNN/ — got {cand}", file=sys.stderr)
        return 2
    block_dir = ROOT / "blocks" / block
    log_path = block_dir / "evals.jsonl"

    source = cand.read_text(encoding="utf-8")
    if not args.diag and (source.count(TRAIN_TIME_LITERAL) != 1 or "FORGE_EVAL_TOKEN" not in source):
        print(f"error: candidate must contain the frozen-header literal "
              f"`{TRAIN_TIME_LITERAL}` exactly once (found {source.count(TRAIN_TIME_LITERAL)}) "
              f"and the eval-gate block (`FORGE_EVAL_TOKEN`). "
              f"Keep baseline.py's PDE-CONSTANTS header byte-identical.", file=sys.stderr)
        return 2

    # budget: pre-charge the wall cap under the file lock (evals.jsonl
    # lags a running eval by its whole training time, so concurrent runs
    # settle on the counter), then reconcile to actual seconds afterwards
    est = TRAIN_WALL_S
    if not args.smoke:
        log_spent = _log_wall_spent(log_path)
        refused = []

        def _claim(spent: float) -> float:
            spent = max(spent, log_spent)
            if spent >= BUDGET:
                refused.append(spent)
                return spent
            return spent + est

        _budget_mutate(block_dir, _claim)
        if refused:
            print(f"error: block {block} has spent {refused[0]:.0f}/{BUDGET:.0f} wall-seconds — "
                  f"budget exhausted. Write your summary (kb/kb2/{block}.md) and finish.",
                  file=sys.stderr)
            return 2

    if args.diag:
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["FORGE_EVAL_TOKEN"] = "task/eval.py"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        lock_file = open(f"/tmp/pinnforge_gpu{args.gpu}.lock", "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            t0 = time.time()
            env["FORGE_T0"] = str(t0)
            try:
                r = subprocess.run([sys.executable, str(cand)], capture_output=True,
                                   text=True, timeout=TRAIN_WALL_S, env=env, check=False)
                out, err = r.stdout, r.stderr
                error = None if r.returncode == 0 else f"exit code {r.returncode}"
            except subprocess.TimeoutExpired as e:
                out, err = (e.stdout or ""), (e.stderr or "")
                error = f"killed at the {TRAIN_WALL_S:.0f}s wall (hung or runaway process)"
            wall_s = time.time() - t0
        finally:
            lock_file.close()
        spent = _budget_mutate(block_dir, lambda s: max(0.0, s - est + wall_s))
        if out:
            print(out[-8000:])
        if err:
            print(err[-3000:], file=sys.stderr)
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "block": block,
            "candidate": str(cand.relative_to(ROOT)),
            "seed": None, "gpu": args.gpu,
            "smoke": False, "diag": True,
            "rRMSE": None, "MSE": None, "step_count": None,
            "train_s": None, "wall_s": round(wall_s, 1),
            "params": None, "error": error,
            "wall_budget": f"{spent:.0f}/{BUDGET:.0f}",
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        return 0

    tmpdir = Path(tempfile.mkdtemp(prefix="forge_eval_"))
    lock_file = None
    try:
        params_tmp = tmpdir / "params.pkl"
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # frozen-header eval gate: without this token a candidate's train()
        # self-limits to CPU/10 s, so eval.py is the only full-fidelity path
        env["FORGE_EVAL_TOKEN"] = "task/eval.py"
        if args.smoke:
            env["JAX_PLATFORMS"] = "cpu"
            env.pop("CUDA_VISIBLE_DEVICES", None)
            code_for_train = tmpdir / "candidate_smoke.py"
            code_for_train.write_text(source.replace(TRAIN_TIME_LITERAL, SMOKE_TRAIN_TIME),
                                      encoding="utf-8")
            wall = SMOKE_WALL_S
        else:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
            code_for_train = cand
            wall = TRAIN_WALL_S
            # blocking per-GPU lock: concurrent evals on the same GPU serialize
            lock_file = open(f"/tmp/pinnforge_gpu{args.gpu}.lock", "w")
            fcntl.flock(lock_file, fcntl.LOCK_EX)

        train_out, train_s = _run_worker("--worker-train",
                                         [str(code_for_train), str(args.seed), str(params_tmp)],
                                         env, wall)
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

        if args.smoke:
            spent = _budget_mutate(block_dir, lambda s: s)   # read-only peek
        else:
            spent = _budget_mutate(block_dir, lambda s: max(0.0, s - est + train_s))

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "block": block,
            "candidate": str(cand.relative_to(ROOT)),
            "seed": args.seed,
            "gpu": None if args.smoke else args.gpu,
            "smoke": bool(args.smoke),
            "rRMSE": None, "MSE": None,
            "step_count": train_out.get("step_count"),
            "train_s": round(train_s, 1),
            "wall_s": 0.0 if args.smoke else round(train_s, 1),
            "params": None,
            "error": train_out.get("error"),
        }

        if "error" not in train_out and params_tmp.is_file():
            if not args.smoke:
                suffix = f"_s{args.seed}" if args.seed != 0 else ""
                params_final = cand.with_name(cand.stem + suffix + ".pkl")
                params_final.write_bytes(params_tmp.read_bytes())
                record["params"] = str(params_final.relative_to(ROOT))
            eval_env = os.environ.copy()
            eval_env["JAX_PLATFORMS"] = "cpu"
            eval_out, _ = _run_worker("--worker-eval", [str(cand), str(params_tmp)],
                                      eval_env, EVAL_WALL_S)
            record["rRMSE"] = eval_out.get("rRMSE")
            record["MSE"] = eval_out.get("MSE")
            if "error" in eval_out:
                record["error"] = eval_out["error"]

        record["wall_budget"] = f"{spent:.0f}/{BUDGET:.0f}"

        block_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        return 0
    finally:
        if lock_file is not None:
            lock_file.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker-train":
        sys.exit(worker_train(*sys.argv[2:5]))
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker-eval":
        sys.exit(worker_eval(*sys.argv[2:4]))
    sys.exit(main())
