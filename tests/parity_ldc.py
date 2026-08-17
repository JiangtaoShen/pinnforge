"""Behavioural parity against the framework this one replaces.

Scoring equivalence is already settled: the archived b00 params, pushed
through the new score service, reproduce the old rRMSE bit for bit. What
this measures is the other half — that *training* a block the new way lands
where training it the old way did.

It cannot be an equality check. Training stops on a wall clock, not a step
count, so a busier machine buys fewer steps and a different set of weights.
The honest question is whether the new framework's b00 falls inside the
spread the old one would have produced, which is why this runs the anchor
several times and reports a distribution rather than a number.

    .venv/bin/python tests/parity_ldc.py [--repeats 3] [--gpu 0]

Not part of the pytest suite: it costs real GPU time and needs an idle
machine to mean anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnforge import paths, scoring
from pinnforge.config import RunConfig
from pinnforge.run import layout
from pinnforge.types import EvalRecord

# What the previous framework measured for this exact baseline, seed 0,
# 150 s wall. Kept here rather than read from the cache so the comparison
# still means something if the cache is ever refreshed.
REFERENCE = {
    "rRMSE": 0.9444145758929638,
    "MSE": 0.04237408124788594,
    "step_count": 133357,
    "train_s": 145.5,
    "source": "PINNForge b00, 2026-07-21, ldc3200_4_0.01604",
}


def measure_once(root: Path, gpu: int, index: int) -> dict:
    """One fresh b00: copy the baseline, evaluate it, read the record back."""
    run_id = f"parity-ldc-{index}"
    run = paths.runs_dir(root) / run_id
    if run.exists():
        shutil.rmtree(run)
    run = layout.create_run(RunConfig(task="ldc"), root=root, run_id=run_id)

    block = run / "blocks" / "b00"
    block.mkdir(parents=True)
    shutil.copy2(run / "task" / "baseline.py", block / "b00_v01.py")

    with scoring.serving(root, run):
        subprocess.run(
            [str(paths.venv_python(root)), "task/eval.py",
             "blocks/b00/b00_v01.py", "--gpu", str(gpu)],
            cwd=str(run), check=False, capture_output=True, text=True,
        )
    recs = EvalRecord.read_all(run / "blocks" / "b00" / "evals.jsonl")
    scored = [r for r in recs if r.scored]
    return scored[-1].raw if scored else {"error": "no scored record"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the parity run dirs")
    args = ap.parse_args()

    root = paths.project_root()
    print(f"reference: {REFERENCE['source']}")
    print(f"  rRMSE {REFERENCE['rRMSE']:.6f}   steps {REFERENCE['step_count']:,}"
          f"   train {REFERENCE['train_s']} s\n")

    rows = []
    for i in range(1, args.repeats + 1):
        rec = measure_once(root, args.gpu, i)
        if rec.get("rRMSE") is None:
            print(f"  run {i}: FAILED — {rec.get('error', 'no score')}")
            continue
        rows.append(rec)
        d_score = rec["rRMSE"] - REFERENCE["rRMSE"]
        d_steps = rec["step_count"] - REFERENCE["step_count"]
        print(f"  run {i}: rRMSE {rec['rRMSE']:.6f} ({d_score:+.6f})   "
              f"steps {rec['step_count']:,} ({d_steps:+,})   "
              f"train {rec['train_s']:.1f} s")

    if not rows:
        print("\nno successful runs — parity cannot be assessed")
        return 1

    scores = [r["rRMSE"] for r in rows]
    steps = [r["step_count"] for r in rows]
    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"\nnew framework: rRMSE {statistics.mean(scores):.6f} ± {spread:.6f} "
          f"(n={len(scores)})   steps {statistics.mean(steps):,.0f}")
    gap = statistics.mean(scores) - REFERENCE["rRMSE"]
    print(f"gap to reference: {gap:+.6f}", end="")
    if spread > 0:
        print(f"  ({abs(gap) / spread:.1f}× the observed spread)")
    else:
        print()

    verdict = abs(gap) <= max(3 * spread, 0.01)
    print("\nVERDICT:", "consistent with the reference"
          if verdict else "OUTSIDE the reference — investigate before publishing")

    if not args.keep:
        for i in range(1, args.repeats + 1):
            shutil.rmtree(paths.runs_dir(root) / f"parity-ldc-{i}", ignore_errors=True)
    detail = root / "parity_ldc.json"
    detail.write_text(
        json.dumps({"reference": REFERENCE, "measured": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"detail: {detail}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
