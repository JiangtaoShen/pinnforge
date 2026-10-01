"""Observe what a task's `eval.py` does, in a form two versions can be diffed on.

A dumper candidate (header literal intact, eval gate present) records, from
inside each worker, the argv, working directory, `sys.path[0]` and the
environment variables the machinery sets. The parent's record is kept too.
Paths are normalised, so a run in one temporary directory compares equal to
a run in another.

`capture.py` stored what the pre-kit `eval.py` of each task did; the parity
test runs the same observation against today's `eval.py` and compares.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

WATCHED = re.compile(r"^(FORGE_|JAX_|XLA_|CUDA_|PINNFORGE_)")
VOLATILE = {"FORGE_T0"}

DUMPER = '''
{literal}
import json, os, sys
# FORGE_EVAL_TOKEN gate: this candidate only records where it runs.
_DUMP = os.environ["PF_DUMP_DIR"]

def _dump(tag):
    snap = {{
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "path0": sys.path[0],
        "env": {{k: v for k, v in os.environ.items()}},
    }}
    with open(os.path.join(_DUMP, tag + ".json"), "w") as f:
        json.dump(snap, f)

def train(rng, eval_callback=None):
    _dump("train")
    return {{"w": [0.0]}}, 1

def predict_fn(params, X):
    _dump("eval")
    import jax.numpy as jnp
    n = X.shape[0]
    return {{k: jnp.zeros(n) for k in ("u", "v", "p")}}
'''


def _norm(value: str, run: Path, tmp: str | None) -> str:
    out = value.replace(str(run), "<RUN>")
    out = re.sub(r"/tmp/forge_eval_[A-Za-z0-9_]+", "<TMP>", out)
    if tmp:
        out = out.replace(tmp, "<TMP>")
    return out


def observe(task_dir: Path, python: Path, mode: str, root: Path) -> dict:
    """Run one evaluation of the dumper through `task_dir`'s eval.py.

    `mode` is "smoke" or "gpu". Needs a score daemon: the caller serves the
    run's queue (see test_evalkit_parity.py).
    """
    from pinnforge import scoring
    from pinnforge.task import registry

    literal = re.search(r'TRAIN_TIME_LITERAL\s*=\s*"([^"]+)"',
                        (task_dir / "eval.py").read_text(encoding="utf-8")).group(1)
    tmp = Path(tempfile.mkdtemp(prefix="pf-golden-"))
    try:
        run = tmp / f"{task_dir.name}_0"
        run.mkdir()
        registry.install(registry.describe(task_dir), run)
        block = run / "blocks" / "b01"
        block.mkdir(parents=True)
        cand = block / "b01_dump.py"
        cand.write_text(DUMPER.format(literal=literal), encoding="utf-8")
        dump = tmp / "dump"
        dump.mkdir()
        env = {k: v for k, v in os.environ.items() if not WATCHED.match(k)}
        env.update(PF_DUMP_DIR=str(dump), FORGE_WALL_BUDGET="3600", FORGE_TASK=task_dir.name)
        args = [str(python), "task/eval.py", str(cand.relative_to(run))]
        args += ["--smoke"] if mode == "smoke" else ["--gpu", "0"]
        with scoring.serving(root, run):
            proc = subprocess.run(args, cwd=str(run), env=env, capture_output=True, text=True,
                                  timeout=900, check=False)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
        record = json.loads(lines[-1]) if lines else {"_stderr": proc.stderr[-2000:]}
        out = {"record_keys": list(record), "rRMSE": record.get("rRMSE"),
               "error": bool(record.get("error"))}
        for tag in ("train", "eval"):
            f = dump / f"{tag}.json"
            if not f.is_file():
                out[tag] = None
                continue
            snap = json.loads(f.read_text(encoding="utf-8"))
            out[tag] = {
                "argv": [_norm(a, run, str(tmp)) for a in snap["argv"]],
                "cwd": _norm(snap["cwd"], run, str(tmp)),
                "path0": _norm(snap["path0"], run, str(tmp)),
                "env": {k: _norm(v, run, str(tmp)) for k, v in sorted(snap["env"].items())
                        if WATCHED.match(k) and k not in VOLATILE},
            }
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
