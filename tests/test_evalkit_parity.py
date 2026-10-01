"""The shared evaluation machinery behaves exactly as the per-task copies did.

`tests/golden/<task>.json` recorded, from inside each worker of the pre-kit
`eval.py`, what a candidate sees: argv, working directory, `sys.path[0]`, the
environment the machinery sets — plus the record's fields and the score.
These tests observe today's `eval.py` the same way and compare. The one
difference allowed is deliberate: cyl's scoring worker now files its requests
under the block they came from (it used to omit the context, so its per-block
cap was one cap for the whole run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pinnforge import gpus, paths

GOLDEN = Path(__file__).parent / "golden"
sys.path.insert(0, str(GOLDEN))

TASKS = ("ldc", "ks", "cyl", "naca")
CONTEXT = {"FORGE_BLOCK": "b01", "FORGE_CANDIDATE": "blocks/b01/b01_dump.py",
           "FORGE_RUN_ID": "cyl_0", "FORGE_SEED": "0"}


def _expected(task: str, mode: str) -> dict:
    want = json.loads((GOLDEN / f"{task}.json").read_text(encoding="utf-8"))[mode]
    if task == "cyl" and want.get("eval"):
        want["eval"]["env"].update(CONTEXT)
        want["eval"]["env"]["FORGE_SMOKE"] = "1" if mode == "smoke" else "0"
    return want


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["smoke", "gpu"])
@pytest.mark.parametrize("task", TASKS)
def test_eval_matches_the_pre_kit_behaviour(task, mode):
    from harness import observe

    root = paths.project_root()
    if not (root / "tasks" / task / "private").is_dir():
        pytest.skip("needs the task's private reference")
    if mode == "gpu" and not gpus.inventory():
        pytest.skip("needs a GPU")
    got = observe(root / "tasks" / task, paths.venv_python(root), mode, root)
    assert got == _expected(task, mode)
