"""Record what each library task's eval.py does today — run before a migration.

    uv run python tests/golden/capture.py

Writes tests/golden/<task>.json. Needs the repo's .venv (JAX) and one GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import observe  # noqa: E402

from pinnforge import paths  # noqa: E402

if __name__ == "__main__":
    root = paths.project_root()
    python = paths.venv_python(root)
    for task in ("ldc", "ks", "cyl", "naca"):
        out = {m: observe(root / "tasks" / task, python, m, root) for m in ("smoke", "gpu")}
        (HERE / f"{task}.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
        print(task, "captured", {m: (o["rRMSE"], o["error"]) for m, o in out.items()})
