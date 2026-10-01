"""Pack the benchmark tasks for a release, and write the catalog that points at them.

    uv run python scripts/build_examples.py [--tag examples-v1] [--out dist/examples]

Each task in `tasks/` with a `private/` reference becomes `<name>.tar.gz` — its
public files and its answers, never its machine-specific anchor cache — and
`pinnforge/examples/catalog.json` records each archive's URL under the release
tag, its size and its sha256. Upload *exactly these files* to that release:
a gzip carries its build time, so a rebuilt archive has a new sha256 and needs
the catalog rewritten with it.

    gh release create examples-v1 dist/examples/*.tar.gz --title "Benchmark tasks"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinnforge.task import examples

REPO = Path(__file__).resolve().parents[1]
BASE = "https://github.com/JiangtaoShen/pinnforge/releases/download"
NAMES = ("ldc", "ks", "cyl", "naca")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="examples-v1")
    ap.add_argument("--out", default=str(REPO / "dist" / "examples"))
    args = ap.parse_args()
    entries = []
    for name in NAMES:
        task = REPO / "tasks" / name
        if not (task / "private").is_dir():
            print(f"skip {name}: no private/ reference here")
            continue
        title = (task / "problem.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        archive, sha = examples.pack(task, Path(args.out))
        entries.append({"name": name, "title": title, "url": f"{BASE}/{args.tag}/{archive.name}",
                        "size": archive.stat().st_size, "sha256": sha})
        print(f"{name}: {archive} ({archive.stat().st_size / 1e6:.1f} MB) {sha[:12]}…")
    catalog = {"tag": args.tag, "examples": entries}
    examples.CATALOG.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"catalog: {examples.CATALOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
