"""`pinnforge agents …` — which harnesses are installed, and do they work."""

from __future__ import annotations

import subprocess

from pinnforge.runtime import registry


def cmd_list(args) -> int:
    print(f"{'runtime':<14} {'command':<14} {'default model':<16} installed")
    for row in registry.detect_available():
        mark = row["path"] or "—"
        # A harness that names no default requires one per run; say so rather
        # than printing a blank column that reads like missing information.
        model = row["default_model"] or "(pass --model)"
        print(f"{row['runtime']:<14} {row['command']:<14} {model:<16} {mark}")
    return 0


def cmd_doctor(args) -> int:
    """Check each harness is on PATH and answers `--version`.

    A missing CLI is the single most common reason a run dies on its first
    block, and it costs nothing to find out beforehand.
    """
    rows = registry.detect_available()
    if args.runtime:
        rows = [r for r in rows if r["runtime"] == registry.canonical(args.runtime)]
        if not rows:
            print(f"unknown runtime: {args.runtime}")
            return 2
    failures = 0
    for row in rows:
        name, cmd, path = row["runtime"], row["command"], row["path"]
        if not path:
            print(f"  [MISS] {name:<14} {cmd} not on PATH")
            failures += 1
            continue
        try:
            out = subprocess.run(
                [cmd, "--version"], capture_output=True, text=True, timeout=args.timeout
            )
            version = (out.stdout or out.stderr).strip().splitlines()
            version = version[0] if version else "(no version output)"
            print(f"  [OK]   {name:<14} {version}")
        except subprocess.TimeoutExpired:
            print(f"  [SLOW] {name:<14} {cmd} --version timed out after {args.timeout}s")
            failures += 1
        except OSError as e:
            print(f"  [FAIL] {name:<14} {e}")
            failures += 1
    print(f"{len(rows)} runtime(s), {failures} problem(s)")
    return 0 if failures == 0 else 1


def register(sub) -> None:
    p = sub.add_parser("agents", help="coding-agent harnesses")
    s = p.add_subparsers(dest="agents_cmd", required=True)

    s.add_parser("list", help="list known runtimes").set_defaults(func=cmd_list)

    doc = s.add_parser("doctor", help="check the harness CLIs are usable")
    doc.add_argument("runtime", nargs="?", default=None)
    doc.add_argument("--timeout", type=int, default=30)
    doc.set_defaults(func=cmd_doctor)
