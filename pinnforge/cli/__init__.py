"""`pinnforge` — the command line.

Everything the framework does is reachable from here, and nothing here needs
a model to decide anything. That is the point: the orchestration is a
program, and the agents are the workers it starts.
"""

from __future__ import annotations

import argparse
import os
import sys

from pinnforge import __version__, paths
from pinnforge.cli import agents_cmd, kb_cmd, run_cmd, task_cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinnforge",
        description="Knowledge-centric multi-agent system for autonomous PINN design.",
    )
    p.add_argument("--version", action="version", version=f"pinnforge {__version__}")
    p.add_argument("--root", default=None, help="project root (default: auto-detect)")
    sub = p.add_subparsers(dest="command", required=True)

    task_cmd.register(sub)
    run_cmd.register(sub)
    kb_cmd.register(sub)
    agents_cmd.register(sub)

    info = sub.add_parser("info", help="show resolved paths and library contents")
    info.set_defaults(func=cmd_info)

    scored = sub.add_parser(
        "scored", help="run the score service — the only reader of the reference"
    )
    scored.add_argument("run_id", help="run whose queue to serve")
    scored.add_argument(
        "--private",
        default=None,
        help="directory holding one <task>/ subdirectory per task's scoring "
        "truth; chown it to this service's user to make the isolation an OS "
        "fact. A directory rather than a filename, so a task renaming its "
        "reference does not silently fall outside the rule",
    )
    scored.add_argument("--once", action="store_true", help="drain the queue and exit")
    scored.set_defaults(func=cmd_scored)
    return p


def cmd_scored(args) -> int:
    import logging

    from pinnforge import scoring

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    root = paths.project_root()
    run = paths.run_root(args.run_id, root)
    if not run.is_dir():
        print(f"no run at {run}", file=sys.stderr)
        return 2
    private = args.private or os.environ.get("FORGE_PRIVATE_DIR")
    print(f"serving scores for {run}" + (f" (private data: {private})" if private else ""))
    try:
        scoring.serve(root, run, private_dir=private, once=args.once)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_info(args) -> int:
    root = paths.project_root()
    print(f"root    {root}")
    print(f"tasks   {paths.tasks_dir(root)}")
    print(f"runs    {paths.runs_dir(root)}")
    print(f"kb1     {paths.kb1_dir(root)}")
    print(f"python  {paths.venv_python(root)}")
    print()
    from pinnforge.task import registry

    tasks = registry.list_tasks(root)
    print(f"{len(tasks)} task(s): {', '.join(t.name for t in tasks) or '—'}")
    runs = paths.list_runs(root)
    print(f"{len(runs)} run(s): {', '.join(runs) or '—'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["PINNFORGE_ROOT"] = args.root
    try:
        return args.func(args)
    except KeyError as e:
        print(str(e).strip('"'), file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
