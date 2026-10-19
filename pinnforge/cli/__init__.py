"""`pinnforge` — the command line.

Everything the framework does is reachable from here, and nothing here needs
a model to decide anything. That is the point: the orchestration is a
program, and the agents are the workers it starts.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

from pinnforge import __version__, paths
from pinnforge.cli import (
    agents_cmd,
    chat_cmd,
    demo_cmd,
    doctor_cmd,
    examples_cmd,
    init_cmd,
    kb_cmd,
    run_cmd,
    setup_cmd,
    task_cmd,
    ui_cmd,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinnforge",
        description="Knowledge-centric multi-agent system for autonomous PINN design.",
    )
    p.add_argument("--version", action="version", version=f"pinnforge {__version__}")
    p.add_argument("--root", default=None, help="project root (default: auto-detect)")
    sub = p.add_subparsers(dest="command", required=True)

    init_cmd.register(sub)
    setup_cmd.register(sub)
    doctor_cmd.register(sub)
    demo_cmd.register(sub)
    task_cmd.register(sub)
    run_cmd.register(sub)
    examples_cmd.register(sub)
    kb_cmd.register(sub)
    agents_cmd.register(sub)
    ui_cmd.register(sub)
    chat_cmd.register(sub)

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
    _hide_suppressed(p)
    return p


def _hide_suppressed(parser: argparse.ArgumentParser) -> None:
    """Keep `help=SUPPRESS` subcommands (`run _drive`) out of every help screen.

    argparse lists a suppressed subcommand anyway — in the usage line's
    `{a,b,...}` and as a `==SUPPRESS==` row — so each subcommand group whose
    members hide is given a metavar naming only the visible ones, and the
    hidden rows are dropped. The commands themselves still parse.
    """
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        rows = action._choices_actions
        shown = [a for a in rows if a.help != argparse.SUPPRESS]
        if len(shown) != len(rows):
            action._choices_actions = shown
            action.metavar = "{" + ",".join(a.dest for a in shown) + "}"
        for sub in set(action.choices.values()):
            _hide_suppressed(sub)


def cmd_scored(args) -> int:
    import logging

    from pinnforge import scoring

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    root = paths.project_root()
    run = paths.run_root(args.run_id, root)
    if not run.is_dir():
        print(f"no run at {run}", file=sys.stderr)
        return 2
    from pinnforge import private as private_home

    private = args.private or private_home.configured(root)
    print(f"serving scores for {run}" + (f" (private data: {private})" if private else ""))
    try:
        scoring.serve(root, run, private_dir=private, once=args.once)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_info(args) -> int:
    from pinnforge import bindings, gpus, settings, workspace

    root = paths.project_root()
    kind = "workspace" if paths.is_workspace(root) else "source checkout"
    print(f"root     {root}  ({kind}, stack {workspace.stack_of(root) or '—'})")
    print(f"tasks    {paths.tasks_dir(root)}")
    print(f"runs     {paths.runs_dir(root)}")
    print(f"kb1      {paths.kb1_dir(root)}")
    try:
        print(f"python   {paths.venv_python(root)}")
    except FileNotFoundError as e:
        print(f"python   — ({e})")
    spec = settings.get(root, "sandbox.gpus", gpus.AUTO)
    print(f"gpus     {gpus.describe(gpus.workspace_pool(root))}  "
          f"(sandbox.gpus: {gpus.render(spec)})")
    try:
        agent = bindings.resolve(root)
        print(f"agent    {agent.runtime} / {agent.model or 'default'}")
    except (KeyError, ValueError) as e:
        print(f"agent    — ({e})")
    print()
    from pinnforge.task import registry

    tasks = registry.list_tasks(root)
    print(f"{len(tasks)} task(s): {', '.join(t.name for t in tasks) or '—'}")
    runs = paths.list_runs(root)
    print(f"{len(runs)} run(s): {', '.join(runs) or '—'}")
    return 0


# Commands that make or repair the environment, or print something static,
# are not interrupted by the version banner.
_NO_BANNER = {"init", "workspace", "completion", "setup", "ui", "chat"}


def _banner(command: str) -> None:
    if command in _NO_BANNER:
        return
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        return
    if note := init_cmd.skew(root):
        print(note, file=sys.stderr)


def _utf8_everywhere() -> None:
    """Windows: print UTF-8, and have every Python started from here do the same.

    Its pipes and files default to the ANSI code page (cp936, cp1252, …),
    which cannot hold a ✓, a ² or most of a traceback: printing one to a pipe
    raised, and a child reading a UTF-8 file without naming the encoding read
    garbage. Children — agents, `eval.py`, its workers — inherit UTF-8 mode,
    so a block behaves the same on every system.
    """
    if os.name != "nt":
        return
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            with contextlib.suppress(AttributeError, ValueError):
                stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_everywhere()
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["PINNFORGE_ROOT"] = args.root
    _banner(args.command)
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
