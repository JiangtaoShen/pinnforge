"""`pinnforge setup` — choose, prove and save the agent that runs blocks,
and the GPUs the workspace's runs use.

One pass, once per machine: find which coding-agent CLIs are installed, pick
an exact model id for each one worth using, prove it answers through the very
adapter a block is started with, and save it as a named binding. Runs then
use the default binding unless told otherwise. In a workspace on a machine
with GPUs the wizard also offers to change which cards it uses;
`pinnforge setup gpus` does only that.
"""

from __future__ import annotations

import re
import sys

from pinnforge import bindings, gpus, paths, settings, workspace
from pinnforge.cli import agents_cmd, gpu_choice
from pinnforge.config import AgentConfig
from pinnforge.runtime import registry

# Offered in the wizard; the runtime's own default always comes first. Any
# exact id can be typed instead — the ping is what decides whether it works.
SUGGESTED = {
    "claude_code": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"],
    "codex": [],
    "cursor_agent": [],
    "opencode": ["deepseek/deepseek-chat", "anthropic/claude-opus-5-5"],
}


def _ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{shown}: ").strip()
    except EOFError:
        return default
    return answer or default


def _yes(prompt: str, default: bool = True) -> bool:
    answer = _ask(prompt + (" (Y/n)" if default else " (y/N)")).lower()
    return default if not answer else answer.startswith("y")


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model.split("/")[-1]).strip("-") or "agent"


def _report() -> list[dict]:
    print("coding-agent CLIs on this machine:")
    rows = registry.detect_available()
    for row in rows:
        mark = "found  " if row["path"] else "missing"
        print(f"  {mark} {row['runtime']:<13} ({row['command']})")
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        root = None
    pool = gpus.workspace_pool(root)
    print(f"\ncompute: {gpus.describe(pool)}"
          + (f"  (sandbox.gpus: {gpus.render(settings.get(root, 'sandbox.gpus', 'auto'))})"
             if root else ""))
    if root:
        print(f"workspace: {root}")
    else:
        print("workspace: none here yet (`pinnforge init <dir>` creates one)")
    data = bindings.load()
    if data["bindings"]:
        print(f"existing bindings: {', '.join(sorted(data['bindings']))} "
              f"(default {data.get('default')})")
    print()
    return rows


def _configure(runtime: str, ping_timeout: float) -> tuple[str, AgentConfig] | None:
    rt = registry.get_runtime(runtime)
    choices = [m for m in [rt.default_model, *SUGGESTED.get(runtime, [])] if m]
    if choices:
        print(f"  models for {runtime}: {', '.join(choices)}")
    model = _ask(f"  model id for {runtime}", choices[0] if choices else "")
    if not model:
        print("  skipped — this runtime needs an exact model id")
        return None
    name = _ask("  name this binding", _slug(model))
    agent = AgentConfig(runtime=runtime, model=model)
    print(f"  pinging {runtime} / {model} …", flush=True)
    ok, detail, _ = agents_cmd.ping(agent, ping_timeout)
    print(f"  {'OK  ' if ok else 'FAIL'} {detail}")
    if not ok and not _yes("  save it anyway?", default=False):
        return None
    return name, agent


def _store_gpus(path, root, spec, inv) -> None:
    settings.write(path, "sandbox.gpus", spec)
    print(f"sandbox.gpus: {gpu_choice.summary(spec, inv)}")
    print(f"  saved in {path}")
    print("  new runs use it; `run start --gpus` overrides it for one run, and a run")
    print("  already created keeps the cards it started with")
    if root is not None and str(path) != str(root / paths.MARKER) \
            and settings.origin(root, "sandbox.gpus") == settings.WORKSPACE_FILE:
        print(f"  note: {root / paths.MARKER} sets its own sandbox.gpus, which wins here")
    stack = workspace.stack_of(root) if root is not None else None
    if stack == "cpu" and gpus.resolve(spec, inv=inv):
        print("  note: this workspace has the CPU build of JAX — "
              "`pinnforge workspace sync --stack cuda12` to train on the cards")


def _wizard_gpus(inv) -> None:
    """Offer to change the workspace's cards, when there is a workspace and a card."""
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        return
    if not inv or not paths.is_workspace(root):
        return
    current = settings.get(root, "sandbox.gpus", gpus.AUTO)
    if not _yes(f"change which GPUs this workspace uses? (now {gpu_choice.label(current)})",
                default=False):
        print()
        return
    spec = gpu_choice.ask(inv, current)
    _store_gpus(root / paths.MARKER, root, spec, inv)
    print()


def cmd_setup(args) -> int:
    rows = _report()
    if args.non_interactive or not sys.stdin.isatty():
        if not args.non_interactive:
            print("(not a terminal — reporting only; use `pinnforge setup agent …` to bind, "
                  "`pinnforge setup gpus …` to choose cards)")
        return 0
    _wizard_gpus(gpus.inventory())
    installed = [r["runtime"] for r in rows if r["path"]]
    if not installed:
        print("No coding-agent CLI found. Install one of: Claude Code (`claude`), Codex,")
        print("Cursor Agent or OpenCode, sign in, then run `pinnforge setup` again.")
        return 1
    made: list[str] = []
    for runtime in installed:
        if not _yes(f"configure {runtime}?", default=(runtime == installed[0])):
            continue
        got = _configure(runtime, args.ping_timeout)
        if got is None:
            continue
        name, agent = got
        bindings.put(name, agent)
        made.append(name)
        print(f"  saved binding {name!r}\n")
    if not made:
        existing = bindings.load()
        if existing["bindings"]:
            print(f"bindings unchanged (default: {existing.get('default')})")
            return 0
        print("no binding saved")
        return 1
    data = bindings.load()
    default = data.get("default") or made[0]
    if len(data["bindings"]) > 1:
        default = _ask("default binding", default)
    bindings.get(default)
    data = bindings.load()
    data["default"] = default
    path = bindings.save(data)
    print(f"\nbindings saved to {path}; default: {default}")
    print("runs use it unless told `--agent <name>`; list them with `pinnforge agents bindings`")
    return 0


def cmd_setup_agent(args) -> int:
    agent = AgentConfig(
        runtime=registry.canonical(args.runtime), model=args.model, command=args.command or ""
    )
    name = args.name or _slug(args.model)
    if not args.no_ping:
        ok, detail, _ = agents_cmd.ping(agent, args.ping_timeout)
        print(f"{'OK  ' if ok else 'FAIL'} {agent.runtime} / {agent.model} — {detail}")
        if not ok and not args.force:
            print("not saved (pass --force to save it anyway)")
            return 1
    bindings.put(name, agent, make_default=args.default)
    print(f"saved binding {name!r}{' (default)' if bindings.default_name() == name else ''}")
    return 0


def cmd_setup_gpus(args) -> int:
    inv = gpus.inventory()
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        root = None
    if args.user:
        path = settings.user_dir() / "config.yaml"
        current = settings.get(None, "sandbox.gpus", gpus.AUTO)
    elif root is None:
        print("not in a workspace — run this inside one, or pass --user to set the "
              "default for every workspace on this machine")
        return 2
    else:
        path = root / paths.MARKER
        current = settings.get(root, "sandbox.gpus", gpus.AUTO)
    if args.spec is None:
        if not sys.stdin.isatty():
            print(f"sandbox.gpus: {gpu_choice.summary(current, inv)}")
            print("set it with: pinnforge setup gpus <all | none | 1 | 0,1 …>")
            return 0
        spec = gpu_choice.ask(inv, current)
    else:
        try:
            spec = gpus.parse_choice(args.spec, inv)
        except ValueError as e:
            print(e)
            return 2
    _store_gpus(path, root, spec, inv)
    return 0


def register(sub) -> None:
    p = sub.add_parser("setup", help="choose the coding agent that runs the rounds, and the GPUs")
    p.add_argument("--non-interactive", action="store_true", help="report only; change nothing")
    p.add_argument("--ping-timeout", type=float, default=120.0)
    p.set_defaults(func=cmd_setup)
    s = p.add_subparsers(dest="setup_cmd")
    ag = s.add_parser("agent", help="create or update one binding without the wizard")
    ag.add_argument("--runtime", required=True)
    ag.add_argument("--model", required=True, help="an exact id, e.g. claude-opus-4-8")
    ag.add_argument("--name", default="", help="binding name (default: from the model id)")
    ag.add_argument("--command", default="", help="path to the CLI, if not on PATH")
    ag.add_argument("--default", action="store_true", help="make it the default binding")
    ag.add_argument("--no-ping", action="store_true", help="save without proving it answers")
    ag.add_argument("--force", action="store_true", help="save even if the ping fails")
    ag.add_argument("--ping-timeout", type=float, default=120.0)
    ag.set_defaults(func=cmd_setup_agent)
    gp = s.add_parser("gpus", help="choose which GPUs this workspace's runs use")
    gp.add_argument("spec", nargs="?", default=None,
                    help="all, none, or card ids such as 1 or 0,1 (asked if omitted)")
    gp.add_argument("--user", action="store_true",
                    help="set the default for every workspace on this machine instead")
    gp.set_defaults(func=cmd_setup_gpus)
