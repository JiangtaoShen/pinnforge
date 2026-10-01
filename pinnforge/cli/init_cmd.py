"""`pinnforge init`, `pinnforge workspace …` and `pinnforge completion`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pinnforge import __version__, gpus, paths, settings, workspace
from pinnforge.cli import gpu_choice


def _record_env(root: Path) -> None:
    """Note which PINNForge the environment holds, for the version banner."""
    version = workspace.venv_version(root)
    stamp = {"pinnforge": version, "tool": __version__, "stack": workspace.stack_of(root)}
    (paths.state_dir(root) / "venv.json").write_text(json.dumps(stamp) + "\n", encoding="utf-8")


def _choose_gpus(args, dest: Path, inv: list[gpus.Gpu]):
    """The operator's cards: `--gpus`, else asked when there is a choice to make.

    None keeps what the workspace file says (`auto` in a new one).
    """
    if args.gpus is not None:
        return gpus.parse_choice(args.gpus, inv)
    if len(inv) > 1 and sys.stdin.isatty():
        current = settings.get(dest if paths.is_workspace(dest) else None, "sandbox.gpus",
                               gpus.AUTO)
        spec = gpu_choice.ask(inv, current)
        print()
        return spec
    return None


def cmd_init(args) -> int:
    dest = Path(args.dir).expanduser().resolve()
    try:
        workspace.check_dest(dest, args.force)
    except FileExistsError as e:
        print(e)
        return 2
    inv = gpus.inventory()
    # Asked first, so the long `uv sync` below runs unattended.
    try:
        spec = _choose_gpus(args, dest, inv)
    except ValueError as e:
        print(f"--gpus: {e}")
        return 2
    existing = paths.is_workspace(dest)
    # Without --gpus the workspace file keeps (or, new, leaves null) its own
    # value, so the cards are whatever the layered settings say: the
    # workspace's, else this machine's default, else auto.
    shown = spec if spec is not None else settings.get(
        dest if existing else None, "sandbox.gpus", gpus.AUTO)
    cpu_only = gpus.render(shown) == "[]"
    # A refresh keeps the environment's build unless told otherwise.
    stack = (args.stack or (workspace.stack_of(dest) if existing else None)
             or ("cpu" if cpu_only else workspace.detect_stack()))
    print(f"creating a PINNForge workspace at {dest}")
    print(f"  gpus    {gpu_choice.summary(shown, inv)}")
    print(f"  stack   {stack}")
    if cpu_only and stack == "cpu" and inv and not args.stack:
        print("          (the CPU build; `pinnforge workspace sync --stack cuda12` if you "
              "give it cards later)")
    if spec is None and len(inv) > 1 and gpus.render(shown) == gpus.AUTO:
        print("          (every visible card; choose with `pinnforge setup gpus`)")
    if not args.no_venv:
        print("  env     `uv sync` — the first CUDA install downloads a few GB\n")
    sys.stdout.flush()
    workspace.init(
        dest, stack_name=stack, venv=not args.no_venv, demo=not args.no_demo, force=args.force,
        gpus_spec=spec,
    )
    if not args.no_venv:
        _record_env(dest)
    failed = _fetch_examples(args.examples, dest)
    rel = dest if dest != Path.cwd() else Path(".")
    print(f"\nworkspace ready: {dest}")
    print("\nnext:")
    if rel != Path("."):
        print(f"  cd {rel}")
    print("  pinnforge demo                 # a full run in minutes, no API key")
    print("  pinnforge setup                # choose the coding agent that runs your blocks")
    print("  pinnforge task new             # describe your PDE; no reference? one is built")
    print("  pinnforge run start -t demo    # real agents on the demo task")
    if failed:
        print(f"\nnot installed: {', '.join(failed)} (see above) — the workspace is usable; "
              "add them later with `pinnforge examples add <name>`, or "
              "`--source <archive>` for a copy you downloaded")
        return 1
    return 0


def _fetch_examples(names: str, dest: Path) -> list[str]:
    """Fetch the `--examples` tasks; returns the ones that could not be installed.

    By now the workspace is built and usable, so a download that fails —
    offline, a release not published yet — is reported, not raised.
    """
    from pinnforge.task import examples

    failed = []
    for name in [n.strip() for n in (names or "").split(",") if n.strip()]:
        if (dest / "tasks" / name / "problem.md").is_file():  # a refresh: keep it
            print(f"example {name}: already installed")
            continue
        print(f"fetching example {name}…")
        try:
            examples.fetch(name, dest / "tasks")
        except (KeyError, ValueError, FileExistsError, OSError) as e:
            why = e.args[0] if isinstance(e, KeyError) and e.args else e
            print(f"  could not install {name}: {why}")
            failed.append(name)
    return failed


def cmd_sync(args) -> int:
    root = paths.project_root()
    if not paths.is_workspace(root):
        print(f"{root} is a source checkout, not a workspace — use `uv sync` there")
        return 2
    if args.upgrade or args.stack:
        stack = workspace.repin(root, args.stack)
        print(f"re-pinned {root}/pyproject.toml to pinnforge {__version__} (stack {stack})")
    code = workspace.sync(root)
    if code == 0:
        _record_env(root)
        print(f"environment in sync: {root / '.venv'}")
    return code


def skew(root: Path) -> str | None:
    """A one-line warning when the workspace environment lags the tool."""
    stamp = root / paths.STATE_DIR / "venv.json"
    if not stamp.is_file():
        return None
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8")).get("pinnforge")
    except (OSError, json.JSONDecodeError):
        return None
    if recorded and recorded != __version__:
        return (
            f"note: this workspace's environment has pinnforge {recorded}, the tool is "
            f"{__version__} — run `pinnforge workspace upgrade`"
        )
    return None


# ─────────────────────────── shell completion ───────────────────────────


def _tree(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    """{"": top-level words, "run": its subcommands, …} from the live parser."""
    out: dict[str, list[str]] = {}

    def subs(p):
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices
        return {}

    def visible(names):
        return sorted(n for n in names if not n.startswith("_"))  # `run _drive` is internal

    top = subs(parser)
    out[""] = visible(top)
    for name, sp in top.items():
        words = visible(subs(sp))
        flags = sorted(
            o for a in sp._actions for o in a.option_strings if o.startswith("--")
        )
        out[name] = words or flags
    return out


def cmd_completion(args) -> int:
    from pinnforge.cli import build_parser

    tree = _tree(build_parser())
    cases = "\n".join(
        f'        {name}) words="{" ".join(words)}" ;;' for name, words in tree.items() if name
    )
    script = f"""# pinnforge completion — eval "$(pinnforge completion {args.shell})"
_pinnforge_complete() {{
    local cur words
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        words="{" ".join(tree[""])}"
    else
        case "${{COMP_WORDS[1]}}" in
{cases}
        *) words="" ;;
        esac
    fi
    COMPREPLY=( $(compgen -W "$words" -- "$cur") )
}}
"""
    if args.shell == "zsh":
        script = "autoload -U +X bashcompinit && bashcompinit\n" + script
    script += "complete -F _pinnforge_complete pinnforge\n"
    print(script)
    return 0


def register(sub) -> None:
    init = sub.add_parser("init", help="create a workspace: settings, environment, demo task")
    init.add_argument("dir", nargs="?", default=".", help="where (default: here)")
    init.add_argument("--stack", choices=workspace.STACKS, default=None,
                      help="JAX build for the environment (default: cuda12 if a GPU is visible)")
    init.add_argument("--gpus", default=None,
                      help="the cards runs may use: all, none, or ids such as 1 or 0,1 "
                      "(asked when several are present)")
    init.add_argument("--examples", default="",
                      help="benchmark tasks to fetch too, e.g. ldc,ks "
                      "(see `pinnforge examples list`)")
    init.add_argument("--no-venv", action="store_true", help="skip creating the environment")
    init.add_argument("--no-demo", action="store_true", help="do not install the demo task")
    init.add_argument("--force", action="store_true", help="refresh an existing workspace")
    init.set_defaults(func=cmd_init)

    ws = sub.add_parser("workspace", help="the workspace's training environment")
    s = ws.add_subparsers(dest="ws_cmd", required=True)
    sync = s.add_parser("sync", help="create or update .venv from pyproject.toml")
    sync.add_argument("--stack", choices=workspace.STACKS, default=None,
                      help="switch the JAX build (re-pins pyproject.toml)")
    sync.set_defaults(func=cmd_sync, upgrade=False)
    up = s.add_parser("upgrade", help="re-pin the environment to this pinnforge, then sync")
    up.add_argument("--stack", choices=workspace.STACKS, default=None)
    up.set_defaults(func=cmd_sync, upgrade=True)

    comp = sub.add_parser("completion", help="print a shell completion script")
    comp.add_argument("shell", choices=("bash", "zsh"))
    comp.set_defaults(func=cmd_completion)
