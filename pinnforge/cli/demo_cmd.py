"""`pinnforge demo` — a whole run in minutes, with no model and no key.

The packaged demo task (a two-scale Poisson problem with an analytic answer)
driven by the scripted `replay` runtime: the anchor is measured, blocks
write and evaluate candidates, summaries are written, and the dashboard shows
it all happening. Everything except the agent is the real machinery, so this
is also the quickest way to see that a workspace works.
"""

from __future__ import annotations

import argparse

from pinnforge import gpus, paths
from pinnforge.cli import run_cmd
from pinnforge.run import lifecycle
from pinnforge.task import examples


def _free_cards(root, share: bool) -> str | None:
    """The visible cards no other run has claimed, as a `--gpus` value.

    A demo is a toy; a real run's training stops on a wall clock, so a demo
    that joins its card quietly costs it accuracy. Left to choose, the demo
    takes only free cards, and refuses rather than share one unasked.
    """
    pool = gpus.workspace_pool(root)
    if not pool or share:
        return ""
    held = lifecycle.leases(pool)
    busy = {g for g, _ in held}
    free = [g for g in pool if g not in busy]
    for g, info in held:
        print(f"GPU {g} is in use by {info.get('run')} (pid {info.get('pid')})")
    if not free:
        print("every card is in use; wait, pick one with --gpus, or join anyway with --share")
        return None
    if not busy:
        return ""  # the workspace's own choice, as it stands
    print(f"the demo takes the free card(s): {','.join(map(str, free))}")
    return ",".join(map(str, free))


def _chosen_cards(text: str) -> tuple[str, list[str]]:
    """`--gpus` as typed (all, none, 1, 0,1, "0 1") -> (`run start --gpus`, `--set` overrides).

    Read the way `init` and `setup gpus` read it. Cards go on as an explicit
    comma list; CPU only — which an empty `--gpus` cannot say, since empty
    means "the workspace's choice" — goes as `sandbox.gpus=[]`.
    """
    inv = gpus.inventory()
    pool = gpus.resolve(gpus.parse_choice(text, inv), inv=inv)
    if not pool:
        return "", ["sandbox.gpus=[]"]
    return ",".join(map(str, pool)), []


def cmd_demo(args) -> int:
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        print("`pinnforge demo` runs inside a workspace. Create one first:")
        print("  pinnforge init ~/pf && cd ~/pf && pinnforge demo")
        return 2
    if not (paths.tasks_dir(root) / examples.DEMO / "problem.md").is_file():
        print("installing the demo task…")
        examples.install_demo(paths.tasks_dir(root))
    overrides: list[str] = []
    if args.gpus:
        try:
            cards, overrides = _chosen_cards(args.gpus)
        except ValueError as e:
            print(f"--gpus: {e}")
            return 2
    else:
        cards = _free_cards(root, args.share)
    if cards is None:
        return 2
    ns = argparse.Namespace(
        task=examples.DEMO,
        blocks=args.blocks,
        agent=None,
        runtime="replay",
        model="",
        gpus=cards,
        budget=str(args.budget),
        no_anchor=False,
        force=False,
        share=args.share,
        set=overrides,
        foreground=args.foreground,
        detach=not args.foreground,
        no_ui=args.no_ui,
    )
    print("demo: a scripted agent plays a fixed playbook on the demo task — no model, no key.")
    return run_cmd.cmd_start(ns)


def register(sub) -> None:
    p = sub.add_parser("demo", help="a complete run in minutes with a scripted agent (no API key)")
    p.add_argument("-n", "--blocks", type=int, default=2, help="rounds after round 0")
    p.add_argument("--budget", type=int, default=240, help="GPU seconds per round (default 240)")
    p.add_argument("--gpus", default="",
                   help="the cards to use: all, none, or ids such as 1 or 0,1 (default: the "
                   "workspace's sandbox.gpus, less any card another run holds)")
    p.add_argument("--share", action="store_true",
                   help="use cards another run is already using")
    p.add_argument("--foreground", action="store_true", help="drive it in this terminal")
    p.add_argument("--no-ui", action="store_true", help="do not start the dashboard")
    p.set_defaults(func=cmd_demo)
