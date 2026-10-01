"""`pinnforge examples …` — the demo and the benchmark tasks."""

from __future__ import annotations

from pinnforge import paths
from pinnforge.task import examples


def cmd_list(args) -> int:
    have = {t.name for t in paths.tasks_dir().iterdir()} if _has_workspace() else set()
    print(f"{'name':<8} {'size':>8}  {'installed':<9}  title")
    print(f"{examples.DEMO:<8} {'—':>8}  {'yes' if examples.DEMO in have else 'no':<9}  "
          "Multi-scale Poisson — analytic reference, runs on a CPU")
    for e in examples.catalog():
        size = f"{e.get('size', 0) / 1e6:.1f} MB" if e.get("size") else "—"
        mark = "yes" if e["name"] in have else "no"
        print(f"{e['name']:<8} {size:>8}  {mark:<9}  {e.get('title', '')}")
    return 0


def _has_workspace() -> bool:
    try:
        return paths.tasks_dir().is_dir()
    except paths.ProjectNotFound:
        return False


def cmd_add(args) -> int:
    tasks = paths.tasks_dir()
    tasks.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        if name == examples.DEMO:
            dest = examples.install_demo(tasks)
        else:
            print(f"fetching {name}…")
            dest = examples.fetch(name, tasks, source=args.source, force=args.force)
        print(f"  {name} -> {dest}")
    print("\nnext: pinnforge task anchor <name> --measure"
          "   (or just `pinnforge run start -t <name>`)")
    return 0


def register(sub) -> None:
    p = sub.add_parser("examples", help="the demo task and the benchmark tasks")
    s = p.add_subparsers(dest="examples_cmd", required=True)
    s.add_parser("list", help="what can be installed").set_defaults(func=cmd_list)
    add = s.add_parser("add", help="install tasks into this workspace")
    add.add_argument("names", nargs="+")
    add.add_argument("--source", default=None,
                     help="archive path or mirror URL instead of the release (one name)")
    add.add_argument("--force", action="store_true", help="replace an installed copy")
    add.set_defaults(func=cmd_add)
