"""`pinnforge task …` — inspect, validate and author task packages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pinnforge import paths, scoring
from pinnforge.config import AgentConfig
from pinnforge.run import anchor
from pinnforge.task import contract, registry


def cmd_list(args) -> int:
    tasks = registry.list_tasks()
    if not tasks:
        print(f"no tasks in {paths.tasks_dir()}")
        return 0
    print(f"{'task':<12} {'train s':>8} {'budget s':>9}  title")
    for t in tasks:
        tt = f"{t.train_time_s:.0f}" if t.train_time_s else "—"
        wb = f"{t.wall_budget_s:.0f}" if t.wall_budget_s else "—"
        print(f"{t.name:<12} {tt:>8} {wb:>9}  {t.title}")
    return 0


def cmd_show(args) -> int:
    t = registry.get_task(args.name)
    print(f"name        {t.name}")
    print(f"path        {t.path}")
    print(f"title       {t.title}")
    print(f"train time  {t.train_time_s} s")
    print(f"budget      {t.wall_budget_s} s")
    print(f"data        {', '.join(t.data_files) or '—'}")
    print(f"valid       {t.valid}")
    return 0


def cmd_validate(args) -> int:
    task_dir = paths.tasks_dir() / args.name
    if not task_dir.is_dir():
        print(f"no such task: {args.name}")
        return 2
    report = contract.validate(task_dir)
    print(report.render())
    return 0 if report.ok else 1


def cmd_smoke(args) -> int:
    """Free CPU dress rehearsal of a task's own baseline.

    Runs in a throwaway directory laid out like a run root, so `eval.py`
    finds `task/` and writes its record where it expects to, without
    creating a real run.

    Two things the directory cannot supply on its own:

    * a **scorer** — `eval.py` submits predictions and blocks for a verdict,
      exactly as it does inside a run, so the smoke must serve the queue the
      way `run start` does;
    * the **task name** — the daemon resolves the reference through it, and
      `eval.py` infers it from the run directory (`runs/ldc_1` -> `ldc`),
      which a `pinnforge-smoke-ldc-XXXX` tempdir cannot spell. `FORGE_TASK`
      is the override it already honours.

    The verdict comes from the record, never from the exit status: the
    contract is that `eval.py` exits 0 whenever a record was *written*,
    failed runs included, so the exit code says nothing about success.
    """
    with scratch_run(args.name) as run:
        block = run / "blocks" / "b00"
        block.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / "task" / "baseline.py", block / "b00_v01.py")
        env = os.environ.copy()
        env["FORGE_TASK"] = args.name
        with scoring.serving(paths.project_root(), run):
            proc = subprocess.run(
                [str(paths.venv_python()), "task/eval.py", "blocks/b00/b00_v01.py", "--smoke"],
                cwd=str(run),
                text=True,
                env=env,
                check=False,  # the record is the verdict, not the exit status
            )
        records = _records(block / "evals.jsonl")

    code, message = smoke_verdict(proc.returncode, records)
    print(message)
    return code


def smoke_verdict(returncode: int, records: list[dict]) -> tuple[int, str]:
    """Did the dress rehearsal actually rehearse? — from the evidence.

    `eval.py` exits 0 whenever a record was written, failed runs included,
    so a zero exit only means the machinery reached the log. What makes a
    smoke meaningful is a record that carries a score and no error.
    """
    if returncode != 0:
        return 1, f"smoke failed (exit {returncode})"
    if not records:
        return 1, "smoke failed: eval.py wrote no record"
    rec = records[-1]
    if rec.get("error"):
        return 1, f"smoke failed: {rec['error']}"
    if rec.get("rRMSE") is None:
        return 1, "smoke failed: the record carries no rRMSE — nothing was scored"
    return 0, (
        f"smoke passed — rRMSE {rec['rRMSE']:.5g} "
        f"({rec.get('step_count', '?')} steps, {rec.get('train_s', 0):.1f} s)"
    )


@contextmanager
def scratch_run(name: str):
    """A throwaway directory laid out like a run root, for one task.

    Named `<task>_0` on purpose: `eval.py` infers which task it is serving
    from the run directory (`runs/ldc_1` -> `ldc`), and the score daemon
    resolves the reference through that name. A `tmpXXXX` directory spells
    no task, so the daemon cannot find the answers and every evaluation
    inside it fails to score.
    """
    task = registry.get_task(name)
    with tempfile.TemporaryDirectory(prefix=f"pinnforge-{name}-") as tmp:
        run = Path(tmp) / f"{name}_0"
        run.mkdir()
        registry.install(task, run)
        yield run


def cmd_anchor(args) -> int:
    """Show, build or rebuild a task's b00 control node.

    The anchor is the score every later block must beat. `run start` builds
    it on demand, but a task is not finished until it has one — this makes
    that step reachable on its own, so authoring a task can end with the
    task ready to load rather than ready to be measured later.
    """
    name = args.name
    cache = anchor.cache_dir(name)
    meta = cache / "meta.json"

    if meta.is_file() and not args.rebuild:
        data = json.loads(meta.read_text(encoding="utf-8"))
        print(f"{name}: anchor rRMSE {data['rRMSE']:.5g} (cached)")
        stale = [
            f
            for f, want in data.get("fingerprint", {}).items()
            if _md5(paths.tasks_dir() / name / f) != want
        ]
        if stale:
            print(f"  STALE — {', '.join(stale)} changed since it was measured")
            print(f"  rebuild with: pinnforge task anchor {name} --rebuild")
            return 1
        print("  task/ matches the measured baseline")
        return 0

    # Before spending a GPU, look for a run that already measured this exact
    # package. Same guard, cheaper source.
    if not args.rebuild and (found := anchor.harvest(name)) is not None:
        print(f"{name}: anchor rRMSE {found.rRMSE:.5g} ({found.detail}) -> {cache}")
        return 0

    # Measuring costs a full training run on a GPU — 55 s for one task, 475 s
    # for another — and two of them on one card contend for the same lock.
    # Asking for it has to be explicit: reporting "no anchor" answers a
    # question about state, and silently starting a GPU job does not.
    if not (args.measure or args.rebuild):
        print(f"{name}: no anchor cached, and no run of this package to harvest one from")
        print(f"  measure it with: pinnforge task anchor {name} --measure")
        return 1

    if args.rebuild and cache.is_dir():
        shutil.rmtree(cache)
    task = registry.get_task(name)
    est = f"~{task.train_time_s:.0f} s of training" if task.train_time_s else "one training run"
    print(f"measuring b00 for {name} on gpu {args.gpu} ({est})…")
    with scratch_run(name) as run, scoring.serving(paths.project_root(), run):
        result = anchor.ensure(run, name, gpu=args.gpu)
    if result.rRMSE is None:
        print(f"  failed: {result.detail}")
        return 1
    print(f"  anchor rRMSE {result.rRMSE:.5g} ({result.detail}) -> {cache}")
    return 0


def _md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def cmd_new(args) -> int:
    from pinnforge import agents

    # A task the run library has already frozen is not a new task. Authoring it
    # again would produce a different definition and break comparability with
    # the runs that exist, so re-use theirs instead — and skip the agent.
    known = (paths.tasks_dir() / args.name / "problem.md").is_file()
    if not known and (found := registry.recover_from_run(args.name)) is not None:
        dest, run = found
        print(f"{args.name} was run before — recovered its definition from {run.name}")
        print(f"  -> {dest}  ({', '.join(sorted(p.name for p in dest.iterdir()))})")
        print("  private/ is not in a run by design; supply the scoring truth before use")
        return 0 if (dest / "private").is_dir() else 1

    sources = Path(args.source).expanduser().resolve()
    if not sources.exists():
        print(f"no such source path: {sources}")
        return 2
    agent = AgentConfig(runtime=args.runtime, model=args.model or "")
    print(f"authoring task {args.name!r} from {sources} using {agent.runtime}…")
    result = agents.author_task(
        args.name,
        sources,
        args.prompt or "",
        agent=agent,
        template_task=args.template,
    )
    print(f"agent exited {result.returncode} after {result.duration_s:.0f} s")
    print(f"log: {result.log_path}")

    # Authoring is not done when the files exist — it is done when the task
    # loads. Each gate reads the previous one's evidence: the contract, then
    # the scoring path end to end, then the anchor every later block beats.
    report = contract.validate(paths.tasks_dir() / args.name)
    print()
    print(report.render())
    if not report.ok:
        return 1
    if args.no_load:
        print("\n--no-load: stopping after validation")
        return 0

    print()
    if (rc := cmd_smoke(args)) != 0:
        return rc
    print()
    return cmd_anchor(_anchor_args(args))


class _anchor_args:
    """Adapt `task new`'s namespace to what `cmd_anchor` reads."""

    def __init__(self, args):
        self.name = args.name
        self.gpu = getattr(args, "gpu", 0)
        self.measure = True
        self.rebuild = True  # a task just authored has no anchor worth keeping


def register(sub) -> None:
    p = sub.add_parser("task", help="inspect, validate and author task packages")
    s = p.add_subparsers(dest="task_cmd", required=True)

    s.add_parser("list", help="list library tasks").set_defaults(func=cmd_list)

    show = s.add_parser("show", help="show one task")
    show.add_argument("name")
    show.set_defaults(func=cmd_show)

    val = s.add_parser("validate", help="check a task against the contract")
    val.add_argument("name")
    val.set_defaults(func=cmd_validate)

    smoke = s.add_parser("smoke", help="free CPU dress rehearsal of a task's baseline")
    smoke.add_argument("name")
    smoke.set_defaults(func=cmd_smoke)

    anc = s.add_parser("anchor", help="show, build or rebuild a task's b00 control node")
    anc.add_argument("name")
    anc.add_argument(
        "--measure", action="store_true", help="build it on a GPU when none is cached"
    )
    anc.add_argument("--rebuild", action="store_true", help="re-measure even if cached")
    anc.add_argument("--gpu", type=int, default=0)
    anc.set_defaults(func=cmd_anchor)

    new = s.add_parser("new", help="author a new task, then carry it to loadable")
    new.add_argument("name")
    new.add_argument("--source", required=True, help="directory of reference files")
    new.add_argument("--prompt", default="", help="what the task should be")
    new.add_argument("--template", default="ldc", help="task package to adapt (default: ldc)")
    new.add_argument("--runtime", default="claude_code")
    new.add_argument("--model", default="")
    new.add_argument("--gpu", type=int, default=0, help="gpu for the anchor measurement")
    new.add_argument(
        "--no-load", action="store_true", help="stop after validation; skip smoke and anchor"
    )
    new.set_defaults(func=cmd_new)
