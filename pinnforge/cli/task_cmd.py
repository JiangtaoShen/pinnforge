"""`pinnforge task …` — inspect, validate and author task packages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pinnforge import gpus, oscompat, paths, scoring
from pinnforge.cli import task_new
from pinnforge.run import anchor
from pinnforge.task import authoring, contract, registry


def cmd_list(args) -> int:
    tasks = registry.list_tasks()
    if not tasks:
        print(f"no tasks in {paths.tasks_dir()}")
        return 0
    print(f"{'task':<12} {'state':<10} {'train s':>8} {'budget s':>9}  title")
    for t in tasks:
        tt = f"{t.train_time_s:.0f}" if t.train_time_s else "—"
        wb = f"{t.wall_budget_s:.0f}" if t.wall_budget_s else "—"
        st = authoring.status(t.path)
        state = "ready" if st is None or st.get("finalized") else f"{st.get('stage')}…"
        if st and st.get("status") == "failed":
            state = f"failed@{st.get('stage')}"
        print(f"{t.name:<12} {state:<10} {tt:>8} {wb:>9}  {t.title}")
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
    report = contract.validate(task_dir, reference=not args.no_reference)
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
    code, message = run_smoke(args.name)
    print(message)
    return code


def run_smoke(name: str) -> tuple[int, str]:
    """The smoke itself: (exit code, verdict)."""
    root = paths.project_root()
    with scratch_run(name) as run:
        block = run / "blocks" / "b00"
        block.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / "task" / "baseline.py", block / "b00_v01.py")
        env = os.environ.copy()
        env["FORGE_TASK"] = name
        with scoring.serving(root, run):
            proc = subprocess.run(
                [str(paths.venv_python(root)), "task/eval.py", "blocks/b00/b00_v01.py",
                 "--smoke"],
                cwd=str(run),
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                check=False,  # the record is the verdict, not the exit status
            )
        records = _records(block / "evals.jsonl")
    return smoke_verdict(proc.returncode, records)


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
    with tempfile.TemporaryDirectory(prefix=f"pinnforge-{name}-",
                                     ignore_cleanup_errors=oscompat.WINDOWS) as tmp:
        run = Path(tmp) / f"{name}_0"
        run.mkdir()
        registry.install(task, run)
        yield run


def cmd_anchor(args) -> int:
    """Show, build, rebuild or rekey a task's b00 control node.

    The anchor is the score every later block must beat. `run start` builds
    it on demand, but a task is not finished until it has one — this makes
    that step reachable on its own, so authoring a task can end with the
    task ready to load rather than ready to be measured later.
    """
    name = args.name
    root = paths.project_root()
    cache = anchor.cache_dir(name, root)
    task_dir = paths.tasks_dir(root) / name
    gpu = _anchor_gpu(args)
    hw = anchor.hardware(gpu)
    meta = anchor.read_meta(name, root)

    if args.rekey:
        # A cache from before hardware classes were recorded is stamped with
        # the class of the card this machine measures on — when it has one.
        stamp = hw if gpu is not None and hw.get("platform") == "gpu" else None
        with scratch_run(name) as run, scoring.serving(root, run):
            result = anchor.rekey(run, name, root, hw=stamp)
        if result.rRMSE is None:
            print(f"{name}: cannot rekey — {result.detail}")
            print(f"  measure it again with: pinnforge task anchor {name} --rebuild")
            return 1
        print(f"{name}: anchor rRMSE {result.rRMSE:.5g} — {result.detail}")
        return 0

    if meta is not None and not args.rebuild:
        print(f"{name}: anchor rRMSE {meta['rRMSE']:.5g} (cached)")
        if why := anchor.staleness(meta, task_dir, name, root, hw):
            print(f"  STALE — {why}")
            print(f"  carry it over with:  pinnforge task anchor {name} --rekey"
                  "   (free, when only the evaluation's plumbing changed)")
            print(f"  or measure again:    pinnforge task anchor {name} --rebuild")
            return 1
        print("  task/ matches the measured baseline")
        return 0

    # Before spending a GPU, look for a run that already measured this exact
    # package. Same guard, cheaper source.
    if not args.rebuild and (found := anchor.harvest(name, root, hw)) is not None:
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

    if args.rebuild:
        anchor.clear(name, root)
    task = registry.get_task(name)
    est = f"~{task.train_time_s:.0f} s of training" if task.train_time_s else "one training run"
    where = f"gpu {gpu}" if gpu is not None else "the CPU"
    print(f"measuring b00 for {name} on {where} ({est})…")
    with scratch_run(name) as run, scoring.serving(root, run):
        # --rebuild means a new measurement: with the cache just cleared,
        # ensure() would otherwise harvest an old run's b00 and call it one.
        result = anchor.ensure(run, name, gpu=gpu, hw=hw, rebuild=args.rebuild)
    if result.rRMSE is None:
        print(f"  failed: {result.detail}")
        return 1
    print(f"  anchor rRMSE {result.rRMSE:.5g} ({result.detail}) -> {cache}")
    return 0


def _anchor_gpu(args) -> int | None:
    """The card to measure on: as asked, else the first of the pool, else the CPU."""
    if getattr(args, "gpu", None) is not None:
        return int(args.gpu)
    pool = gpus.workspace_pool(paths.project_root())
    return pool[0] if pool else None


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def register(sub) -> None:
    p = sub.add_parser("task", help="inspect, validate and author task packages")
    s = p.add_subparsers(dest="task_cmd", required=True)

    s.add_parser("list", help="list library tasks").set_defaults(func=cmd_list)

    from pinnforge.cli.delete_cmd import cmd_task_delete

    dl = s.add_parser("delete", help="delete a task: its package, its answers, its anchor")
    dl.add_argument("name")
    dl.add_argument("--with-runs", action="store_true", help="delete the runs made from it too")
    dl.add_argument("-y", "--yes", action="store_true", help="do not ask first")
    dl.set_defaults(func=cmd_task_delete)

    show = s.add_parser("show", help="show one task")
    show.add_argument("name")
    show.set_defaults(func=cmd_show)

    val = s.add_parser("validate", help="check a task against the contract")
    val.add_argument("name")
    val.add_argument("--no-reference", action="store_true",
                     help="check everything but the answers (while authoring)")
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
    anc.add_argument(
        "--rekey",
        action="store_true",
        help="carry a stale anchor over without a GPU: allowed when the baseline is "
        "unchanged and re-scoring its cached params reproduces the score exactly",
    )
    anc.add_argument("--gpu", type=int, default=None, help="default: first GPU of the pool")
    anc.set_defaults(func=cmd_anchor)

    task_new.register(s)
