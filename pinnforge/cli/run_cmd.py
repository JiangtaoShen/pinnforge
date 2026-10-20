"""`pinnforge run …` — start, resume, inspect and stop runs.

A run is driven by exactly one process, `pinnforge run _drive <id>`, which
holds the run's lock, serves its scores, installs its anchor and dispatches
its blocks. `run start` and `run resume` launch that process detached — in a
tmux session of its own, or in the background with a log — so a run outlives
the terminal that started it; `--foreground` keeps it here instead.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from pinnforge import bindings, gpus, integrity, paths, scoring, settings
from pinnforge.config import RunConfig, apply_overrides
from pinnforge.orchestrator import SETTLED, EnvironmentFailure, Orchestrator, verify_block
from pinnforge.run import anchor, layout, ledger, lifecycle
from pinnforge.runtime import registry as runtime_registry
from pinnforge.task import authoring, contract
from pinnforge.types import EvalRecord, best_score

# ─────────────────────────────── configuration ───────────────────────────────


def _build_config(args, root: Path, inv: list | None = None) -> RunConfig:
    """Settings files, then flags, then `--set` — see pinnforge/settings.py.

    Raises ValueError for a `--gpus` that names no card here or is not a
    list of cards, or for a malformed `--set`.
    """
    layered = settings.merged(root)
    agent = bindings.resolve(
        root,
        agent=getattr(args, "agent", None),
        runtime=getattr(args, "runtime", None),
        model=getattr(args, "model", None) or None,
    )
    sandbox = dict(layered.get("sandbox") or {})
    sandbox.setdefault("gpus", "auto")
    d = {
        "task": args.task,
        "blocks": args.blocks,
        "agent": agent.to_dict(),
        "sandbox": sandbox,
    }
    for key in ("stall_minutes", "poll_seconds"):
        if layered.get(key) is not None:
            d[key] = layered[key]
    if args.gpus:
        # The same words `pinnforge init` and `setup gpus` take: all, none,
        # "0,1" or "0 1" — not only comma-separated ids.
        d["sandbox"]["gpus"] = gpus.parse_choice(
            args.gpus, gpus.inventory() if inv is None else inv
        )
    if args.budget:
        d["sandbox"]["wall_budget_s"] = float(args.budget)
    apply_overrides(d, args.set or [])
    return RunConfig.from_dict(d)


def _logging(log: Path | None = None) -> None:
    # stdout, not stderr: a detached run tees stdout into its log file
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )


def _detach_wanted(args, root: Path) -> bool:
    if getattr(args, "foreground", False):
        return False
    if getattr(args, "detach", False):
        return True
    return bool(settings.get(root, "run.detach", True))


# ─────────────────────────────── start / resume ───────────────────────────────


def cmd_start(args) -> int:
    root = paths.project_root()
    inv = gpus.inventory()
    try:
        cfg = _build_config(args, root, inv)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    # Worked out now, while a refusal is still free: everything that can turn
    # a start down has to run before the run directory exists, or a refused
    # start leaves an orphan run behind.
    source = _gpu_source(args, root)
    task_dir = paths.tasks_dir(root) / cfg.task
    if task_dir.is_dir() and not authoring.is_ready(task_dir):
        st = authoring.status(task_dir) or {}
        print(f"task {cfg.task!r} is still being authored (stage {st.get('stage')}: "
              f"{st.get('status')}) — see `pinnforge task status {cfg.task}`")
        return 2
    # Settle the model before anything is built: a harness that names no
    # default has to be told one, and finding that out after the run directory
    # and a measured anchor exist would cost a GPU for nothing.
    model = runtime_registry.resolve_model(
        runtime_registry.get_runtime(cfg.agent.runtime), cfg.agent.model
    )
    # A block is complete only once it has a real evaluation, so a budget that
    # cannot pay for one makes every block fail after the agent has been paid.
    per_run = contract.train_time(task_dir) if task_dir.is_dir() else None
    budget = cfg.sandbox.wall_budget_s
    if per_run and budget is not None and budget < per_run:
        print(f"--budget {budget:.0f} cannot pay for one evaluation of {cfg.task!r}, which "
              f"trains for {per_run:.0f} s; give it at least that, typically 4-10x")
        return 2
    if lost := gpus.missing(cfg.sandbox.gpus, inv):
        have = ", ".join(str(g.index) for g in inv) or "none"
        print(f"GPU {', '.join(map(str, lost))} is not on this machine (it has: {have}); "
              f"the cards came from {source}")
        print("choose others with `pinnforge setup gpus`, or for this run with --gpus")
        return 2
    busy = lifecycle.leases(cfg.sandbox.gpus)
    if busy and not args.share:
        for g, info in busy:
            print(f"GPU {g} is in use by {info.get('run')} (pid {info.get('pid')})")
        print("two runs on one card queue each other's evaluations; start anyway with "
              "--share, or pick other cards with --gpus")
        return 2
    # create_run adopts the task's declared per-block budget when the caller
    # did not ask for a different one.
    run = layout.create_run(cfg, root)
    cfg = RunConfig.load(paths.config_path(run))
    print(f"run {run.name} created at {run}")
    print(f"  agent    {cfg.agent.runtime} / {model}")
    print(f"  compute  {gpus.describe(cfg.sandbox.gpus, inv)}, "
          f"{cfg.sandbox.wall_budget_s:.0f} s per block, {cfg.blocks} block(s)")
    print(f"           cards from {source}")
    flags = ["--no-anchor"] if args.no_anchor else []
    flags += ["--force"] if args.force else []
    return _launch(args, root, run, ["run", "_drive", run.name, *flags])


def _gpu_source(args, root: Path) -> str:
    """Where a run's cards were chosen, so an operator knows what to change.

    In the order they take effect: `--set` is applied last, then `--gpus`,
    then the settings files. Never raises — it is asked before the run is
    built, and a description is not worth refusing a start over.
    """
    for item in reversed(getattr(args, "set", None) or []):
        key, _, value = str(item).partition("=")
        if key.strip() == "sandbox.gpus":
            return f"--set sandbox.gpus={value.strip()}"
    if getattr(args, "gpus", ""):
        return f"--gpus {args.gpus}"
    raw = settings.get(root, "sandbox.gpus", gpus.AUTO)
    where = settings.origin(root, "sandbox.gpus")
    if where is None:
        return "the default (auto: every visible card)"
    try:
        spec = gpus.render(raw)
    except (TypeError, ValueError):
        spec = str(raw)
    return f"sandbox.gpus: {spec} in {where}"


def cmd_resume(args) -> int:
    root = paths.project_root()
    run, _, _ = layout.load_run(args.run_id, root)
    if (h := lifecycle.holder(run)) is not None:
        print(f"{args.run_id} is still being driven (pid {h.get('pid')}) — "
              f"`pinnforge run attach {args.run_id}` to watch it")
        return 2
    command = ["run", "_drive", run.name, "--resume"]
    if args.blocks is not None:
        command += ["-n", str(args.blocks)]
    return _launch(args, root, run, command)


def _launch(args, root: Path, run: Path, command: list[str]) -> int:
    """Run `command` (this CLI) detached, or here in the foreground."""
    log = paths.logs_dir(run) / "orchestrator.log"
    if not _detach_wanted(args, root):
        if url := _ensure_ui(args, root):
            print(f"  dashboard  {url}")
        return cmd_drive(build_drive_args([*command[2:], "--log", str(log)]))
    name = lifecycle.session_name(root, run.name)
    info = lifecycle.detach(
        lifecycle.self_command("--root", str(root), *command, "--log", str(log)),
        cwd=root,
        log=log,
        name=name,
        record=run / lifecycle.DETACH_FILE,
    )
    where = "in tmux" if info["mode"] == "tmux" else f"in the background (pid {info['pid']})"
    print(f"\n{run.name} is running {where}; it outlives this terminal.")
    if url := _ensure_ui(args, root):
        print(f"  dashboard  {url}")
    print(f"  follow     pinnforge run attach {run.name}"
          + ("      (Ctrl-b d detaches again)" if info["mode"] == "tmux" else ""))
    print(f"  status     pinnforge run status {run.name}")
    print(f"  stop       pinnforge run stop {run.name}")
    return 0


def _ensure_ui(args, root: Path) -> str | None:
    if getattr(args, "no_ui", False) or not settings.get(root, "ui.autostart", True):
        return None
    try:
        from pinnforge.ui import server
    except ImportError:
        return None
    try:
        return server.ensure_running(root)
    except OSError as e:
        print(f"  (dashboard not started: {e})", file=sys.stderr)
        return None


# ─────────────────────────────── the driver ───────────────────────────────


def _drive_arguments(p) -> None:
    p.add_argument("run_id")
    p.add_argument("--no-anchor", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("-n", "--blocks", type=int, default=None)
    p.add_argument("--log", default=None)


def build_drive_args(argv: list[str]) -> argparse.Namespace:
    """The namespace `_drive` would get from the command line."""
    p = argparse.ArgumentParser(prog="pinnforge run _drive")
    _drive_arguments(p)
    return p.parse_args(argv)


class _Tee:
    """A stream into the run's log as well as the terminal, whatever happens to the terminal."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", buffering=1)
        # Detached without tmux, the terminal *is* the log: writing to both
        # printed every line twice.
        if _same_file(stream, self.file):
            self.stream = None

    def write(self, text: str) -> int:
        self.file.write(text)
        if self.stream is not None:
            with contextlib.suppress(OSError, ValueError):
                self.stream.write(text)
        return len(text)

    def flush(self) -> None:
        self.file.flush()
        if self.stream is not None:
            with contextlib.suppress(OSError, ValueError):
                self.stream.flush()

    def isatty(self) -> bool:
        return False


def _same_file(stream, other) -> bool:
    try:
        a, b = os.fstat(stream.fileno()), os.fstat(other.fileno())
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _log_crashes(log: Path) -> None:
    """Send stderr — and with it any traceback, the score thread's included — to the log too.

    Only stdout used to be tee'd, so a detached driver that crashed left a
    log ending mid-sentence and the reason on a terminal nobody was watching.
    """
    sys.stderr = _Tee(sys.stderr, log)


def cmd_drive(args) -> int:
    """Own one run: lock it, serve its scores, install its anchor, run its blocks.

    Everything that starts or resumes a run ends up here, in the foreground or
    in a detached session, so a run behaves the same however it was launched.
    """
    root = paths.project_root()
    run, cfg, _ = layout.load_run(args.run_id, root)
    log = Path(args.log) if args.log else None
    if log is not None:
        sys.stdout = _Tee(sys.stdout, log)
        _log_crashes(log)
    _logging()
    try:
        lock = lifecycle.RunLock(run).acquire()
    except lifecycle.RunLocked as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    lifecycle.install_signal_handlers(
        escalate=lambda: lifecycle.kill_group(lock.info.get("agent_pgid"))
    )
    lifecycle.claim(cfg.sandbox.gpus, run)
    # Which block the score service answers for: the driver's word, so a block
    # cannot open a fresh score cap by filing under a block id of its choosing.
    dispatch = scoring.Dispatch()
    try:
        try:
            # The daemon has to be up before the anchor is measured: measuring
            # runs eval.py, which files a score request like any evaluation.
            with scoring.serving(root, run, dispatch=dispatch) as server:
                rc = _execute(args, run, cfg, lock, server=server, dispatch=dispatch)
        finally:
            lifecycle.release_claim(cfg.sandbox.gpus)  # the film needs no card
        _film_after(root, run, lock)
        return rc
    except KeyboardInterrupt:
        print("\ninterrupted — state checkpointed; resume with:")
        print(f"  pinnforge run resume {run.name}")
        return 130
    finally:
        lock.release()


def _recorded(lock):
    """An `on_start` that records an agent's process group in the run's lock."""
    return lambda pgid: lock.update(agent_pgid=pgid)


def _film_after(root: Path, run: Path, lock) -> None:
    """A run that finished gets its discovery film, its own agent writing the script.

    Best effort, under the run's lock: the run is done whether or not its film
    could be made, and `pinnforge run film <id>` makes it again.
    """
    _, cfg, state = layout.load_run(run.name, root)
    if not cfg.film or state.status != "finished":
        return
    from pinnforge import discovery

    print(f"discovery film: {cfg.agent.runtime} is writing its script ...")
    try:
        film = discovery.make(root, run, cfg.agent, on_start=_recorded(lock))
    except KeyboardInterrupt:
        print(f"discovery film interrupted; make it with: pinnforge run film {run.name}")
        return
    except Exception as e:  # the run itself is done
        logging.getLogger(__name__).warning("the discovery film could not be made",
                                            exc_info=True)
        print(f"discovery film not made ({e}); try: pinnforge run film {run.name}")
        return
    finally:
        lock.update(agent_pgid=None)
    _film_said(film)


def _film_said(film) -> None:
    how = "the agent's script" if film.scripted_by == "agent" else "the draft script"
    print(f"discovery film: {film.path} ({how})")
    for p in film.problems:
        print(f"  {p}")


def _execute(args, run, cfg, lock=None, server=None, dispatch=None) -> int:
    _, _, state = layout.load_run(run.name)
    if not args.resume:
        # Remembered, so `run resume` starts the run the way `run start` did:
        # resume has no flags of its own for these.
        state.no_anchor, state.force = bool(args.no_anchor), bool(args.force)
        state.save(paths.state_path(run))
    no_anchor = bool(args.no_anchor or state.no_anchor or _anchor_skipped_before(run))
    force = bool(args.force or state.force)

    # A resumed run gets its anchor too, if it has none yet: one whose start
    # was interrupted mid-measurement used to resume straight into blocks
    # with no yardstick at all — and with no refusal either.
    if not no_anchor and not (args.resume and _anchor_scored(run)):
        if args.resume:
            print("b00 has no scored record — installing the anchor before resuming")
            _clear_anchor(run)
        print("installing b00 anchor…")
        first = cfg.sandbox.gpus[0] if cfg.sandbox.gpus else None
        if dispatch is not None:
            dispatch.block = "b00"
        try:
            result = anchor.ensure(run, cfg.task, gpu=first, hw=cfg.hw or None)
        finally:
            if dispatch is not None:
                dispatch.block = None
        where = "restored from cache" if result.restored else result.detail
        if result.rRMSE is None:
            print(f"  b00 unavailable: {result.detail}")
            if not force:
                print("  refusing to run blocks without an anchor (use --force to override)")
                return 1
        else:
            print(f"  b00 rRMSE {result.rRMSE:.5g} ({where})")

    _, cfg, state = layout.load_run(run.name)
    if args.blocks is not None:
        n = args.blocks
    else:
        n = _remaining(run, cfg, state) if args.resume else cfg.blocks
    if n <= 0:
        print(f"{run.name}: nothing to resume ({state.status})")
        return 0
    if args.resume:
        print(f"resuming {run.name}: {n} block(s) to go")
    orch = Orchestrator(run, cfg, state)
    # Checked between segments: a dead in-process scorer is restarted, a gone
    # standalone one stops the run.
    orch.scorer = server if server is not None else scoring.Standalone(run)
    orch.dispatch = dispatch
    if lock is not None:
        orch.on_dispatch = lambda h: lock.update(agent_pgid=h._pgid())
        orch.on_segment_end = lambda: lock.update(agent_pgid=None)
    try:
        done = orch.run_blocks(n)
    except EnvironmentFailure as e:
        print(f"environment failure: {e}")
        return 1
    print(f"finished {len(done)}/{n} block(s)")
    return _report(run.name)


def _anchor_skipped_before(run: Path) -> bool:
    """A run from before the flags were recorded that was started with `--no-anchor`.

    Its blocks ran with no b00 directory ever made — every other start makes
    one before the first block — so resuming it must not set off a GPU
    measurement that its start deliberately skipped.
    """
    try:
        raw = json.loads(paths.state_path(run).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict) or "no_anchor" in raw:
        return False
    blocks = [b for b in layout.existing_block_ids(run) if b != "b00"]
    return bool(blocks) and not paths.block_dir(run, "b00").exists()


def _anchor_scored(run: Path) -> bool:
    return best_score(EvalRecord.read_all(paths.evals_path(run, "b00"))) is not None


def _clear_anchor(run: Path) -> None:
    """Drop what an interrupted anchor measurement left, so it starts clean.

    A half-measured b00 holds a pre-charged budget, a candidate copy and
    perhaps records without a score; measured again on top of them, its
    records would mix two measurements.
    """
    shutil.rmtree(paths.block_dir(run, "b00"), ignore_errors=True)
    with contextlib.suppress(FileNotFoundError):
        paths.summary_path(run, "b00").unlink()


def _remaining(run, cfg, state=None) -> int:
    """Blocks still owed: the configured count minus the slots already used.

    A slot is used by a block verified done, and equally by one the loop
    closed as `short` or `failed`: those had their dispatches, and counting
    them as still owed made a resume run them again — or, re-adopted, spend
    the extra slots on the very block that had just been given up on.
    """
    if state is None:
        _, _, state = layout.load_run(run.name)
    per_run = contract.train_time(run / "task") or 0.0
    used = 0
    for b in layout.existing_block_ids(run):
        if b == "b00":
            continue
        bs = state.blocks.get(b)
        if (bs is not None and bs.status in SETTLED) or verify_block(run, b, cfg, per_run).done:
            used += 1
    return max(0, cfg.blocks - used)


# ─────────────────────────────── reading ───────────────────────────────


def _live(run: Path) -> str:
    h = lifecycle.holder(run)
    return f"live (pid {h.get('pid')})" if h else "not running"


def cmd_list(args) -> int:
    runs = paths.list_runs(task=args.task)
    if not runs:
        print("no runs")
        return 0
    print(f"{'run':<16} {'status':<10} {'live':<5} {'blocks':>7}  best rRMSE")
    for run_id in runs:
        run, _, state = layout.load_run(run_id)
        best, block = ledger.overall_best(run)
        shown = f"{best:.5g} ({block})" if best is not None else "—"
        n = len([b for b in layout.existing_block_ids(run) if b != "b00"])
        live = "yes" if lifecycle.holder(run) else ""
        print(f"{run_id:<16} {state.status:<10} {live:<5} {n:>7}  {shown}")
    return 0


def cmd_status(args) -> int:
    run, cfg, state = layout.load_run(args.run_id)
    per_run = contract.train_time(run / "task") or 0.0
    print(f"run      {state.run_id}   task {state.task}   status {state.status}   "
          f"driver {_live(run)}")
    resolved = sorted({b.model_resolved for b in state.blocks.values() if b.model_resolved})
    asked = cfg.agent.model or "default"
    ran = f" -> {', '.join(resolved)}" if resolved and resolved != [asked] else ""
    print(f"agent    {cfg.agent.runtime} / {asked}{ran}")
    print(f"sandbox  gpus {cfg.sandbox.gpus}   budget {cfg.sandbox.wall_budget_s:.0f} s/block")
    print()
    print(f"{'block':<7} {'status':<9} {'wall s':>9} {'evals':>6} {'summary':>8}  best")
    for block_id in layout.existing_block_ids(run):
        v = verify_block(run, block_id, cfg, per_run)
        recs = EvalRecord.read_all(paths.evals_path(run, block_id))
        best = best_score(recs)
        bs = state.blocks.get(block_id)
        if bs:
            status = bs.status
        elif block_id == "b00":
            status = lifecycle.anchor_status(run, best is not None)
        else:
            status = "—"
        print(
            f"{block_id:<7} {status:<9} {v.spent:>9.0f} {v.scored_evals:>6} "
            f"{'yes' if v.has_summary else 'no':>8}  "
            f"{format(best, '.5g') if best is not None else '—'}"
        )
    return 0


def cmd_log(args) -> int:
    run, _, _ = layout.load_run(args.run_id)
    rows = []
    for block_id in layout.existing_block_ids(run):
        for rec in EvalRecord.read_all(paths.evals_path(run, block_id)):
            if rec.scored and rec.rRMSE is not None:
                rows.append((rec.rRMSE, block_id, rec.candidate, rec.wall_s))
    rows.sort()
    if not rows:
        print("no scored evaluations yet")
        return 0
    print(f"{'rRMSE':>12}  {'block':<6} {'wall s':>7}  candidate")
    for score, block_id, cand, wall in rows[: args.number]:
        print(f"{score:>12.5g}  {block_id:<6} {wall:>7.0f}  {cand}")
    return 0


def _report(run_id: str) -> int:
    run, _, _ = layout.load_run(run_id)
    path = ledger.write_run_summary(run)
    best, block = ledger.overall_best(run)
    print(f"summary: {path}")
    if best is not None:
        print(f"overall best: {best:.5g} in {block}")
    return 0


def cmd_audit(args) -> int:
    """Was the record tampered with? The artefact, not the assurance."""
    run, _, _ = layout.load_run(args.run_id)
    report = integrity.audit(run)
    unverified = report.get("unverified", [])
    if report["segments_checked"] == 0 and not unverified:
        print(f"{args.run_id}: no integrity manifests — nothing has been dispatched yet")
        return 0
    print(f"{args.run_id}: {report['segments_checked']} dispatch segment(s) verified")
    if report["clean"]:
        print("  clean — every protected file matched its pre-dispatch hash")
        return 0
    for r in report["violations"]:
        print(f"  block {r['block']} segment {r['segment']}:")
        for v in r["violations"]:
            detail = f"  ({v['detail']})" if v.get("detail") else ""
            print(f"    {v['kind']:<10} {v['path']}{detail}")
    if unverified:
        print(f"  {len(unverified)} dispatch segment(s) were never verified:")
        for u in unverified:
            print(f"    block {u['block']} segment {u['segment']}  ({u['reason']})")
        if lifecycle.holder(run) is not None:
            print("  (the run is live: the segment being dispatched now is verified when it ends)")
    if report["violations"]:
        print(f"\n{len(report['violations'])} segment(s) violated the record — "
              "results from this run cannot be trusted as they stand")
    else:
        print("\nno violation found, but the record is not fully checked")
    return 1


def cmd_stop(args) -> int:
    """Stop a run the way Ctrl-C would, then reconcile what its state says.

    A live run is stopped by signalling the process that holds its lock: it
    interrupts the agent's whole process group, books the segment and writes
    the integrity verdict before it exits. Only when nobody holds the lock is
    `state.json` edited here — editing it under a live orchestrator was
    overwritten at its next checkpoint, which is why stop never used to stop.

    A stopped run has no running blocks in it, so any left marked that way are
    reconciled too. Otherwise `run status` keeps reporting a block as running
    long after its process is gone, which is the one thing that display exists
    to tell you.
    """
    run, _, _ = layout.load_run(args.run_id)
    outcome = lifecycle.stop(run, force=args.force)
    if outcome.startswith(("still stopping", "refused")) or lifecycle.holder(run) is not None:
        # whatever was said, a held lock is a live driver: its state is its own
        print(f"{args.run_id}: {outcome}")
        return 1
    _, _, state = layout.load_run(args.run_id)
    stale = [b for b in state.blocks.values() if b.status == "running"]
    if state.status == "running" or stale:
        state.status = "stopped"
        for bs in stale:
            bs.status = "interrupted"
            bs.exit_reason = bs.exit_reason or "run stopped while this block was dispatched"
        state.save(paths.state_path(run))
    note = f" ({', '.join(b.block_id for b in stale)} marked interrupted)" if stale else ""
    print(f"{args.run_id}: {outcome}{note}")
    return 0


def cmd_attach(args) -> int:
    run, _, _ = layout.load_run(args.run_id)
    return lifecycle.attach(run / lifecycle.DETACH_FILE)


def cmd_watch(args) -> int:
    import sys as _sys

    from pinnforge.ui import api, tui

    root = paths.project_root()
    api.ROOT_REAL = str(root.resolve())
    if not _sys.stdout.isatty() or args.once:
        for _, text in tui.lines(root, args.run_id):
            print(text)
        return 0
    return tui.watch(root, args.run_id)


def cmd_report(args) -> int:
    from pinnforge.ui import report

    root = paths.project_root()
    layout.load_run(args.run_id, root)
    out = Path(args.output or f"{args.run_id}-report.html")
    out.write_text(report.render(root, args.run_id), encoding="utf-8")
    print(f"report: {out.resolve()} ({out.stat().st_size / 1e6:.1f} MB) — "
          "self-contained; open it in any browser")
    return 0


def cmd_film(args) -> int:
    """Make a run's discovery film again: its agent rewrites the script, or, with
    `--render-only`, the script already written is rendered again."""
    from pinnforge import discovery

    root = paths.project_root()
    run, cfg, _ = layout.load_run(args.run_id, root)
    if args.render_only:
        if not (run / discovery.PACKAGE / "facts.json").is_file():
            print(f"{run.name} has no film package yet; make one with: "
                  f"pinnforge run film {run.name}")
            return 1
        _film_said(discovery.render(root, run))
        return 0
    # Under the run's lock, like a block: no driver at the same time, and the
    # agent's group recorded where `run stop` and the next driver look for it.
    try:
        lock = lifecycle.RunLock(run).acquire()
    except lifecycle.RunLocked as e:
        print(f"error: {e} — its film is made when the run finishes")
        return 2
    lifecycle.install_signal_handlers(
        escalate=lambda: lifecycle.kill_group(lock.info.get("agent_pgid")))
    try:
        print(f"discovery film: {cfg.agent.runtime} is writing its script ...")
        _film_said(discovery.make(root, run, cfg.agent, timeout_s=args.timeout * 60,
                                  on_start=_recorded(lock)))
    finally:
        lock.release()
    return 0


def cmd_show(args) -> int:
    run, _, _ = layout.load_run(args.run_id)
    path = paths.evals_path(run, args.block)
    if not path.is_file():
        print(f"no evals for {args.block}")
        return 2
    for rec in EvalRecord.read_all(path):
        print(json.dumps(rec.raw))
    summary = paths.summary_path(run, args.block)
    if summary.is_file():
        print()
        print(summary.read_text(encoding="utf-8"))
    return 0


def _launch_flags(p) -> None:
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true", help="drive the run in this terminal")
    mode.add_argument("--detach", action="store_true", help="return at once (the default)")
    p.add_argument("--no-ui", action="store_true", help="do not start the dashboard")


def register(sub) -> None:
    p = sub.add_parser("run", help="start, resume and inspect runs")
    s = p.add_subparsers(dest="run_cmd", required=True)

    start = s.add_parser("start", help="create a discovery run and carry out its rounds")
    start.add_argument("-t", "--task", required=True)
    start.add_argument("-n", "--blocks", type=int, default=1, help="rounds after round 0")
    start.add_argument("--agent", default=None,
                       help="agent binding from `pinnforge setup` (default: the default binding)")
    start.add_argument("--runtime", default=None, help="harness CLI, instead of a binding")
    start.add_argument(
        "--model",
        default="",
        help="model id for every round, e.g. claude-opus-4-8. Prefer an exact id "
        "over an alias like `opus`: an alias moves between model generations and "
        "two runs that share it are not comparable",
    )
    start.add_argument(
        "--gpus",
        default="",
        help="the cards for this run: all, none, or ids such as 1, 0,1 or '0 1' "
        "(default: the workspace's sandbox.gpus — see `pinnforge setup gpus`)",
    )
    start.add_argument("--budget", default="", help="GPU budget of a round, in wall seconds")
    start.add_argument("--no-anchor", action="store_true", help="skip round 0, the anchor")
    start.add_argument("--force", action="store_true", help="run the rounds even without an anchor")
    start.add_argument("--share", action="store_true",
                       help="start even if another run is using the same GPUs")
    start.add_argument("--set", action="append", metavar="k.v=x", help="config override")
    _launch_flags(start)
    start.set_defaults(func=cmd_start)

    res = s.add_parser("resume", help="continue an interrupted run")
    res.add_argument("run_id")
    res.add_argument("-n", "--blocks", type=int, default=None, help="rounds still to run")
    _launch_flags(res)
    res.set_defaults(func=cmd_resume)

    drive = s.add_parser("_drive", help=argparse.SUPPRESS)
    _drive_arguments(drive)
    drive.set_defaults(func=cmd_drive)

    att = s.add_parser("attach", help="watch a detached run (its tmux session, or its log)")
    att.add_argument("run_id")
    att.set_defaults(func=cmd_attach)

    wt = s.add_parser("watch", help="the dashboard's essentials in the terminal")
    wt.add_argument("run_id")
    wt.add_argument("--once", action="store_true", help="print once and exit")
    wt.set_defaults(func=cmd_watch)

    rp = s.add_parser("report", help="one run as a self-contained HTML file")
    rp.add_argument("run_id")
    rp.add_argument("-o", "--output", default=None, help="default: <run>-report.html")
    rp.set_defaults(func=cmd_report)

    fm = s.add_parser("film", help="make a run's discovery film again (runs/<id>/discovery.html)")
    fm.add_argument("run_id")
    fm.add_argument("--render-only", action="store_true",
                    help="render the script already written, without an agent")
    fm.add_argument("--timeout", type=float, default=45, help="minutes the agent may take")
    fm.set_defaults(func=cmd_film)

    from pinnforge.cli.delete_cmd import cmd_run_delete

    dl = s.add_parser("delete", help="delete a run: its folder and its score ledger")
    dl.add_argument("run_id")
    dl.add_argument("-y", "--yes", action="store_true", help="do not ask first")
    dl.set_defaults(func=cmd_run_delete)

    lst = s.add_parser("list", help="list runs")
    lst.add_argument("--task", default=None)
    lst.set_defaults(func=cmd_list)

    st = s.add_parser("status", help="per-round status of one run")
    st.add_argument("run_id")
    st.set_defaults(func=cmd_status)

    lg = s.add_parser("log", help="scored evaluations, best first")
    lg.add_argument("run_id")
    lg.add_argument("-n", "--number", type=int, default=20)
    lg.set_defaults(func=cmd_log)

    sh = s.add_parser("show", help="one round's evaluation records and summary")
    sh.add_argument("run_id")
    sh.add_argument("block")
    sh.set_defaults(func=cmd_show)

    au = s.add_parser("audit", help="verify that no round altered another's record")
    au.add_argument("run_id")
    au.set_defaults(func=cmd_audit)

    stop = s.add_parser("stop", help="stop a run the way Ctrl-C would")
    stop.add_argument("run_id")
    stop.add_argument("--force", action="store_true",
                      help="kill the run and its agent if it does not stop in 10 s")
    stop.set_defaults(func=cmd_stop)
