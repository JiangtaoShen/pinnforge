"""`pinnforge run …` — start, resume, inspect and stop runs."""

from __future__ import annotations

import json
import logging

from pinnforge import integrity, paths, scoring
from pinnforge.config import AgentConfig, RunConfig, SandboxConfig, apply_overrides
from pinnforge.orchestrator import EnvironmentFailure, Orchestrator, verify_block
from pinnforge.run import anchor, layout, ledger
from pinnforge.runtime import registry as runtime_registry
from pinnforge.task import contract
from pinnforge.types import EvalRecord, best_score


def _build_config(args) -> RunConfig:
    d = {
        "task": args.task,
        "blocks": args.blocks,
        "agent": AgentConfig(runtime=args.runtime, model=args.model or "").to_dict(),
        "sandbox": SandboxConfig().to_dict(),
    }
    if args.gpus:
        d["sandbox"]["gpus"] = [int(g) for g in args.gpus.split(",") if g.strip()]
    if args.budget:
        d["sandbox"]["wall_budget_s"] = float(args.budget)
    apply_overrides(d, args.set or [])
    return RunConfig.from_dict(d)


def cmd_start(args) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )
    cfg = _build_config(args)
    # Settle the model before anything is built: a harness that names no
    # default has to be told one, and finding that out after the run directory
    # and a measured anchor exist would cost a GPU for nothing.
    runtime_registry.resolve_model(runtime_registry.get_runtime(cfg.agent.runtime), cfg.agent.model)
    # create_run adopts the task's declared per-block budget when the caller
    # did not ask for a different one.
    run = layout.create_run(cfg)
    cfg = RunConfig.load(paths.config_path(run))
    print(f"run {run.name} created at {run} (budget {cfg.sandbox.wall_budget_s:.0f} s/block)")

    # The daemon has to be up before the anchor is measured: measuring runs
    # eval.py, and eval.py files a score request like any other evaluation.
    with scoring.serving(paths.project_root(), run):
        return _execute(args, run, cfg)


def _execute(args, run, cfg) -> int:
    if not args.no_anchor:
        print("installing b00 anchor…")
        result = anchor.ensure(run, cfg.task, gpu=cfg.sandbox.gpus[0])
        where = "restored from cache" if result.restored else result.detail
        if result.rRMSE is None:
            print(f"  b00 unavailable: {result.detail}")
            if not args.force:
                print("  refusing to run blocks without an anchor (use --force to override)")
                return 1
        else:
            print(f"  b00 rRMSE {result.rRMSE:.5g} ({where})")

    _, cfg, state = layout.load_run(run.name)
    orch = Orchestrator(run, cfg, state)
    try:
        done = orch.run_blocks(cfg.blocks)
    except EnvironmentFailure as e:
        print(f"environment failure: {e}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — state checkpointed; resume with:")
        print(f"  pinnforge run resume {run.name}")
        return 130
    print(f"finished {len(done)}/{cfg.blocks} block(s)")
    return _report(run.name)


def cmd_resume(args) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )
    run, cfg, state = layout.load_run(args.run_id)
    n = args.blocks if args.blocks is not None else _remaining(run, cfg)
    if n <= 0:
        print(f"{args.run_id}: nothing to resume ({state.status})")
        return 0
    print(f"resuming {args.run_id}: {n} block(s) to go")
    orch = Orchestrator(run, cfg, state)
    try:
        with scoring.serving(paths.project_root(), run):
            orch.run_blocks(n)
    except EnvironmentFailure as e:
        print(f"environment failure: {e}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — state checkpointed")
        return 130
    return _report(args.run_id)


def _remaining(run, cfg) -> int:
    """Blocks still owed: the configured count minus those verified done."""
    per_run = contract.train_time(run / "task") or 0.0
    done = sum(
        1
        for b in layout.existing_block_ids(run)
        if b != "b00" and verify_block(run, b, cfg, per_run).done
    )
    return max(0, cfg.blocks - done)


def cmd_list(args) -> int:
    runs = paths.list_runs(task=args.task)
    if not runs:
        print("no runs")
        return 0
    print(f"{'run':<16} {'status':<10} {'blocks':>7}  best rRMSE")
    for run_id in runs:
        run, _, state = layout.load_run(run_id)
        best, block = ledger.overall_best(run)
        shown = f"{best:.5g} ({block})" if best is not None else "—"
        n = len([b for b in layout.existing_block_ids(run) if b != "b00"])
        print(f"{run_id:<16} {state.status:<10} {n:>7}  {shown}")
    return 0


def cmd_status(args) -> int:
    run, cfg, state = layout.load_run(args.run_id)
    per_run = contract.train_time(run / "task") or 0.0
    print(f"run      {state.run_id}   task {state.task}   status {state.status}")
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
        status = bs.status if bs else ("done" if block_id == "b00" else "—")
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
    if report["segments_checked"] == 0:
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
    print(f"\n{len(report['violations'])} segment(s) violated the record — "
          "results from this run cannot be trusted as they stand")
    return 1


def cmd_stop(args) -> int:
    """Mark a run stopped. Live processes are in their own group; if one is
    still attached, interrupt the CLI that owns it rather than this state.

    A stopped run has no running blocks in it, so any left marked that way are
    reconciled too. Otherwise `run status` keeps reporting a block as running
    long after its process is gone, which is the one thing that display exists
    to tell you.
    """
    run, _, state = layout.load_run(args.run_id)
    state.status = "stopped"
    stale = [b for b in state.blocks.values() if b.status == "running"]
    for bs in stale:
        bs.status = "interrupted"
        bs.exit_reason = bs.exit_reason or "run stopped while this block was dispatched"
    state.save(paths.state_path(run))
    note = f" ({', '.join(b.block_id for b in stale)} marked interrupted)" if stale else ""
    print(f"{args.run_id}: marked stopped{note}")
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


def register(sub) -> None:
    p = sub.add_parser("run", help="start, resume and inspect runs")
    s = p.add_subparsers(dest="run_cmd", required=True)

    start = s.add_parser("start", help="create a run and execute its blocks")
    start.add_argument("-t", "--task", required=True)
    start.add_argument("-n", "--blocks", type=int, default=1)
    start.add_argument("--runtime", default="claude_code")
    start.add_argument(
        "--model",
        default="",
        help="model id for every block, e.g. claude-opus-4-8 (default) or "
        "claude-opus-5. Prefer an exact id over an alias like `opus`: an "
        "alias moves between model generations and two runs that share it "
        "are not comparable",
    )
    start.add_argument("--gpus", default="", help="comma-separated GPU ids, e.g. 0,1")
    start.add_argument("--budget", default="", help="per-block GPU wall seconds")
    start.add_argument("--no-anchor", action="store_true", help="skip the b00 control node")
    start.add_argument("--force", action="store_true", help="run blocks even without an anchor")
    start.add_argument("--set", action="append", metavar="k.v=x", help="config override")
    start.set_defaults(func=cmd_start)

    res = s.add_parser("resume", help="continue an interrupted run")
    res.add_argument("run_id")
    res.add_argument("-n", "--blocks", type=int, default=None)
    res.set_defaults(func=cmd_resume)

    lst = s.add_parser("list", help="list runs")
    lst.add_argument("--task", default=None)
    lst.set_defaults(func=cmd_list)

    st = s.add_parser("status", help="per-block status of one run")
    st.add_argument("run_id")
    st.set_defaults(func=cmd_status)

    lg = s.add_parser("log", help="scored evaluations, best first")
    lg.add_argument("run_id")
    lg.add_argument("-n", "--number", type=int, default=20)
    lg.set_defaults(func=cmd_log)

    sh = s.add_parser("show", help="one block's records and summary")
    sh.add_argument("run_id")
    sh.add_argument("block")
    sh.set_defaults(func=cmd_show)

    au = s.add_parser("audit", help="verify no block altered another's record")
    au.add_argument("run_id")
    au.set_defaults(func=cmd_audit)

    stop = s.add_parser("stop", help="mark a run stopped")
    stop.add_argument("run_id")
    stop.set_defaults(func=cmd_stop)
