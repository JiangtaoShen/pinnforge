"""`pinnforge run delete` and `pinnforge task delete`: show what goes, ask, then remove it.

The dashboard never deletes anything (it is read-only, and any local process
can reach it), so it hands the operator these commands instead.
"""

from __future__ import annotations

from pinnforge import paths, removal


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _show(plan: removal.Plan, indent: str = "  ") -> None:
    for p in plan.paths:
        print(f"{indent}{p!s:<64} {_size(removal.size(p)):>9}")
    if plan.ledger_lines is not None:
        ledger, queue = plan.ledger_lines
        print(f"{indent}{queue}'s lines in {ledger}")


def _go(plan: removal.Plan, yes: bool) -> int:
    blockers = plan.blockers + [b for r in plan.runs for b in r.blockers]
    if blockers:
        print(f"{plan.what} cannot be deleted now:")
        for b in blockers:
            print(f"  {b}")
        return 1
    print(f"{plan.what} will be deleted:")
    _show(plan)
    if plan.runs:
        print("  with its runs:")
        for r in plan.runs:
            _show(r, "    ")
    for n in plan.notes:
        print(f"note: {n}")
    if not yes:
        try:
            answer = input("This cannot be undone. Delete? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return 1
    errors = removal.carry_out(plan)
    if errors:
        print(f"{plan.what} was only partly deleted:")
        for e in errors:
            print(f"  {e}")
        return 1
    print(f"deleted {plan.what}" + (f" and {len(plan.runs)} run(s)" if plan.runs else ""))
    return 0


def cmd_run_delete(args) -> int:
    try:
        plan = removal.run_plan(paths.project_root(), args.run_id)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 2
    return _go(plan, args.yes)


def cmd_task_delete(args) -> int:
    try:
        plan = removal.task_plan(paths.project_root(), args.name, with_runs=args.with_runs)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}")
        return 2
    return _go(plan, args.yes)
