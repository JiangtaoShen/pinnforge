"""Choosing which GPUs a workspace's runs use — at `init`, in `setup`, or `setup gpus`.

The choice is the operator's, made once and written down as `sandbox.gpus`:
on a shared machine "all of them" is rarely right, and a card someone else
is training on makes every wall-clock-bounded run on it look worse. So the
prompt shows each card with what is on it right now, and stores exactly
what was picked.
"""

from __future__ import annotations

from pinnforge import gpus
from pinnforge.run import lifecycle


def label(spec) -> str:
    """How a stored choice reads as an answer: all, none, or `0,1`."""
    if gpus.render(spec) == gpus.AUTO:
        return "all"
    pool = gpus.resolve(spec, inv=[])
    return ",".join(map(str, pool)) if pool else "none"


def table(inv: list[gpus.Gpu]) -> list[str]:
    """One line per card: index, model, memory, and who is using it now."""
    load = gpus.usage()
    held = {}
    for g, info in lifecycle.leases([c.index for c in inv]):
        held.setdefault(g, []).append(str(info.get("run", "")).rsplit("/", 1)[-1])
    width = max((len(g.name) for g in inv), default=0)
    rows = []
    for g in inv:
        line = f"  [{g.index}]  {g.name:<{width}}  {g.memory_mb / 1024:>3.0f} GB"
        if g.index in load:
            mem, util = load[g.index]
            line += f"   now: {mem / 1024:.1f} GB used, {util}% busy"
        if g.index in held:
            line += f"   (PINNForge run {', '.join(held[g.index])})"
        rows.append(line)
    return rows


def ask(inv: list[gpus.Gpu], current="auto") -> str | list[int]:
    """Ask until the answer names cards this machine has; returns the value to store."""
    print("GPUs on this machine:")
    for row in table(inv):
        print(row)
    print()
    print("Which may this workspace's runs use?")
    print("  all     every card visible when a run starts (follows CUDA_VISIBLE_DEVICES)")
    print("  1       one card")
    print("  0,1     exactly these cards; with several, a block evaluates on them in parallel")
    print("  none    CPU only")
    default = label(current)
    while True:
        try:
            answer = input(f"gpus [{default}]: ").strip()
        except EOFError:
            answer = ""
        try:
            return gpus.parse_choice(answer or default, inv)
        except ValueError as e:
            print(f"  {e}")


def summary(spec, inv: list[gpus.Gpu]) -> str:
    """`[1] — 1 × NVIDIA RTX A6000 (48 GB) [1]`, for confirming a choice."""
    return f"{gpus.render(spec)} — {gpus.describe(gpus.resolve(spec, inv=inv), inv)}"
