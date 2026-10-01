"""The GPU pool a run may use, found rather than assumed.

The pool used to default to `(0, 1)` — the machine this was built on — so a
single-GPU workstation got as far as its first block and died there, with a
run directory and a measured anchor already paid for. Now `gpus: auto`
resolves at `run start` to what is actually visible:

1. `CUDA_VISIBLE_DEVICES`, if set — integers or `GPU-…` UUIDs, mapped onto
   `nvidia-smi` indices; empty or `-1` means "no GPU";
2. otherwise every GPU `nvidia-smi` lists;
3. otherwise none: CPU mode, which runs, slowly — fine for the demo task.

Which cards a workspace uses is the operator's call, made at `pinnforge
init` (or later with `pinnforge setup gpus`) and stored as `sandbox.gpus` in
`pinnforge.yaml`: `auto`, an explicit list such as `[1]` or `[0, 2, 3]`, or
`[]` for CPU only. `run start --gpus` overrides it for one run.

The resolved list is frozen into `run.yaml` like every other setting, so a
run never changes pools under a resume. Indices are `nvidia-smi`'s, which are
PCI-bus order; blocks are given `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that CUDA
numbers the cards the same way when `eval.py --gpu N` selects one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

AUTO = "auto"
ALL_WORDS = ("auto", "all")
NONE_WORDS = ("none", "cpu")


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_mb: int


def inventory(timeout: float = 20.0) -> list[Gpu]:
    """Every GPU `nvidia-smi` can see, in its (PCI) order. Empty if none."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(Gpu(int(parts[0]), parts[1], parts[2], int(float(parts[3]))))
        except ValueError:
            continue
    return gpus


def visible(env: dict[str, str] | None = None, inv: list[Gpu] | None = None) -> list[int] | None:
    """The indices `CUDA_VISIBLE_DEVICES` allows, or None when it is unset."""
    env = os.environ if env is None else env
    raw = env.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw in ("", "-1", "none", "NoDevFiles"):
        return []
    inv = inventory() if inv is None else inv
    out: list[int] = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            continue
        if token.isdigit():
            out.append(int(token))
            continue
        # A UUID, or a unique prefix of one, as CUDA itself accepts.
        matches = [g.index for g in inv if g.uuid == token or g.uuid.startswith(token)]
        if len(matches) == 1:
            out.append(matches[0])
    return out


def usage(timeout: float = 10.0) -> dict[int, tuple[int, int]]:
    """{index: (memory used in MB, utilisation %)} right now — to choose by."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    found = {}
    for line in out.stdout.splitlines():
        try:
            i, mem, util = (int(float(p)) for p in line.split(","))
        except ValueError:
            continue
        found[i] = (mem, util)
    return found


def resolve(spec, env: dict[str, str] | None = None, inv: list[Gpu] | None = None) -> list[int]:
    """Turn a configured pool — `auto`, a list, an int, `"0,1"`, `"0 1"`, `none` — into indices."""
    if isinstance(spec, str) and spec.strip().lower() in NONE_WORDS:
        return []
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ALL_WORDS):
        inv = inventory() if inv is None else inv
        allowed = visible(env, inv)
        known = [g.index for g in inv]
        if allowed is None:
            return known
        if not known:
            return allowed  # no nvidia-smi to check against; trust the variable
        return [i for i in allowed if i in known]
    try:
        if isinstance(spec, str):  # commas and/or spaces, as `parse_choice` accepts
            return [int(t) for t in re.split(r"[,\s]+", spec.strip()) if t]
        if isinstance(spec, int):
            return [spec]
        return [int(g) for g in spec]
    except (TypeError, ValueError):
        raise ValueError(
            f"sandbox.gpus: {spec!r} is not a GPU choice — use auto, none, or card ids "
            "such as [1] or [0, 2]"
        ) from None


def parse_choice(text: str, inv: list[Gpu]) -> str | list[int]:
    """What an operator typed for "which GPUs" -> the value `sandbox.gpus` stores.

    `all` / `auto` / nothing -> `"auto"`; `none` / `cpu` -> `[]`; `1`, `0,2`
    or `0 2 3` -> exactly those cards, each of which must exist here.
    """
    t = text.strip().lower()
    if t in ("", *ALL_WORDS):
        return AUTO
    if t in NONE_WORDS:
        return []
    try:
        chosen = sorted({int(x) for x in re.split(r"[,\s]+", t) if x})
    except ValueError:
        raise ValueError(
            f"{text!r} is not a list of GPUs — e.g. 0, or 0,1, or all, or none"
        ) from None
    if lost := missing(chosen, inv):
        have = ", ".join(str(g.index) for g in inv) or "none"
        raise ValueError(f"no GPU {', '.join(map(str, lost))} on this machine (it has: {have})")
    return chosen


def render(spec) -> str:
    """A `sandbox.gpus` value as it reads in YAML and on screen: auto, [1], []."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ALL_WORDS):
        return AUTO
    pool = resolve(spec, inv=[])  # an int, a string such as "0 1" or "none", or a list
    return "[" + ", ".join(map(str, pool)) + "]"


def missing(pool: list[int], inv: list[Gpu]) -> list[int]:
    """Cards in `pool` this machine does not have.

    No `nvidia-smi` means no usable card at all, so then every one is missing.
    """
    known = {g.index for g in inv}
    return [i for i in pool if i not in known]


def workspace_pool(root: Path | None, inv: list[Gpu] | None = None) -> list[int]:
    """The cards the workspace's settings give its runs."""
    from pinnforge import settings

    return resolve(settings.get(root, "sandbox.gpus", AUTO), inv=inv)


def hw_class(pool: list[int], inv: list[Gpu] | None = None) -> dict:
    """What a measurement on this pool was measured *on*.

    An anchor is a wall-clock-bounded training run, so its number belongs to
    the hardware as much as to the task: a CPU-mode b00 restored into a run on
    A6000s would make every block look brilliant.
    """
    if not pool:
        return {"platform": "cpu"}
    inv = inventory() if inv is None else inv
    by_index = {g.index: g for g in inv}
    names = sorted({by_index[i].name for i in pool if i in by_index})
    return {"platform": "gpu", "gpu": names[0] if len(names) == 1 else names or "unknown"}


def describe(pool: list[int], inv: list[Gpu] | None = None) -> str:
    """One line for humans: `2 × NVIDIA RTX A6000 (48 GB) [0, 1]`."""
    if not pool:
        return "CPU only"
    inv = inventory() if inv is None else inv
    by_index = {g.index: g for g in inv}
    names = sorted({by_index[i].name for i in pool if i in by_index})
    if len(names) == 1:
        mem = by_index[pool[0]].memory_mb / 1024 if pool[0] in by_index else 0
        size = f" ({mem:.0f} GB)" if mem else ""
        return f"{len(pool)} × {names[0]}{size} {pool}"
    return f"{len(pool)} GPU(s) {pool}"
