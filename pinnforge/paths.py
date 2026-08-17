"""Where things live.

Two roots matter:

* the **project root** — this repo: `tasks/`, `runs/`, `kb1/`, `charter/`,
  `.venv/`;
* a **run root** — `runs/<task>_<n>/`, which is also the working directory
  every block agent is started in.

That second point is the whole trick behind harness portability. The block
charter says `task/`, `kb1/`, `blocks/bNN/`, `.venv/bin/python task/eval.py`,
and those paths resolve identically inside a run root as they did in the
single-task layout this replaces — so the charter needs no rewriting and a
block behaves the same whichever CLI is driving it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

RUN_ID_RE = re.compile(r"^(?P<task>.+)_(?P<n>\d+)$")


class ProjectNotFound(FileNotFoundError):
    """Raised instead of silently treating the working directory as a project.

    Falling back to the cwd made a mistyped path look like an empty library
    ("no tasks in /tmp/tasks") rather than a missing project.
    """


def project_root(start: Path | None = None) -> Path:
    """Locate the project root: the nearest ancestor holding `tasks/`.

    `PINNFORGE_ROOT` overrides the search, which is what lets a block agent
    (whose cwd is a run root, not the project) still find the library.
    """
    if env := os.environ.get("PINNFORGE_ROOT"):
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for base in [here, *here.parents]:
        if (base / "tasks").is_dir() and (base / "pinnforge").is_dir():
            return base
    raise ProjectNotFound(
        f"no PINNForge project at {here} or any parent directory.\n"
        "Run from inside a project, or point PINNFORGE_ROOT at one."
    )


def tasks_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "tasks"


def runs_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "runs"


def kb1_dir(root: Path | None = None) -> Path:
    """The knowledge corpus a block reads before choosing a line of attack.

    Ships inside the package, like the prompts: a corpus the framework does
    not carry is a framework that arrives unable to do the one thing blocks
    start every session by doing. A `kb1/` at the project root overrides it
    wholesale, so a deployment can bring its own corpus without editing an
    installed package — the same arrangement as `charter/`.
    """
    override = (root or project_root()) / "kb1"
    if override.is_dir():
        return override
    return Path(__file__).parent / "kb1"


def prompts_dir() -> Path:
    """Everything this framework says to an agent, in one place.

    Three files: the block charter, and the prompts for the task-authoring
    and kb1-distilling agents. They ship inside the package so an installed
    wheel is self-contained.
    """
    return Path(__file__).parent / "prompts"


def charter_template(root: Path | None = None) -> Path:
    """The block-charter template.

    A `charter/block.md.template` at the project root overrides the packaged
    one — the charter is the piece a deployment may legitimately want to
    tune, and doing that should not mean editing an installed package.
    """
    override = (root or project_root()) / "charter" / "block.md.template"
    if override.is_file():
        return override
    return prompts_dir() / "block.md.template"


def venv_python(root: Path | None = None) -> Path:
    """Interpreter the task's `eval.py` runs under.

    Falls back to the ambient interpreter when the project venv is absent so
    that validation and tests work in a bare checkout.
    """
    cand = (root or project_root()) / ".venv" / "bin" / "python"
    if cand.is_file():
        return cand
    import sys

    return Path(sys.executable)


# ──────────────────────────── run ids ────────────────────────────


def parse_run_id(run_id: str) -> tuple[str, int]:
    """`ks_2` -> `("ks", 2)`. Task names may contain underscores."""
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"malformed run id {run_id!r}; expected <task>_<n>")
    return m.group("task"), int(m.group("n"))


def next_run_id(task: str, root: Path | None = None) -> str:
    """`<task>_<n+1>`, where n is the highest existing run for that task."""
    base = runs_dir(root)
    highest = 0
    if base.is_dir():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            try:
                name, n = parse_run_id(child.name)
            except ValueError:
                continue
            if name == task:
                highest = max(highest, n)
    return f"{task}_{highest + 1}"


def run_root(run_id: str, root: Path | None = None) -> Path:
    return runs_dir(root) / run_id


def list_runs(root: Path | None = None, task: str | None = None) -> list[str]:
    """Existing run ids, ordered by task then run number."""
    base = runs_dir(root)
    if not base.is_dir():
        return []
    found: list[tuple[str, int]] = []
    for child in base.iterdir():
        if not (child / "run.yaml").is_file():
            continue
        try:
            name, n = parse_run_id(child.name)
        except ValueError:
            continue
        if task is None or name == task:
            found.append((name, n))
    return [f"{name}_{n}" for name, n in sorted(found)]


# ──────────────────────── paths inside a run ────────────────────────


def blocks_dir(run: Path) -> Path:
    return run / "blocks"


def kb2_dir(run: Path) -> Path:
    """Block summaries. Kept at `blocks/kb2/` so the charter's own paths hold."""
    return run / "blocks" / "kb2"


def block_dir(run: Path, block_id: str) -> Path:
    return run / "blocks" / block_id


def summary_path(run: Path, block_id: str) -> Path:
    return kb2_dir(run) / f"{block_id}.md"


def evals_path(run: Path, block_id: str) -> Path:
    return block_dir(run, block_id) / "evals.jsonl"


def budget_path(run: Path, block_id: str) -> Path:
    return block_dir(run, block_id) / ".budget"


def state_path(run: Path) -> Path:
    return run / "state.json"


def config_path(run: Path) -> Path:
    return run / "run.yaml"


def ledger_path(run: Path) -> Path:
    return run / "blocks" / "run_usage.jsonl"


def run_summary_path(run: Path) -> Path:
    return run / "blocks" / "run_summary.md"


def logs_dir(run: Path) -> Path:
    return run / "logs"
