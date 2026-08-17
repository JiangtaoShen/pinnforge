"""The task contract — what makes a directory a valid PINNForge task.

The framework touches a task only through this interface, so a package that
honours it plugs in without framework changes. Validation is mechanical and
runs before any GPU time is spent: a task that disagrees with itself (a wall
budget in `problem.md` that the frozen header contradicts, a smoke literal
that changes the line length) wastes a whole run before anyone notices.

`problem.md` is also checked for *shape*. The prose paradigm is the point of
the corpus — every task states the same seven sections in the same order, so
a block agent reading its second task already knows where to look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CONTRACT_FILES = ("problem.md", "baseline.py", "eval.py")

# Where the scoring truth lives. A directory rather than a list of filenames:
# a name-based rule silently stops covering the answers the moment a task
# calls them something else, which is exactly how the old charter ended up
# forbidding a file that no longer existed.
PRIVATE_DIR = "private"

# The seven sections, in order. Nothing added, dropped, renamed or reordered.
REQUIRED_SECTIONS = (
    "## Equation",
    "## Domain",
    "## Boundary conditions",
    "## Initial condition",
    "## Scoring",
    "## Environment",
    "## Time budget",
)

# Verbatim in every task. Whitespace is normalised before comparing, so the
# line wrapping is free but the words are not.
INTRO = (
    "Solve the PDE with a genuine physics-informed neural network (PINN). "
    "The space/time derivatives entering the PDE residual come from automatic "
    "differentiation of the neural network(s). Classical methods may solve it "
    "efficiently, but the value of this task is probing the limits of PINN itself."
)

ENVIRONMENT_BODY = (
    "Two GPUs (A6000 class). PINNs must be built on the frozen JAX stack "
    "pinned in `pyproject.toml`; extensions via `uv add` if the pins survive."
)

_TRAIN_TIME_LITERAL = re.compile(r'TRAIN_TIME_LITERAL\s*=\s*"([^"]+)"')
_SMOKE_TRAIN_TIME = re.compile(r'SMOKE_TRAIN_TIME\s*=\s*"([^"]+)"')
_BUDGET_DEFAULT = re.compile(r'FORGE_WALL_BUDGET"\s*,\s*"([\d.]+)"')
_TRAIN_TIME_VALUE = re.compile(r"TRAIN_TIME\s*=\s*([\d.]+)")


@dataclass
class ValidationReport:
    task: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))

    @property
    def failures(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"task: {self.task}", ""]
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
        lines.append("")
        lines.append("ALL CHECKS PASSED" if self.ok else f"{len(self.failures)} CHECK(S) FAILED")
        return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.split())


def validate(task_dir: Path) -> ValidationReport:
    """Check one task directory against the contract. Never raises."""
    rep = ValidationReport(task=task_dir.name)

    for name in CONTRACT_FILES:
        rep.check(f"{name} present", (task_dir / name).is_file())
    if rep.failures:
        return rep  # nothing else is meaningful without the three files

    problem = (task_dir / "problem.md").read_text(encoding="utf-8")
    baseline = (task_dir / "baseline.py").read_text(encoding="utf-8")
    evalsrc = (task_dir / "eval.py").read_text(encoding="utf-8")

    _check_problem(rep, problem)
    _check_baseline(rep, baseline)
    _check_eval(rep, evalsrc)
    _check_data(rep, task_dir)
    _check_agreement(rep, problem, baseline, evalsrc)
    return rep


def _check_data(rep: ValidationReport, task_dir: Path) -> None:
    """The public/private split the score service rests on."""
    rep.check(
        "ships eval_grid.csv (public: where to predict)",
        (task_dir / "eval_grid.csv").is_file(),
    )
    private = task_dir / PRIVATE_DIR
    held = sorted(p.name for p in private.iterdir()) if private.is_dir() else []
    rep.check(
        f"ships the scoring truth under {PRIVATE_DIR}/",
        bool(held),
        ", ".join(held) or f"{PRIVATE_DIR}/ is missing or empty",
    )


def _check_problem(rep: ValidationReport, problem: str) -> None:
    title = problem.splitlines()[0] if problem.strip() else ""
    rep.check(
        "title is '# <ABBREV> — <full name>'",
        bool(re.match(r"^#\s+[A-Z0-9]+\s+—\s+\S", title)),
        title[:60],
    )

    found = [s for s in REQUIRED_SECTIONS if s in problem]
    rep.check(
        "all seven sections present",
        len(found) == len(REQUIRED_SECTIONS),
        f"{len(found)}/{len(REQUIRED_SECTIONS)}",
    )
    order = [problem.index(s) for s in REQUIRED_SECTIONS if s in problem]
    rep.check("sections in canonical order", order == sorted(order))

    extra = [
        line.strip()
        for line in problem.splitlines()
        if line.startswith("## ") and line.strip() not in REQUIRED_SECTIONS
    ]
    rep.check("no extra top-level sections", not extra, ", ".join(extra))
    rep.check(
        "no '###' subsections",
        "\n### " not in problem,
        f"{problem.count(chr(10) + '### ')} found",
    )
    rep.check("intro paragraph is verbatim", INTRO in _norm(problem))
    rep.check("environment section is verbatim", ENVIRONMENT_BODY in _norm(problem))
    # normalised: the corpus wraps prose, so "Lower is\nbetter." is the same
    # sentence as "Lower is better."
    flat = _norm(problem)
    rep.check(
        "scoring states rRMSE, lower is better",
        "rRMSE" in flat and "Lower is better" in flat,
    )


def _check_baseline(rep: ValidationReport, baseline: str) -> None:
    has = lambda pat: re.search(pat, baseline, re.MULTILINE) is not None  # noqa: E731
    rep.check("defines train()", has(r"^def train\("))
    rep.check("defines predict_fn()", has(r"^def predict_fn\("))
    rep.check("eval gate present", "FORGE_EVAL_TOKEN" in baseline)
    rep.check(
        "no RNG in the dataset path",
        "default_rng" not in baseline and "np.random" not in baseline,
    )
    rep.check("frozen header marker present", "DO NOT MODIFY" in baseline)


def _check_eval(rep: ValidationReport, evalsrc: str) -> None:
    lit = _TRAIN_TIME_LITERAL.search(evalsrc)
    smoke = _SMOKE_TRAIN_TIME.search(evalsrc)
    rep.check("declares TRAIN_TIME_LITERAL", lit is not None)
    rep.check("declares SMOKE_TRAIN_TIME", smoke is not None)
    if lit and smoke:
        rep.check(
            "smoke replacement has the same length",
            len(lit.group(1)) == len(smoke.group(1)),
            f"{len(lit.group(1))} vs {len(smoke.group(1))}",
        )
    rep.check("reads FORGE_WALL_BUDGET", "FORGE_WALL_BUDGET" in evalsrc)
    rep.check("logs the rRMSE field", "rRMSE" in evalsrc)
    rep.check("maintains the .budget counter", ".budget" in evalsrc)
    # The scoring split: the half that executes candidate code must not be
    # the half that reads the answers.
    rep.check(
        "defines score_predictions() for the score service",
        re.search(r"^def score_predictions\(", evalsrc, re.MULTILINE) is not None,
    )
    rep.check("submits predictions instead of scoring inline", "submit_score" in evalsrc)


def _check_agreement(
    rep: ValidationReport, problem: str, baseline: str, evalsrc: str
) -> None:
    """Cross-file agreement — where task swaps historically went wrong."""
    lit = _TRAIN_TIME_LITERAL.search(evalsrc)
    if lit:
        rep.check(
            "TRAIN_TIME literal appears exactly once in baseline",
            baseline.count(lit.group(1)) == 1,
            f"'{lit.group(1)}' x{baseline.count(lit.group(1))}",
        )
        want = lit.group(1).split("=")[-1].strip()
        rep.check(
            "problem.md states the same TRAIN_TIME",
            f"TRAIN_TIME = {want}" in _norm(problem),
            want,
        )
    m = _TRAIN_TIME_VALUE.search(problem.split("## Time budget")[-1]) if problem else None
    rep.check("time budget section quotes a number", m is not None, m.group(1) if m else "")


def train_time(task_dir: Path) -> float | None:
    """Per-run wall in seconds, read from `eval.py`'s frozen-header literal."""
    m = _TRAIN_TIME_LITERAL.search((task_dir / "eval.py").read_text(encoding="utf-8"))
    if not m:
        return None
    try:
        return float(m.group(1).split("=")[-1].strip())
    except ValueError:
        return None


def wall_budget(task_dir: Path) -> float | None:
    """Per-block GPU budget default declared by the task's `eval.py`."""
    m = _BUDGET_DEFAULT.search((task_dir / "eval.py").read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None
