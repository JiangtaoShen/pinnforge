"""Authoring a task — a staged pipeline whose state lives on disk.

Stages:

    intake           what the task is: the PDE, its domain and conditions,
                     the scored field, and where a reference comes from
    author           an agent writes problem.md, baseline.py, eval.py and the
                     public grid — never the answers
    reference        the answers: the operator's data, ingested; or, when
                     there is none, a dedicated agent computes them with a
                     classical solver at several resolutions
    reference_check  the framework itself checks the reference
    validate         the task contract
    smoke            the scoring path end to end, free, on the CPU
    anchor           b00 — the score every block must beat
    finalize         the task is released to `run start`

With the operator's data the reference comes first — it fixes the grid the
author then writes the task around; without it the author chooses the grid
and the reference is computed on it.

The state is `tasks/<name>/.authoring/state.json`, rewritten after every
transition, so an interrupted authoring session resumes where it stopped
(`task new <name> --resume`) and the dashboard can follow it live. A task with
no authoring record at all predates this pipeline and is ready as it stands.

Logs: the author's transcript is public (`.authoring/author.log` — it never
sees the answers); the reference agent's is not (`private/logs/`), because a
solver's output *is* the answers. Nor is anything a stage raised: an
exception's text can name the operator's data or a solver's output, so the
public record says only that the stage crashed, and the traceback goes to the
private `logs/pipeline-errors.log`.
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import oscompat, paths, private

STATE_DIR = ".authoring"
# What the framework itself writes into `.authoring/` while a stage runs — the
# only files there an agent's session may not be blamed for changing.
FRAMEWORK_FILES = ("state.json", "state.tmp", "lock", "detach.json", "intake.md")
ERROR_LOG = "pipeline-errors.log"  # under the private directory's logs/
STAGES = (
    "intake",
    "author",
    "reference",
    "reference_check",
    "validate",
    "smoke",
    "anchor",
    "finalize",
)
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


def state_path(task_dir: Path) -> Path:
    return Path(task_dir) / STATE_DIR / "state.json"


def status(task_dir: Path) -> dict | None:
    """The authoring record, or None for a task that never had one."""
    path = state_path(task_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stage": "?", "status": "unreadable", "finalized": False}


def is_ready(task_dir: Path) -> bool:
    """May `run start` use this task? Unfinished authoring says no."""
    st = status(task_dir)
    return st is None or bool(st.get("finalized"))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class Inputs:
    """What the operator asked for — everything a resumed session needs again."""

    describe: str = ""
    reference: str = ""  # operator data: a path or URL; "" = build one
    sources: list[str] = field(default_factory=list)
    template: str = ""
    agent: dict = field(default_factory=dict)
    gpu: str = "0"
    ref_budget_s: float = 4 * 3600.0
    author_budget_s: float = 2 * 3600.0
    ref_tol: float = 1e-3
    accept_reference: str = ""
    no_anchor: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> Inputs:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class Pipeline:
    """One task's authoring, stage by stage, checkpointed after each."""

    def __init__(self, root: Path, name: str, out=print):
        if not NAME_RE.fullmatch(name):
            raise ValueError(
                f"task name {name!r} must be letters, digits, '_' or '-' — the score "
                "service resolves the reference through it"
            )
        self.root = Path(root)
        self.name = name
        self.task_dir = paths.tasks_dir(self.root) / name
        self.out = out
        self.state: dict = status(self.task_dir) or {}
        self._lock_fd: int | None = None

    # ── state ──

    @property
    def inputs(self) -> Inputs:
        return Inputs.from_dict(self.state.get("inputs", {}))

    def _save(self) -> None:
        path = state_path(self.task_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated"] = _now()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        oscompat.replace(tmp, path)

    def begin(self, inputs: Inputs) -> None:
        self.state = {
            "task": self.name,
            "created": _now(),
            "mode": "ingest" if inputs.reference else "build",
            "stage": "intake",
            "status": "running",
            "finalized": False,
            "inputs": inputs.to_dict(),
            "stages": {s: {"status": "pending"} for s in STAGES},
        }
        self._save()

    def plan(self) -> list[str]:
        return plan_for(self.state.get("mode", "build"))

    def _mark(self, stage: str, status_: str, note: str = "", **extra) -> None:
        st = self.state["stages"].setdefault(stage, {})
        if status_ == "running":
            st["started"] = _now()
        elif status_ in ("done", "failed", "skipped"):
            st["ended"] = _now()
        st["status"] = status_
        if note:
            st["note"] = note
        st.update(extra)
        self.state["stage"] = stage
        self.state["status"] = "failed" if status_ == "failed" else (
            "running" if status_ == "running" else self.state.get("status", "running")
        )
        self._save()

    # ── lock ──

    def lock(self) -> None:
        d = self.task_dir / STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d / "lock", os.O_RDWR | os.O_CREAT | oscompat.O_BINARY, 0o644)
        try:
            oscompat.lock(fd, block=False)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(
                f"task {self.name!r} is already being authored by another process "
                f"(`pinnforge task status {self.name}`)"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fd = fd

    def unlock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    # ── running ──

    def redo(self, stage: str) -> list[str]:
        """Set `stage` and every stage after it back to pending. Returns them."""
        plan = self.plan()
        if stage not in plan:
            raise ValueError(f"no stage {stage!r} — the stages are: {', '.join(plan)}")
        again = plan[plan.index(stage):]
        for s in again:
            self.state["stages"][s] = {"status": "pending"}
        self.state["finalized"] = False
        self.state["status"] = "running"
        self._save()
        return again

    def private_log(self, name: str, text: str) -> Path:
        """Append `text` to `<private directory>/logs/<name>` — never a public record."""
        logs = private.locate(self.task_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / name
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"── {_now()} ──\n{text.rstrip()}\n")
        return path

    # ── running ──

    def run(self, steps, before=None) -> int:
        """Run the remaining stages with `steps[stage]()` -> (ok, note).

        `before()`, if given, runs once the lock is held and before any stage —
        whatever must not race another session of the same task.
        """
        self.lock()
        try:
            if before is not None:
                before()
            for stage in self.plan():
                if self.state["stages"].get(stage, {}).get("status") in ("done", "skipped"):
                    continue
                self.out(f"\n── {stage.replace('_', ' ')} ──")
                self._mark(stage, "running")
                try:
                    ok, note = steps[stage]()
                except KeyboardInterrupt:
                    self._mark(stage, "failed", "interrupted")
                    raise
                except Exception as e:  # a stage's crash is that stage's failure
                    # Its text is not published: "no reference data at /home/op/…"
                    # is a path to the answers. The category is; the rest is private.
                    try:
                        where = self.private_log(ERROR_LOG, f"stage {stage}:\n"
                                                 + traceback.format_exc())
                    except OSError:
                        where = "(the private log could not be written)"
                    ok, note = False, (f"the stage crashed ({type(e).__name__}); the details "
                                       "are in the private pipeline-errors log")
                    self.out(f"stage {stage} crashed: {type(e).__name__} — see {where}")
                if ok is None:
                    self._mark(stage, "skipped", note)
                    continue
                self._mark(stage, "done" if ok else "failed", note)
                if not ok:
                    self.out(f"stage {stage} failed: {note}")
                    self.out(f"fix the cause, then: pinnforge task new {self.name} --resume")
                    return 1
            self.state["finalized"] = True
            self.state["status"] = "done"
            self._save()
            return 0
        finally:
            self.unlock()


def plan_for(mode: str) -> list[str]:
    """The stage order: with the operator's data, the reference fixes the grid first."""
    if mode == "ingest":
        return ["intake", "reference", "author", "reference_check", "validate",
                "smoke", "anchor", "finalize"]
    return list(STAGES)


def render_intake(name: str, inputs: Inputs) -> str:
    """The intake as the agents read it.

    Where the operator's reference is never appears here: the author agent
    reads this, and so can anything that reads `.authoring/`.
    """
    lines = [f"Task name: `{name}`", "", inputs.describe.strip() or "(no description given)"]
    if inputs.reference:
        lines += ["", "Reference: supplied by the operator."]
    else:
        lines += ["", "Reference: none supplied — it will be computed with a classical solver."]
    return "\n".join(lines) + "\n"


INTAKE_QUESTIONS = (
    ("equation", "The PDE (plain text or LaTeX), with every constant's value"),
    ("domain", "The domain — intervals, geometry, time window"),
    ("conditions", "Boundary conditions, and initial conditions if time-dependent"),
    ("scored", "Which field(s) the score should be on (e.g. u; or velocity [u; v])"),
    ("notes", "Anything else the problem needs: parameter values, the paper it is from…"),
)

# Said before the questions: the description becomes the author agent's brief
# and part of the task package, so a solution in it is a solution handed out.
INTAKE_NOTE = (
    "Describe the problem, not its solution: if you know the exact solution, give it\n"
    "as the reference (a data file, or a .py/.txt with the formula) when asked for one."
)


def interview(ask) -> str:
    """Ask the operator the intake questions; `ask(prompt)` returns an answer."""
    parts = []
    for key, question in INTAKE_QUESTIONS:
        answer = ask(question).strip()
        if answer:
            parts.append(f"**{key.capitalize()}.** {answer}")
    return "\n\n".join(parts)
