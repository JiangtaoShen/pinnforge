"""`pinnforge task new` — from a described PDE to a task `run start` accepts.

Say what the problem is — in the terminal (asked question by question), with
`--describe "..."` / `--describe @spec.md`, or through a harness plugin that
turned a conversation into that description — and where a reference solution
is, if there is one. The pipeline in `pinnforge.task.authoring` does the rest:
an author agent writes the package, the reference is ingested or built and
checked, and the task is validated, smoked and anchored.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from pinnforge import agents, bindings, gpus, oscompat, paths, private, scoring, workspace
from pinnforge.config import AgentConfig
from pinnforge.run import anchor, lifecycle
from pinnforge.task import audit, authoring, contract, examples, reference

BUILD_MODE = """\
## The grid, and where the answers come from

No reference exists yet. After you finish, a separate agent computes one with
a classical numerical solver on **exactly the rows of your `eval_grid.csv`**,
at several resolutions, and the framework checks it converged. So choose the
grid with that in mind: a tensor-product grid for a box (a solver can hit it
directly), fine enough to judge a network — on the order of 100–300 points
per dimension, fewer along time — and covering the whole domain, boundaries
included. Declare `REF_COLUMNS` as the fields the solver must produce. Do not
create `private/`.
"""

INGEST_MODE = """\
## The grid is already fixed

The operator supplied the reference, and the points it is scored at are
written: `eval_grid.csv`, {rows} rows, columns {grid_columns}, of which the
first {input_columns} are coordinates; the reference holds the fields
{ref_columns}. Do not modify `eval_grid.csv` — not even re-save or sort it:
the framework checks it is byte for byte the grid it was given. Set
`INPUT_COLUMNS = {input_columns}` and `REF_COLUMNS = {ref_tuple}`, and write
the task around this grid. That is all you are told about the reference, by
design.
"""

# Under the private directory's logs/.
AGENT_RECORD = "agent.json"  # the running agent's process group
AUDIT_LOG = "author-audit.log"  # what an untrusted author reached for
GUARD_LOG = "reference-guard.log"  # what a reference agent wrote where it may not

# A column name that may reach the author and the public record: a short
# identifier, no long runs of digits (a header is not a place for numbers).
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}")


def _ident(s) -> bool:
    return isinstance(s, str) and bool(_IDENT.fullmatch(s)) and not re.search(r"\d{3}", s)


def _same(a: Path, b: Path) -> bool:
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


def _numbered(parent: Path, prefix: str) -> Path:
    """`<parent>/<prefix>-<n>` for the first n not taken."""
    n = 1
    while (parent / f"{prefix}-{n}").exists() or (parent / f"{prefix}-{n}").is_symlink():
        n += 1
    return parent / f"{prefix}-{n}"


def _free(parent: Path, stem: str, suffix: str = "") -> Path:
    """`<parent>/<stem><suffix>`, or `<stem>-2<suffix>`, … — the first not taken."""
    target, n = parent / f"{stem}{suffix}", 2
    while target.exists() or target.is_symlink():
        target, n = parent / f"{stem}-{n}{suffix}", n + 1
    return target


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def grid_shape(task_dir: Path, private_home: Path, info: dict) -> tuple[dict, str]:
    """The ingested grid's shape, read off the files themselves: (shape, why it is not usable).

    The ingest agent saw the answers, so its account of the grid (`ingest.json`)
    is not repeated to the author or the public record. The column names come
    from `eval_grid.csv`'s header, the row count from its lines, the number of
    reference columns from `ref_values.npy`; of `ingest.json` only a whole
    number (how many leading columns are coordinates) and identifiers (the
    reference columns' names, as many as the array has) are taken.
    """
    import numpy as np

    grid = Path(task_dir) / "eval_grid.csv"
    ref = Path(private_home) / "ref_values.npy"
    if not grid.is_file() or grid.is_symlink():
        return {}, "the ingest agent wrote no eval_grid.csv"
    if not ref.is_file():
        return {}, "the ingest agent wrote no ref_values.npy"
    with grid.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip()
    names = [c.strip() for c in header.split(",")]
    if not header or not all(_ident(c) for c in names) or len(set(names)) != len(names):
        return {}, ("eval_grid.csv's header must name each column once, with a short "
                    "identifier (a letter or _, then letters, digits or _; at most 32 characters)")
    rows = reference.grid_rows(grid)
    try:
        arr = np.load(ref, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return {}, "ref_values.npy is not a readable .npy array"
    if arr.ndim not in (1, 2) or arr.shape[0] != rows:
        return {}, (f"ref_values.npy has {arr.shape[0] if arr.ndim else 0} row(s) for "
                    f"{rows} grid rows" if arr.ndim in (1, 2) else
                    "ref_values.npy must be a table: one row per grid row")
    n_ref = 1 if arr.ndim == 1 else int(arr.shape[1])
    n_in = info.get("input_columns")
    if isinstance(n_in, bool) or not isinstance(n_in, int) or not 1 <= n_in <= len(names):
        return {}, ("ingest.json's input_columns must be a whole number from 1 to the grid's "
                    f"{len(names)} columns")
    cols = info.get("ref_columns")
    if (not isinstance(cols, list) or len(cols) != n_ref or not all(_ident(c) for c in cols)
            or len(set(cols)) != len(cols)):
        return {}, (f"ingest.json's ref_columns must name each of the reference's {n_ref} "
                    "column(s), once, with a short identifier")
    return {"grid_columns": names, "input_columns": n_in, "ref_columns": list(cols),
            "rows": rows}, ""


def _group_works_in(pgid, where) -> bool | None:
    """Is a member of group `pgid` working under `where`? None where that cannot be told.

    A recorded group number can since have been reused by something else; an
    authoring agent's group has the directory it was started in as its cwd.
    """
    proc = Path("/proc")
    if not where or not (proc / "self" / "stat").is_file():
        return None
    root = os.path.realpath(where)
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "stat").read_text(encoding="utf-8", errors="replace")
            if int(raw[raw.rfind(")") + 2 :].split()[2]) != int(pgid):
                continue
            cwd = os.readlink(d / "cwd")
        except (OSError, ValueError, IndexError):
            continue
        if cwd == root or cwd.startswith(root.rstrip("/") + "/"):
            return True
    return False


def _describe(args) -> str:
    text = args.describe or args.prompt or ""
    if text.startswith("@"):
        text = Path(text[1:]).expanduser().read_text(encoding="utf-8")
    return text


def _template(root: Path, name: str) -> Path:
    if not name:
        return examples.template_dir()
    lib = paths.tasks_dir(root) / name
    if (lib / "problem.md").is_file():
        return lib
    p = Path(name).expanduser()
    if (p / "problem.md").is_file():
        return p.resolve()
    raise FileNotFoundError(f"no template task {name!r} (a workspace task name or a directory)")


def _ask(question: str) -> str:
    print(f"\n{question}:")
    try:
        return input("> ")
    except EOFError:
        return ""


class Steps:
    """The pipeline's stages for one task; each returns (ok, note).

    The answers live in the task's private directory — `tasks/<t>/private/`,
    or `<private_dir>/<t>/` when the deployment keeps them outside the
    library — and every stage reads and writes them there. A stage's note is
    public (the authoring state, the dashboard), so it says what happened in
    the framework's words only; what an agent wrote, what it reached for and
    what a crash said go to the private directory's `logs/`.
    """

    def __init__(self, pipe: authoring.Pipeline, root: Path):
        self.pipe = pipe
        self.root = Path(root)
        self.name = pipe.name
        self.task_dir = pipe.task_dir
        self.inputs = pipe.inputs
        self.agent = AgentConfig.from_dict(self.inputs.agent)
        self.private_home = private.locate(self.task_dir)
        self.in_tree = self.task_dir / contract.PRIVATE_DIR
        self.auth_dir = self.task_dir / authoring.STATE_DIR
        self._stray = False
        self._warning = ""

    def mapping(self) -> dict:
        return {s: getattr(self, s) for s in authoring.STAGES}

    @property
    def home_in_tree(self) -> bool:
        return _same(self.private_home, self.in_tree)

    def recover(self) -> None:
        """Before any stage, under the pipeline's lock: no agent of an earlier session is
        still alive, and answers a crash left sealed away go back where they belong."""
        self._refuse_live_agent()
        self._unseal()

    def _noted(self, note: str) -> str:
        return note + (f" — WARNING: {self._warning}" if self._warning else "")

    # ── stages ──

    def intake(self):
        text = authoring.render_intake(self.name, self.inputs)
        self.auth_dir.mkdir(parents=True, exist_ok=True)
        # bytes, as `_restore_framework` compares them: never Windows' \r\n
        (self.auth_dir / "intake.md").write_bytes(text.encode("utf-8"))
        print(text)
        return True, ""

    def _intake_text(self) -> str:
        # Rendered from the inputs, not read back from `.authoring/intake.md`:
        # that file sits where agents can write.
        return authoring.render_intake(self.name, self.inputs)

    def author(self):
        ingest = self.pipe.state.get("mode") == "ingest"
        if ingest:
            shape = self.pipe.state.get("reference") or {}
            if not shape.get("grid_columns"):
                return False, ("no ingested grid to write the task around — "
                               "`--resume --redo reference`")
            if why := reference.grid_drift(self.task_dir, self.private_home, restore=True):
                return False, why
            names = [c for c in shape.get("grid_columns", []) if _ident(c)]
            cols = [c for c in shape.get("ref_columns", []) if _ident(c)]
            mode = INGEST_MODE.format(
                rows=int(shape.get("rows") or 0),
                grid_columns=", ".join(f"`{c}`" for c in names),
                input_columns=int(shape.get("input_columns") or 0),
                ref_columns=", ".join(f"`{c}`" for c in cols),
                ref_tuple=repr(tuple(cols)),
            )
        else:
            mode = BUILD_MODE
        template = _template(self.root, self.inputs.template)
        log = self.auth_dir / "author.log"
        records_before = self._records()
        print(f"author agent: {self.agent.runtime} / {self.agent.model or 'default'} "
              f"(template {template.name}; transcript {log})")
        started = time.time()
        with self._sealed() as home_now:
            run = agents.author_task(
                self.name,
                intake=self._intake_text(),
                sources=[Path(s) for s in self.inputs.sources],
                mode=mode,
                agent=self.agent,
                template=template,
                log_path=log,
                root=self.root,
                timeout_s=self.inputs.author_budget_s,
                record=home_now / "logs" / AGENT_RECORD,
            )
        print(f"agent finished in {run.duration_s / 60:.1f} min (exit {run.returncode}"
              f"{', timed out' if run.timed_out else ''})")
        hits = self._audit(log)
        if self._stray:
            hits.append(audit.Hit(audit.STRAY, "made a private/ directory in the task "
                                               "(quarantined under the private logs)"))
        if hits:
            # Its transcript may now hold what it saw, and its package is not
            # to be trusted: both leave the public tree.
            self._distrust(log, hits, records_before, ingest)
            return False, ("the author agent reached for the answers ("
                           + "; ".join(audit.categories(hits))
                           + ") — its package was moved into the private logs (details in "
                           f"{AUDIT_LOG}); `--resume` writes it again")
        if run.timed_out or run.returncode != 0:
            why = "it timed out" if run.timed_out else f"exit {run.returncode}"
            return False, (f"the author agent did not finish ({why}) — whatever it left is not "
                           "accepted; `--resume` runs it again")
        if unwritten := self._unwritten(started, ingest):
            return False, (f"the author agent did not write {', '.join(unwritten)} in this "
                           "session — `--resume` runs it again")
        if ingest and (why := reference.grid_drift(self.task_dir, self.private_home,
                                                   restore=True)):
            return False, f"the author agent changed eval_grid.csv: {why}; `--resume` runs it again"
        rep = contract.validate(self.task_dir, reference=False)
        if not rep.ok:
            names = "; ".join(n for n, _, _ in rep.failures)
            return False, f"the package fails the contract: {names}"
        if not reference.ref_columns(self.task_dir):
            return False, "eval.py declares no REF_COLUMNS"
        return True, f"{len(rep.checks)} contract checks pass"

    def _unwritten(self, started: float, ingest: bool) -> list[str]:
        """Package files this author session did not write — absent, links, or older than it."""
        want = list(contract.CONTRACT_FILES) + ([] if ingest else ["eval_grid.csv"])
        out = []
        for name in want:
            f = self.task_dir / name
            try:
                st = f.lstat()
            except OSError:
                out.append(name)
                continue
            # a second of slack for filesystems that keep coarse timestamps
            if f.is_symlink() or not f.is_file() or st.st_mtime < started - 1.0:
                out.append(name)
        return out

    def _records(self) -> dict[str, str]:
        out = {}
        if self.auth_dir.is_dir():
            for f in self.auth_dir.iterdir():
                if f.is_file() and not f.is_symlink() and f.name not in authoring.FRAMEWORK_FILES:
                    out[f.name] = _md5(f)
        return out

    def _distrust(self, log: Path, hits: list[audit.Hit], records_before: dict[str, str],
                  ingest: bool) -> Path:
        """Move an untrusted author's package, transcript and records into the private logs."""
        logs = self.private_home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        dest = _numbered(logs, "untrusted-package")
        dest.mkdir(parents=True)
        with lifecycle.critical():
            for p in sorted(self.task_dir.iterdir()):
                if p.name in (contract.PRIVATE_DIR, authoring.STATE_DIR, ".anchor"):
                    continue
                shutil.move(str(p), str(dest / p.name))
            if log.is_file() or log.is_symlink():
                shutil.move(str(log), str(_free(logs, "author.untrusted", ".log")))
            for name, digest in self._records().items():
                if records_before.get(name) != digest:  # what it wrote into .authoring/
                    (dest / authoring.STATE_DIR).mkdir(exist_ok=True)
                    shutil.move(str(self.auth_dir / name), str(dest / authoring.STATE_DIR / name))
            if ingest:
                reference.grid_drift(self.task_dir, self.private_home, restore=True)
        self.pipe.private_log(AUDIT_LOG, f"author package moved to {dest}\n" + "\n".join(
            f"[{h.category}] {h.detail}" for h in hits))
        return dest

    # ── the answers stay out of the author's reach ──

    def _sealed_home(self) -> Path:
        return paths.state_dir(self.root) / "sealed" / self.name

    def _park_stray(self) -> Path:
        """Move what stands at `tasks/<t>/private` aside: an agent made it, not the framework."""
        target = audit._unique(self._sealed_home(), "stray")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.in_tree), str(target))
        return target

    def _quarantine(self, src: Path, prefix: str) -> Path:
        """Move `src` under the private directory's logs, under a name of its own.

        Never into `.authoring/` (public), and never where the answers live:
        a stray `ref_values.npy` must not be mistaken for the task's.
        """
        logs = self.private_home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        dest = _numbered(logs, prefix)
        shutil.move(str(src), str(dest))
        return dest

    def _refuse_live_agent(self) -> None:
        for rec in (self._sealed_home() / "private" / "logs" / AGENT_RECORD,
                    self.private_home / "logs" / AGENT_RECORD):
            info = agents.read_record(rec)
            if not info:
                continue
            pgid = info.get("pgid")
            if agents.group_alive(pgid) and _group_works_in(pgid, info.get("cwd")) is not False:
                raise RuntimeError(
                    f"an agent of an earlier session of {self.name} (process group {pgid}) is "
                    f"still running — stop it first ({_stop_hint(pgid)}), then resume"
                )
            rec.unlink(missing_ok=True)

    def _unseal(self) -> None:
        held = self._sealed_home() / "private"
        with lifecycle.critical():
            if held.is_dir() or held.is_symlink():
                stray = None
                if self.in_tree.exists() or self.in_tree.is_symlink():
                    stray = self._park_stray()  # made while the answers were away
                shutil.move(str(held), str(self.in_tree))
                if stray is not None:
                    self._quarantine(stray, "stray-private")
            sealed = self._sealed_home()
            if sealed.is_dir():
                for left in sorted(sealed.glob("stray*")):  # a crash between the two moves
                    self._quarantine(left, "stray-private")

    @contextlib.contextmanager
    def _sealed(self):
        """Keep the answers out of the task directory while the author agent works in it.

        In ingest mode the answers exist before the author starts, in the very
        directory it is told to write in, where a listing shows them. Agents of
        the operator's own user cannot be walled off, so this is paired with
        `_audit`, which reads what the agent actually reached for.

        Only a private directory inside the task directory is moved; an
        external one (`private_dir`) is outside the author's tree already.
        Yields where the private directory is while sealed. Whatever the agent
        leaves at `tasks/<t>/private` is a stray — quarantined in the private
        logs, never taken for the answers — and the answers go back only once
        the agent's process group is gone.
        """
        held = self._sealed_home() / "private"
        moved = False
        with lifecycle.critical():
            if held.exists() or held.is_symlink():
                raise RuntimeError(f"answers are already sealed away at {held}")
            held.parent.mkdir(parents=True, exist_ok=True)
            if self.in_tree.is_dir() or self.in_tree.is_symlink():
                shutil.move(str(self.in_tree), str(held))  # a link moves as the link
                moved = True
            elif self.home_in_tree:
                held.mkdir()
                moved = True
        self._stray = False
        try:
            yield held if self.home_in_tree else self.private_home
        finally:
            with lifecycle.critical():
                pgid = agents.live_pgid()
                if pgid is not None and agents.group_alive(pgid):
                    raise RuntimeError(
                        f"the author agent's process group {pgid} survived being stopped; the "
                        f"answers stay sealed at {held} until it is gone "
                        f"({_stop_hint(pgid, kill=True)}, then --resume)"
                    )
                stray = None
                if self.in_tree.exists() or self.in_tree.is_symlink():
                    stray = self._park_stray()
                if moved:
                    shutil.move(str(held), str(self.in_tree))
                if stray is not None:
                    self._quarantine(stray, "stray-private")
                    self._stray = True

    def _audit(self, log: Path) -> list[audit.Hit]:
        """What, of the answers, the author agent's tool calls reached for (see `audit`)."""
        if not log.is_file():
            return []
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        operator = ""
        with contextlib.suppress(FileNotFoundError, ValueError, KeyError, OSError):
            operator = reference.source_location(self.private_home, self.inputs.reference)
        scope = audit.Scope(self.root, self.name, cwd=self.task_dir,
                            private_home=self.private_home, operator=operator)
        return audit.audit(self.agent.runtime, lines, scope)

    # ── an agent that saw the answers leaves nothing outside the private directory ──

    def _public(self) -> dict[str, str]:
        """{path: digest} of everything in the task directory an agent that saw the answers
        may not touch: the package, `.authoring/` (but the framework's own files) and
        `.anchor/`."""
        out: dict[str, str] = {}
        framework = {f"{authoring.STATE_DIR}/{n}" for n in authoring.FRAMEWORK_FILES}
        for dirpath, dirnames, filenames in os.walk(self.task_dir):
            here = Path(dirpath)
            if here == self.task_dir:
                dirnames[:] = [d for d in dirnames if d != "__pycache__" and not (
                    d == contract.PRIVATE_DIR and self.home_in_tree)]
            links = [d for d in dirnames if (here / d).is_symlink()]
            for name in filenames + links:
                f = here / name
                rel = f.relative_to(self.task_dir).as_posix()
                if rel in framework:
                    continue
                if f.is_symlink():
                    out[rel] = "link:" + os.readlink(f)
                elif f.is_file():
                    out[rel] = _md5(f)
        return out

    def _guard(self, before: dict[str, str], allowed: set[str]) -> str:
        """Why the reference agent's changes to the task directory are not allowed, or "".

        What it created or changed is moved into the private logs at once —
        left in place, a changed `problem.md` would be the next snapshot's
        "before". In build mode a package file it changed or removed sends the
        author stage back to pending, so `--resume` writes the package again.
        """
        after = self._public()
        changed = sorted(k for k in set(before) | set(after)
                         if before.get(k) != after.get(k) and k not in allowed)
        if not changed:
            return ""
        dest = _numbered(self.private_home / "logs", "reference-guard")
        with lifecycle.critical():
            for rel in changed:
                src = self.task_dir / rel
                if src.exists() or src.is_symlink():
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dest / rel))
        self.pipe.private_log(GUARD_LOG, "created or changed outside the private directory "
                              f"(moved to {dest}):\n" + "\n".join(changed))
        note = (f"the reference agent created or changed {len(changed)} file(s) outside the "
                "private directory — it saw the answers, so they were moved into the private "
                f"logs ({GUARD_LOG})")
        touched = [k for k in changed if k in before
                   and not k.startswith((authoring.STATE_DIR + "/", ".anchor/"))]
        if touched and self.pipe.state.get("mode") != "ingest":
            self.pipe.state["stages"]["author"] = {"status": "pending"}
            note += "; the package is written again on `--resume`"
        return note

    def _framework_files(self) -> dict[str, bytes]:
        out = {}
        for name in ("intake.md", "detach.json"):
            with contextlib.suppress(OSError):
                out[name] = (self.auth_dir / name).read_bytes()
        return out

    def _restore_framework(self, saved: dict[str, bytes]) -> None:
        """Put back what the framework wrote in `.authoring/`, should an agent have changed it.

        The guard leaves these files to the framework, which may rewrite them
        while an agent runs; so rather than fail the stage, the framework's
        own content is written over whatever an agent left there.
        """
        restored = []
        intake = self.auth_dir / "intake.md"
        text = authoring.render_intake(self.name, self.inputs).encode()
        with contextlib.suppress(OSError):
            if not intake.is_file() or intake.is_symlink() or intake.read_bytes() != text:
                intake.unlink(missing_ok=True)
                intake.write_bytes(text)
                restored.append("intake.md")
        detach = self.auth_dir / "detach.json"
        if "detach.json" in saved:  # written before the agent started: nobody else rewrites it
            with contextlib.suppress(OSError):
                if detach.is_symlink() or detach.read_bytes() != saved["detach.json"]:
                    detach.unlink(missing_ok=True)
                    detach.write_bytes(saved["detach.json"])
                    restored.append("detach.json")
        if restored:
            self.pipe.private_log(GUARD_LOG, "restored the framework's " + ", ".join(restored))

    def _sweep(self, workdir: Path, started: float, logs: Path) -> None:
        """Take what the harness kept of the agent's session out of its store."""
        with lifecycle.critical():
            try:
                moved, warning = audit.sweep_sessions(
                    self.agent.runtime, workdir, started, logs / "harness-sessions",
                    log=logs / "reference.log")
            except OSError as e:
                moved, warning = [], (f"the harness's session files could not be moved "
                                      f"({type(e).__name__}); remove the sessions run in the "
                                      "task's private reference directory from its store")
            if moved:
                self.pipe.private_log("harness-sessions.log", "\n".join(moved))
            # a cache the agent may have written by importing the package
            shutil.rmtree(self.task_dir / "__pycache__", ignore_errors=True)
        if warning:
            print(f"WARNING: {warning}")
            self._warning = warning

    def reference(self):
        uv = workspace.uv_binary()
        python = reference.solver_env(paths.state_dir(self.root), uv)
        logs = self.private_home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        workdir = self.private_home / "reference"
        ingest = self.pipe.state.get("mode") == "ingest"
        before = self._public()
        saved = self._framework_files()
        self._warning = ""
        started = time.time()
        sha = ""
        try:
            if ingest:
                location = reference.source_location(self.private_home, self.inputs.reference)
                data, sha = reference.fetch(location, workdir / "user")
                print(f"operator data copied into the private directory (sha256 {sha[:12]}…)")
                run = agents.ingest_reference(
                    self.name, intake=self._intake_text(), user_data=data, agent=self.agent,
                    python=python, uv=uv, budget_s=self.inputs.ref_budget_s,
                    log_path=logs / "reference.log", root=self.root,
                    record=logs / AGENT_RECORD,
                )
            else:
                pool = gpus.workspace_pool(self.root)
                gpu = str(pool[0]) if pool else ""
                print(f"reference agent: {self.agent.runtime}; solver env {python}; "
                      f"time limit {self.inputs.ref_budget_s / 60:.0f} min")
                run = agents.build_reference(
                    self.name, agent=self.agent, python=python, uv=uv, gpu=gpu,
                    tol=self.inputs.ref_tol, budget_s=self.inputs.ref_budget_s,
                    log_path=logs / "reference.log", root=self.root,
                    record=logs / AGENT_RECORD,
                )
            print(f"agent finished in {run.duration_s / 60:.1f} min (exit {run.returncode}"
                  f"{', timed out' if run.timed_out else ''})")
        finally:
            self._sweep(workdir, started, logs)
            self._restore_framework(saved)
        if why := self._guard(before, {"eval_grid.csv"} if ingest else set()):
            return False, self._noted(why)
        if ingest:
            return self._ingested(workdir, run, sha)
        if not (self.private_home / "ref_values.npy").is_file():
            return False, self._noted("the reference agent wrote no ref_values.npy")
        return True, self._noted("built")

    def _ingested(self, workdir: Path, run, sha: str):
        """Accept what the ingest agent wrote: read off the files, checked before the author."""
        import numpy as np

        info_path = workdir / "ingest.json"
        if not info_path.is_file():
            return False, self._noted(f"the ingest agent wrote no ingest.json "
                                      f"(exit {run.returncode})")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            info = None
        if not isinstance(info, dict):
            return False, self._noted("ingest.json is not a JSON object")
        info["sha256"] = sha
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        if why := contract.grid_problem(self.task_dir / "eval_grid.csv"):
            return False, self._noted(f"{why} — `--resume --redo reference` runs the ingest again")
        shape, why = grid_shape(self.task_dir, self.private_home, info)
        if why:
            return False, self._noted(why)
        # Before the author ever reads the grid: does it carry the answers?
        ref = np.load(self.private_home / "ref_values.npy", allow_pickle=False)
        ref = np.asarray(ref, dtype=np.float64).reshape(shape["rows"], -1)
        only = ("eval_grid.csv",)
        leaks = (contract.published_copies(self.task_dir, ref, shape["input_columns"], only)
                 + contract.printed_copies(self.task_dir, ref, only))
        if leaks:
            moved = self._quarantine(self.task_dir / "eval_grid.csv", "leaky-eval_grid")
            self.pipe.private_log(GUARD_LOG, f"eval_grid.csv held the reference (moved to "
                                  f"{moved}): " + "; ".join(leaks))
            return False, self._noted(
                "the ingested eval_grid.csv holds the reference — it was moved into the private "
                "logs; fix the ingest and `--resume`")
        grid_sha = reference.pin_grid(self.task_dir, self.private_home)
        # The ingest agent saw the answers: of what it wrote, only the grid's
        # shape as the files show it leaves the private directory.
        self.pipe.state["reference"] = {**shape, "grid_sha256": grid_sha}
        self.pipe._save()
        return True, self._noted(
            f"ingested: {shape['rows']} rows; grid {', '.join(shape['grid_columns'])}; "
            f"reference {', '.join(shape['ref_columns'])}")

    def reference_check(self):
        ingest = self.pipe.state.get("mode") == "ingest"
        if ingest:
            if why := reference.grid_drift(self.task_dir, self.private_home, restore=True):
                return False, f"{why} — `--resume` checks it again"
            verdict = reference.check_user(
                self.task_dir, self.private_home, self.pipe.state.get("reference", {})
            )
        else:
            python = paths.venv_bin(paths.state_dir(self.root) / "refenv")
            verdict = reference.check(
                self.task_dir, self.private_home, tol=self.inputs.ref_tol,
                python=python if python.is_file() else None,
            )
        accepted = ""
        if verdict["verdict"] != "pass" and self.inputs.accept_reference:
            accepted = self.inputs.accept_reference
            verdict["accepted_by_operator"] = accepted
        reference.record(verdict, self.private_home, self.auth_dir / "reference.json")
        _print_verdict(verdict)
        if verdict["verdict"] != "pass" and not accepted:
            note = "; ".join(verdict["reasons"]) or "reference check failed"
            if not ingest:
                # the reference stage is done, so a plain --resume would only
                # check the same levels again
                note += (" — `--resume --redo reference` re-runs the builder "
                         "(or `--resume --accept-reference REASON` releases it as it is)")
            return False, note
        est = verdict.get("estimate")
        return True, (f"estimated error {est:.1e}" if est else "operator data") + (
            f" (accepted: {accepted})" if accepted else ""
        )

    def validate(self):
        rep = contract.validate(self.task_dir)
        if not rep.ok:
            return False, "; ".join(n for n, _, _ in rep.failures)
        return True, f"{len(rep.checks)} checks pass"

    def smoke(self):
        from pinnforge.cli.task_cmd import run_smoke

        code, message = run_smoke(self.name)
        print(message)
        return code == 0, message

    def anchor(self):
        if self.inputs.no_anchor:
            return None, "skipped (--no-load)"
        from pinnforge.cli.task_cmd import scratch_run

        pool = gpus.workspace_pool(self.root)
        gpu = pool[0] if pool else None
        hw = anchor.hardware(gpu)
        anchor.clear(self.name, self.root)
        print(f"measuring b00 on {'gpu ' + str(gpu) if gpu is not None else 'the CPU'}…")
        with scratch_run(self.name) as run, scoring.serving(self.root, run):
            result = anchor.ensure(run, self.name, gpu=gpu, hw=hw)
        if result.rRMSE is None:
            return False, result.detail
        note = f"anchor rRMSE {result.rRMSE:.4g}"
        verdict_path = self.auth_dir / "reference.json"
        if verdict_path.is_file():
            est = json.loads(verdict_path.read_text(encoding="utf-8")).get("estimate")
            if est and est > 0.01 * result.rRMSE:
                note += (f" — WARNING: the reference error ({est:.1e}) is more than 1% of it; "
                         "blocks may hit the reference's own error before the PINN's")
        print(note)
        return True, note

    def finalize(self):
        return True, "released to run start"


def _print_verdict(v: dict) -> None:
    if v.get("source") == "built":
        diffs = ", ".join(f"{d:.2e}" for d in v.get("differences", []))
        order = v.get("order")
        print(f"reference check: {v['verdict'].upper()}")
        print(f"  method      {v.get('method') or '—'}")
        print(f"  level diffs {diffs or '—'}   (task metric, each level vs the next)")
        print(f"  order       {order:.2f}" if order else "  order       —")
        if v.get("estimate") is not None:
            print(f"  estimate    {v['estimate']:.2e}   (tolerance {v.get('tol'):.0e})")
        if "reproducible" in v:
            print(f"  reproduced  {'yes' if v['reproducible'] else 'NO'}")
    else:
        print(f"reference check (operator data): {v['verdict'].upper()} — {v.get('rows')} rows")
    for r in v.get("reasons", []):
        print(f"  ✗ {r}")


def _merge_flags(pipe: authoring.Pipeline, args, root: Path) -> list[str]:
    """Flags given to `--resume` override what the authoring started with. Returns their names."""
    inputs = pipe.state.setdefault("inputs", {})
    changed = []
    if args.ref_budget is not None:
        inputs["ref_budget_s"] = float(args.ref_budget) * 60
        changed.append("--ref-budget")
    if args.author_budget is not None:
        inputs["author_budget_s"] = float(args.author_budget) * 60
        changed.append("--author-budget")
    if args.ref_tol is not None:
        inputs["ref_tol"] = float(args.ref_tol)
        changed.append("--ref-tol")
    if args.accept_reference is not None:
        inputs["accept_reference"] = args.accept_reference
        changed.append("--accept-reference")
    if args.no_load:
        inputs["no_anchor"] = True
        changed.append("--no-load")
    if args.agent or args.runtime:
        cfg = bindings.resolve(root, agent=args.agent, runtime=args.runtime,
                               model=args.model or None)
        inputs["agent"] = cfg.to_dict()
        changed.append("--agent/--runtime" + ("/--model" if args.model else ""))
    elif args.model:
        cfg = AgentConfig.from_dict(inputs.get("agent") or {})
        cfg.model = args.model
        inputs["agent"] = cfg.to_dict()
        changed.append("--model")
    if changed:
        pipe._save()
    return changed


def _stop_hint(pgid, kill: bool = False) -> str:
    """How the operator stops process group `pgid` by hand."""
    if oscompat.WINDOWS:
        return f"`taskkill /T /F /PID {pgid}`"
    return f"`kill -{'KILL' if kill else 'TERM'} -- -{pgid}`"


def _reference_on_command_line() -> bool:
    return any(a == "--reference" or a.startswith("--reference=") for a in sys.argv[1:])


def cmd_new(args) -> int:
    if getattr(args, "log", None):
        from pinnforge.cli.run_cmd import _Tee

        class _ErrTee(_Tee):
            def __getattr__(self, name):  # isatty(), fileno(): the terminal's
                return getattr(self.stream, name)

        # stderr too: a crash's traceback belongs in the pipeline's log
        sys.stdout = _Tee(sys.stdout, Path(args.log))
        sys.stderr = _ErrTee(sys.stderr, Path(args.log))
    root = paths.project_root()
    name = args.name
    if not name:
        if not sys.stdin.isatty():
            print("error: name the task: pinnforge task new <name> --describe ...", file=sys.stderr)
            return 2
        name = _ask("Task name (letters, digits, - and _)").strip()
    task_dir = paths.tasks_dir(root) / name
    pipe = authoring.Pipeline(root, name)

    if args.recover:
        from pinnforge.task import registry

        found = registry.recover_from_run(name, root)
        if found is None:
            print(f"no run of {name!r} to recover a definition from")
            return 1
        dest, run = found
        print(f"recovered {name}'s definition from {run.name} -> {dest}")
        print("  the private directory is not in a run by design; supply the scoring truth "
              "before use")
        return 0 if (private.locate(dest) / "ref_values.npy").is_file() else 1

    redo = getattr(args, "redo", None)
    resume = bool(args.resume or redo)
    if resume:
        if not pipe.state:
            print(f"no authoring of {name!r} to resume")
            return 2
        try:
            pipe.lock()
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            if changed := _merge_flags(pipe, args, root):
                print(f"resuming with {', '.join(changed)} as given now")
            if redo:
                again = pipe.redo(redo)
                print(f"redoing {', '.join(again)}")
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        finally:
            pipe.unlock()
    else:
        if pipe.state or (task_dir / "problem.md").is_file():
            print(f"task {name!r} already exists — `--resume` to continue its authoring, "
                  "or choose another name")
            return 2
        describe = _describe(args)
        if not describe:
            if not sys.stdin.isatty():
                print("error: say what the task is: --describe \"...\" or --describe @file.md",
                      file=sys.stderr)
                return 2
            print("Describe the problem. (An empty answer skips a question.)")
            print(authoring.INTAKE_NOTE)
            describe = authoring.interview(_ask)
            if not args.reference:
                ref = _ask("Reference solution — a data file, directory or URL, or a .py/.txt "
                           "with the exact formula (empty: build one)")
                args.reference = ref.strip()
        agent = bindings.resolve(root, agent=args.agent, runtime=args.runtime,
                                 model=args.model or None)
        pool = gpus.workspace_pool(root)
        if args.reference:
            # Where the answers are is itself kept with them, never in the
            # authoring record the author agent and the dashboard read.
            reference.remember_source(private.locate(task_dir), args.reference)
        inputs = authoring.Inputs(
            describe=describe,
            reference="operator" if args.reference else "",
            sources=[str(Path(s).expanduser().resolve()) for s in (args.source or [])],
            template=args.template or "",
            agent=agent.to_dict(),
            gpu=str(pool[0]) if pool else "",
            ref_budget_s=float(args.ref_budget if args.ref_budget is not None else 240) * 60,
            author_budget_s=float(args.author_budget if args.author_budget is not None
                                  else 120) * 60,
            ref_tol=float(args.ref_tol if args.ref_tol is not None else reference.DEFAULT_TOL),
            accept_reference=args.accept_reference or "",
            no_anchor=bool(args.no_load),
        )
        pipe.begin(inputs)
        if args.reference and not args.detach and _reference_on_command_line():
            # `ps` shows every ancestor's command line to every process of
            # this user — the author agent included. Carry on as a process
            # whose command line does not hold the path to the answers.
            if oscompat.WINDOWS:
                # There is no exec: a process cannot shed its command line, and
                # the launcher that started it keeps its own. So the pipeline
                # goes on detached, and both of them end.
                print("note: given --reference, the pipeline continues detached on Windows, "
                      "so that no process's command line holds the path to the answers")
                args.detach = True
            else:
                cmd = lifecycle.self_command("--root", str(root), "task", "new", name,
                                             "--resume")
                if args.log:
                    cmd += ["--log", str(args.log)]
                sys.stdout.flush()
                sys.stderr.flush()
                os.execv(cmd[0], cmd)

    if args.detach:
        # The pipeline's console names the reference's method and provenance —
        # the operator's to read, so it is kept with the answers.
        log = private.locate(task_dir) / "logs" / "pipeline.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        info = lifecycle.detach(
            lifecycle.self_command("--root", str(root), "task", "new", name, "--resume",
                                   "--log", str(log)),
            cwd=root, log=log, name=lifecycle.session_name(root, name, kind="task"),
            record=task_dir / authoring.STATE_DIR / "detach.json",
        )
        where = "in tmux" if info["mode"] == "tmux" else f"in the background (pid {info['pid']})"
        print(f"authoring {name} {where}")
        print(f"  follow   pinnforge task status {name} --follow")
        return 0

    # SIGTERM and SIGHUP (a closed terminal, `tmux kill-session`) unwind like
    # Ctrl-C: the agent's group is stopped and the answers unsealed before
    # this process goes. A second signal kills the agent outright.
    lifecycle.install_signal_handlers(escalate=agents.kill_live)
    steps = Steps(pipe, root)
    try:
        code = pipe.run(steps.mapping(), before=steps.recover)
    except KeyboardInterrupt:
        print(f"\ninterrupted — the agent was stopped; resume with:\n"
              f"  pinnforge task new {name} --resume")
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if code == 0:
        print(f"\ntask {name} is ready.")
        print(f"  pinnforge run start -t {name} -n 3")
    return code


def cmd_status(args) -> int:
    root = paths.project_root()
    task_dir = paths.tasks_dir(root) / args.name
    while True:
        st = authoring.status(task_dir)
        if st is None:
            ready = (task_dir / "problem.md").is_file()
            print(f"{args.name}: {'ready (no authoring record)' if ready else 'no such task'}")
            return 0 if ready else 2
        _print_status(args.name, st, task_dir)
        done = st.get("finalized") or st.get("status") == "failed"
        if not args.follow or done:
            return 0 if st.get("finalized") else (1 if st.get("status") == "failed" else 0)
        time.sleep(3)
        print()


def _print_status(name: str, st: dict, task_dir: Path) -> None:
    mode = "operator reference" if st.get("mode") == "ingest" else "reference built by an agent"
    head = "READY" if st.get("finalized") else st.get("status", "?").upper()
    print(f"{name}: {head}   ({mode})")
    for stage in authoring.plan_for(st.get("mode", "build")):
        s = st.get("stages", {}).get(stage, {})
        mark = {"done": "✓", "failed": "✗", "running": "…", "skipped": "-"}.get(
            s.get("status"), " ")
        note = f"  {s['note']}" if s.get("note") else ""
        print(f"  [{mark}] {stage:<16}{note}")


def cmd_reference(args) -> int:
    """Check a task's reference — what the reference agent runs on itself."""
    root = paths.project_root()
    task_dir = paths.tasks_dir(root) / args.name
    python = paths.venv_bin(paths.state_dir(root) / "refenv")
    tol = args.tol
    if tol is None:
        st = authoring.status(task_dir) or {}
        tol = float(st.get("inputs", {}).get("ref_tol") or reference.DEFAULT_TOL)
    verdict = reference.check(
        task_dir, private.locate(task_dir), tol=tol,
        python=python if python.is_file() else None, rerun=not args.no_rerun,
    )
    _print_verdict(verdict)
    return 0 if verdict["verdict"] == "pass" else 1


def register(s) -> None:
    new = s.add_parser(
        "new",
        help="author a task from a described PDE: package, reference, checks, anchor",
    )
    new.add_argument("name", nargs="?", default=None)
    new.add_argument("--describe", default="",
                     help='the problem — text, or @file.md (asked interactively if omitted)')
    new.add_argument("--prompt", default="", help=argparse_suppress())
    new.add_argument("--reference", default="",
                     help="the operator's reference solution: a file, directory or URL "
                     "(omit it and an agent builds one)")
    new.add_argument("--source", action="append", default=None,
                     help="papers or notes for the author (repeatable)")
    new.add_argument("--template", default="",
                     help="task to adapt (default: the packaged demo)")
    new.add_argument("--agent", default=None, help="agent binding for the authoring agents")
    new.add_argument("--runtime", default=None)
    # Defaults are None so `--resume` can tell a flag given now from one left
    # out: given ones override what the authoring started with.
    new.add_argument("--model", default=None)
    new.add_argument("--ref-budget", type=float, default=None,
                     help="minutes the reference agent may take (default 240)")
    new.add_argument("--author-budget", type=float, default=None,
                     help="minutes the author agent may take (default 120)")
    new.add_argument("--ref-tol", type=float, default=None,
                     help="largest acceptable estimated reference error (relative L2; "
                     f"default {reference.DEFAULT_TOL:g})")
    new.add_argument("--accept-reference", default=None, metavar="REASON",
                     help="release the task even if the reference check fails, recording why")
    new.add_argument("--no-load", action="store_true", default=None,
                     help="stop before measuring the anchor")
    new.add_argument("--resume", action="store_true",
                     help="continue an interrupted authoring (flags given now override)")
    new.add_argument("--redo", default=None, metavar="STAGE",
                     help="with --resume: run STAGE and every stage after it again "
                     f"({', '.join(authoring.STAGES)})")
    new.add_argument("--recover", action="store_true",
                     help="rebuild a task's public half from its newest run instead")
    new.add_argument("--detach", action="store_true",
                     help="continue in the background after the intake")
    new.add_argument("--log", default=None, help=argparse_suppress())
    new.set_defaults(func=cmd_new)

    st = s.add_parser("status", help="where a task's authoring stands")
    st.add_argument("name")
    st.add_argument("--follow", action="store_true", help="keep watching until it ends")
    st.set_defaults(func=cmd_status)

    ref = s.add_parser("reference", help="check a built reference's convergence")
    ref.add_argument("name")
    ref.add_argument("--check", action="store_true", help="(the default action)")
    ref.add_argument("--tol", type=float, default=None)
    ref.add_argument("--no-rerun", action="store_true", help="skip re-running solve.py")
    ref.set_defaults(func=cmd_reference)


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS
