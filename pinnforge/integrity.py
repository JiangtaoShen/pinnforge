"""Proving the record was not tampered with.

A run's conclusions rest on files a block could edit but must not: earlier
blocks' evaluations, their summaries, the task package, the corpus. The
charter forbids touching them and blocks have obeyed — but "we asked them
not to" is a weak thing to put in a paper, and a violation would be silent.

So each dispatch is bracketed. Before the agent starts, everything outside
its own workspace is hashed; after it exits, the hashes are checked. The
block's own ledger is checked differently: `evals.jsonl` may grow, never
change, so its previous content must survive as a prefix.

This detects rather than prevents, deliberately. Prevention means a
filesystem jail, and a jail has to enumerate every path a block legitimately
writes — its workspace, the score queue, temporary files, the GPU locks, the
harness's own session state, and whatever `uv add` touches, which the
charter explicitly permits. Miss one and a block that used to work now
fails, which is precisely the behavioural drift this framework cannot
afford. Verification costs a block nothing and produces an artefact a
reader can check.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import oscompat, paths

MANIFEST_DIR = ".integrity"
CHUNK = 1 << 20


@dataclass
class Violation:
    path: str
    kind: str  # modified | removed | truncated
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.path}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class Manifest:
    """Hashes of everything the block being dispatched must leave alone."""

    block: str
    files: dict[str, str] = field(default_factory=dict)
    # `evals.jsonl` of the active block may grow; its prefix may not change.
    ledger: str = ""
    ledger_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "block": self.block,
            "files": self.files,
            "ledger": self.ledger,
            "ledger_bytes": self.ledger_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        return cls(
            block=d.get("block", ""),
            files=dict(d.get("files") or {}),
            ledger=d.get("ledger", ""),
            ledger_bytes=int(d.get("ledger_bytes", 0)),
        )


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _protected(run: Path, block: str, root: Path) -> list[Path]:
    """Everything the dispatched block is not allowed to change.

    Its own workspace is absent by design — that is the one place the charter
    tells it to write.
    """
    out: list[Path] = []

    task = run / "task"
    if task.is_dir():
        out += [p for p in task.rglob("*") if p.is_file()]

    charter = run / "block.md"
    if charter.is_file():
        out.append(charter)

    blocks = paths.blocks_dir(run)
    if blocks.is_dir():
        for child in blocks.iterdir():
            if child.name == "kb2":
                # Earlier summaries are frozen; this block writes only its own.
                out += [p for p in child.glob("*.md") if p.stem != block]
            elif child.is_dir() and child.name != block:
                out += [p for p in child.rglob("*") if p.is_file()]

    # The run's own copy of the corpus; a run created before corpora were
    # copied links to the shared one, which is then what it reads.
    kb1 = run / "kb1"
    if not kb1.is_dir():
        kb1 = paths.kb1_dir(root)
    if kb1.is_dir():
        out += list(kb1.glob("*.md"))

    return out


def snapshot(run: Path, block: str, root: Path | None = None) -> Manifest:
    """Hash the protected set just before a block is dispatched."""
    root = root or paths.project_root()
    man = Manifest(block=block)
    for p in _protected(run, block, root):
        try:
            man.files[str(p)] = _digest(p)
        except OSError:
            continue  # vanished between listing and hashing; verify will catch it

    ledger = paths.evals_path(run, block)
    if ledger.is_file():
        data = ledger.read_bytes()
        man.ledger = hashlib.sha256(data).hexdigest()
        man.ledger_bytes = len(data)
    return man


def verify(run: Path, man: Manifest) -> list[Violation]:
    """Re-check the manifest after the block exits."""
    out: list[Violation] = []
    for path_s, want in man.files.items():
        p = Path(path_s)
        if not p.is_file():
            out.append(Violation(path_s, "removed"))
            continue
        try:
            got = _digest(p)
        except OSError as e:
            out.append(Violation(path_s, "removed", str(e)))
            continue
        if got != want:
            out.append(Violation(path_s, "modified"))

    ledger = paths.evals_path(run, man.block)
    if man.ledger_bytes:
        if not ledger.is_file():
            out.append(Violation(str(ledger), "removed", "the block's own ledger"))
        else:
            data = ledger.read_bytes()
            if len(data) < man.ledger_bytes:
                out.append(
                    Violation(str(ledger), "truncated",
                              f"{len(data)} < {man.ledger_bytes} bytes")
                )
            elif hashlib.sha256(data[: man.ledger_bytes]).hexdigest() != man.ledger:
                out.append(
                    Violation(str(ledger), "modified",
                              "earlier evaluations were rewritten")
                )
    return out


def manifest_path(run: Path, block: str, segment: int) -> Path:
    d = run / MANIFEST_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{block}.{segment}.json"


def result_path(run: Path, block: str, segment: int) -> Path:
    return run / MANIFEST_DIR / f"{block}.{segment}.result.json"


_ENTRY = re.compile(r"^(?P<block>b\d+)\.(?P<segment>\d+)(?P<result>\.result)?\.json$")


def _entries(run: Path):
    """(block, segment, is_result, path) for every manifest and verdict on disk."""
    d = run / MANIFEST_DIR
    if not d.is_dir():
        return
    for f in d.iterdir():
        if m := _ENTRY.match(f.name):
            yield m["block"], int(m["segment"]), bool(m["result"]), f


def next_segment(run: Path, block: str) -> int:
    """The first segment number no earlier dispatch of `block` has used.

    Segments are numbered per block across every invocation of the driver,
    not per `_run_one`: numbering from 0 again in a resumed block wrote its
    first manifest and verdict over the ones an earlier dispatch had left —
    and with them, any violation that dispatch was caught in.
    """
    used = [seg for b, seg, _, _ in _entries(run) if b == block]
    return max(used, default=-1) + 1


def segment_of(path: Path) -> int:
    m = _ENTRY.match(Path(path).name)
    if m is None:
        raise ValueError(f"not an integrity manifest: {path}")
    return int(m["segment"])


def save(run: Path, man: Manifest, segment: int | None = None) -> Path:
    """Write the manifest — never over one already written.

    With no `segment`, the next free number for the block is taken, and a
    name that turns out to be taken (one planted in advance) moves it on
    rather than being written over.
    """
    seg = next_segment(run, man.block) if segment is None else segment
    data = (json.dumps(man.to_dict(), indent=2) + "\n").encode()
    while True:
        path = manifest_path(run, man.block, seg)
        try:
            fd = oscompat.open_file(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if segment is not None:
                raise
            seg += 1
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path


def record_result(run: Path, block: str, segment: int, violations: list[Violation]) -> Path:
    """Write the verdict next to the manifest — the artefact a reader checks.

    Through a fresh temporary and a rename: `.integrity/` sits in the run
    directory the block works in, so a verdict planted there in advance is
    replaced, never written through.
    """
    path = result_path(run, block, segment)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        {
            "block": block,
            "segment": segment,
            "clean": not violations,
            "violations": [{"path": v.path, "kind": v.kind, "detail": v.detail}
                           for v in violations],
        },
        indent=2,
    ) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        oscompat.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def audit(run: Path) -> dict:
    """Summarise every verdict in a run — backs `pinnforge run audit`.

    A manifest without a verdict is reported, not skipped: a dispatch whose
    result was never written is an unchecked one, and counting only verdicts
    made such a run look smaller and fully checked rather than partly
    unchecked. (The segment a live run is dispatching right now is one of
    these until it ends.)
    """
    checked, dirty, unverified = 0, [], []
    manifests, verdicts = {}, {}
    for block, seg, is_result, f in _entries(run):
        (verdicts if is_result else manifests)[(block, seg)] = f
    for key in sorted(verdicts, key=lambda k: (int(k[0][1:]), k[1])):
        try:
            r = json.loads(verdicts[key].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            r = None
        if not isinstance(r, dict):
            unverified.append({"block": key[0], "segment": key[1], "reason": "unreadable verdict"})
            continue
        checked += 1
        if not r.get("clean", True):
            dirty.append(r)
    for key in sorted(manifests, key=lambda k: (int(k[0][1:]), k[1])):
        if key not in verdicts:
            unverified.append({"block": key[0], "segment": key[1], "reason": "no verdict"})
    return {
        "segments_checked": checked,
        "violations": dirty,
        "unverified": unverified,
        "clean": not dirty and not unverified,
    }
