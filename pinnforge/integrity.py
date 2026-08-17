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
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import paths

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


def save(run: Path, man: Manifest, segment: int) -> Path:
    path = manifest_path(run, man.block, segment)
    path.write_text(json.dumps(man.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def record_result(run: Path, block: str, segment: int, violations: list[Violation]) -> Path:
    """Write the verdict next to the manifest — the artefact a reader checks."""
    path = run / MANIFEST_DIR / f"{block}.{segment}.result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "block": block,
                "segment": segment,
                "clean": not violations,
                "violations": [{"path": v.path, "kind": v.kind, "detail": v.detail}
                               for v in violations],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def audit(run: Path) -> dict:
    """Summarise every verdict in a run — backs `pinnforge run audit`."""
    d = run / MANIFEST_DIR
    checked, dirty = 0, []
    if d.is_dir():
        for f in sorted(d.glob("*.result.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            checked += 1
            if not r.get("clean", True):
                dirty.append(r)
    return {"segments_checked": checked, "violations": dirty, "clean": not dirty}
