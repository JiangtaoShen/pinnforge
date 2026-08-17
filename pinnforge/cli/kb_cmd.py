"""`pinnforge kb1 …` — grow and inspect the knowledge corpus."""

from __future__ import annotations

import re
from pathlib import Path

from pinnforge import paths
from pinnforge.config import AgentConfig

REQUIRED_SECTIONS = ("## TL;DR", "## Problem", "## Method", "## Results")

# The corpus a block reads is only usable if its nodes are comparable: the 130
# hand-written ones run 3.6–6.3 KB (median 5.0). A node far outside that has
# either padded past what a block can afford to read or dropped the runnable
# Method that makes it worth opening. The band is wider than the observed
# spread so it catches malformed output, not style.
MIN_NODE_B = 3 * 1024
MAX_NODE_B = 8 * 1024


def _nodes(kb1: Path) -> list[Path]:
    return sorted(p for p in kb1.glob("*.md") if p.name != "INDEX.md")


def cmd_list(args) -> int:
    kb1 = paths.kb1_dir()
    nodes = _nodes(kb1)
    print(f"{len(nodes)} node(s) in {kb1}")
    for p in nodes[: args.number]:
        title = ""
        for line in p.read_text(encoding="utf-8").splitlines()[:10]:
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"')
                break
        print(f"  {p.name}  {title[:70]}")
    return 0


def cmd_check(args) -> int:
    """Validate the corpus's shape — the paradigm blocks rely on."""
    kb1 = paths.kb1_dir()
    nodes = _nodes(kb1)
    bad = 0
    for p in nodes:
        text = p.read_text(encoding="utf-8")
        problems = []
        if not text.startswith("---"):
            problems.append("no frontmatter")
        for sec in REQUIRED_SECTIONS:
            if sec not in text:
                problems.append(f"missing {sec}")
        extra = [
            ln.strip()
            for ln in text.splitlines()
            if ln.startswith("## ") and ln.strip() not in REQUIRED_SECTIONS
        ]
        if extra:
            problems.append("extra sections: " + ", ".join(extra))
        if not re.match(r"^\d{3}_\d{4}_", p.name):
            problems.append("filename is not NNN_YYYY_…")
        size = len(text.encode("utf-8"))
        if not MIN_NODE_B <= size <= MAX_NODE_B:
            problems.append(
                f"{size / 1024:.1f} KB is outside the corpus band "
                f"({MIN_NODE_B // 1024}–{MAX_NODE_B // 1024} KB)"
            )
        if problems:
            bad += 1
            print(f"  [FAIL] {p.name}: {'; '.join(problems)}")
    index = kb1 / "INDEX.md"
    if not index.is_file():
        print("  [FAIL] INDEX.md missing")
        bad += 1
    else:
        entries = index.read_text(encoding="utf-8").splitlines()
        listed = {
            ln.split("**")[1] for ln in entries if ln.startswith("- **") and "**" in ln[4:]
        }
        names = {p.name for p in nodes}
        # Name the difference rather than the count: the index carries a
        # hand-tuned TL;DR snippet per entry, so it is repaired by editing
        # those lines, and "lists 129 for 130 nodes" does not say which.
        if missing := sorted(names - listed):
            print(f"  [FAIL] INDEX.md is missing {len(missing)}: {', '.join(missing[:5])}")
            bad += 1
        if orphan := sorted(listed - names):
            print(f"  [FAIL] INDEX.md lists {len(orphan)} node(s) that do not exist: "
                  f"{', '.join(orphan[:5])}")
            bad += 1
    print(f"{len(nodes)} node(s), {bad} problem(s)")
    return 0 if bad == 0 else 1


def cmd_add(args) -> int:
    from pinnforge import agents

    sources = Path(args.source).expanduser().resolve()
    if not sources.exists():
        print(f"no such source path: {sources}")
        return 2
    agent = AgentConfig(runtime=args.runtime, model=args.model or "")
    before = len(_nodes(paths.kb1_dir()))
    print(f"distilling {sources} into kb1 using {agent.runtime}…")
    result = agents.distill_kb1(sources, args.prompt or "", agent=agent)
    after = len(_nodes(paths.kb1_dir()))
    print(f"agent exited {result.returncode} after {result.duration_s:.0f} s")
    print(f"nodes: {before} → {after}")
    print(f"log: {result.log_path}")
    return 0 if result.ok else 1


def register(sub) -> None:
    p = sub.add_parser("kb1", help="the fixed knowledge corpus")
    s = p.add_subparsers(dest="kb_cmd", required=True)

    lst = s.add_parser("list", help="list knowledge nodes")
    lst.add_argument("-n", "--number", type=int, default=20)
    lst.set_defaults(func=cmd_list)

    s.add_parser("check", help="validate node shape and the index").set_defaults(func=cmd_check)

    add = s.add_parser("add", help="distil papers into new nodes")
    add.add_argument("--source", required=True, help="directory of papers")
    add.add_argument("--prompt", default="")
    add.add_argument("--runtime", default="claude_code")
    add.add_argument("--model", default="")
    add.set_defaults(func=cmd_add)
