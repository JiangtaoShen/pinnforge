#!/usr/bin/env python3
"""PINNForge plugin SessionStart hook.

Two jobs, both cheap and dependency-free:
  1. Detect whether the `pinnforge` CLI is installed and reachable on PATH.
  2. Inject a small block of context so the agent knows PINNForge is available
     and which skill to reach for (authoring vs. running vs. harness setup).

Output contract (Claude Code / Codex SessionStart hook): print a JSON object on
stdout with hookSpecificOutput.additionalContext. That string is added to the
session context. The session is never blocked (exit 0 always).

This hook is operator-facing only. It must never run inside a block: blocks are
started in a run directory by the orchestrator, with their own charter, and
anything this injected there would change what a block is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def _version(path: str) -> str | None:
    """Best-effort `pinnforge --version`; None if it does not answer quickly."""
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode == 0:
        return (out.stdout or out.stderr).strip() or None
    return None


def _installed(version: str | None) -> str:
    ver = f" ({version})" if version else ""
    return (
        f"# PINNForge is available\n\n"
        f"The `pinnforge` CLI is installed{ver}. PINNForge runs autonomous coding "
        f"agents as **blocks**: one agent, a fixed GPU-wall budget, one written "
        f"summary, in series, with knowledge compounding through the summaries.\n\n"
        f"When the user wants to **run or manage blocks** (`pinnforge run start / "
        f"status / resume / log / show / audit`), use the `running-pinnforge-blocks` "
        f"skill. When they want to **author a task** (`problem.md` + `baseline.py` + "
        f"`eval.py`), use `creating-a-pinnforge-task`. For picking and authenticating "
        f"a coding-agent CLI, use `setting-up-harnesses`. For what-it-is and the fast "
        f"path from zero, use `pinnforge-quickstart`.\n\n"
        f"Do not memorise flags: run `pinnforge --help` or `pinnforge <cmd> --help`, "
        f"and let the skills drive the workflow.\n\n"
        f"**Never edit a run directory while it is running.** `runs/<id>/` is a live "
        f"block's workspace, and the orchestrator hashes it before every dispatch.\n"
    )


def _missing() -> str:
    return (
        "# PINNForge CLI not found\n\n"
        "The `pinnforge` CLI is not on PATH. If the user asks to author or run a "
        "PINNForge task, install it first:\n\n"
        "```bash\n"
        "git clone git@github.com:JiangtaoShen/pinnforge.git && cd pinnforge\n"
        "uv sync\n"
        "```\n\n"
        "Then the `pinnforge-quickstart` skill covers the path from there to a "
        "launched run.\n"
    )


def main() -> int:
    # A block agent must never receive operator context. The orchestrator sets
    # PINNFORGE_RUN for every block process, so its presence means "this session
    # is a block, stay out of it".
    if os.environ.get("PINNFORGE_RUN"):
        return 0
    path = shutil.which("pinnforge")
    context = _installed(_version(path)) if path else _missing()
    json.dump(
        {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}},
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
