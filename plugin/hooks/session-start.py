#!/usr/bin/env python3
"""PINNForge plugin SessionStart hook.

Two jobs, both cheap and dependency-free:
  1. Find the `pinnforge` CLI — on PATH, or in the environment of a workspace
     at or above the current directory.
  2. Inject a small block of context so the agent knows PINNForge is available
     and which skill or subagent to reach for.

Output contract (Claude Code / Codex SessionStart hook): print a JSON object on
stdout with hookSpecificOutput.additionalContext. The session is never blocked
(exit 0 always).

This hook is operator-facing only. It must never speak inside an agent the
framework itself started — a block (`PINNFORGE_RUN`), or an author, reference
or ping agent (`PINNFORGE_ROLE`): anything it injected there would change what
that agent is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def _find() -> tuple[str | None, Path | None]:
    """The CLI, and the workspace we are in (if any)."""
    workspace = None
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        if (base / "pinnforge.yaml").is_file():
            workspace = base
            break
    on_path = shutil.which("pinnforge")
    if on_path:
        return on_path, workspace
    if workspace is not None:
        local = workspace / ".venv" / "bin" / "pinnforge"
        if local.is_file():
            return str(local), workspace
    return None, workspace


def _installed(path: str, version: str | None, workspace: Path | None) -> str:
    ver = f" ({version})" if version else ""
    cmd = "pinnforge" if shutil.which("pinnforge") == path else path
    where = (f"You are in the workspace `{workspace}`." if workspace
             else "No workspace here yet — `pinnforge init <dir>` creates one.")
    return (
        f"# PINNForge is available\n\n"
        f"The `{cmd}` CLI is installed{ver}. {where} PINNForge runs autonomous "
        f"coding agents as **blocks** to design physics-informed neural networks; "
        f"every candidate is scored by rRMSE against a reference solution.\n\n"
        f"- **A PDE to solve** (\"use pinnforge on this equation\"): delegate to the "
        f"`pinnforge-task-author` subagent if available, else the "
        f"`creating-a-pinnforge-task` skill — `pinnforge task new`.\n"
        f"- **Run / watch / stop** blocks: `running-pinnforge-blocks` — "
        f"`run start`, `ui`, `run status`, `run stop`, `run resume`.\n"
        f"- **A stuck or failing run**: the `pinnforge-run-doctor` subagent.\n"
        f"- **Which agent/model runs blocks**: `setting-up-harnesses` — `pinnforge setup`.\n"
        f"- **First time**: `pinnforge-quickstart` — `pinnforge demo`, `pinnforge doctor`.\n\n"
        f"Do not memorise flags: `{cmd} <cmd> --help`. Never edit `runs/<id>/` while a "
        f"run is live; never start a run (it spends GPU hours and credits) unless "
        f"the user asked for one.\n"
    )


def _missing() -> str:
    return (
        "# PINNForge CLI not found\n\n"
        "The `pinnforge` CLI is not installed. If the user wants PINNForge, install it:\n\n"
        "```bash\n"
        "curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh\n"
        "pinnforge init ~/pf && cd ~/pf && pinnforge demo\n"
        "```\n\n"
        "The `pinnforge-quickstart` skill covers the path from there to a launched run.\n"
    )


def main() -> int:
    # An agent the framework started — a block, or an author, reference or
    # ping agent — must never receive operator context.
    if os.environ.get("PINNFORGE_RUN") or os.environ.get("PINNFORGE_ROLE"):
        return 0
    path, workspace = _find()
    context = _installed(path, _version(path), workspace) if path else _missing()
    event = "SessionStart"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event,
                                             "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)  # a broken hook must never block a session
