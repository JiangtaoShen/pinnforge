"""`pinnforge chat` — talk to your coding agent with PINNForge already loaded.

Starts the default agent binding's CLI interactively in the workspace with the
PINNForge plugin (skills, the task-author and run-doctor subagents, the
session hook) loaded for this session only — nothing to install, and nothing
written into the workspace, where every block would read it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from pinnforge import bindings, oscompat, paths


def plugin_dir() -> Path | None:
    """The plugin shipped in the wheel, or the checkout's own `plugin/`."""
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "_plugin", here.parent / "plugin"):
        if (cand / ".claude-plugin" / "plugin.json").is_file():
            return cand
    return None


# Why Codex, Cursor and OpenCode get the operator plugin per project only.
PER_PROJECT_WHY = (
    "Not globally (~/.config/opencode, `codex plugin add`, user rules): blocks run inside "
    "a workspace's runs/<id>/ with the same CLI, which reads your user-level config and "
    "often its parent directories' too — so they would load the operator's instructions. "
    "Only Claude Code and Pi blocks are started with the operator's plugin, skills and "
    "instruction files switched off."
)


def install_steps(runtime: str, plugin: Path | None) -> list[str]:
    """Shell commands that install the operator plugin into the current project."""
    src = str(plugin) if plugin is not None else "<pinnforge plugin directory>"
    if runtime == "opencode":
        return [f"mkdir -p .opencode/agent && cp {src}/opencode/pinnforge.md .opencode/agent/"]
    if runtime == "cursor_agent":
        return [f"mkdir -p .cursor/rules && cp {src}/cursor/pinnforge.mdc .cursor/rules/"]
    if runtime in ("codex", "pi"):  # both read AGENTS.md and the Agent Skills layout
        return [f"cp {src}/AGENTS.md .",
                f"mkdir -p .agents/skills && cp -r {src}/skills/. .agents/skills/"]
    return []


def cmd_chat(args) -> int:
    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        root = Path.cwd()
    agent = bindings.resolve(root if paths.is_workspace(root) else None, agent=args.agent)
    plugin = plugin_dir()
    if agent.runtime != "claude_code":
        print(f"`pinnforge chat` starts Claude Code; your binding is {agent.runtime}.")
        if steps := install_steps(agent.runtime, plugin):
            print(f"To drive PINNForge from {agent.runtime}, install the operator plugin per "
                  "project: in your own project directory — not a PINNForge workspace, and "
                  "not a directory above one — run")
            for step in steps:
                print(f"  {step}")
            print(PER_PROJECT_WHY)
        return 2
    binary = agent.command or shutil.which("claude")
    if not binary:
        print("Claude Code (`claude`) is not installed — see `pinnforge setup`.")
        return 2
    argv = [binary]
    if plugin is not None:
        argv += ["--plugin-dir", str(plugin)]
    if agent.model:
        argv += ["--model", agent.model]
    argv += args.rest
    os.chdir(root)
    if oscompat.WINDOWS:
        # exec there is a new process and an exit: the shell would take the
        # console back while Claude Code is still reading from it
        return subprocess.call(oscompat.resolve_command(argv))
    os.execvp(argv[0], argv)
    return 0  # not reached


def register(sub) -> None:
    p = sub.add_parser("chat", help="open your coding agent with PINNForge loaded")
    p.add_argument("--agent", default=None, help="agent binding (default: the default binding)")
    p.add_argument("rest", nargs="*", help="passed on to the agent CLI")
    p.set_defaults(func=cmd_chat)
