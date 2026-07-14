#!/usr/bin/env python3
"""PreToolUse hook: .py files may only be written under blocks/, task/,
or .claude/ — blocks shadow-evaluator helper scripts (block.md §3)."""
import json
import os
import sys

data = json.load(sys.stdin)
path = (data.get("tool_input") or {}).get("file_path") or ""
if path.endswith(".py"):
    root = os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    target = os.path.realpath(path)
    allowed = tuple(os.path.join(root, d) + os.sep for d in ("blocks", "task", ".claude"))
    if not target.startswith(allowed):
        json.dump({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "block.md §3: candidates under blocks/bNN/ are the only .py "
                "files you may write — no helper scripts (scratchpad//tmp included). "
                "Run diagnostics inline or via eval.py --smoke."
            )}}, sys.stdout)
sys.exit(0)
