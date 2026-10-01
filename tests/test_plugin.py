"""The harness plugin: manifests that load, a hook that stays out of blocks,
and documentation whose commands the CLI actually accepts."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from pinnforge.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugin"
DOCS = [*PLUGIN.glob("skills/*/SKILL.md"), *PLUGIN.glob("agents/*.md"), PLUGIN / "AGENTS.md",
        PLUGIN / "opencode" / "pinnforge.md", PLUGIN / "cursor" / "pinnforge.mdc",
        REPO / "README.md"]


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    head = text.split("---\n", 2)[1]
    import yaml

    return yaml.safe_load(head)


def test_manifests_parse_and_point_at_real_files():
    manifests = (
        REPO / ".claude-plugin" / "marketplace.json",
        PLUGIN / ".claude-plugin" / "plugin.json",
        PLUGIN / ".codex-plugin" / "plugin.json",
        REPO / ".agents" / "plugins" / "marketplace.json",
    )
    for f in manifests:
        json.loads(f.read_text(encoding="utf-8"))
    codex = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert (PLUGIN / codex["skills"]).is_dir() and (PLUGIN / codex["hooks"]).is_file()
    assert (PLUGIN / "hooks" / "hooks.json").is_file()


def test_skills_and_subagents_declare_themselves():
    for skill in PLUGIN.glob("skills/*/SKILL.md"):
        fm = _frontmatter(skill)
        assert fm["name"] == skill.parent.name and len(fm["description"]) > 80
    agents = sorted(PLUGIN.glob("agents/*.md"))
    assert [a.stem for a in agents] == ["pinnforge-run-doctor", "pinnforge-task-author"]
    for a in agents:
        fm = _frontmatter(a)
        assert fm["name"] == a.stem and fm["description"] and fm["tools"]
    assert "Write" not in _frontmatter(PLUGIN / "agents" / "pinnforge-run-doctor.md")["tools"]


@pytest.mark.parametrize("var", ["PINNFORGE_RUN", "PINNFORGE_ROLE"])
def test_the_hook_says_nothing_inside_framework_agents(var):
    env = {**os.environ, var: "x"}
    out = subprocess.run([sys.executable, str(PLUGIN / "hooks" / "session-start.py")],
                         capture_output=True, text=True, env=env, check=True)
    assert out.stdout == ""


def test_the_hook_speaks_valid_json_to_the_operator(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PINNFORGE_")}
    out = subprocess.run([sys.executable, str(PLUGIN / "hooks" / "session-start.py")],
                         capture_output=True, text=True, env=env, cwd=tmp_path, check=True)
    ctx = json.loads(out.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "PINNForge" in ctx


def _commands(text: str):
    """Single-line `pinnforge …` commands inside fenced code blocks."""
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        for raw in block.splitlines():
            raw = re.sub(r"\s+#.*$", "", raw.strip())
            for line in (part.strip() for part in re.split(r"&&|\|", raw)):
                if not line.startswith("pinnforge "):
                    continue
                line = re.sub(r"<[^>]+>", "x", line).replace("…", "x")
                if line.count('"') % 2 or line.count("'") % 2 or line.endswith("\\"):
                    continue  # continued on the next line; not a whole command
                yield line


def test_every_documented_command_parses():
    parser = build_parser()
    seen = 0
    for doc in DOCS:
        for cmd in _commands(doc.read_text(encoding="utf-8")):
            seen += 1
            try:
                parser.parse_args(shlex.split(cmd)[1:])
            except SystemExit:
                pytest.fail(f"{doc.relative_to(REPO)}: `{cmd}` is not a valid command")
    assert seen > 30
