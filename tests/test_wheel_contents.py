"""What `uv tool install` actually delivers.

The installed package must be complete on its own — the corpus blocks read,
the prompts, the dashboard, the harness plugin, the demo task and the stack
pins — because a user who installed it has no checkout to fall back on. And
it must carry nothing that is not meant to ship.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv to build the wheel")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(out), str(REPO)], check=True,
                   capture_output=True, text=True)
    (path,) = out.glob("pinnforge-*.whl")
    with zipfile.ZipFile(path) as z:
        return z.namelist()


def test_the_corpus_ships_whole(wheel):
    nodes = [n for n in wheel if n.startswith("pinnforge/kb1/") and n.endswith(".md")
             and not n.endswith("INDEX.md")]
    assert len(nodes) >= 130
    assert "pinnforge/kb1/INDEX.md" in wheel


@pytest.mark.parametrize("name", [
    "pinnforge/prompts/block.md.template",
    "pinnforge/prompts/task_author.md",
    "pinnforge/prompts/reference_builder.md",
    "pinnforge/prompts/reference_ingest.md",
    "pinnforge/ui/static/index.html",
    "pinnforge/ui/static/app.js",
    "pinnforge/ui/static/app.css",
    "pinnforge/examples/demo/eval.py",
    "pinnforge/examples/catalog.json",
    "pinnforge/templates/stack.toml",
    "pinnforge/task/evalkit.py",
    "pinnforge/_plugin/.claude-plugin/plugin.json",
    "pinnforge/_plugin/agents/pinnforge-task-author.md",
    "pinnforge/_plugin/skills/pinnforge-quickstart/SKILL.md",
])
def test_everything_a_workspace_needs_is_inside(wheel, name):
    assert name in wheel


def test_nothing_that_must_not_ship(wheel):
    assert not [n for n in wheel if "_QUARANTINE" in n or "/private/" in n or n.startswith("tasks/")
                or n.startswith("runs/")]
