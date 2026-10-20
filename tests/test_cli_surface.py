"""What the operator sees when something is off: doctor, info, help, chat — and the
packaging details that decide what gets installed at all."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pytest
import yaml

from pinnforge import bindings, gpus, settings, workspace
from pinnforge.cli import build_parser, chat_cmd, doctor_cmd, init_cmd, main
from pinnforge.config import AgentConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with no environment and no GPUs, entered, with a private ~/.config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.delenv("FORGE_PRIVATE_DIR", raising=False)
    monkeypatch.setattr(gpus, "inventory", lambda timeout=20.0: [])
    root = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False, demo=False)
    monkeypatch.chdir(root)
    return root


# ─────────────────────────────── a broken settings file ───────────────────────────────


def test_doctor_reports_a_broken_settings_file_instead_of_crashing(ws, capsys):
    (ws / "pinnforge.yaml").write_text("sandbox:\n  gpus: [0\n", encoding="utf-8")
    found = doctor_cmd.checks()
    bad = [f for f in found if f.name == "settings"]
    assert len(bad) == 1 and bad[0].level == "fail"
    assert str(ws / "pinnforge.yaml") in bad[0].detail and "invalid YAML" in bad[0].detail
    assert {"tool", "workspace", "kb1"} <= {f.name for f in found}, "the rest still ran"
    assert doctor_cmd.cmd_doctor(argparse.Namespace(ping=False, json=False)) == 1
    assert "[FAIL] settings" in capsys.readouterr().out


def test_doctor_names_a_broken_user_file_too(ws):
    user = settings.user_dir()
    user.mkdir(parents=True)
    (user / "config.yaml").write_text("ui: [\n", encoding="utf-8")
    bad = [f for f in doctor_cmd.checks() if f.name == "settings"]
    assert [str(user / "config.yaml") in f.detail for f in bad] == [True]


def test_info_and_setup_print_the_problem_not_a_traceback(ws, capsys):
    (ws / "pinnforge.yaml").write_text("ui:\n  port: 1\n  - x\n", encoding="utf-8")
    assert main(["info"]) == 2
    err = capsys.readouterr().err
    assert f"error: {ws / 'pinnforge.yaml'}: invalid YAML" in err
    assert main(["setup", "--non-interactive"]) == 2
    assert "invalid YAML" in capsys.readouterr().err


# ─────────────────────────────── doctor's other checks ───────────────────────────────


def test_doctor_reads_an_int_pool_and_fails_a_nonsense_one(ws):
    settings.write(ws / "pinnforge.yaml", "sandbox.gpus", 1)
    gpu = [f for f in doctor_cmd.checks() if f.name == "gpus"]
    assert gpu and gpu[0].level == "fail" and "names GPU 1" in gpu[0].detail
    settings.write(ws / "pinnforge.yaml", "sandbox.gpus", "first")
    gpu = [f for f in doctor_cmd.checks() if f.name == "gpus"]
    assert gpu[0].level == "fail" and "not a GPU choice" in gpu[0].detail


def test_doctor_warns_about_cursor_rules_above_the_runs(tmp_path):
    root = tmp_path / "proj" / "ws"
    root.mkdir(parents=True)
    (tmp_path / "proj" / ".cursor" / "rules").mkdir(parents=True)
    (root / ".cursor" / "rules").mkdir(parents=True)
    (root / "AGENTS.md").write_text("x", encoding="utf-8")
    found = doctor_cmd._ambient(root)
    assert root / ".cursor" / "rules" in found
    assert tmp_path / "proj" / ".cursor" / "rules" in found
    assert root / "AGENTS.md" in found


def test_doctor_names_where_a_missing_private_dir_is_set(ws):
    user = settings.user_dir()
    user.mkdir(parents=True)
    (user / "config.yaml").write_text(f"private_dir: {ws / 'nowhere'}\n", encoding="utf-8")
    pd = [f for f in doctor_cmd.checks() if f.name == "private_dir"]
    assert pd and pd[0].level == "fail" and str(user / "config.yaml") in pd[0].fix


# ─────────────────────────────── help and completion ───────────────────────────────


def test_hidden_subcommands_stay_out_of_help_but_still_parse(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    assert "_drive" not in out and "SUPPRESS" not in out
    assert "{start,resume," in out and "audit" in out
    args = build_parser().parse_args(["run", "_drive", "ldc_1", "--resume"])
    assert args.run_id == "ldc_1"


def test_completion_offers_no_internal_commands():
    tree = init_cmd._tree(build_parser())
    assert "start" in tree["run"] and "_drive" not in tree["run"]


# ─────────────────────────────── chat on another harness ───────────────────────────────


@pytest.mark.parametrize("runtime", ["opencode", "codex", "cursor_agent", "pi"])
def test_chat_installs_the_operator_plugin_per_project_only(ws, capsys, runtime):
    bindings.put("other", AgentConfig(runtime=runtime, model="m"))
    assert chat_cmd.cmd_chat(argparse.Namespace(agent="other", rest=[])) == 2
    out = capsys.readouterr().out
    assert "per project" in out and "not a PINNForge workspace" in out
    assert "~/.config/opencode/agent" not in out.split("Not globally")[0]
    assert "codex plugin add" not in out.split("Not globally")[0]
    for step in chat_cmd.install_steps(runtime, chat_cmd.plugin_dir()):
        assert step in out
        assert "~" not in step and "codex plugin" not in step


def test_the_plugin_docs_never_install_globally():
    """Only Claude Code blocks switch the operator plugin off, so the others go per project."""
    import re

    for doc in (REPO / "plugin" / "README.md", REPO / "plugin" / "opencode" / "pinnforge.md"):
        text = doc.read_text(encoding="utf-8")
        code = "\n".join(re.findall(r"```[a-z]*\n(.*?)```", text, re.S))
        code += "\n".join(re.findall(r"`((?:mkdir|cp) [^`\n]*)`", text))  # inline commands
        assert "~/.config/opencode" not in code, doc
        assert "codex plugin add" not in code, doc
    assert ".opencode/agent" in (REPO / "plugin" / "README.md").read_text(encoding="utf-8")


# ─────────────────────────────── packaging ───────────────────────────────


def test_ci_runs_the_fast_suite_without_resyncing_the_jax_stack():
    ci = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    job = ci["jobs"]["test"]
    assert job["env"]["UV_NO_SYNC"] == "1", "`uv run` would install the stack group again"
    runs = [step.get("run", "") for step in job["steps"]]
    assert "uv sync --no-default-groups --group dev" in runs


def test_python_is_bounded_where_the_pinned_numpy_has_wheels():
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["requires-python"] == ">=3.11,<3.14"
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"].replace(" ", "") == ">=3.11,<3.14"
