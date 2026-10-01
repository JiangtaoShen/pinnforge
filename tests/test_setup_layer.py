"""Settings layering, agent bindings and GPU resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pinnforge import bindings, gpus, settings
from pinnforge.config import AgentConfig


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A private ~/.config and a workspace, so nothing touches the real ones."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# ─────────────────────────────── settings ───────────────────────────────


def test_workspace_overrides_user_and_keeps_the_rest(home):
    user = settings.user_dir()
    user.mkdir(parents=True)
    (user / "config.yaml").write_text(
        "sandbox: {gpus: [0], wall_budget_s: 900}\nui: {port: 9000}\n", encoding="utf-8"
    )
    (home / "pinnforge.yaml").write_text("sandbox: {gpus: auto}\n", encoding="utf-8")
    assert settings.get(home, "sandbox.gpus") == "auto"
    assert settings.get(home, "sandbox.wall_budget_s") == 900  # merged, not replaced
    assert settings.get(home, "ui.port") == 9000
    assert settings.get(home, "ui.host", "127.0.0.1") == "127.0.0.1"


def test_a_null_setting_means_the_default(home):
    (home / "pinnforge.yaml").write_text("private_dir: null\n", encoding="utf-8")
    assert settings.get(home, "private_dir", "x") == "x"


def test_a_null_never_erases_the_layer_below():
    base = {"agent": "opus", "sandbox": {"gpus": [1], "wall_budget_s": 900}, "ui": {"port": 1}}
    over = {"agent": None, "sandbox": {"gpus": None, "wall_budget_s": 60}, "private_dir": None,
            "run": {"detach": None, "x": {"y": None}}}
    assert settings.deep_merge(base, over) == {
        "agent": "opus", "sandbox": {"gpus": [1], "wall_budget_s": 60}, "ui": {"port": 1},
        "run": {"x": {}}}


def test_the_machine_file_holds_in_a_workspace_init_made(home, monkeypatch):
    """The template's nulls are "unset", so ~/.config/pinnforge/config.yaml decides."""
    from pinnforge import private, workspace

    user = settings.user_dir()
    user.mkdir(parents=True)
    (user / "config.yaml").write_text(
        "agent: ds\nprivate_dir: /srv/answers\n"
        "sandbox: {gpus: [1], wall_budget_s: 900}\n", encoding="utf-8")
    ws = workspace.init(home / "ws", stack_name="cpu", venv=False, demo=False)
    raw = settings.workspace(ws)
    assert raw["agent"] is None and raw["private_dir"] is None
    assert raw["sandbox"] == {"gpus": None, "wall_budget_s": None}

    assert settings.get(ws, "sandbox.gpus") == [1]
    assert settings.get(ws, "sandbox.wall_budget_s") == 900
    assert settings.merged(ws)["sandbox"] == {"gpus": [1], "wall_budget_s": 900}
    assert settings.origin(ws, "sandbox.gpus") == str(user / "config.yaml")
    assert settings.origin(ws, "ui.port") == settings.WORKSPACE_FILE

    bindings.put("opus", AgentConfig(runtime="claude_code", model="claude-opus-4-8"))
    bindings.put("ds", AgentConfig(runtime="opencode", model="deepseek/deepseek-chat"))
    assert bindings.resolve(ws).runtime == "opencode", "the machine's agent, not the default"

    monkeypatch.delenv(private.ENV, raising=False)
    assert private.configured(ws) == Path("/srv/answers")
    assert private.locate(ws / "tasks" / "ldc") == Path("/srv/answers/ldc")
    monkeypatch.setenv(private.ENV, "/elsewhere")
    assert private.configured(ws) == Path("/elsewhere"), "the environment still wins"
    monkeypatch.delenv(private.ENV)

    settings.write(ws / "pinnforge.yaml", "private_dir", "/ws/answers")
    assert private.configured(ws) == Path("/ws/answers"), "the workspace's own wins"


def test_a_settings_file_that_does_not_parse_names_itself(home):
    (home / "pinnforge.yaml").write_text("sandbox:\n  gpus: [0\nui: {port: 1}\n",
                                         encoding="utf-8")
    with pytest.raises(ValueError, match=r"pinnforge.yaml: invalid YAML — .*\(line \d+"):
        settings.get(home, "sandbox.gpus")
    (home / "pinnforge.yaml").write_bytes(b"ui:\n  host: caf\xe9\n")
    with pytest.raises(ValueError, match=r"pinnforge\.yaml: not UTF-8"):
        settings.workspace(home)
    user = settings.user_dir()
    user.mkdir(parents=True)
    (user / "agents.yaml").write_text("bindings: {opus: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"agents\.yaml: invalid YAML"):
        bindings.load()


def test_a_settings_file_must_be_a_mapping(home):
    (home / "pinnforge.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        settings.workspace(home)


# ─────────────────────────────── bindings ───────────────────────────────


def test_first_binding_becomes_the_default(home):
    bindings.put("opus", AgentConfig(runtime="claude", model="claude-opus-4-8"))
    bindings.put("ds", AgentConfig(runtime="opencode", model="deepseek/deepseek-chat"))
    assert bindings.default_name() == "opus"
    assert bindings.get("opus").runtime == "claude_code"  # aliases are canonicalised


def test_resolution_order(home):
    bindings.put("opus", AgentConfig(runtime="claude_code", model="claude-opus-4-8"))
    bindings.put("ds", AgentConfig(runtime="opencode", model="deepseek/deepseek-chat"))
    # default binding
    assert bindings.resolve(home).model == "claude-opus-4-8"
    # the workspace names another binding
    (home / "pinnforge.yaml").write_text("agent: ds\n", encoding="utf-8")
    assert bindings.resolve(home).runtime == "opencode"
    # --runtime beats the workspace; --agent beats everything
    assert bindings.resolve(home, runtime="codex").runtime == "codex"
    assert bindings.resolve(home, agent="opus", runtime="codex").runtime == "claude_code"
    # --model alone keeps the runtime the chain chose
    got = bindings.resolve(home, model="deepseek/deepseek-reasoner")
    assert (got.runtime, got.model) == ("opencode", "deepseek/deepseek-reasoner")


def test_no_bindings_falls_back_to_the_builtin_default(home):
    assert bindings.resolve(home).runtime == "claude_code"


def test_an_unknown_binding_names_the_known_ones(home):
    bindings.put("opus", AgentConfig(runtime="claude_code", model="m"))
    with pytest.raises(KeyError, match="opus"):
        bindings.get("nope")


def test_removing_the_default_picks_another(home):
    bindings.put("a", AgentConfig(runtime="claude_code", model="m"))
    bindings.put("b", AgentConfig(runtime="codex", model="n"))
    bindings.remove("a")
    assert bindings.default_name() == "b"


# ─────────────────────────────── GPUs ───────────────────────────────

INV = [gpus.Gpu(0, "GPU-aaaa1111", "NVIDIA RTX A6000", 49140),
       gpus.Gpu(1, "GPU-bbbb2222", "NVIDIA RTX A6000", 49140)]


@pytest.mark.parametrize(
    ("value", "want"),
    [(None, [0, 1]), ("1", [1]), ("", []), ("-1", []), ("GPU-bbbb", [1]),
     ("0,GPU-bbbb2222", [0, 1])],
)
def test_auto_follows_cuda_visible_devices(value, want):
    env = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}
    assert gpus.resolve("auto", env=env, inv=INV) == want


def test_explicit_pools_pass_through():
    assert gpus.resolve([1], inv=INV) == [1]
    assert gpus.resolve("0,1", inv=INV) == [0, 1]


def test_no_gpu_means_cpu_mode():
    assert gpus.resolve("auto", env={}, inv=[]) == []
    assert gpus.hw_class([]) == {"platform": "cpu"}


def test_hardware_class_names_the_card():
    assert gpus.hw_class([0, 1], INV) == {"platform": "gpu", "gpu": "NVIDIA RTX A6000"}
    assert "2 × NVIDIA RTX A6000" in gpus.describe([0, 1], INV)


def test_cpu_mode_keeps_blocks_off_hidden_cards():
    """`eval.py --gpu 0` would re-expose a hidden card; JAX_PLATFORMS must not."""
    from pinnforge.config import RunConfig, SandboxConfig
    from pinnforge.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = RunConfig(task="t", sandbox=SandboxConfig(gpus=[], wall_budget_s=60))
    orch.root = orch.run = None
    assert orch._block_env()["JAX_PLATFORMS"] == "cpu"
    orch.cfg = RunConfig(task="t", sandbox=SandboxConfig(gpus=[0], wall_budget_s=60))
    assert "JAX_PLATFORMS" not in orch._block_env()


def test_a_budget_too_small_for_one_evaluation_is_refused(tmp_path, monkeypatch, capsys):
    import argparse

    from pinnforge.cli import run_cmd
    from pinnforge.task import examples

    (tmp_path / "pinnforge.yaml").write_text("sandbox:\n  gpus: []\n", encoding="utf-8")
    examples.install_demo(tmp_path / "tasks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    ns = argparse.Namespace(
        task="demo", blocks=1, agent=None, runtime="replay", model="", gpus="",
        budget="30", no_anchor=False, force=False, share=False, set=[],
        foreground=False, detach=True, no_ui=True,
    )
    assert run_cmd.cmd_start(ns) == 2
    assert "cannot pay for one evaluation" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists(), "refused before anything was built"
