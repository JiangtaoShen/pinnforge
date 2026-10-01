"""Which GPUs a workspace uses is the operator's choice, written down once."""

from __future__ import annotations

import argparse

import pytest
import yaml

from pinnforge import gpus, settings, workspace

TWO = [gpus.Gpu(0, "GPU-a", "NVIDIA RTX A6000", 49140),
       gpus.Gpu(1, "GPU-b", "NVIDIA RTX A6000", 49140)]


@pytest.fixture
def machine(monkeypatch):
    """A two-card machine, whatever this one has."""
    monkeypatch.setattr(gpus, "inventory", lambda timeout=20.0: list(TWO))
    monkeypatch.setattr(gpus, "usage", lambda timeout=10.0: {0: (35, 0), 1: (20480, 97)})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return TWO


@pytest.mark.parametrize("typed, stored", [
    ("", "auto"), ("all", "auto"), ("AUTO", "auto"),
    ("none", []), ("cpu", []),
    ("1", [1]), ("0,1", [0, 1]), ("1 0", [0, 1]), ("1, 1", [1]),
])
def test_what_an_operator_types_is_what_is_stored(typed, stored):
    assert gpus.parse_choice(typed, TWO) == stored


@pytest.mark.parametrize("typed, says", [
    ("2", "no GPU 2"), ("0,5", "no GPU 5"), ("x", "not a list"),
])
def test_a_card_the_machine_lacks_is_refused(typed, says):
    with pytest.raises(ValueError, match=says):
        gpus.parse_choice(typed, TWO)


def test_stored_choices_resolve_and_read_back():
    assert gpus.resolve("none") == [] and gpus.resolve([]) == []
    assert gpus.resolve("all", env={}, inv=TWO) == [0, 1]
    assert [gpus.render(s) for s in ("auto", [1], [], "0,1")] == ["auto", "[1]", "[]", "[0, 1]"]
    assert gpus.missing([1, 3], TWO) == [3]
    assert gpus.missing([0], []) == [0], "no nvidia-smi: no card is usable"
    assert gpus.describe([]) == "CPU only"


def test_an_int_or_a_spaced_string_is_a_pool_too():
    """`gpus: 1` in a hand-edited file, or `"0 1"`, as `parse_choice` accepts."""
    assert gpus.render(1) == "[1]" and gpus.resolve(1, inv=[]) == [1]
    assert gpus.render("0 1") == "[0, 1]" and gpus.resolve("0 1", inv=[]) == [0, 1]
    assert gpus.resolve(" 0, 2 ", inv=[]) == [0, 2]
    assert gpus.render("none") == "[]"
    with pytest.raises(ValueError, match="not a GPU choice"):
        gpus.resolve("first", inv=[])


def test_info_and_setup_read_an_int_pool(tmp_path, machine, monkeypatch, capsys):
    from pinnforge.cli import main

    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False, demo=False)
    settings.write(ws / "pinnforge.yaml", "sandbox.gpus", 1)
    monkeypatch.chdir(ws)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert main(["info"]) == 0
    assert "(sandbox.gpus: [1])" in capsys.readouterr().out
    assert main(["setup", "gpus"]) == 0
    assert "sandbox.gpus: [1] — 1 × NVIDIA RTX A6000" in capsys.readouterr().out


def test_a_setting_is_changed_on_its_own_line(tmp_path):
    f = tmp_path / "pinnforge.yaml"
    f.write_text(workspace.WORKSPACE_YAML.format(stack="cpu", version="0"), encoding="utf-8")
    before = f.read_text(encoding="utf-8").splitlines()

    settings.write(f, "sandbox.gpus", [0, 1])
    after = f.read_text(encoding="utf-8").splitlines()
    changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
    assert changed == [("  gpus: null", "  gpus: [0, 1]")], "every comment survives"
    assert yaml.safe_load(f.read_text())["sandbox"]["gpus"] == [0, 1]

    settings.write(f, "ui.port", 9000)
    line = next(ln for ln in f.read_text().splitlines() if "port:" in ln)
    assert line == "  port: 9000", line


def test_a_missing_key_is_added_under_its_parent(tmp_path):
    f = tmp_path / "pinnforge.yaml"
    f.write_text("# mine\nsandbox:\n  wall_budget_s: 600  # mine too\nui:\n  port: 1\n",
                 encoding="utf-8")
    settings.write(f, "sandbox.gpus", [1])
    text = f.read_text(encoding="utf-8")
    assert "# mine\n" in text and "# mine too" in text
    assert yaml.safe_load(text) == {"sandbox": {"gpus": [1], "wall_budget_s": 600},
                                    "ui": {"port": 1}}

    odd = tmp_path / "odd.yaml"
    odd.write_text("sandbox: null\n", encoding="utf-8")
    settings.write(odd, "sandbox.gpus", [])  # a shape a line edit cannot fix: re-dumped
    assert yaml.safe_load(odd.read_text()) == {"sandbox": {"gpus": []}}

    fresh = tmp_path / "new" / "config.yaml"
    settings.write(fresh, "sandbox.gpus", "auto")
    assert yaml.safe_load(fresh.read_text()) == {"sandbox": {"gpus": "auto"}}


def test_a_missing_key_is_indented_like_its_siblings(tmp_path):
    f = tmp_path / "pinnforge.yaml"
    f.write_text("# top\nsandbox:\n    # budget\n    wall_budget_s: 600  # mine\n"
                 "ui:\n    port: 1\n", encoding="utf-8")
    settings.write(f, "sandbox.gpus", [1])
    text = f.read_text(encoding="utf-8")
    assert "    gpus: [1]\n" in text
    assert "# top" in text and "# budget" in text and "# mine" in text, "not re-dumped"
    assert yaml.safe_load(text) == {"sandbox": {"gpus": [1], "wall_budget_s": 600},
                                    "ui": {"port": 1}}
    settings.write(f, "ui.theme.dark", True)  # two levels missing, the file's own unit
    text = f.read_text(encoding="utf-8")
    assert "    theme:\n        dark: true\n" in text and "# budget" in text


def test_only_the_named_key_is_changed_never_a_deeper_namesake(tmp_path):
    f = tmp_path / "pinnforge.yaml"
    f.write_text("sandbox:\n  env:\n    gpus: custom   # an env var, not the pool\n"
                 "  wall_budget_s: 600\n", encoding="utf-8")
    settings.write(f, "sandbox.gpus", [0])
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert data == {"sandbox": {"env": {"gpus": "custom"}, "gpus": [0], "wall_budget_s": 600}}
    assert "# an env var, not the pool" in f.read_text(encoding="utf-8")


def test_an_edit_that_changes_anything_else_is_not_kept(tmp_path, monkeypatch):
    """The whole document is compared, not just the key that was set."""
    f = tmp_path / "pinnforge.yaml"
    f.write_text("# note\nsandbox:\n  gpus: auto\nui:\n  port: 1\n", encoding="utf-8")

    def sloppy(lines, keys, value):  # sets the key, and also loses ui.port
        return ["sandbox:\n", f"  gpus: {settings._flow(value)}\n"]

    monkeypatch.setattr(settings, "_set_lines", sloppy)
    settings.write(f, "sandbox.gpus", [1])
    assert yaml.safe_load(f.read_text(encoding="utf-8")) == {"sandbox": {"gpus": [1]},
                                                              "ui": {"port": 1}}


def test_init_stores_the_choice_and_a_refresh_keeps_the_operators_settings(tmp_path):
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False, gpus_spec=[1])
    assert settings.get(ws, "sandbox.gpus") == [1]
    assert "`pinnforge setup gpus`" in (ws / "pinnforge.yaml").read_text()

    settings.write(ws / "pinnforge.yaml", "ui.port", 9100)
    workspace.init(ws, stack_name="cpu", venv=False, force=True)
    assert settings.get(ws, "sandbox.gpus") == [1], "a refresh is not a reset"
    assert settings.get(ws, "ui.port") == 9100
    workspace.init(ws, stack_name="cpu", venv=False, force=True, gpus_spec="auto")
    assert settings.get(ws, "sandbox.gpus") == "auto"


def _init_args(dest, gpus_flag):
    return argparse.Namespace(dir=str(dest), stack="cpu", examples="", no_venv=True,
                              no_demo=True, force=False, gpus=gpus_flag)


def test_init_takes_the_cards_from_the_flag(tmp_path, machine, capsys):
    from pinnforge.cli import init_cmd

    assert init_cmd.cmd_init(_init_args(tmp_path / "a", "1")) == 0
    assert settings.get(tmp_path / "a", "sandbox.gpus") == [1]
    assert "[1] — 1 × NVIDIA RTX A6000" in capsys.readouterr().out

    assert init_cmd.cmd_init(_init_args(tmp_path / "b", "7")) == 2
    assert "no GPU 7" in capsys.readouterr().out
    assert not (tmp_path / "b").exists(), "nothing is built on a bad choice"


def test_init_asks_when_there_is_a_choice(tmp_path, machine, monkeypatch, capsys):
    from pinnforge.cli import init_cmd

    answers = iter(["3", "0"])  # a card it lacks, then a real one
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(init_cmd.sys.stdin, "isatty", lambda: True)
    assert init_cmd.cmd_init(_init_args(tmp_path / "ws", None)) == 0
    out = capsys.readouterr().out
    assert "[1]  NVIDIA RTX A6000   48 GB   now: 20.0 GB used, 97% busy" in out, out
    assert "no GPU 3" in out
    assert settings.get(tmp_path / "ws", "sandbox.gpus") == [0]


def test_setup_gpus_changes_the_workspace_or_the_machine_default(
        tmp_path, machine, monkeypatch, capsys):
    from pinnforge.cli import setup_cmd

    ws = workspace.init(tmp_path / "ws", stack_name="cuda12", venv=False)
    monkeypatch.chdir(ws)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def setup(spec, user=False):
        return setup_cmd.cmd_setup_gpus(argparse.Namespace(spec=spec, user=user))

    assert setup("0,1") == 0
    assert settings.get(ws, "sandbox.gpus") == [0, 1]
    assert setup("none") == 0
    assert settings.get(ws, "sandbox.gpus") == []
    assert setup("4") == 2
    assert settings.get(ws, "sandbox.gpus") == [], "a refused choice changes nothing"

    capsys.readouterr()
    assert setup("1", user=True) == 0
    assert settings.user()["sandbox"]["gpus"] == [1]
    assert "pinnforge.yaml sets its own sandbox.gpus" in capsys.readouterr().out


def test_a_run_on_a_card_the_machine_lacks_is_refused(tmp_path, machine, monkeypatch, capsys):
    from pinnforge.cli import run_cmd
    from pinnforge.task import examples

    (tmp_path / "pinnforge.yaml").write_text("sandbox:\n  gpus: [0, 3]\n", encoding="utf-8")
    examples.install_demo(tmp_path / "tasks")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    ns = argparse.Namespace(
        task="demo", blocks=1, agent=None, runtime="replay", model="", gpus="",
        budget="", no_anchor=False, force=False, share=False, set=[],
        foreground=False, detach=True, no_ui=True,
    )
    assert run_cmd.cmd_start(ns) == 2
    out = capsys.readouterr().out
    assert "GPU 3 is not on this machine" in out
    assert "sandbox.gpus: [0, 3] in pinnforge.yaml" in out
    assert not (tmp_path / "runs").exists()


def test_the_dashboard_dims_cards_the_workspace_does_not_use(tmp_path, machine):
    from pinnforge.ui import api

    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False, gpus_spec=[1])
    view = api.overview(ws)
    assert [g["index"] for g in view["gpus"]] == [0, 1]
    assert view["pool"] == [1]


def test_a_new_workspace_follows_the_machine_default(tmp_path, machine, monkeypatch, capsys):
    """`setup gpus --user` must reach every workspace `init` made without a choice."""
    from pinnforge.cli import init_cmd, setup_cmd

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert setup_cmd.cmd_setup_gpus(argparse.Namespace(spec="1", user=True)) == 0
    capsys.readouterr()

    assert init_cmd.cmd_init(_init_args(tmp_path / "ws", None)) == 0  # not a tty: no question
    ws = tmp_path / "ws"
    assert "[1] — 1 × NVIDIA RTX A6000" in capsys.readouterr().out
    assert settings.workspace(ws)["sandbox"]["gpus"] is None, "left to the machine"
    assert settings.get(ws, "sandbox.gpus") == [1]
    assert settings.origin(ws, "sandbox.gpus") == str(settings.user_dir() / "config.yaml")
    assert gpus.workspace_pool(ws, TWO) == [1]

    monkeypatch.chdir(ws)
    assert setup_cmd.cmd_setup_gpus(argparse.Namespace(spec="0", user=True)) == 0
    out = capsys.readouterr().out
    assert "sets its own sandbox.gpus" not in out
    assert settings.get(ws, "sandbox.gpus") == [0], "the machine default moved; so did it"

    assert setup_cmd.cmd_setup_gpus(argparse.Namespace(spec="none", user=False)) == 0
    assert settings.origin(ws, "sandbox.gpus") == settings.WORKSPACE_FILE
    assert settings.get(ws, "sandbox.gpus") == []
    assert setup_cmd.cmd_setup_gpus(argparse.Namespace(spec="all", user=True)) == 0
    assert "sets its own sandbox.gpus, which wins here" in capsys.readouterr().out
    assert settings.get(ws, "sandbox.gpus") == []


def test_demo_reads_gpus_the_way_init_does(tmp_path, machine, monkeypatch, capsys):
    from pinnforge.cli import demo_cmd, run_cmd

    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    monkeypatch.chdir(ws)
    monkeypatch.delenv("PINNFORGE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    seen = []
    monkeypatch.setattr(run_cmd, "cmd_start", lambda ns: seen.append(ns) or 0)

    def demo(flag):
        ns = argparse.Namespace(blocks=1, budget=240, gpus=flag, share=False,
                                foreground=False, no_ui=True)
        return demo_cmd.cmd_demo(ns)

    for flag, cards, sets in [("all", "0,1", []), ("0 1", "0,1", []), ("1", "1", []),
                              ("none", "", ["sandbox.gpus=[]"])]:
        assert demo(flag) == 0
        assert (seen[-1].gpus, seen[-1].set) == (cards, sets), flag
    assert demo("7") == 2
    assert "--gpus: no GPU 7" in capsys.readouterr().out
    from pinnforge.cli import build_parser

    sub = next(a for a in build_parser()._actions if hasattr(a, "choices") and a.choices
               and "demo" in a.choices)
    assert "workspace's sandbox.gpus" in " ".join(sub.choices["demo"].format_help().split())
