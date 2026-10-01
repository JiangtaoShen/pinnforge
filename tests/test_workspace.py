"""Workspaces: the marker, the pinned environment, and `pinnforge init`."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pinnforge import paths, workspace
from pinnforge.task import contract, examples

REPO = Path(__file__).resolve().parents[1]


def test_the_dev_environment_pins_the_same_stack_a_workspace_gets():
    """One source of truth: templates/stack.toml, mirrored by the repo's `stack` group."""
    s = workspace.stack()
    group = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))[
        "dependency-groups"]["stack"]
    assert sorted(group) == sorted([*s["common"], *s["cuda12"]])


def test_the_workspace_pyproject_pins_and_constrains_the_stack(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_direct_url", lambda: None)
    text = workspace.render_pyproject(tmp_path, "cpu")
    data = tomllib.loads(text)
    deps = data["project"]["dependencies"]
    assert "jax==0.4.33" in deps and "pinnforge" in deps
    assert "jax==0.4.33" in data["tool"]["uv"]["constraint-dependencies"]
    assert data["tool"]["uv"]["package"] is False
    assert data["project"]["requires-python"] == ">=3.11,<3.13"


@pytest.mark.parametrize(
    ("info", "want"),
    [
        ({"url": "https://github.com/x/pinnforge",
          "vcs_info": {"vcs": "git", "commit_id": "abc123"}},
         'git = "https://github.com/x/pinnforge", rev = "abc123"'),
        ({"url": "file:///src/pinnforge", "dir_info": {"editable": True}},
         'path = "/src/pinnforge", editable = true'),
    ],
)
def test_pinnforge_is_pinned_to_the_tools_own_source(tmp_path, monkeypatch, info, want):
    monkeypatch.setattr(workspace, "_direct_url", lambda: info)
    requirement, source, _ = workspace.pinnforge_source(tmp_path)
    assert requirement == "pinnforge"
    assert want in source


def test_without_an_install_record_pinnforge_never_comes_from_an_index(tmp_path, monkeypatch):
    """PyPI's `pinnforge` is somebody else's project: the fallback is our own git tag."""
    from pinnforge import __version__

    for info in (None, {"url": "file:///gone/pinnforge-0.2.0-py3-none-any.whl",
                        "archive_info": {}}):
        monkeypatch.setattr(workspace, "_direct_url", lambda info=info: info)
        requirement, source, note = workspace.pinnforge_source(tmp_path)
        assert requirement == "pinnforge", "a bare name with a source, never `==version`"
        assert source == (f'{{ git = "https://github.com/JiangtaoShen/pinnforge", '
                          f'tag = "v{__version__}" }}')
        assert f"v{__version__}" in note and "index" in note
        data = tomllib.loads(workspace.render_pyproject(tmp_path, "cpu"))
        assert data["tool"]["uv"]["sources"]["pinnforge"] == {
            "git": "https://github.com/JiangtaoShen/pinnforge", "tag": f"v{__version__}"}
        assert not [d for d in data["project"]["dependencies"] if d.startswith("pinnforge=")]


def test_an_archive_installed_from_a_url_is_pinned_to_that_url(tmp_path, monkeypatch):
    url = "https://example.org/dist/pinnforge-0.2.0-py3-none-any.whl"
    monkeypatch.setattr(workspace, "_direct_url",
                        lambda: {"url": url, "archive_info": {"hashes": {"sha256": "ab"}}})
    requirement, source, _ = workspace.pinnforge_source(tmp_path)
    assert (requirement, source) == ("pinnforge", f'{{ url = "{url}" }}')
    data = tomllib.loads(workspace.render_pyproject(tmp_path, "cpu"))
    assert data["tool"]["uv"]["sources"]["pinnforge"] == {"url": url}


def test_init_never_replaces_a_pyproject_it_did_not_write(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    mine = '[project]\nname = "mine"\nversion = "1"\n'
    (proj / "pyproject.toml").write_text(mine, encoding="utf-8")
    with pytest.raises(FileExistsError, match="not written by `pinnforge init`"):
        workspace.init(proj, stack_name="cpu", venv=False, force=True)
    assert (proj / "pyproject.toml").read_text(encoding="utf-8") == mine
    assert not paths.is_workspace(proj), "refused before anything was written"

    # a source checkout marked as a workspace: its package pyproject is not ours either
    (proj / paths.MARKER).write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        workspace.init(proj, stack_name="cpu", venv=False, force=True)
    with pytest.raises(FileExistsError):
        workspace.repin(proj, "cpu")
    assert (proj / "pyproject.toml").read_text(encoding="utf-8") == mine

    # its own, it refreshes
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    assert workspace.owns_pyproject(ws)
    workspace.init(ws, stack_name="cpu", venv=False, force=True)
    assert workspace.repin(ws) == "cpu"


def test_a_refresh_keeps_the_workspaces_stack(tmp_path, monkeypatch):
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    monkeypatch.setattr(workspace, "detect_stack", lambda: "cuda12")
    workspace.init(ws, venv=False, force=True)
    assert workspace.stack_of(ws) == "cpu", "re-detected instead of kept"
    workspace.init(ws, stack_name="cuda12", venv=False, force=True)
    assert workspace.stack_of(ws) == "cuda12", "an explicit stack still switches it"


def test_init_lays_out_a_workspace_without_the_environment(tmp_path):
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    for d in ("tasks", "runs", "logs", ".pinnforge"):
        assert (ws / d).is_dir()
    assert paths.is_workspace(ws)
    assert (ws / "pyproject.toml").is_file() and (ws / ".gitignore").is_file()
    demo = ws / "tasks" / "demo"
    assert contract.validate(demo).ok
    assert paths.project_root(ws / "tasks" / "demo") == ws.resolve()


def test_init_refuses_to_clobber(tmp_path):
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    with pytest.raises(FileExistsError):
        workspace.init(ws, stack_name="cpu", venv=False)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        workspace.init(busy, stack_name="cpu", venv=False)


def test_a_workspace_without_its_environment_says_how_to_fix_it(tmp_path, monkeypatch):
    ws = workspace.init(tmp_path / "ws", stack_name="cpu", venv=False)
    monkeypatch.delenv("PINNFORGE_AMBIENT_PYTHON", raising=False)
    with pytest.raises(FileNotFoundError, match="workspace sync"):
        paths.venv_python(ws)


def test_the_demo_reference_is_the_analytic_solution(tmp_path):
    import numpy as np

    task = examples.install_demo(tmp_path)
    grid = np.loadtxt(task / "eval_grid.csv", delimiter=",", skiprows=1)
    ref = np.load(task / "private" / "ref_values.npy")
    assert ref.shape == (examples.GRID_N**2, 1)
    assert np.allclose(ref[:, 0], examples.demo_solution(grid[:, 0], grid[:, 1]))
    # zero on the walls, as problem.md states
    on_wall = (grid[:, 0] == 0) | (grid[:, 0] == 1) | (grid[:, 1] == 0) | (grid[:, 1] == 1)
    assert np.abs(ref[on_wall, 0]).max() < 1e-12


def test_an_example_archive_round_trips_and_is_verified(tmp_path, monkeypatch):
    task = examples.install_demo(tmp_path / "lib", name="toy")
    (task / ".anchor").mkdir()
    (task / ".anchor" / "meta.json").write_text("{}", encoding="utf-8")
    archive, sha = examples.pack(task, tmp_path / "dist")
    monkeypatch.setattr(examples, "catalog", lambda: [{"name": "toy", "url": "x", "sha256": sha}])
    dest = examples.fetch("toy", tmp_path / "tasks", source=str(archive))
    assert (dest / "private" / "ref_values.npy").is_file()
    assert not (dest / ".anchor").exists(), "a machine's anchor does not travel"
    wrong = [{"name": "toy", "url": "x", "sha256": "0" * 64}]
    monkeypatch.setattr(examples, "catalog", lambda: wrong)
    with pytest.raises(ValueError, match="sha256"):
        examples.fetch("toy", tmp_path / "other", source=str(archive))


def test_an_unreachable_example_says_how_to_supply_it(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    url = "https://github.com/JiangtaoShen/pinnforge/releases/download/examples-v1/toy.tar.gz"
    monkeypatch.setattr(examples, "catalog",
                        lambda: [{"name": "toy", "url": url, "sha256": "0" * 64}])

    def not_found(req, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", not_found)
    with pytest.raises(ValueError, match="HTTP 404") as e:
        examples.fetch("toy", tmp_path / "tasks")
    assert url in str(e.value) and "--source" in str(e.value)

    def offline(req, timeout=None):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(ValueError, match="Name or service not known"):
        examples.fetch("toy", tmp_path / "tasks")
    with pytest.raises(ValueError, match="could not fetch"):
        examples.fetch("toy", tmp_path / "tasks", source=str(tmp_path / "missing.tar.gz"))
    assert not (tmp_path / "tasks" / "toy").exists()


def test_init_reports_a_failed_example_and_keeps_the_workspace(tmp_path, monkeypatch, capsys):
    import argparse

    from pinnforge.cli import init_cmd

    def unreachable(name, tasks_dir, source=None, force=False):
        raise ValueError(f"{name}: could not fetch https://x/{name}.tar.gz (HTTP 404 Not Found)")

    monkeypatch.setattr(examples, "fetch", unreachable)
    monkeypatch.setattr(init_cmd.gpus, "inventory", lambda timeout=20.0: [])
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    ns = argparse.Namespace(dir=str(tmp_path / "ws"), stack="cpu", examples="ldc,nope",
                            no_venv=True, no_demo=True, force=False, gpus=None)
    assert init_cmd.cmd_init(ns) == 1
    out = capsys.readouterr().out
    assert "could not install ldc: ldc: could not fetch" in out
    assert "not installed: ldc, nope" in out and "workspace ready" in out
    assert paths.is_workspace(tmp_path / "ws")

    (tmp_path / "ws" / "tasks" / "ks").mkdir()
    (tmp_path / "ws" / "tasks" / "ks" / "problem.md").write_text("x", encoding="utf-8")
    ns.examples, ns.force = "ks", True
    assert init_cmd.cmd_init(ns) == 0, "a refresh keeps an installed example"
    assert "example ks: already installed" in capsys.readouterr().out
