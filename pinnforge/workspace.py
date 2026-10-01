"""Workspaces — where tasks, runs and the training environment live.

The tool is installed once (`uv tool install`), with nothing heavier than
PyYAML and NumPy; the JAX stack blocks train on belongs to a *workspace*,
which `pinnforge init` creates anywhere:

    <workspace>/
      pinnforge.yaml     marks the workspace; its settings (see settings.py)
      pyproject.toml     the frozen stack, pinned; blocks may `uv add` beside it
      uv.lock, .venv/    the environment every eval.py runs in
      tasks/             the task library (the demo task is installed at once)
      runs/  logs/       experiments, and the agents' transcripts
      .pinnforge/        internal state: locks, ledgers, dashboard

The environment holds PINNForge itself as well as JAX, pinned to the very
same source as the tool that created it — the same git commit, the same
local checkout, the same wheel — so the score client a legacy task imports
speaks the tool's protocol. `pinnforge workspace sync` re-pins it after the
tool is upgraded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

from pinnforge import __version__, gpus, paths

STACK_FILE = Path(__file__).resolve().parent / "templates" / "stack.toml"
STACKS = ("cuda12", "cpu")

WORKSPACE_YAML = """\
# PINNForge workspace. Settings here apply to every run started from this
# directory; ~/.config/pinnforge/config.yaml holds per-machine defaults, and
# command-line flags override both. A key left null here is unset: the
# per-machine file decides it, else the built-in default. Each run freezes
# what it used into runs/<id>/run.yaml, so editing this file never changes an
# existing run.
version: 1

# Which coding agent drives the blocks: the name of a binding created by
# `pinnforge setup`, or a mapping such as {{runtime: claude_code, model: claude-opus-4-8}}.
# null: the per-machine file's `agent`, else your default binding.
agent: null

sandbox:
  # Which GPUs runs may use: `auto` = every card visible when a run starts
  # (narrowed by CUDA_VISIBLE_DEVICES); a list such as [1] or [0, 2] = exactly
  # those cards; [] = CPU only; null = this machine's default (set with
  # `pinnforge setup gpus --user`), else auto. Change it with
  # `pinnforge setup gpus`, or for one run with `run start --gpus`.
  gpus: null
  wall_budget_s: null   # GPU seconds per block; null = the machine's default, else the task's

# Keep the scoring truth outside the workspace, e.g. owned by another user
# who runs `pinnforge scored`; tasks' answers then live in <private_dir>/<task>/.
# null: the per-machine file's, else tasks/<task>/private/. FORGE_PRIVATE_DIR wins.
private_dir: null

ui:
  host: 127.0.0.1
  port: 8765
  autostart: true       # `run start` opens the dashboard if it is not running

run:
  detach: true          # `run start` returns at once; the run lives in tmux

# This workspace's environment: stack `{stack}`, created by pinnforge {version}.
"""

GITIGNORE = """\
# PINNForge workspace
.venv/
.pinnforge/
runs/
logs/
tasks/*/private/
tasks/*/.anchor/
tasks/*/.authoring/
"""


def stack() -> dict[str, Any]:
    return tomllib.loads(STACK_FILE.read_text(encoding="utf-8"))


def detect_stack() -> str:
    """`cuda12` on Linux with an NVIDIA GPU visible, `cpu` everywhere else."""
    if sys.platform.startswith("linux") and gpus.inventory():
        return "cuda12"
    return "cpu"


def _direct_url() -> dict | None:
    try:
        dist = metadata.distribution("pinnforge")
    except metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    return json.loads(text) if text else None


# Where PINNForge comes from when the install record cannot say. Never a
# package index: the name `pinnforge` on PyPI belongs to an unrelated project,
# so `pinnforge==<version>` would install somebody else's code into every
# environment blocks train in.
GIT_URL = "https://github.com/JiangtaoShen/pinnforge"


def pinnforge_source(root: Path) -> tuple[str, str | None, str]:
    """How the workspace environment pins PINNForge: requirement, uv source, note.

    Read from the running tool's own install record (PEP 610), so the
    workspace gets exactly what the operator installed, however they did it.
    Without a reusable record it is this version's tag in the GitHub
    repository; it is never resolved from an index.
    """
    info = _direct_url()
    url = str((info or {}).get("url", ""))
    if info is not None:
        if vcs := info.get("vcs_info"):
            rev = vcs.get("commit_id") or vcs.get("requested_revision") or ""
            src = f'{{ git = "{url}", rev = "{rev}" }}'
            return "pinnforge", src, f"git {url} @ {rev[:12]}"
        if url.startswith("file://"):
            local = Path(url[len("file://"):])
            if "dir_info" in info:
                editable = bool((info.get("dir_info") or {}).get("editable"))
                src = f'{{ path = "{local}", editable = {"true" if editable else "false"} }}'
                return "pinnforge", src, f"{'editable ' if editable else ''}checkout {local}"
            if local.is_file():  # a wheel: keep a copy, the original may be a cache
                keep = paths.state_dir(root) / "wheels" / local.name
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, keep)
                return "pinnforge", f'{{ path = "{keep}" }}', f"wheel {local.name}"
        elif url and "dir_info" not in info:
            # A wheel or sdist fetched from a URL: the same archive again.
            return "pinnforge", f'{{ url = "{url}" }}', f"archive {url}"
    why = "no install record" if info is None else f"source {url or '?'} not reusable"
    tag = f"v{__version__}"
    src = f'{{ git = "{GIT_URL}", tag = "{tag}" }}'
    return "pinnforge", src, (f"git {GIT_URL} @ {tag} — {why}; never the index, where "
                              "`pinnforge` is an unrelated project")


PYPROJECT_HEADER = "# The environment every block of this workspace trains in, written by"


def owns_pyproject(root: Path) -> bool:
    """Whether `root/pyproject.toml` is absent or one `pinnforge init` wrote.

    Anything else — a project's own pyproject, this package's in a source
    checkout — is not the workspace's to replace, `--force` or not.
    """
    py = Path(root) / "pyproject.toml"
    if not py.is_file():
        return True
    try:
        text = py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return text.startswith(PYPROJECT_HEADER)


def _refuse_foreign_pyproject(root: Path) -> None:
    if not owns_pyproject(root):
        raise FileExistsError(
            f"{Path(root) / 'pyproject.toml'} was not written by `pinnforge init`, so it is "
            "not replaced (`--force` does not change that). Create the workspace in a "
            "directory of its own, or move that file away first."
        )


def render_pyproject(root: Path, stack_name: str) -> str:
    if stack_name not in STACKS:
        raise ValueError(f"unknown stack {stack_name!r} (choose from {', '.join(STACKS)})")
    s = stack()
    requirement, source, note = pinnforge_source(root)
    deps = [requirement, *s["common"], *s[stack_name]]
    pins = [d for d in [*s["common"], *s[stack_name]] if "==" in d and "@" not in d]
    # constraints name packages, not extras
    pins = [d.split("[")[0] + "==" + d.split("==")[1] if "[" in d else d for d in pins]

    def block(items: list[str]) -> str:
        return "\n".join(f'    "{d}",' for d in items)

    out = [
        PYPROJECT_HEADER,
        f"# `pinnforge init` ({note}). The stack is frozen: blocks may `uv add`",
        "# extensions, and the constraints below keep the pins where they are.",
        "[project]",
        'name = "pinnforge-workspace"',
        'version = "0"',
        f'requires-python = "{s["python"]}"',
        "dependencies = [",
        block(deps),
        "]",
        "",
        "[tool.uv]",
        "package = false",
        "constraint-dependencies = [",
        block(pins),
        "]",
    ]
    if source:
        out += ["", "[tool.uv.sources]", f"pinnforge = {source}"]
    return "\n".join(out) + "\n"


def uv_binary() -> str:
    found = shutil.which("uv")
    if found:
        return found
    raise FileNotFoundError(
        "uv is not on PATH — install it (https://docs.astral.sh/uv/) or re-run install.sh"
    )


def sync(root: Path, frozen: bool = False) -> int:
    """Create or update the workspace's `.venv` from its pyproject.

    Output streams straight to the terminal: the CUDA wheels are a couple of
    gigabytes the first time, and a silent minute looks like a hang.
    """
    cmd = [uv_binary(), "sync", "--project", str(root)]
    if frozen:
        cmd.append("--frozen")
    drop = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    return subprocess.run(cmd, cwd=str(root), env=env, check=False).returncode


def stack_of(root: Path) -> str | None:
    """Which stack a workspace's pyproject was written for."""
    py = Path(root) / "pyproject.toml"
    if not py.is_file():
        return None
    text = py.read_text(encoding="utf-8")
    if "jax[cuda12]" in text:
        return "cuda12"
    if '"jax==' in text:
        return "cpu"
    return None


def venv_version(root: Path) -> str | None:
    """The PINNForge version installed in a workspace's environment."""
    python = Path(root) / ".venv" / "bin" / "python"
    if not python.is_file():
        return None
    try:
        out = subprocess.run(
            [str(python), "-c",
             "import importlib.metadata as m; print(m.version('pinnforge'))"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def check_dest(dest: Path, force: bool = False) -> None:
    """Raise FileExistsError if `init` must not write to `dest`."""
    if paths.is_workspace(dest) and not force:
        raise FileExistsError(
            f"{dest} is already a PINNForge workspace (pass --force to refresh it)"
        )
    if dest.exists() and any(dest.iterdir()) and not force and not paths.is_workspace(dest):
        raise FileExistsError(
            f"{dest} exists and is not empty — choose a new directory, or pass --force"
        )
    _refuse_foreign_pyproject(dest)


def init(
    dest: Path,
    *,
    stack_name: str | None = None,
    venv: bool = True,
    demo: bool = True,
    force: bool = False,
    gpus_spec: str | list[int] | None = None,
) -> Path:
    """Create a workspace at `dest` and (unless `venv=False`) its environment.

    `gpus_spec` is the operator's choice of cards (`auto`, `[1]`, `[]`…),
    stored as `sandbox.gpus`; None leaves the file's value alone (null in a
    new one: the machine's default, else auto). Refreshing an existing
    workspace (`force`) keeps its `pinnforge.yaml` — the settings in it are
    the operator's, not the template's — and, unless told otherwise, its
    stack. A `pyproject.toml` that `init` did not write is never replaced.
    """
    from pinnforge.task import examples

    dest = Path(dest).expanduser().resolve()
    check_dest(dest, force)
    stack_name = stack_name or stack_of(dest) or detect_stack()
    for d in ("tasks", "runs", "logs", paths.STATE_DIR):
        (dest / d).mkdir(parents=True, exist_ok=True)
    marker = dest / paths.MARKER
    if not marker.is_file():
        marker.write_text(WORKSPACE_YAML.format(stack=stack_name, version=__version__),
                          encoding="utf-8")
    if gpus_spec is not None:
        from pinnforge import settings

        settings.write(marker, "sandbox.gpus", gpus_spec)
    (dest / "pyproject.toml").write_text(render_pyproject(dest, stack_name), encoding="utf-8")
    if not (dest / ".gitignore").is_file():
        (dest / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if demo and not (dest / "tasks" / examples.DEMO / "problem.md").is_file():
        examples.install_demo(dest / "tasks")
    if venv:
        code = sync(dest)
        if code != 0:
            raise RuntimeError(
                f"`uv sync` failed (exit {code}); fix the cause above, then run "
                f"`pinnforge workspace sync` in {dest}"
            )
    return dest


def repin(root: Path, stack_name: str | None = None) -> str:
    """Rewrite the workspace pyproject for the running tool; returns the stack."""
    _refuse_foreign_pyproject(root)
    stack_name = stack_name or stack_of(root) or detect_stack()
    (Path(root) / "pyproject.toml").write_text(render_pyproject(root, stack_name), encoding="utf-8")
    return stack_name
