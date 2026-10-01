"""Where a task's scoring truth lives.

Two places are legitimate. By default it sits in the task library, under
`tasks/<task>/private/`, which the installer never copies into a run. A
hardened deployment moves it further, into a directory owned by the score
service's own user (`private_dir` in `pinnforge.yaml` or the per-machine
config file, `FORGE_PRIVATE_DIR`, or `pinnforge scored --private`), so that
no block process can open it at all.

Resolution used to be spread across the score daemon, the contract and the
CLI, and they disagreed: `run start` never passed the configured directory
on, so the in-process daemon quietly scored against the library copy, and a
task missing from the private directory fell back to it without a word. One
resolver, and a configured directory that lacks the task is an error rather
than a fallback — a fallback is how an isolation boundary stops being one
without anybody noticing.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "FORGE_PRIVATE_DIR"


def configured(root: Path | None = None) -> Path | None:
    """The deployment's external private directory, if one is configured.

    The environment wins over the settings files, so a service started with
    `FORGE_PRIVATE_DIR` set means what it says whatever the files hold. Then
    `private_dir` as the layered settings give it: the workspace's
    `pinnforge.yaml`, else the per-machine `~/.config/pinnforge/config.yaml`
    (with no workspace, the per-machine file alone).
    """
    if env := os.environ.get(ENV):
        return Path(env).expanduser()
    from pinnforge import settings

    value = settings.get(root, "private_dir")
    return Path(str(value)).expanduser() if value else None


def reference_dir(root: Path, task: str, private_dir: Path | None = None) -> Path:
    """The directory holding `task`'s answers.

    With an external directory configured, `<private_dir>/<task>/` must exist:
    scoring against the library copy instead would be scoring against answers
    the deployment meant to keep somewhere else.
    """
    if private_dir is not None:
        where = Path(private_dir) / task
        if not where.is_dir():
            raise FileNotFoundError(
                f"private directory {private_dir} has no {task}/ — the scoring truth "
                f"for {task!r} must live there when a private directory is configured"
            )
        return where
    return Path(root) / "tasks" / task / "private"


def locate(task_dir: Path) -> Path:
    """`reference_dir` for a task given by its library directory."""
    task_dir = Path(task_dir)
    root = task_dir.parent.parent
    private_dir = configured(root)
    if private_dir is not None:
        return Path(private_dir) / task_dir.name
    return task_dir / "private"
