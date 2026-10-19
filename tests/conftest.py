"""Test-wide conventions.

Several suites build runs from the library tasks under `tasks/`, whose
references are not in the repository (`tasks/*/private/` is gitignored — the
answers are published separately). On a fresh clone those suites are skipped
with a reason, rather than failing on data the clone was never meant to have;
`pinnforge examples add ldc ks cyl naca --force` supplies it. `--force`
because the clone already has `tasks/<t>/`: it replaces each task's public
files with the identical released ones, adds `private/`, and drops a cached
`.anchor/`, which the next run or `task anchor --measure` measures again.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# As the session found it: `pinnforge`'s entry point turns UTF-8 mode on for its
# children on Windows, and a test that runs it must not do so for later tests.
_PYTHONUTF8 = os.environ.get("PYTHONUTF8")
NEEDS_LIBRARY = {"test_run", "test_orchestrator", "test_integrity", "test_contract",
                 "test_reuse"}


@pytest.fixture(autouse=True)
def _no_operator_config(tmp_path_factory, monkeypatch):
    """Never read the developer's own `~/.config/pinnforge/`.

    A workspace's null settings fall through to the per-machine file, so a
    `private_dir`, a GPU choice or an agent binding there would otherwise
    change what a test sees. Tests that need one set `XDG_CONFIG_HOME` again.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    if _PYTHONUTF8 is None:
        monkeypatch.delenv("PYTHONUTF8", raising=False)


@pytest.fixture
def symlink():
    """`symlink(link, target)`: plant a symbolic link, or skip where the system refuses one.

    Windows makes links only with Developer Mode or an elevated shell; what a
    test plants to prove a link is refused cannot be planted there otherwise.
    """
    def make(link: Path, target: Path, target_is_directory: bool = False) -> Path:
        try:
            Path(link).symlink_to(target, target_is_directory=target_is_directory)
        except OSError as e:
            pytest.skip(f"this system refuses symbolic links ({type(e).__name__})")
        return Path(link)

    return make


def plant_link(link: Path, target: Path) -> Path:
    """A link at `target`, as a block would plant one where the framework reads.

    A symbolic link where the system allows one. On Windows without the
    privilege, what a block there can make instead — a hard link to a file,
    a junction to a directory — neither of which needs any.
    """
    link, target = Path(link), Path(target)
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        if os.name != "nt":
            raise
        if target.is_dir():
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            os.link(target, link)
    return link


@pytest.fixture
def plant():
    """`plant(link, target)`: `plant_link`, for a test to call."""
    return plant_link


def pytest_collection_modifyitems(config, items):
    missing = [t for t in ("ldc", "ks", "naca") if not (REPO / "tasks" / t / "private").is_dir()]
    if not missing:
        return
    skip = pytest.mark.skip(
        reason=f"library task references missing ({', '.join(missing)}); "
        "`pinnforge examples add ldc ks cyl naca --force` supplies them (it replaces "
        "each task's public files with identical ones and drops a cached .anchor/)"
    )
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in NEEDS_LIBRARY:
            item.add_marker(skip)
