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

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
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
