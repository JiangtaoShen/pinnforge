"""Charter rendering and prompt parity.

The block charter is the one artefact that must not drift: a block reading a
different charter is a different experiment. These tests pin the parts that
are supposed to be constant and check that the parts that are supposed to
track the task actually do.
"""

from __future__ import annotations

import re
from pathlib import Path

from pinnforge import charter, paths
from pinnforge.config import RunConfig, SandboxConfig
from pinnforge.runtime import registry as runtime_registry

# A frozen copy of the previous framework's `block.md`, vendored so the guard
# does not depend on a sibling checkout. It used to point at
# /home/jiangtao/PINNForge/block.md — a live file whose budget is hand-edited
# on every task swap there, so this test passed or failed according to which
# task that other project happened to have installed.
REFERENCE_CHARTER = Path(__file__).parent / "data" / "block.md.reference"


def _reference_budget(text: str) -> float:
    """The budget the reference charter states — rendering must match it."""
    m = re.search(r"budget of \*\*([0-9.]+) s", text)
    assert m, "reference charter does not state a budget"
    return float(m.group(1))


def _cfg(**kw) -> RunConfig:
    cfg = RunConfig(task="ldc")
    cfg.sandbox = SandboxConfig(**kw) if kw else SandboxConfig()
    return cfg


def test_placeholders_all_substituted():
    text = charter.render_charter(_cfg())
    assert "{{" not in text and "}}" not in text


def test_budget_appears_from_config():
    text = charter.render_charter(_cfg(wall_budget_s=3600))
    assert "3600 s of\nGPU-run wall time" in text
    assert "**Budget:** 3600 s" in text
    assert "7200" not in text


def test_charter_states_the_truth_is_elsewhere():
    """No filename to go stale: the reference is simply not in the run."""
    text = charter.render_charter(_cfg())
    assert "The scoring truth is **not here**" in text
    assert "never runs your code" in text
    assert "ref_data" not in text and "Never read" not in text


def test_single_gpu_wording():
    text = charter.render_charter(_cfg(gpus=[1]))
    assert "`--gpu 1`" in text
    assert "one (`--gpu 1`)" in text
    assert "& wait" not in text


def test_two_gpu_wording_matches_the_original():
    text = charter.render_charter(_cfg(gpus=[0, 1]))
    assert "two (`--gpu 0` / `--gpu 1`)" in text
    assert "both GPUs only within one blocking command" in text
    assert "(`… --gpu 0 & … --gpu 1 & wait`)" in text


def test_only_intended_lines_differ_from_the_original():
    """Everything except the two deliberately generalised lines is verbatim.

    Rendered with the budget and GPU pool the reference itself states, so the
    substituted constants cancel out and only real drift shows up.
    """
    text = REFERENCE_CHARTER.read_text(encoding="utf-8")
    original = text.splitlines()
    rendered = charter.render_charter(
        _cfg(gpus=[0, 1], wall_budget_s=_reference_budget(text))
    ).splitlines()
    import difflib

    changed = [
        ln for ln in difflib.unified_diff(original, rendered, lineterm="", n=0)
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    ]
    # Only two things may differ from the original charter:
    # 1. the scoring-truth sentence — the reference is no longer in the run;
    # 2. `run_in_background`, a Claude-Code tool name that means nothing to
    #    another harness.
    assert all(
        any(k in ln for k in ("ref_data", "scoring truth", "score service",
                              "answers", "runs your code", "Never read",
                              "run_in_background", "background jobs"))
        for ln in changed
    ), changed


def test_block_prompt_is_runtime_independent():
    """Every harness gets the same instruction — otherwise runs cannot be compared."""
    run = Path("/tmp/pinnforge-run")
    prompt = charter.block_prompt("b03", run)
    assert "You are PINNForge block b03" in prompt
    assert f"{run}/block.md" in prompt
    assert f"{run}/blocks/b03/" in prompt
    assert "foreground" in prompt
    # The prompt takes no runtime argument, so the way this could regress is a
    # harness name leaking into the text — a tool name, a CLI flag, a vendor.
    lowered = prompt.lower()
    for name in runtime_registry.known_runtimes():
        rt = runtime_registry.get_runtime(name)
        assert name not in lowered, name
        assert rt.default_command not in lowered, rt.default_command


def test_crashed_prompt_adds_recovery_instructions():
    prompt = charter.block_prompt("b02", Path("/tmp/r"), crashed=True)
    assert "already exists from a crashed run" in prompt
    assert "spend only the remaining budget" in prompt


def test_resume_and_repair_prompts():
    r = charter.resume_prompt("b01", 1200, 7000, has_summary=False)
    assert "1200/7000 wall-seconds spent" in r
    assert "summary missing" in r
    fix = charter.summary_repair_prompt("b01")
    assert "Do not run any evaluations." in fix


def test_template_ships_inside_the_package():
    """An installed wheel must be self-contained: no charter, no runs."""
    packaged = paths.prompts_dir() / "block.md.template"
    assert packaged.is_file()
    assert paths.charter_template() == packaged


def test_project_root_can_override_the_template(tmp_path, monkeypatch):
    """Tuning the charter must not mean editing an installed package."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "pinnforge").mkdir()
    override = tmp_path / "charter" / "block.md.template"
    override.parent.mkdir()
    override.write_text("# custom charter\n", encoding="utf-8")
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    assert paths.charter_template() == override


def test_kb1_ships_inside_the_package():
    """The corpus travels with the framework, like the prompts.

    A block opens `kb1/INDEX.md` before it chooses anything; an install that
    arrives without the corpus arrives unable to start.
    """
    packaged = Path(paths.__file__).parent / "kb1"
    assert (packaged / "INDEX.md").is_file()
    assert len(list(packaged.glob("*.md"))) > 100


def test_project_root_kb1_overrides_the_packaged_corpus(tmp_path, monkeypatch):
    """A deployment may bring its own corpus without editing the package."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "pinnforge").mkdir()
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    assert paths.kb1_dir() == Path(paths.__file__).parent / "kb1"
    (tmp_path / "kb1").mkdir()
    assert paths.kb1_dir() == tmp_path / "kb1"
