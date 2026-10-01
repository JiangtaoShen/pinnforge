"""Re-running a task must cost nothing it has already paid for.

A run freezes the task it was given and keeps the b00 it measured. So a
library that lost its cached anchor, or its copy of the package, can take
both back from the run library instead of spending a GPU or an authoring
session — and, more to the point, gets back the *same* definition, which is
what keeps the new runs comparable with the old.
"""

from __future__ import annotations

import json

import pytest

from pinnforge import paths
from pinnforge.run import anchor
from pinnforge.task import registry

CONTRACT = ("problem.md", "baseline.py", "eval.py")


def _seed_task(root, name="toy", body="x = 1\n"):
    task = root / "tasks" / name
    (task / "private").mkdir(parents=True)
    for f in CONTRACT:
        (task / f).write_text(body, encoding="utf-8")
    (task / "grid.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (task / "private" / "answers.npy").write_bytes(b"secret")
    return task


def _seed_run(root, name="toy", n=1, score=0.5, fingerprint_from=None):
    run = root / "runs" / f"{name}_{n}"
    (run / "task").mkdir(parents=True)
    src = fingerprint_from or (root / "tasks" / name)
    for f in sorted(src.iterdir()):
        if f.is_file():
            (run / "task" / f.name).write_bytes(f.read_bytes())
    b00 = run / "blocks" / "b00"
    b00.mkdir(parents=True)
    (b00 / "b00_v01.py").write_text("x = 1\n", encoding="utf-8")
    (b00 / "b00_v01.pkl").write_bytes(b"params")
    (b00 / ".budget").write_text("42.0\n", encoding="utf-8")
    (b00 / "evals.jsonl").write_text(
        json.dumps({"block": "b00", "smoke": False, "diag": False,
                    "wall_s": 42.0, "rRMSE": score}) + "\n",
        encoding="utf-8",
    )
    (run / "blocks" / "kb2").mkdir(parents=True)
    (run / "blocks" / "kb2" / "b00.md").write_text("# b00\n", encoding="utf-8")
    return run


def _project(tmp_path, monkeypatch):
    (tmp_path / "pinnforge").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("PINNFORGE_ROOT", str(tmp_path))
    return tmp_path


def test_anchor_is_harvested_from_a_matching_run(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    _seed_run(root, score=0.37)

    result = anchor.harvest("toy", root)
    assert result is not None
    assert result.rRMSE == 0.37
    assert "toy_1" in result.detail

    cache = anchor.cache_dir("toy", root)
    for f in (*anchor.ANCHOR_FILES, "b00.md", "meta.json"):
        assert (cache / f).is_file(), f
    assert json.loads((cache / "meta.json").read_text())["rRMSE"] == 0.37


def test_a_run_of_different_contract_files_is_not_harvested(tmp_path, monkeypatch):
    """The same guard as `restore`: a cheaper source, not a different one."""
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    run = _seed_run(root)
    (run / "task" / "baseline.py").write_text("x = 2  # changed\n", encoding="utf-8")

    assert anchor.harvest("toy", root) is None
    assert not anchor.cache_dir("toy", root).exists()


def test_the_newest_matching_run_wins(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    old = _seed_run(root, n=1, score=0.9)
    new = _seed_run(root, n=2, score=0.4)
    import os

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    assert anchor.harvest("toy", root).rRMSE == 0.4


def test_no_run_means_no_harvest(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    assert anchor.harvest("toy", root) is None


def test_definition_is_recovered_from_a_run(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    _seed_run(root)
    original = {f: (task / f).read_bytes() for f in (*CONTRACT, "grid.csv")}
    for f in (*CONTRACT, "grid.csv"):
        (task / f).unlink()

    found = registry.recover_from_run("toy", root)
    assert found is not None
    dest, run = found
    assert run.name == "toy_1"
    for f, want in original.items():
        assert (dest / f).read_bytes() == want, f


def test_recovery_cannot_bring_back_the_answers(tmp_path, monkeypatch):
    """`private/` is never copied into a run — that is the split, not a gap."""
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    _seed_run(root)
    import shutil

    shutil.rmtree(task)

    dest, _ = registry.recover_from_run("toy", root)
    assert (dest / "problem.md").is_file()
    assert not (dest / "private").exists()


def test_runs_of_ignores_other_tasks(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root, "toy")
    _seed_task(root, "other")
    _seed_run(root, "toy")
    _seed_run(root, "other")
    assert [r.name for r in registry.runs_of("toy", root)] == ["toy_1"]


def test_paths_still_resolve_under_a_test_root(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    assert paths.tasks_dir() == root / "tasks"
    assert paths.runs_dir() == root / "runs"


def test_a_new_anchor_replaces_the_cache_whole(tmp_path, monkeypatch):
    """Another run's save never leaves one run's params beside another's score."""
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    first = _seed_run(root, n=1, score=0.9)
    second = _seed_run(root, n=2, score=0.4)
    (second / "blocks" / "b00" / "b00_v01.pkl").write_bytes(b"second params")
    (second / "blocks" / "kb2" / "b00.md").unlink()

    anchor.save(first, "toy", root)
    anchor.save(second, "toy", root)
    cache = anchor.cache_dir("toy", root)
    assert (cache / "b00_v01.pkl").read_bytes() == b"second params"
    assert anchor.read_meta("toy", root)["rRMSE"] == 0.4
    assert not (cache / "b00.md").exists(), "nothing of the first anchor survives"
    assert sorted(p.name for p in task.iterdir() if p.name.startswith(".")) == [".anchor"], (
        "no staging directory is left behind, and no lock file sits among the task's files"
    )

    anchor.clear("toy", root)
    assert anchor.read_meta("toy", root) is None


# ─────────────── what an anchor carries, and what it is filed under ───────────────

LEGACY = {"ts": "2026-01-01T00:00:00Z", "block": "b00", "candidate": "blocks/b00/b00_v01.py",
          "seed": 0, "gpu": "0", "smoke": False, "rRMSE": 0.5, "MSE": 0.0123,
          "n_scored": 65536, "step_count": 10, "train_s": 42.0, "wall_s": 42.0,
          "params": "blocks/b00/b00_v01.pkl", "error": None, "wall_budget": "42/3600"}


def _keys(path):
    return [list(json.loads(ln)) for ln in path.read_text(encoding="utf-8").splitlines()]


def test_legacy_record_fields_never_travel_with_an_anchor(tmp_path, monkeypatch):
    """MSE beside rRMSE, with the grid's row count, is the reference's norm."""
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    run = _seed_run(root)
    (run / "blocks" / "b00" / "evals.jsonl").write_text(
        json.dumps(LEGACY) + "\nnot a record\n[1, 2]\n", encoding="utf-8"
    )
    assert anchor.harvest("toy", root) is not None  # -> save()
    cache = anchor.cache_dir("toy", root) / "evals.jsonl"
    want = [k for k in LEGACY if k not in ("MSE", "n_scored")]
    assert _keys(cache) == [want], "only today's record fields, in their order"

    # a cache written before the whitelist existed is cleaned on the way out too
    cache.write_text(json.dumps(LEGACY) + "\n", encoding="utf-8")
    new = root / "runs" / "toy_2"
    (new / "task").mkdir(parents=True)
    for f in (root / "tasks" / "toy").iterdir():
        if f.is_file():
            (new / "task" / f.name).write_bytes(f.read_bytes())
    assert anchor.restore(new, "toy", root).restored
    assert _keys(new / "blocks" / "b00" / "evals.jsonl") == [want]


def test_an_unrecorded_hardware_class_is_not_a_match(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    task = _seed_task(root)
    meta = {"fingerprint": registry.fingerprint(task), "reference": ""}
    assert "unrecorded hardware" in anchor.staleness(meta, task, "toy", root,
                                                     {"platform": "cpu"})
    assert anchor.staleness(meta, task, "toy", root, None) == "", "nothing asked, nothing refused"
    meta["hw"] = {"platform": "cpu"}
    assert anchor.staleness(meta, task, "toy", root, {"platform": "cpu"}) == ""


def test_a_run_that_recorded_no_hardware_is_not_harvested_as_this_machines(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    _seed_run(root)  # no run.yaml, so no class
    gpu_class = {"platform": "gpu", "gpu": "NVIDIA RTX A6000"}
    assert anchor.harvest("toy", root, gpu_class) is None
    assert anchor.harvest("toy", root) is not None
    assert "hw" not in anchor.read_meta("toy", root), "and it is not stamped with one"


def test_rebuild_measures_instead_of_harvesting(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    _seed_run(root, score=0.37)
    scratch = root / "scratch" / "toy_0"
    (scratch / "task").mkdir(parents=True)
    measured = []

    def measure(run, root=None, gpu=0):
        measured.append(run)
        return anchor.AnchorResult(None, False, "stub: not measured")

    monkeypatch.setattr(anchor, "measure", measure)
    result = anchor.ensure(scratch, "toy", root, gpu=None, hw=None, rebuild=True)
    assert measured == [scratch] and result.detail == "stub: not measured"
    assert not anchor.cache_dir("toy", root).exists(), "no old run was harvested"


def test_ensure_reports_the_anchor_it_installed(tmp_path, monkeypatch):
    """Another run's save between harvest and restore must not be misreported."""
    root = _project(tmp_path, monkeypatch)
    _seed_task(root)
    cpu = {"platform": "cpu"}
    anchor.save(_seed_run(root, score=0.2), "toy", root, hw=cpu)  # what the cache holds: 0.2
    real, calls = anchor.restore, []

    def restore(*a, **kw):
        calls.append(1)
        return None if len(calls) == 1 else real(*a, **kw)  # first look: nothing cached yet

    monkeypatch.setattr(anchor, "restore", restore)
    monkeypatch.setattr(anchor, "harvest",
                        lambda task, root=None, hw=None: anchor.AnchorResult(0.37, True, "h"))
    monkeypatch.setattr(anchor, "measure", lambda *a, **kw: pytest.fail("measured"))
    new = root / "runs" / "toy_9"
    (new / "task").mkdir(parents=True)
    for f in (root / "tasks" / "toy").iterdir():
        if f.is_file():
            (new / "task" / f.name).write_bytes(f.read_bytes())
    result = anchor.ensure(new, "toy", root, gpu=None, hw=cpu)
    assert result.restored and result.rRMSE == 0.2, (
        "the score of what was installed, not of what was harvested"
    )


def test_the_anchor_is_measured_on_the_card_nvidia_smi_names(tmp_path, monkeypatch):
    run = tmp_path / "toy_1"
    (run / "task").mkdir(parents=True)
    (run / "task" / "baseline.py").write_text("x = 1\n", encoding="utf-8")
    envs = []

    def group(cmd, cwd, env=None):
        import subprocess

        envs.append(env)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(anchor, "_run_group", group)
    monkeypatch.setattr(anchor.paths, "venv_python", lambda root=None: "python")
    anchor.measure(run, tmp_path, gpu=1)
    assert envs[-1]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    anchor.measure(run, tmp_path, gpu=None)
    assert envs[-1]["JAX_PLATFORMS"] == "cpu"


def _rekey_setup(root, monkeypatch, recorded_hw=None):
    _seed_task(root)
    run = _seed_run(root, score=0.5)
    (run / "blocks" / "b00" / "evals.jsonl").write_text(json.dumps(LEGACY) + "\n",
                                                         encoding="utf-8")
    anchor.save(run, "toy", root, hw=recorded_hw)
    # the cache as an older tool left it: legacy fields, and the package changed since
    cache = anchor.cache_dir("toy", root)
    (cache / "evals.jsonl").write_text(json.dumps(LEGACY) + "\n", encoding="utf-8")
    scratch = root / "scratch" / "toy_0"
    (scratch / "task").mkdir(parents=True)
    for f in (root / "tasks" / "toy").iterdir():
        if f.is_file():
            (scratch / "task" / f.name).write_bytes(f.read_bytes())

    def rescore(cmd, **kw):
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, json.dumps({"rRMSE": 0.5}) + "\n", "")

    monkeypatch.setattr(anchor.subprocess, "run", rescore)
    monkeypatch.setattr(anchor.paths, "venv_python", lambda root=None: "python")
    return scratch, cache


def test_rekey_stamps_a_classless_cache_with_this_machines_card(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path, monkeypatch)
    scratch, cache = _rekey_setup(root, monkeypatch)
    card = {"platform": "gpu", "gpu": "NVIDIA RTX A6000"}
    result = anchor.rekey(scratch, "toy", root, hw=card)
    assert result.rRMSE == 0.5
    assert "assumed to have been measured on this machine" in result.detail
    assert anchor.read_meta("toy", root)["hw"] == card
    want = [k for k in LEGACY if k not in ("MSE", "n_scored")]
    assert _keys(cache / "evals.jsonl") == [want], "rekeying brings the records up to date"


def test_rekey_without_a_card_leaves_the_class_unknown(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    scratch, _ = _rekey_setup(root, monkeypatch)
    assert anchor.rekey(scratch, "toy", root, hw=None).rRMSE == 0.5
    assert "hw" not in anchor.read_meta("toy", root)
    recorded = {"platform": "gpu", "gpu": "Old card"}
    (tmp_path / "two").mkdir()
    root2 = _project(tmp_path / "two", monkeypatch)
    scratch2, _ = _rekey_setup(root2, monkeypatch, recorded_hw=recorded)
    anchor.rekey(scratch2, "toy", root2, hw={"platform": "gpu", "gpu": "New card"})
    assert anchor.read_meta("toy", root2)["hw"] == recorded, "a recorded class is never replaced"
