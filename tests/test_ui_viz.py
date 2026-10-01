"""The field archiver writes inside a run a block can write to — possibly as a
more privileged user than the block. It must never write through a link the
block planted, never hang on a FIFO, never append to a hard link, and when
anything under `.viz/` is not what it made, stop archiving without raising
into the score daemon.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import types

import numpy as np
import pytest

from pinnforge.ui import viz

COORDS = np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), -1).reshape(-1, 2)


def _within(fn, fifos=(), seconds=5.0):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        for f in fifos:
            with contextlib.suppress(OSError):
                os.close(os.open(f, os.O_RDONLY | os.O_NONBLOCK))
                os.close(os.open(f, os.O_WRONLY | os.O_NONBLOCK))
        pytest.fail(f"{fn} hung")
    if "error" in box:
        raise box["error"]
    return box.get("value")


@pytest.fixture()
def run(tmp_path):
    d = tmp_path / "runs" / "toy_1"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def payload(tmp_path):
    p = tmp_path / "queue" / "p.npz"
    p.parent.mkdir()
    np.savez(p, u=np.arange(len(COORDS), dtype=np.float64))
    return p


@pytest.fixture()
def elsewhere(tmp_path):
    d = tmp_path / "elsewhere"
    d.mkdir()
    (d / "precious.txt").write_text("untouched", encoding="utf-8")
    return d


def _req(block="b01", candidate="blocks/b01/b01_a.py", **kw):
    return {"block": block, "candidate": candidate, "smoke": False, "seed": "0", **kw}


def _untouched(d):
    assert sorted(p.name for p in d.iterdir()) == ["precious.txt"]
    assert (d / "precious.txt").read_text() == "untouched"


def test_archiving_works_normally(run, payload):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    arch(_req(), {"rRMSE": 0.5}, payload)
    arch(_req(), {"rRMSE": 0.4}, payload)
    index = [json.loads(x) for x in (run / ".viz" / "index.jsonl").read_text().splitlines()]
    assert [r["frame"] for r in index] == [0, 1]
    assert index[0]["candidate"] == "blocks/b01/b01_a.py" and index[0]["seed"] == "0"
    assert sorted(p.name for p in (run / ".viz" / "frames").iterdir()) == \
        ["00000.npz", "00001.npz"]
    # a second archiver (a restarted daemon) carries on from the index
    again = viz.Archiver(run, COORDS, 2, ["u"])
    assert again.count == 2 and again.best_run == 0.4 and not again.disabled


def test_a_symlinked_viz_dir_is_refused(run, payload, elsewhere):
    (run / ".viz").symlink_to(elsewhere)
    arch = viz.Archiver(run, COORDS, 2, ["u"])  # must not raise
    assert arch.disabled
    arch(_req(), {"rRMSE": 0.1}, payload)
    _untouched(elsewhere)


def test_a_viz_path_that_is_a_file_is_refused(run, payload):
    (run / ".viz").write_text("not a dir", encoding="utf-8")
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    assert arch.disabled
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert (run / ".viz").read_text() == "not a dir"


def test_a_viz_dir_owned_by_someone_else_is_refused(run, payload, monkeypatch):
    (run / ".viz").mkdir()
    euid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: euid + 12345)
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    assert arch.disabled
    assert list((run / ".viz").iterdir()) == []


def test_symlinked_frames_dir_is_refused(run, payload, elsewhere):
    (run / ".viz").mkdir()
    (run / ".viz" / "frames").symlink_to(elsewhere)
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert arch.disabled
    _untouched(elsewhere)
    assert not (run / ".viz" / "index.jsonl").exists()


def test_a_symlinked_index_is_refused(run, payload, elsewhere):
    (run / ".viz").mkdir()
    (run / ".viz" / "index.jsonl").symlink_to(elsewhere / "precious.txt")
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    assert arch.disabled
    arch(_req(), {"rRMSE": 0.1}, payload)
    _untouched(elsewhere)


def test_an_index_linked_after_start_is_not_appended_through(run, payload, elsewhere):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    assert not arch.disabled
    (run / ".viz" / "index.jsonl").symlink_to(elsewhere / "precious.txt")
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert arch.disabled
    _untouched(elsewhere)


def test_a_hard_linked_index_is_refused(run, payload, elsewhere):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    os.link(elsewhere / "precious.txt", run / ".viz" / "index.jsonl")
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert arch.disabled
    assert (elsewhere / "precious.txt").read_text() == "untouched"
    os.unlink(run / ".viz" / "index.jsonl")
    (run / ".viz").rename(run / ".viz.old")
    (run / ".viz.old" / "frames").mkdir(exist_ok=True)
    (run / ".viz").mkdir()
    os.link(elsewhere / "precious.txt", run / ".viz" / "index.jsonl")
    assert viz.Archiver(run, COORDS, 2, ["u"]).disabled  # read refused too


def test_a_fifo_index_neither_hangs_the_start_nor_the_append(run, payload):
    (run / ".viz").mkdir()
    fifo = run / ".viz" / "index.jsonl"
    os.mkfifo(fifo)
    arch = _within(lambda: viz.Archiver(run, COORDS, 2, ["u"]), fifos=[fifo])
    assert arch.disabled
    # planted after start: the append (O_NONBLOCK, no reader) refuses at once
    os.unlink(fifo)
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    os.mkfifo(fifo)
    _within(lambda: arch(_req(), {"rRMSE": 0.1}, payload), fifos=[fifo])
    assert arch.disabled
    # and with a reader on the other end, it is still not a regular file
    os.unlink(fifo)
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        _within(lambda: arch(_req(), {"rRMSE": 0.1}, payload), fifos=[fifo])
        assert arch.disabled
        assert os.read(reader, 100) == b""
    finally:
        os.close(reader)


def test_files_planted_at_output_names_are_replaced_not_written_through(run, payload,
                                                                        elsewhere):
    (run / ".viz" / "frames").mkdir(parents=True)
    (run / ".viz" / "grid.npz").symlink_to(elsewhere / "precious.txt")
    (run / ".viz" / "frames" / "00000.npz").symlink_to(elsewhere / "precious.txt")
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert not arch.disabled
    _untouched(elsewhere)
    for f in (run / ".viz" / "grid.npz", run / ".viz" / "frames" / "00000.npz"):
        assert f.is_file() and not f.is_symlink()


@pytest.mark.parametrize("req", [
    _req(block="../b01"), _req(block="b1"), _req(block=None), _req(block=["b01"]),
    _req(candidate="../../etc/passwd"), _req(candidate="/abs/path.py"),
    _req(candidate="blocks/b01/../x.py"), _req(candidate="blocks//x.py"),
    _req(candidate="blocks/b01/x y.py"), _req(candidate="a" * 129),
    _req(candidate={"x": 1}), _req(candidate="blocks/b01/\n.py"),
])
def test_names_from_the_request_are_validated_before_they_are_written(run, payload, req):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    arch(req, {"rRMSE": 0.1}, payload)
    assert not (run / ".viz" / "index.jsonl").exists()
    assert not arch.disabled


def test_an_empty_candidate_is_kept_as_empty(run, payload):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    arch(_req(candidate=""), {"rRMSE": 0.1}, payload)
    entry = json.loads((run / ".viz" / "index.jsonl").read_text())
    assert entry["candidate"] == "" and entry["block"] == "b01"


def test_a_payload_that_is_a_fifo_link_or_hard_link_is_skipped(run, payload, tmp_path):
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    fifo = tmp_path / "queue" / "fifo.npz"
    os.mkfifo(fifo)
    _within(lambda: arch(_req(), {"rRMSE": 0.1}, fifo), fifos=[fifo])
    link = tmp_path / "queue" / "link.npz"
    link.symlink_to(payload)
    arch(_req(), {"rRMSE": 0.1}, link)
    os.link(payload, tmp_path / "queue" / "twin.npz")
    arch(_req(), {"rRMSE": 0.1}, payload)
    assert not (run / ".viz" / "index.jsonl").exists()
    assert not arch.disabled


def test_malformed_index_lines_are_skipped_on_replay(run):
    (run / ".viz").mkdir()
    (run / ".viz" / "index.jsonl").write_text("\n".join([
        json.dumps({"frame": "7", "rRMSE": "0.1", "block": "b01"}),
        json.dumps([1, 2]), "garbage",
        json.dumps({"frame": 1e30, "rRMSE": 0.2, "block": "b01"}),
        json.dumps({"frame": 3, "rRMSE": 0.3, "block": "../x"}),
        json.dumps({"frame": 4, "rRMSE": True, "block": "b02"}),
    ]) + "\n", encoding="utf-8")
    arch = viz.Archiver(run, COORDS, 2, ["u"])
    assert not arch.disabled
    assert arch.count == 5 and arch.best_run == 0.2 and arch.best_block == {"b01": 0.2}


def test_archiver_for_writes_grid_json_without_following_links(run, tmp_path, elsewhere):
    csv = tmp_path / "grid.csv"
    csv.write_text("x,y\n" + "\n".join(f"{a},{b}" for a, b in COORDS), encoding="utf-8")
    metric = types.SimpleNamespace(module=types.SimpleNamespace(
        GRID_CSV=str(csv), FIELDS=["u"], INPUT_COLUMNS=2))
    (run / ".viz").mkdir()
    (run / ".viz" / "grid.json").symlink_to(elsewhere / "precious.txt")
    arch = viz.archiver_for(run, metric)
    assert arch is not None
    _untouched(elsewhere)
    grid = json.loads((run / ".viz" / "grid.json").read_text())
    assert grid["coords"] == ["x", "y"] and grid["points"] == len(COORDS)
    # and with .viz itself planted, there is simply no archiver
    other = tmp_path / "runs" / "toy_2"
    other.mkdir()
    (other / ".viz").symlink_to(elsewhere)
    assert viz.archiver_for(other, metric) is None
    _untouched(elsewhere)
