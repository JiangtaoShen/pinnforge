"""What the dashboard reads out of a run a block can write anything into.

Hard links (which no path check sees through), FIFOs (which hang whoever
opens them), links through a directory, malformed records, decompression
and alias bombs, secrets in agent text, and HTML in a report's data.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import threading
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

from pinnforge.ui import api, feeds, report, server, tui

CANARY = "CANARY-5d1c0b"


def _workspace(root):
    task = root / "tasks" / "toy"
    (task / "private").mkdir(parents=True)
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    (task / "problem.md").write_text("# TOY — toy\n", encoding="utf-8")
    (task / "private" / "secret.md").write_text(CANARY, encoding="utf-8")
    (task / "private" / "leak.jsonl").write_text(
        json.dumps({"ts": "z", "rRMSE": 0.01, "error": CANARY, "candidate": CANARY}) + "\n",
        encoding="utf-8")
    (task / "private" / "leak.log").write_text(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": CANARY}]}}) + "\n", encoding="utf-8")
    run = root / "runs" / "toy_1"
    for d in ("blocks/b01", "blocks/kb2", "logs", ".viz/frames"):
        (run / d).mkdir(parents=True)
    (run / "run.yaml").write_text(
        "task: toy\nblocks: 2\nagent: {runtime: claude_code, model: m}\n"
        "sandbox: {gpus: [0], wall_budget_s: 100}\n", encoding="utf-8")
    (run / "state.json").write_text(json.dumps(
        {"status": "running", "blocks": {"b01": {"status": "running"}}}), encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(json.dumps(
        {"ts": "2026-01-01T00:00:00Z", "candidate": "blocks/b01/b01_a.py",
         "rRMSE": 0.25, "wall_s": 10.0}) + "\n", encoding="utf-8")
    (run / "blocks" / "kb2" / "b01.md").write_text("# b01\n", encoding="utf-8")
    (run / "logs" / "b01.0.log").write_text(json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}) + "\n",
        encoding="utf-8")
    return root


@pytest.fixture()
def ws(tmp_path):
    root = _workspace(tmp_path / "ws")
    api.ROOT_REAL = os.path.realpath(root)
    return root


def _serve(ws):
    srv = server.make_server(ws, "127.0.0.1", 0, None)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()
    return srv


@pytest.fixture()
def served(ws):
    srv = _serve(ws)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _within(fn, fifos=(), seconds=5.0):
    """fn() must return promptly; if it hangs on a FIFO, unblock it and fail."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # surfaced below
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        for f in fifos:  # let the stuck reader go before failing
            with contextlib.suppress(OSError):
                os.close(os.open(f, os.O_WRONLY | os.O_NONBLOCK))
        pytest.fail(f"{fn} hung")
    if "error" in box:
        raise box["error"]
    return box.get("value")


ENDPOINTS = ["/api/overview", "/api/runs/toy_1", "/api/runs/toy_1/summary/b01",
             "/api/runs/toy_1/summary/b03", "/api/runs/toy_1/activity",
             "/api/runs/toy_1/activity?block=b01", "/api/runs/toy_1/activity?block=b03",
             "/api/runs/toy_1/frame/0", "/api/tasks/toy", "/api/tasks/toy/activity"]


# ─────────────────────────────── hard links ───────────────────────────────


def test_hard_links_to_the_answers_are_not_read(ws, served):
    run, private = ws / "runs" / "toy_1", ws / "tasks" / "toy" / "private"
    (run / "blocks" / "b03").mkdir()
    os.link(private / "leak.jsonl", run / "blocks" / "b03" / "evals.jsonl")
    os.link(private / "secret.md", run / "blocks" / "kb2" / "b03.md")
    os.link(private / "leak.log", run / "logs" / "b03.0.log")
    os.link(private / "leak.log", run / "logs" / "b01.7.log")
    (ws / "tasks" / "toy" / ".authoring").mkdir()
    os.link(private / "leak.log", ws / "tasks" / "toy" / ".authoring" / "author.log")
    for path in ENDPOINTS:
        status, body = _get(served + path)
        assert CANARY not in body, path
        assert status in (200, 404), (path, status)
    detail = json.loads(_get(served + "/api/runs/toy_1")[1])
    b03 = next(b for b in detail["blocks"] if b["id"] == "b03")
    assert b03["evals"] == 0 and b03["summary"] is False


def test_the_safe_reader_refuses_links_fifos_and_the_answers(tmp_path, plant):
    from pinnforge.ui import safe

    f = tmp_path / "ok.txt"
    f.write_text("fine", encoding="utf-8")
    assert safe.read_bytes(f, 100) == b"fine"
    os.link(f, tmp_path / "twin.txt")
    assert safe.read_bytes(f, 100) is None, "two links: either could be the answers"
    plant(tmp_path / "sym.txt", tmp_path / "twin.txt")
    assert safe.read_bytes(tmp_path / "sym.txt", 100) is None
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "pipe")
        assert _within(lambda: safe.read_bytes(tmp_path / "pipe", 100),
                       fifos=[tmp_path / "pipe"]) is None
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "x").write_text("no", encoding="utf-8")
    assert safe.read_bytes(tmp_path / "private" / "x", 100) is None
    assert safe.read_bytes(tmp_path / "ok.txt", 100, root_real=str(tmp_path / "elsewhere")) \
        is None


# ─────────────────────────────── FIFOs ───────────────────────────────


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this system")
def test_fifos_planted_where_the_dashboard_reads_do_not_hang_it(ws):
    run = ws / "runs" / "toy_1"
    (run / "blocks" / "b02").mkdir()
    fifos = [run / "blocks" / "b02" / "evals.jsonl", run / "blocks" / "b01" / ".budget",
             run / "blocks" / "kb2" / "b02.md", run / "logs" / "b02.0.log",
             run / ".lock", run / ".viz" / "index.jsonl", run / ".viz" / "grid.json",
             run / ".viz" / "grid.npz", run / "blocks" / "run_usage.jsonl"]
    for f in fifos:
        os.mkfifo(f)
    (ws / "tasks" / "toy" / ".authoring").mkdir()
    for name in ("author.log", "state.json"):
        os.mkfifo(ws / "tasks" / "toy" / ".authoring" / name)
        fifos.append(ws / "tasks" / "toy" / ".authoring" / name)
    srv = _within(lambda: _serve(ws))
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for path in [*ENDPOINTS, "/api/runs/toy_1/activity?block=b02",
                     "/api/runs/toy_1/summary/b02"]:
            status, _ = _within(lambda p=path: _get(base + p), fifos=fifos, seconds=8)
            assert status in (200, 404), path
        d = json.loads(_get(base + "/api/runs/toy_1")[1])
        assert d["live"] is False and d["frames"] == []
    finally:
        srv.shutdown()
        srv.server_close()


def test_feeds_tail_refuses_what_api_refuses(tmp_path):
    log = tmp_path / "b01.0.log"
    log.write_text("hello\n", encoding="utf-8")
    assert [e["text"] for e in feeds.tail(log, None, "opencode")[0]] == ["hello"]
    os.link(log, tmp_path / "b01.1.log")
    assert feeds.tail(log, None, "opencode") == ([], 0)
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "b01.2.log")
        assert _within(lambda: feeds.tail(tmp_path / "b01.2.log", None, "opencode"),
                       fifos=[tmp_path / "b01.2.log"]) == ([], 0)
    hidden = tmp_path / ".scoreq"
    hidden.mkdir()
    (hidden / "b01.0.log").write_text("secret\n", encoding="utf-8")
    assert feeds.tail(hidden / "b01.0.log", None, "opencode") == ([], 0)
    outside = tmp_path / "other.log"
    outside.write_text("x\n", encoding="utf-8")
    assert feeds.tail(outside, None, "opencode", root_real=str(tmp_path / "ws")) == ([], 0)


def test_a_log_directory_linked_at_the_answers_is_not_tailed(ws, served, plant):
    run = ws / "runs" / "toy_1"
    private = ws / "tasks" / "toy" / "private"
    (private / "b01.9.log").write_text((private / "leak.log").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    os.rename(run / "logs", run / "logs.real")
    plant(run / "logs", private)
    plant(ws / "tasks" / "toy" / ".authoring", private)
    (private / "author.log").write_text((private / "leak.log").read_text(encoding="utf-8"),
                                        encoding="utf-8")
    for path in ("/api/runs/toy_1/activity", "/api/runs/toy_1/activity?block=b01",
                 "/api/tasks/toy/activity", "/api/tasks/toy"):
        _, body = _get(served + path)
        assert CANARY not in body, path
    assert api.activity(ws, "toy_1", "b01", None)["events"] == []
    assert api.author_activity(ws, "toy", None)["events"] == []


# ─────────────────────────────── malformed records ───────────────────────────────


def _hostile(ws):
    run = ws / "runs" / "toy_1"
    lines = [
        {"ts": "a", "rRMSE": "0.1", "wall_s": "10", "error": {"x": 1}},
        {"ts": 5, "rRMSE": True, "wall_s": True, "train_s": [1]},
        {"ts": None, "rRMSE": 1e400, "wall_s": -1e400, "candidate": ["x"]},
        {"ts": "b", "rRMSE": 0.2, "gpu": {"x": 1}, "candidate": 7},
        [1, 2], "string", 3, None,
    ]
    body = "\n".join(json.dumps(x) for x in lines) + '\n{"rRMSE": NaN}\n{"rRMSE": Infinity}\n'
    (run / "blocks" / "b01" / "evals.jsonl").write_text(body, encoding="utf-8")
    (run / "blocks" / "b01" / ".budget").write_text("inf", encoding="utf-8")
    (run / "blocks" / "b00").mkdir()
    (run / "blocks" / "b00" / "evals.jsonl").write_text(
        json.dumps({"rRMSE": "0.5"}) + "\n" + json.dumps({"rRMSE": 0.5}) + "\n",
        encoding="utf-8")
    (run / "blocks" / "b00" / ".budget").write_text("nan", encoding="utf-8")
    (run / "blocks" / "run_usage.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"block": "b01", "tokens": "many", "cost_usd": "1", "duration_ms": None,
         "tool_uses": True, "model": ["x"]},
        {"block": "b01", "tokens": 1e300, "cost_usd": 1e308},
        {"block": "b01", "tokens": 10, "cost_usd": 1e308},
        {"block": {"x": 1}}, [1]]) + "\n", encoding="utf-8")
    (run / "state.json").write_text(json.dumps(
        {"status": {"x": 1}, "blocks": [1, 2], "created_at": 5}), encoding="utf-8")
    (run / ".integrity").mkdir()
    (run / ".integrity" / "b01.1.result.json").write_text("[1]", encoding="utf-8")
    (run / ".viz" / "index.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"frame": "0", "rRMSE": "x"}, {"frame": -1}, {"frame": 1e30}, [1],
        {"frame": 2, "block": "../x", "rRMSE": 0.3, "run_best": "yes"}]) + "\n",
        encoding="utf-8")
    (run / ".viz" / "grid.json").write_text('{"coords": "xy", "dims": "two"}', encoding="utf-8")
    return run


def test_malformed_records_do_not_break_the_pages(ws, served):
    _hostile(ws)
    for path in ENDPOINTS:
        status, body = _get(served + path)
        assert status in (200, 404), (path, status, body[:200])
    ov = json.loads(_get(served + "/api/overview")[1])
    run = ov["runs"][0]
    assert run["best"] == 0.2 and run["anchor"] == 0.5 and run["status"] == "?"
    d = json.loads(_get(served + "/api/runs/toy_1")[1])
    assert [e["rRMSE"] for e in d["evals"] if e["block"] == "b01"].count(None) >= 4
    assert all(isinstance(e["candidate"], str) for e in d["evals"])
    assert all(e["error"] is None or isinstance(e["error"], str) for e in d["evals"])
    b01 = next(b for b in d["blocks"] if b["id"] == "b01")
    assert b01["best"] == 0.2 and isinstance(b01["spent"], float)
    assert b01["tokens"] == 10 and b01["cost_usd"] == 0.0  # an overflowing sum is no cost
    assert d["frames"] == [{"frame": 2, "block": "", "candidate": "", "rRMSE": 0.3,
                            "run_best": False}]
    assert d["grid"]["coords"] == [] and d["grid"]["dims"] == 2


def test_a_report_and_the_terminal_view_survive_malformed_records(ws):
    _hostile(ws)
    html = report.render(ws, "toy_1")
    assert "__PF_SNAPSHOT__" in html
    rows = tui.lines(ws, "toy_1")
    assert rows and rows[0][1].startswith("PINNForge")


def test_an_unreadable_run_is_a_card_not_a_broken_page(ws, monkeypatch):
    real = api.run_summary

    def flaky(root, run):
        if run.name == "toy_1":
            raise TypeError("unexpected shape")
        return real(root, run)

    monkeypatch.setattr(api, "run_summary", flaky)
    ov = api.overview(ws)
    assert ov["runs"] == [api._unreadable("toy_1")]
    monkeypatch.setattr(api, "task_summary", lambda root, t: 1 / 0)
    ov = api.overview(ws)
    assert ov["tasks"][0]["name"] == "toy" and ov["tasks"][0]["ready"] is False


def test_state_and_lease_records_of_the_wrong_shape_are_empty(ws, monkeypatch, tmp_path, plant):
    run = ws / "runs" / "toy_1"
    (run / "state.json").write_text("[1, 2, 3]", encoding="utf-8")
    d = api.run_detail(ws, "toy_1")
    assert d["status"] == "?" and d["blocks"][0]["status"] == "—"
    assert api.activity(ws, "toy_1", None, None)["block"] == "b01"
    leases = tmp_path / "leases"
    leases.mkdir()
    for i, body in enumerate(["[1]", '"x"', '{"pid": [1]}', '{"pid": -1, "run": "x"}',
                              '{"pid": true}', "{bad", f'{{"pid": {os.getpid()}, "run": 5}}']):
        (leases / f"gpu0.{i}.json").write_text(body, encoding="utf-8")
    plant(leases / "gpu0.90.json", leases / "gpu0.0.json")
    if hasattr(os, "mkfifo"):
        os.mkfifo(leases / "gpu0.91.json")
    (leases / "gpu0.99.json").write_text(
        json.dumps({"pid": os.getpid(), "run": "/x/runs/toy_7"}), encoding="utf-8")
    from pinnforge.run import lifecycle

    monkeypatch.setattr(lifecycle, "LEASE_DIR", leases)
    assert _within(lambda: api._leases([0, 1]), fifos=[leases / "gpu0.91.json"]) == \
        {0: "toy_7"}


# ─────────────────────────────── run.yaml ───────────────────────────────


def test_yaml_aliases_are_refused(tmp_path, ws):
    bomb = "a: &a [x, x, x, x, x, x, x, x, x]\n" + "".join(
        f"{c}: &{c} [*{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}, *{p}]\n"
        for p, c in zip("abcdefgh", "bcdefghi", strict=True))
    f = ws / "runs" / "toy_1" / "run.yaml"
    f.write_text("task: toy\n" + bomb, encoding="utf-8")
    assert _within(lambda: api._yaml(f)) == {}
    f.write_text("task: toy\nagent: &x {runtime: r}\nother: *x\n", encoding="utf-8")
    assert api._yaml(f) == {}
    f.write_text("task: toy\nagent: &x {runtime: r}\n", encoding="utf-8")  # anchor alone is fine
    assert api._yaml(f)["agent"] == {"runtime": "r"}
    assert api.run_detail(ws, "toy_1")["task"] == "toy"


def test_only_whitelisted_config_is_served(ws):
    (ws / "runs" / "toy_1" / "run.yaml").write_text(
        "task: toy\nblocks: 2\n"
        f"agent: {{runtime: claude_code, model: m, env: {{KEY: {CANARY}}}, extra: {CANARY}}}\n"
        f"sandbox: {{gpus: [0, true, '1', 2], wall_budget_s: 100, private_dir: {CANARY}}}\n"
        f"hw: {{gpu: A100, nested: {{x: {CANARY}}}}}\n"
        f"reference: {CANARY}\nsecret_env: {CANARY}\n", encoding="utf-8")
    d = api.run_detail(ws, "toy_1")
    assert set(d["config"]) == {"agent", "gpus", "wall_budget_s", "hw", "task", "blocks"}
    assert d["config"]["agent"] == {"runtime": "claude_code", "model": "m"}
    assert d["config"]["gpus"] == [0, 2] and d["config"]["hw"] == {"gpu": "A100"}
    assert d["config"]["task"] == "toy" and d["config"]["blocks"] == 2
    assert CANARY not in json.dumps(d)
    (ws / "runs" / "toy_1" / "run.yaml").write_text(
        "agent: [1]\nsandbox: 5\nblocks: x\nhw: [1]\ntask: {a: 1}\n", encoding="utf-8")
    d = api.run_detail(ws, "toy_1")
    assert d["config"] == {"agent": {"runtime": "", "model": ""}, "gpus": [],
                           "wall_budget_s": 0.0, "hw": {}, "task": "", "blocks": 0}


# ─────────────────────────────── frames ───────────────────────────────


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _write_frame(ws, grid: bytes, frame: bytes):
    viz = ws / "runs" / "toy_1" / ".viz"
    (viz / "grid.npz").write_bytes(grid)
    (viz / "frames" / "00000.npz").write_bytes(frame)


def test_a_good_frame_is_served(ws):
    coords = np.random.default_rng(0).random((100, 2)).astype(np.float32)
    _write_frame(ws, _npz_bytes(coords=coords, n=np.array(100)),
                 _npz_bytes(u=np.arange(100, dtype=np.float16)))
    out = api.frame(ws, "toy_1", 0)
    assert len(out["coords"]) == 2 and len(out["fields"]["u"]["values"]) == 100


def test_a_decompression_bomb_frame_is_refused(ws):
    coords = np.zeros((10, 2), dtype=np.float32)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("u.npy", b"\0" * (200 * 2**20))  # 200 MB of zeros, ~200 kB on disk
    assert len(buf.getvalue()) < 2**20
    _write_frame(ws, _npz_bytes(coords=coords), buf.getvalue())
    with pytest.raises(api.NotFound):
        _within(lambda: api.frame(ws, "toy_1", 0), seconds=10)


def test_an_npy_header_cannot_declare_a_huge_array(ws):
    coords = np.zeros((10, 2), dtype=np.float32)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                  "shape": (10**13,)})
    lie = header.getvalue() + b"\0" * 80  # 80 bytes claiming to be 80 TB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("u.npy", lie)
    _write_frame(ws, _npz_bytes(coords=coords), buf.getvalue())
    with pytest.raises(api.NotFound):
        api.frame(ws, "toy_1", 0)
    # an object array would need pickle — never
    with zipfile.ZipFile(buf2 := io.BytesIO(), "w") as zf:
        h = io.BytesIO()
        np.lib.format.write_array_header_1_0(h, {"descr": "|O", "fortran_order": False,
                                                 "shape": (1,)})
        zf.writestr("u.npy", h.getvalue() + b"\0" * 8)
    _write_frame(ws, _npz_bytes(coords=coords), buf2.getvalue())
    with pytest.raises(api.NotFound):
        api.frame(ws, "toy_1", 0)


def test_a_grid_beyond_max_points_is_refused(ws):
    from pinnforge.ui import viz

    n = viz.MAX_POINTS + 1
    _write_frame(ws, _npz_bytes(coords=np.zeros((n, 2), dtype=np.float32)),
                 _npz_bytes(u=np.zeros(n, dtype=np.float16)))
    with pytest.raises(api.NotFound):
        api.frame(ws, "toy_1", 0)
    _write_frame(ws, b"not a zip at all", _npz_bytes(u=np.zeros(3)))
    with pytest.raises(api.NotFound):
        api.frame(ws, "toy_1", 0)


# ─────────────────────────────── redaction ───────────────────────────────

KEY = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"


def test_agent_strings_are_redacted_everywhere_they_are_served(ws):
    run = ws / "runs" / "toy_1"
    (run / "blocks" / "b01" / "evals.jsonl").write_text(json.dumps(
        {"ts": "t", "rRMSE": None, "error": f"crashed; ANTHROPIC_API_KEY={KEY}"}) + "\n",
        encoding="utf-8")
    (run / "blocks" / "kb2" / "b01.md").write_text(f"# b01\nused {KEY}\n", encoding="utf-8")
    d = api.run_detail(ws, "toy_1")
    assert KEY not in json.dumps(d) and "•••" in d["evals"][0]["error"]
    md = api.summary_md(ws, "toy_1", "b01")["markdown"]
    assert KEY not in md and "•••" in md
    assert KEY not in report.render(ws, "toy_1")


def test_redaction_happens_before_the_clip():
    text = "x" * 385 + " " + "sk-ant-" + "A" * 30
    line = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": text}]}})
    out = feeds.parse("claude_code", [line])[0]["text"]
    assert "AAAAAA" not in out and "sk-ant" not in out and "•••" in out


def test_odd_transcript_lines_are_skipped_not_fatal():
    lines = [
        json.dumps({"type": "assistant", "message": ["not", "a", "dict"]}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": 5}]}}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "kept"}]}}),
    ]
    assert [e["text"] for e in feeds.parse("claude_code", lines)] == ["kept"]
    codex = [json.dumps({"item": {"type": "x", "command": ["ls", 3, None]}})]
    assert feeds.parse("codex", codex)[0]["text"] == "ls 3 None"


# ─────────────────────────────── the report ───────────────────────────────


def test_a_report_escapes_every_markup_character_in_its_data(ws):
    nasty = "</script><!--<script>alert(1)</script>&amp;]]>"
    run = ws / "runs" / "toy_1"
    (run / "blocks" / "kb2" / "b01.md").write_text(f"# {nasty}\n", encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(json.dumps(
        {"ts": "t", "rRMSE": 0.3, "candidate": f"blocks/b01/{nasty}.py",
         "error": nasty}) + "\n", encoding="utf-8")
    html = report.render(ws, "toy_1")
    start = html.index("window.__PF_SNAPSHOT__ = ") + len("window.__PF_SNAPSHOT__ = ")
    data = html[start: html.index(";\n", start)]
    assert not set(data) & set("<>&")
    snap = json.loads(data)
    assert nasty in snap["/api/runs/toy_1/summary/b01"]["markdown"]
    assert report.script_json("<&>") == '"\\u003c\\u0026\\u003e"'


def test_the_terminal_view_strips_control_characters(ws):
    (ws / "runs" / "toy_1" / "logs" / "b01.0.log").write_text(json.dumps(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "hi \x1b]52;c;ZXZpbA==\x07 \x1b[2J there"}]}}) + "\n",
        encoding="utf-8")
    rows = tui.lines(ws, "toy_1")
    assert all("\x1b" not in text and "\x07" not in text for _, text in rows)
    assert any("there" in text for _, text in rows)
