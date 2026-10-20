"""The dashboard: what it shows, and — above all — what it never gives away.

A block can reach the dashboard (`curl localhost`), and can plant symlinks in
its own workspace. So the tests plant a canary in the private reference and
in the score queue, crawl every endpoint, and require the canary never to
appear; plant links pointing at it and require they are not followed; and
require the server to refuse anything but GET from the expected host.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from pinnforge.ui import api, feeds, server, viz

CANARY = "CANARY-7f3e91"


@pytest.fixture()
def ws(tmp_path, plant):
    root = tmp_path / "ws"
    (root / "tasks" / "toy" / "private").mkdir(parents=True)
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    task = root / "tasks" / "toy"
    (task / "problem.md").write_text("# TOY — toy\n", encoding="utf-8")
    (task / "private" / "ref_values.npy").write_bytes(CANARY.encode() * 4)
    (task / "private" / "secret.md").write_text(CANARY, encoding="utf-8")
    run = root / "runs" / "toy_1"
    (run / "blocks" / "b01").mkdir(parents=True)
    (run / "blocks" / "kb2").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / ".scoreq").mkdir()
    (run / ".scoreq" / "scores.jsonl").write_text(CANARY, encoding="utf-8")
    (run / "run.yaml").write_text(
        "task: toy\nblocks: 1\nagent: {runtime: claude_code, model: m}\n"
        "sandbox: {gpus: [0], wall_budget_s: 100}\n", encoding="utf-8")
    (run / "state.json").write_text(json.dumps(
        {"task": "toy", "run_id": "toy_1", "status": "running",
         "blocks": {"b01": {"status": "running"}}}), encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(json.dumps(
        {"ts": "2026-01-01T00:00:00Z", "block": "b01", "candidate": "blocks/b01/b01_a.py",
         "smoke": False, "rRMSE": 0.25, "wall_s": 10.0}) + "\n", encoding="utf-8")
    (run / "blocks" / "kb2" / "b01.md").write_text("# b01 — <script>alert(1)</script>\n",
                                                   encoding="utf-8")
    (run / "logs" / "b01.0.log").write_text(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "using key sk-ant-abcdefghijklmnop1234"}]}}) + "\n",
        encoding="utf-8")
    # a block plants links at the answers, where the dashboard reads
    (run / "blocks" / "b02").mkdir()
    plant(run / "blocks" / "b02" / "evals.jsonl", task / "private" / "secret.md")
    plant(run / "blocks" / "kb2" / "b02.md", task / "private" / "secret.md")
    plant(run / ".viz", task / "private")
    return root


@pytest.fixture()
def served(ws):
    srv = server.make_server(ws, "127.0.0.1", 0, None)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _get(url, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


ENDPOINTS = [
    "/api/ping", "/api/overview", "/api/gpus", "/api/runs/toy_1", "/api/runs/toy_1/summary/b01",
    "/api/runs/toy_1/summary/b02", "/api/runs/toy_1/activity", "/api/runs/toy_1/activity?block=b02",
    "/api/runs/toy_1/frame/0", "/api/tasks/toy", "/api/tasks/toy/activity",
    "/", "/app.js", "/fonts.css",
]


def test_no_endpoint_ever_carries_the_answers(served):
    for path in ENDPOINTS:
        status, body = _get(served + path)
        assert CANARY not in body, f"{path} leaked the reference"
        assert status in (200, 404), (path, status)


def test_planted_links_are_not_followed(served):
    assert _get(served + "/api/runs/toy_1/summary/b02")[0] == 404
    detail = json.loads(_get(served + "/api/runs/toy_1")[1])
    b02 = next(b for b in detail["blocks"] if b["id"] == "b02")
    assert b02["evals"] == 0


def test_only_get_from_the_expected_host(served):
    assert _get(served + "/api/overview", method="POST")[0] == 405
    assert _get(served + "/api/overview", method="DELETE")[0] == 405
    assert _get(served + "/api/overview", headers={"Host": "evil.example:80"})[0] == 421


def test_names_in_requests_cannot_walk_the_tree(served):
    for path in ("/api/runs/..%2F..%2Ftasks%2Ftoy%2Fprivate", "/api/tasks/..",
                 "/api/runs/toy_1/summary/..%2F..%2Frun"):
        assert _get(served + path)[0] == 404


def test_agent_text_is_redacted_and_served_as_data(served):
    body = json.loads(_get(served + "/api/runs/toy_1/activity")[1])
    text = " ".join(e["text"] for e in body["events"])
    assert "sk-ant-abcdefghijklmnop1234" not in text and "•••" in text
    summary = json.loads(_get(served + "/api/runs/toy_1/summary/b01")[1])["markdown"]
    assert "<script>" in summary  # delivered as text; the page never parses it as HTML
    csp = urllib.request.urlopen(served + "/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("style-src")[0]


def test_a_token_guards_a_non_loopback_dashboard(ws):
    srv = server.make_server(ws, "127.0.0.1", 0, "s3cret")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert _get(base + "/api/overview")[0] == 401
        assert _get(base + "/api/overview?token=s3cret")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_run_detail_reads_the_record(ws):
    api.ROOT_REAL = str(ws.resolve())
    d = api.run_detail(ws, "toy_1")
    assert d["best"] == 0.25 and d["evals"][0]["cum_wall"] == 10.0
    assert d["blocks"][0]["status"] == "running"


# ─────────────────────────────── feeds ───────────────────────────────


def test_claude_stream_json_becomes_events():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-4-8"}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}}),
        json.dumps({"type": "result", "num_turns": 3, "total_cost_usd": 0.5}),
    ]
    ev = feeds.parse("claude_code", lines)
    assert [e["kind"] for e in ev] == ["info", "tool", "info"]
    assert ev[1]["text"] == "Bash: ls" and "$0.50" in ev[2]["text"]


def test_opencode_output_skips_logs_and_command_output():
    lines = ["timestamp=1 level=INFO message=x", "\x1b[0m$ \x1b[0mls -la", "total 44",
             "\x1b[0m", "Now the summary.", "\x1b[0m→ \x1b[0mRead block.md"]
    ev = feeds.parse("opencode", lines)
    assert [(e["kind"], e["text"]) for e in ev] == [
        ("tool", "ls -la"), ("text", "Now the summary."), ("tool", "Read block.md")]


def test_tail_resumes_from_its_offset(tmp_path):
    log = tmp_path / "b01.0.log"
    log.write_text("one\n", encoding="utf-8")
    ev, off = feeds.tail(log, None, "opencode")
    assert [e["text"] for e in ev] == ["one"]
    with log.open("a", encoding="utf-8") as fh:
        fh.write("two\npartial")
    ev, off2 = feeds.tail(log, off, "opencode")
    assert [e["text"] for e in ev] == ["two"] and off2 < log.stat().st_size


# ─────────────────────────────── field archive ───────────────────────────────


def test_only_improvements_are_archived(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    coords = np.stack(np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50)), -1).reshape(-1, 2)
    arch = viz.Archiver(run, coords, 2, ["u"])
    pay = tmp_path / "p.npz"
    np.savez(pay, u=np.arange(len(coords), dtype=np.float64))
    for block, score in (("b01", 0.5), ("b01", 0.6), ("b01", 0.4), ("b02", 0.45), ("b02", 0.3)):
        arch({"block": block, "candidate": "c", "smoke": False}, {"rRMSE": score}, pay)
    arch({"block": "b03", "smoke": True}, {"rRMSE": 0.01}, pay)
    text = (run / ".viz" / "index.jsonl").read_text(encoding="utf-8")
    index = [json.loads(ln) for ln in text.splitlines()]
    assert [r["rRMSE"] for r in index] == [0.5, 0.4, 0.45, 0.3]
    assert [r["run_best"] for r in index] == [True, True, False, True]
    assert len(list((run / ".viz" / "frames").glob("*.npz"))) == 4


def test_a_report_is_self_contained_and_leaks_nothing(ws):
    from pinnforge.ui import report

    html = report.render(ws, "toy_1")
    assert "<script src=" not in html and 'href="app.css"' not in html, "nothing fetched"
    assert "url('fonts/" not in html and 'src="colalab.png"' not in html
    assert "__PF_SNAPSHOT__" in html and '"/api/runs/toy_1"' in html
    assert CANARY not in html
    assert "</script><script>alert" not in html  # agent text cannot close the data script
    here = json.dumps(str(ws / "runs" / "toy_1"))[1:-1]
    assert here not in html, "where the run lives on this machine stays here"


def test_chat_finds_the_plugin():
    from pinnforge.cli import chat_cmd

    plugin = chat_cmd.plugin_dir()
    assert plugin is not None and (plugin / "agents" / "pinnforge-task-author.md").is_file()
