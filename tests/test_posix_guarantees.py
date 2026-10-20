"""Guarantees the POSIX side kept through the Windows port, pinned down where it slipped."""

from __future__ import annotations

import argparse
import json
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pinnforge import oscompat, paths
from pinnforge.run import lifecycle

posix_only = pytest.mark.skipif(oscompat.WINDOWS, reason="POSIX semantics")


# ─────────────────────── run stop: never a stop that did not happen ───────────────────────


class _Clock:
    """`time` for lifecycle with sleeps that pass instantly."""

    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, *a):
        import time

        return time.strftime(*a)


@pytest.fixture
def held_by_another_user(tmp_path, monkeypatch):
    """A run whose lock is held by a driver this user cannot signal."""
    lock = lifecycle.RunLock(tmp_path).acquire()
    monkeypatch.setattr(lifecycle, "time", _Clock())
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(oscompat, "request_stop", lambda pid: False)

    def refuse(pid):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(oscompat, "kill", refuse)
    yield tmp_path
    lock.release()


def test_a_driver_that_cannot_be_signalled_is_not_reported_stopped(held_by_another_user):
    run = held_by_another_user
    said = lifecycle.stop(run)
    assert said.startswith("refused"), said
    assert lifecycle.holder(run) is not None


def test_a_forced_stop_that_could_not_kill_says_so(held_by_another_user):
    run = held_by_another_user
    said = lifecycle.stop(run, force=True)
    assert said.startswith("still stopping"), f"'{said}' — the lock is still held"


def test_run_stop_leaves_a_live_drivers_state_alone(held_by_another_user, monkeypatch):
    from pinnforge.cli import run_cmd

    run = held_by_another_user
    state = types.SimpleNamespace(status="running", blocks={}, save=lambda p: saved.append(p))
    saved: list = []
    monkeypatch.setattr(run_cmd.layout, "load_run", lambda run_id, root=None: (run, None, state))
    monkeypatch.setattr(lifecycle, "stop", lambda run, force=False: "killed")  # claims success
    assert run_cmd.cmd_stop(argparse.Namespace(run_id="x_1", force=True)) == 1
    assert saved == [] and state.status == "running", "a held lock is a live driver's state"


# ─────────────────────── ui: only this workspace's dashboard ───────────────────────


def _ping_server(path: str):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"app": "pinnforge", "path": path}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_another_workspaces_dashboard_on_the_port_is_not_ours(tmp_path, monkeypatch):
    from pinnforge.ui import server

    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    for d in (mine, theirs):
        (d / ".pinnforge").mkdir(parents=True)
    monkeypatch.setattr(server.oscompat, "pid_alive", lambda pid: True)  # a reused pid
    for serving, expect in ((theirs, None), (mine, "found")):
        srv = _ping_server(str(serving))
        try:
            server.state_file(mine).write_text(json.dumps(
                {"pid": os.getpid(), "host": "127.0.0.1", "port": srv.server_address[1]}))
            got = server.running(mine)
            assert (got is not None) == (expect == "found"), serving
        finally:
            srv.shutdown()


# ─────────────────────── file:// references on POSIX ───────────────────────


@posix_only
@pytest.mark.parametrize("url, path", [
    ("file:///srv/a%20b/ref.npy", "/srv/a b/ref.npy"),
    ("file://localhost/srv/ref.npy", "/srv/ref.npy"),
    ("file://~/data/ref.npy", "~/data/ref.npy"),
    ("file://data/ref.npy", "data/ref.npy"),
    ("/plain/path.npy", "/plain/path.npy"),
])
def test_file_urls_name_the_file_they_always_named(url, path):
    assert paths.local_path(url) == path


# ─────────────────────── temp dirs: errors are errors off Windows ───────────────────────


def test_temporary_directories_hide_cleanup_errors_only_on_windows():
    root = Path(__file__).resolve().parents[1] / "pinnforge"
    loose = [str(p.relative_to(root)) for p in root.rglob("*.py")
             if "ignore_cleanup_errors=True" in p.read_text(encoding="utf-8")]
    assert loose == [], "use ignore_cleanup_errors=oscompat.WINDOWS"


# ─────────────────────── the film's agent is a recorded agent ───────────────────────


def test_the_film_agent_is_recorded_in_the_run_lock_while_it_runs(tmp_path, monkeypatch):
    from pinnforge import discovery
    from pinnforge.cli import run_cmd

    lock = lifecycle.RunLock(tmp_path).acquire()
    seen = {}

    def make(root, run, agent, timeout_s=None, on_start=None):
        on_start(4242)
        seen["during"] = lifecycle.holder(tmp_path).get("agent_pgid")
        return types.SimpleNamespace(path=run / "discovery.html", scripted_by="agent",
                                     problems=[])

    monkeypatch.setattr(discovery, "make", make)
    cfg = types.SimpleNamespace(film=True, agent=types.SimpleNamespace(runtime="stub"))
    state = types.SimpleNamespace(status="finished")
    monkeypatch.setattr(run_cmd.layout, "load_run",
                        lambda run_id, root=None: (tmp_path, cfg, state))
    try:
        run_cmd._film_after(tmp_path, tmp_path, lock)
        assert seen["during"] == 4242, "a driver that dies outright leaves a group to stop"
        assert lifecycle.holder(tmp_path).get("agent_pgid") is None, "and it is cleared after"
    finally:
        lock.release()


def test_run_film_waits_for_the_driver(tmp_path, monkeypatch, capsys):
    from pinnforge.cli import run_cmd

    lock = lifecycle.RunLock(tmp_path).acquire()
    cfg = types.SimpleNamespace(agent=types.SimpleNamespace(runtime="stub"))
    monkeypatch.setattr(run_cmd.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(run_cmd.layout, "load_run",
                        lambda run_id, root=None: (tmp_path, cfg, None))
    try:
        rc = run_cmd.cmd_film(argparse.Namespace(run_id="x_1", render_only=False, timeout=1))
        assert rc == 2 and "made when the run finishes" in capsys.readouterr().out
    finally:
        lock.release()
