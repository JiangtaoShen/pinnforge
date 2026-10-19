"""The dashboard's HTTP front: who gets the token cookie, what a request with a
body gets (the connection closed, nothing smuggled behind it), what an error
says (never an exception's text), and how long a client may hold a thread.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import socket
import stat
import threading
import time

import pytest

from pinnforge.ui import api, server

# Permission bits mean nothing on Windows, which guards a file with its ACL.
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")

TOKEN = "s3cret-token"


def _workspace(root):
    run = root / "runs" / "toy_1"
    (run / "blocks" / "b01").mkdir(parents=True)
    (root / "tasks" / "toy").mkdir(parents=True)
    (root / "tasks" / "toy" / "problem.md").write_text("# TOY\n", encoding="utf-8")
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    (run / "run.yaml").write_text("task: toy\nblocks: 1\nagent: {runtime: claude_code}\n",
                                  encoding="utf-8")
    (run / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run / "blocks" / "b01" / "evals.jsonl").write_text(
        json.dumps({"ts": "t", "rRMSE": 0.5, "wall_s": 1.0}) + "\n", encoding="utf-8")
    return root


@pytest.fixture()
def ws(tmp_path):
    return _workspace(tmp_path / "ws")


def _serve(ws, token=None):
    srv = server.make_server(ws, "127.0.0.1", 0, token)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return srv


@pytest.fixture()
def guarded(ws):
    srv = _serve(ws, TOKEN)
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def open_srv(ws):
    srv = _serve(ws)
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, headers=None, body=None):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read().decode("utf-8", "replace")
    finally:
        conn.close()


def _raw(srv, data: bytes, wait=5.0) -> bytes:
    """Send raw bytes; read until the server closes (or `wait` runs out)."""
    port = srv.server_address[1]
    out = b""
    with socket.create_connection(("127.0.0.1", port), timeout=wait) as s:
        s.sendall(data)
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                chunk = s.recv(65536)
            except ConnectionResetError:
                break
            except TimeoutError:
                pytest.fail("the server kept the connection open")
            if not chunk:
                break
            out += chunk
        else:
            pytest.fail("the server kept the connection open")
    return out


# ─────────────────────────────── the token cookie ───────────────────────────────


def test_the_cookie_comes_only_with_a_200_to_a_valid_token(guarded):
    port = guarded.server_address[1]
    name = f"pf_token_{port}"
    status, headers, _ = _req(guarded, "GET", f"/api/overview?token={TOKEN}")
    assert status == 200
    cookie = headers.get("Set-Cookie", "")
    assert cookie.startswith(f"{name}={TOKEN};")
    for attr in ("HttpOnly", "SameSite=Strict", "Path=/"):
        assert attr in cookie
    # the cookie alone opens the door, but is not handed out again
    status, headers, _ = _req(guarded, "GET", "/api/overview", {"Cookie": f"{name}={TOKEN}"})
    assert status == 200 and "Set-Cookie" not in headers
    # the page itself gets it too (that is how the browser receives it)
    status, headers, _ = _req(guarded, "GET", f"/?token={TOKEN}")
    assert status == 200 and headers.get("Set-Cookie", "").startswith(f"{name}=")


@pytest.mark.parametrize("method,path,headers", [
    ("GET", "/api/overview", {}),                                     # 401: no token
    ("GET", "/api/overview?token=wrong", {}),                          # 401: wrong token
    ("GET", f"/api/overview?token={TOKEN}", {"Host": "evil.example"}),  # 421
    ("POST", f"/api/overview?token={TOKEN}", {}),                      # 405
    ("DELETE", f"/api/overview?token={TOKEN}", {}),                    # 405
    ("GET", f"/api/runs/nope?token={TOKEN}", {}),                      # 404
    ("GET", f"/api/overview?token={TOKEN}", {"Content-Length": "3"}),  # 400
])
def test_no_error_response_ever_carries_the_token(guarded, method, path, headers):
    body = b"abc" if headers.get("Content-Length") == "3" else None
    status, got, text = _req(guarded, method, path, headers, body)
    assert status >= 400
    assert "Set-Cookie" not in got
    assert TOKEN not in text and TOKEN not in json.dumps(got)


def test_a_cookie_for_another_port_or_the_old_name_is_not_accepted(guarded):
    for cookie in (f"pf_token={TOKEN}", f"pf_token_1={TOKEN}", f"pf_token_{TOKEN}"):
        assert _req(guarded, "GET", "/api/overview", {"Cookie": cookie})[0] == 401


def test_non_ascii_tokens_are_refused_not_crashed(guarded):
    port = guarded.server_address[1]
    assert _req(guarded, "GET", "/api/overview?token=%C3%A9%E2%82%AC")[0] == 401
    raw = (f"GET /api/overview HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
           f"Cookie: pf_token_{port}=\xe9\xff\r\nConnection: close\r\n\r\n").encode("latin-1")
    assert _raw(guarded, raw).startswith(b"HTTP/1.1 401")


# ─────────────────────────────── bodies and keep-alive ───────────────────────────────


def test_a_post_body_cannot_smuggle_a_second_request(open_srv):
    port = open_srv.server_address[1]
    smuggled = b"GET /api/overview HTTP/1.1\r\nHost: evil.example\r\n\r\n"
    data = (f"POST /api/overview HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Length: {len(smuggled)}\r\n\r\n").encode() + smuggled
    out = _raw(open_srv, data)
    assert out.startswith(b"HTTP/1.1 400")
    assert out.count(b"HTTP/1.1 ") == 1, "only one answer: the body was never parsed"
    assert b"Connection: close" in out


@pytest.mark.parametrize("framing", ["Content-Length: {n}", "Transfer-Encoding: chunked"])
def test_a_get_with_a_body_is_refused_and_closed(open_srv, framing):
    port = open_srv.server_address[1]
    smuggled = b"GET /api/overview HTTP/1.1\r\nHost: evil.example\r\n\r\n"
    head = framing.format(n=len(smuggled))
    data = (f"GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{head}\r\n\r\n").encode()
    out = _raw(open_srv, data + smuggled)
    assert out.startswith(b"HTTP/1.1 400")
    assert out.count(b"HTTP/1.1 ") == 1


def test_every_refusal_closes_the_connection(open_srv):
    port = open_srv.server_address[1]
    # a refused request followed, on the same connection, by one that would succeed
    for first in (b"GET /api/overview HTTP/1.1\r\nHost: evil.example\r\n\r\n",
                  b"DELETE /api/overview HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port,
                  b"GET /api/runs/nope HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port,
                  b"BREW /api/overview HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port):
        second = b"GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port
        out = _raw(open_srv, first + second)
        assert out.count(b"HTTP/1.1 ") == 1, first
        assert b'"app"' not in out
    # while a good request keeps the connection open for the next
    good = b"GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n\r\n" % port
    last = b"GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n" % port
    assert _raw(open_srv, good + last).count(b"HTTP/1.1 200") == 2


# ─────────────────────────────── errors say nothing ───────────────────────────────


@pytest.mark.parametrize("exc", [
    ValueError("internal detail /home/secret/path"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "internal detail /home/secret/path"),
    KeyError("internal detail /home/secret/path"),
    RuntimeError("internal detail /home/secret/path"),
])
def test_an_unexpected_failure_is_a_generic_500(open_srv, monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(api, "run_detail", boom)
    status, headers, body = _req(open_srv, "GET", "/api/runs/toy_1")
    assert status == 500
    assert "internal detail" not in body and "secret" not in body
    assert json.loads(body) == {"error": "internal error"}
    assert headers.get("Connection") == "close"


def test_oversized_or_unicode_numbers_in_a_url_are_not_a_crash(open_srv):
    port = open_srv.server_address[1]
    for path in ("/api/runs/toy_1/frame/" + "9" * 5000, "/api/runs/toy_1/frame/\xb2",
                 "/api/runs/toy_1/activity?after=" + "9" * 5000,
                 "/api/runs/toy_1/activity?after=%C2%B2"):
        req = (f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
               "Connection: close\r\n\r\n").encode("latin-1")
        out = _raw(open_srv, req)
        assert out.startswith((b"HTTP/1.1 200", b"HTTP/1.1 404")), (path, out[:40])


# ─────────────────────────────── holding threads ───────────────────────────────


def test_the_handler_has_a_socket_timeout():
    assert server.Handler.timeout == server.SOCKET_TIMEOUT_S <= 120


def test_an_idle_connection_is_dropped(open_srv):
    open_srv.RequestHandlerClass.timeout = 0.3
    port = open_srv.server_address[1]
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        t0 = time.time()
        with contextlib.suppress(ConnectionResetError):
            assert s.recv(1024) == b""
        assert time.time() - t0 < 4


def test_an_event_stream_ends_and_frees_its_slot(open_srv, monkeypatch):
    monkeypatch.setattr(server, "STREAM_LIFETIME_S", 0.5)
    monkeypatch.setattr(api, "gpu_stats", lambda: [])
    port = open_srv.server_address[1]
    sem = open_srv.RequestHandlerClass.streams
    req = f"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
    out = _raw(open_srv, req, wait=8)
    assert out.startswith(b"HTTP/1.1 200") and b"retry:" in out
    deadline = time.time() + 3
    while time.time() < deadline and sem._value != server.MAX_STREAMS:
        time.sleep(0.05)
    assert sem._value == server.MAX_STREAMS


# ─────────────────────────────── ui.json ───────────────────────────────


@posix_only
def test_ui_json_is_private_from_its_first_byte(tmp_path):
    old = os.umask(0)
    try:
        f = tmp_path / "ui.json"
        server._write_private(f, '{"token": "x"}\n')
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
        # and replacing an old, wider record narrows it
        f.chmod(0o644)
        server._write_private(f, '{"token": "y"}\n')
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
        assert json.loads(f.read_text(encoding="utf-8")) == {"token": "y"}
    finally:
        os.umask(old)


@posix_only
def test_serve_records_its_token_privately(ws, monkeypatch):
    seen = {}

    class Fake:
        server_address = ("0.0.0.0", 4321)

        def serve_forever(self, **_):
            f = server.state_file(ws)
            seen["mode"] = stat.S_IMODE(f.stat().st_mode)
            seen["info"] = json.loads(f.read_text(encoding="utf-8"))

        def server_close(self):
            pass

    monkeypatch.setattr(server, "make_server", lambda *a, **k: Fake())
    old = os.umask(0)
    try:
        server.serve(ws, "0.0.0.0", 4321)
    finally:
        os.umask(old)
    assert seen["mode"] == 0o600 and seen["info"]["token"]
    assert not server.state_file(ws).exists()
