"""`pinnforge ui` — the dashboard, served from the standard library alone.

One process per workspace, started on demand by `run start` (or by hand) and
left running: it only reads. Its state — pid, address, token — is kept in
`.pinnforge/ui.json` so a second `pinnforge ui` finds it instead of starting
another.

It is reachable from every process on the machine, blocks included, so it
is built to give nothing away and to take nothing in:

* only GET and HEAD; everything else is refused, and so is any request that
  carries a body — and every refusal closes the connection, so nothing left
  unread on it can be taken for a second request;
* JSON assembled from known files (see `api.py`), never a file named by a
  request; the page itself is three packaged static files;
* a `Host` check against DNS rebinding, and a token whenever it listens on
  anything but loopback — the cookie that carries it is handed out only in
  answer to a request that presented the token itself;
* a strict Content-Security-Policy — no inline script, no third-party
  origin — because summaries and transcripts shown on the page were written
  by agents;
* errors say what kind of error, never an exception's text;
* a socket timeout, a cap on open event streams and a lifetime for each, so
  idle or stalled clients cannot hold every thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pinnforge import __version__, oscompat, paths, settings
from pinnforge.ui import api, safe

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
         "/app.css": "app.css", "/favicon.svg": "favicon.svg", "/fonts.css": "fonts.css",
         # the packaged typefaces, by name: a fixed list, never a path from the request
         **{f"/fonts/{p.name}": f"fonts/{p.name}" for p in (STATIC / "fonts").glob("*.woff2")}}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".woff2": "font/woff2"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")
MAX_STREAMS = 16
STREAM_LIFETIME_S = 30 * 60  # then the stream ends; EventSource reconnects by itself
SOCKET_TIMEOUT_S = 60
LOOPBACK = ("127.0.0.1", "localhost", "::1")
REASONS = {400: "bad request", 401: "token required — use the URL `pinnforge ui` printed",
           404: "not found", 405: "this dashboard is read-only", 408: "request timed out",
           413: "request bodies are not accepted", 414: "request line too long",
           417: "request bodies are not accepted",
           421: "unexpected Host header", 431: "request headers too large",
           500: "internal error", 501: "not implemented", 503: "too many open dashboards"}

logger = logging.getLogger(__name__)


def state_file(root: Path) -> Path:
    return paths.state_dir(root) / "ui.json"


class Handler(BaseHTTPRequestHandler):
    server_version = "pinnforge-ui"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # `StreamRequestHandler` applies this to the socket: a client that sends
    # nothing (or reads nothing) for this long is dropped, not waited on.
    timeout = SOCKET_TIMEOUT_S

    # set on the class by `make_server`
    root: Path = Path(".")
    token: str | None = None
    cookie: str = "pf_token"
    hosts: frozenset[str] = frozenset()
    streams = threading.BoundedSemaphore(MAX_STREAMS)

    # per request: hand the cookie out with this (200) answer?
    _give_cookie = False

    def log_message(self, *args) -> None:  # the terminal is not a web log
        pass

    # ── gatekeeping ──

    def _has_body(self) -> bool:
        """A body this server would never read — so the request is refused."""
        if self.headers.get("Transfer-Encoding") is not None:
            return True
        lengths = self.headers.get_all("Content-Length") or []
        return len(lengths) > 1 or any(v.strip() != "0" for v in lengths)

    def handle_expect_100(self) -> bool:
        """`Expect: 100-continue` asks leave to send a body; it is never given."""
        self._error(417)
        return False

    def _authorised(self) -> bool:
        self._give_cookie = False
        host = (self.headers.get("Host") or "").strip().lower()
        if self.hosts and host not in self.hosts:
            self._error(421)
            return False
        if self.token is None:
            return True
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        from_query = (query.get("token") or [None])[0]
        if from_query is not None:
            if _same(from_query, self.token):
                self._give_cookie = True
                return True
        elif _same(self._cookie_value() or "", self.token):
            return True
        self._error(401)
        return False

    def _cookie_value(self) -> str | None:
        for raw in self.headers.get_all("Cookie") or []:
            for part in raw.split(";"):
                name, sep, value = part.strip().partition("=")
                if sep and name == self.cookie:
                    return value
        return None

    def _headers(self, status: int, ctype: str, length: int, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        if status == 200 and self._give_cookie and self.token is not None:
            # only to a request that showed the token itself, and per port:
            # two dashboards on one host must not overwrite each other's
            self.send_header("Set-Cookie",
                             f"{self.cookie}={self.token}; HttpOnly; SameSite=Strict; Path=/")
        if status >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _error(self, status: int, message: str | None = None) -> None:
        body = json.dumps({"error": message or REASONS.get(status, "error")}).encode()
        self.close_connection = True
        self._headers(status, "application/json", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None):
        """The base class's own refusals (bad request line, oversized headers,
        unknown method), in this server's shape: JSON, generic text, the same
        security headers, and the connection closed."""
        self.close_connection = True
        with contextlib.suppress(OSError):
            self._error(code)

    # ── methods ──

    def do_GET(self) -> None:
        self._give_cookie = False
        if self._has_body():
            self._error(400)
            return
        if not self._authorised():
            return
        url = urllib.parse.urlsplit(self.path)
        route, query = url.path, urllib.parse.parse_qs(url.query)
        try:
            if route in PAGES:
                self._static(PAGES[route])
                return
            if route == "/api/events":
                self._events()
                return
            data = self._api(route, query)
            if data is None:
                raise api.NotFound(route)
            body = json.dumps(_finite(data), allow_nan=False, default=str).encode()
        except api.NotFound:
            self._error(404)
            return
        except Exception:
            # what went wrong goes to the log, never to the client
            logger.exception("request failed: %s", route)
            self._error(500)
            return
        self._headers(200, "application/json", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET

    def _refuse(self) -> None:
        self.close_connection = True
        self._error(400 if self._has_body() else 405)

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _refuse

    def _static(self, name: str) -> None:
        f = STATIC / name
        body = f.read_bytes()
        self._headers(200, TYPES.get(f.suffix, "application/octet-stream"), len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _api(self, route: str, query: dict):
        parts = [p for p in route.split("/") if p]
        if parts[:1] != ["api"]:
            return None
        parts = parts[1:]
        root = self.root

        def after():
            raw = (query.get("after") or [""])[0]
            return int(raw) if _digits(raw, 18) else None

        match parts:
            case ["ping"]:
                return {"app": "pinnforge", "version": __version__, "workspace": root.name,
                        "path": str(root)}
            case ["overview"]:
                return api.overview(root)
            case ["gpus"]:
                stats = api.gpu_stats()
                return {"gpus": stats, "leases": api._leases([g["index"] for g in stats])}
            case ["runs", run_id]:
                return api.run_detail(root, run_id)
            case ["runs", run_id, "summary", block]:
                return api.summary_md(root, run_id, block)
            case ["runs", run_id, "activity"]:
                block = (query.get("block") or [None])[0]
                return api.activity(root, run_id, block, after())
            case ["runs", run_id, "frame", n] if _digits(n, 7):
                return api.frame(root, run_id, int(n))
            case ["tasks", name]:
                return api.task_detail(root, name)
            case ["tasks", name, "activity"]:
                return api.author_activity(root, name, after())
        return None

    def _events(self) -> None:
        """Server-sent events: what changed, and the GPUs, as they happen.

        A stream lasts at most `STREAM_LIFETIME_S`; the browser's EventSource
        reconnects on its own (after the `retry` it is told), so a page left
        open stays live while an abandoned connection does not hold a slot.
        """
        if not self.streams.acquire(blocking=False):
            self._error(503)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command == "HEAD":
                return
            self.wfile.write(b"retry: 3000\n\n")
            self.wfile.flush()
            last = api.fingerprint(self.root)
            beat = gpu_t = 0.0
            deadline = time.monotonic() + STREAM_LIFETIME_S
            while time.monotonic() < deadline:
                now = time.time()
                if now - gpu_t >= 2.0:
                    self._send("gpu", {"gpus": api.gpu_stats()})
                    gpu_t = now
                sig = api.fingerprint(self.root)
                changed = sorted(k for k in set(sig) | set(last) if sig.get(k) != last.get(k))
                if changed:
                    self._send("change", {"keys": changed})
                    last = sig
                if now - beat >= 15.0:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    beat = now
                time.sleep(1.0)
        except OSError:  # the client went away (or stalled past the timeout)
            pass
        except Exception:  # the headers are out: end the stream, do not answer again
            logger.exception("event stream failed")
        finally:
            self.streams.release()
            self.close_connection = True

    def _send(self, event: str, data) -> None:
        payload = json.dumps(_finite(data), allow_nan=False)
        self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode())
        self.wfile.flush()


def _same(given: str, token: str) -> bool:
    """Constant-time, and as bytes: a non-ASCII guess must not raise."""
    return secrets.compare_digest(given.encode("utf-8", "replace"),
                                  token.encode("utf-8", "replace"))


def _digits(raw: str, most: int) -> bool:
    return bool(raw) and len(raw) <= most and raw.isascii() and raw.isdigit()


def _finite(value):
    """JSON has no NaN or infinity; say None instead of emitting invalid JSON."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


# Windows' SO_REUSEADDR lets a second socket bind a port another process is
# already listening on, and requests then reach either of them: there a port
# is taken exclusively. Elsewhere SO_REUSEADDR only skips TIME_WAIT, as meant.
_BIND_OPT = (getattr(socket, "SO_EXCLUSIVEADDRUSE", None) if oscompat.WINDOWS
             else socket.SO_REUSEADDR)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = not oscompat.WINDOWS

    def server_bind(self) -> None:
        if oscompat.WINDOWS and _BIND_OPT is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, _BIND_OPT, 1)
        super().server_bind()


def make_server(root: Path, host: str, port: int, token: str | None) -> ThreadingHTTPServer:
    api.ROOT_REAL = os.path.realpath(root)
    wide = host in ("0.0.0.0", "::", "")
    handler = type("PinnforgeHandler", (Handler,), {
        "root": Path(root), "token": token, "hosts": frozenset(),
        "streams": threading.BoundedSemaphore(MAX_STREAMS),
    })
    server = _Server((host or "0.0.0.0", port), handler)
    server.daemon_threads = True
    handler.cookie = f"pf_token_{server.server_address[1]}"
    if not wide:
        # the Host a browser sends names the port actually bound (port 0 = any)
        bound = server.server_address[1]
        names = {"127.0.0.1", "localhost", "[::1]", host}
        handler.hosts = frozenset(f"{n}:{bound}" for n in names)
    return server


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve in the foreground until interrupted, recording where in ui.json."""
    token = None if host in LOOPBACK else secrets.token_urlsafe(24)
    server = make_server(root, host, port, token)
    port = server.server_address[1]
    record = state_file(root)
    _write_private(record, json.dumps({"pid": os.getpid(), "host": host, "port": port,
                                       "token": token, "version": __version__}) + "\n")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with contextlib.suppress(OSError):
            if (_read_record(record) or {}).get("pid") == os.getpid():
                record.unlink()


def _read_record(path: Path) -> dict | None:
    """`ui.json`, read without following a link or waiting on a FIFO."""
    raw = safe.read_bytes(path, 64 * 1024, check_path=False, single_link=False)
    try:
        info = json.loads(raw) if raw else None
    except (ValueError, RecursionError):
        return None
    return info if isinstance(info, dict) else None


def _write_private(path: Path, text: str) -> None:
    """Write `path` readable by its owner alone — from the first byte.

    A temporary file created 0600 (never wider, whatever the umask), then
    renamed over the old record: the token is never in a file anyone else
    could open, not even for the moment between a write and a chmod.
    """
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        oscompat.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def url_of(info: dict) -> str:
    host = info.get("host") or "127.0.0.1"
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    base = f"http://{shown}:{info['port']}/"
    return base + (f"?token={info['token']}" if info.get("token") else "")


def running(root: Path) -> dict | None:
    """The dashboard serving this workspace, if one answers."""
    info = _read_record(state_file(root))
    if info is None:
        return None
    if not oscompat.pid_alive(info.get("pid")):
        return None
    try:
        host = "127.0.0.1" if info.get("host") in ("0.0.0.0", "::", "") else info["host"]
        q = f"?token={info['token']}" if info.get("token") else ""
        with urllib.request.urlopen(f"http://{host}:{info['port']}/api/ping{q}", timeout=2) as r:
            if json.loads(r.read().decode()).get("app") == "pinnforge":
                return info
    except (OSError, ValueError):
        return None
    return None


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if _BIND_OPT is not None:
                s.setsockopt(socket.SOL_SOCKET, _BIND_OPT, 1)
            try:
                s.bind(("127.0.0.1" if host in LOOPBACK else "", port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {start}–{start + 19}")


def ensure_running(root: Path, host: str | None = None, port: int | None = None) -> str:
    """The dashboard's URL — starting it in the background if it is not up."""
    from pinnforge.run import lifecycle

    if (info := running(root)) is not None:
        return url_of(info)
    host = host or str(settings.get(root, "ui.host", "127.0.0.1"))
    port = _free_port(host, int(port or settings.get(root, "ui.port", 8765)))
    log = paths.state_dir(root) / "ui.log"
    with log.open("a", encoding="utf-8") as out:
        oscompat.popen_detached(
            lifecycle.self_command("--root", str(root), "ui", "--serve", "--host", host,
                                   "--port", str(port)),
            cwd=str(root), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
    deadline = time.time() + 10
    while time.time() < deadline:
        if (info := running(root)) is not None:
            return url_of(info)
        time.sleep(0.2)
    raise OSError(f"the dashboard did not come up; see {log}")


def stop(root: Path) -> bool:
    info = running(root)
    if info is None:
        return False
    return oscompat.request_stop(info["pid"])
