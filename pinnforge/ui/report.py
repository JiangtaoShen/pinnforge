"""`pinnforge run report` — one run as a single, self-contained HTML file.

The dashboard's own page, with the answers to every request it would make
embedded in it: the run's detail, every block summary, the agents' activity,
every archived prediction frame. Opened from disk, attached to an e-mail or
dropped into a chat, it renders exactly as the live page did — charts, field
animation and all — with no server behind it.

Everything embedded went through the same read-only API the live dashboard
uses, so a report carries nothing the dashboard would not show: never the
reference, never an error field.
"""

from __future__ import annotations

import contextlib
import html
import json
from pathlib import Path

from pinnforge.ui import api

STATIC = Path(__file__).resolve().parent / "static"

SHIM = """
window.__PF_STATIC__ = true;
(function () {
  var snap = window.__PF_SNAPSHOT__;
  function key(u) {
    var url = new URL(u, "http://report.local");
    var block = url.searchParams.get("block");
    return url.pathname + (block ? "?block=" + block : "");
  }
  window.fetch = function (u) {
    var url = new URL(u, "http://report.local");
    var k = key(u);
    var body = snap[k];
    if (body !== undefined && url.searchParams.get("after") !== null && body.events) {
      body = { block: body.block, log: body.log, events: [], offset: body.offset };
    }
    return Promise.resolve({
      ok: body !== undefined, status: body === undefined ? 404 : 200,
      json: function () { return Promise.resolve(body); },
    });
  };
  window.EventSource = function () {
    this.addEventListener = function () {};
    this.close = function () {};
  };
  if (!location.hash) location.hash = "#/run/" + encodeURIComponent(window.__PF_RUN__);
})();
"""


def snapshot(root: Path, run_id: str) -> dict:
    api.ROOT_REAL = str(Path(root).resolve())
    detail = api.run_detail(root, run_id)
    snap: dict[str, object] = {
        "/api/ping": {"app": "pinnforge", "workspace": Path(root).name},
        f"/api/runs/{run_id}": detail,
        "/api/gpus": {"gpus": [], "leases": {}},
    }
    overview = api.overview(root)
    overview["runs"] = [r for r in overview["runs"] if r["id"] == run_id]
    overview["gpus"], overview["leases"] = [], {}
    snap["/api/overview"] = overview
    for b in detail["blocks"]:
        if b["summary"]:
            with contextlib.suppress(api.NotFound):
                snap[f"/api/runs/{run_id}/summary/{b['id']}"] = api.summary_md(root, run_id,
                                                                               b["id"])
        snap[f"/api/runs/{run_id}/activity?block={b['id']}"] = api.activity(root, run_id,
                                                                              b["id"], None)
    for fr in detail["frames"]:
        try:
            snap[f"/api/runs/{run_id}/frame/{fr['frame']}"] = api.frame(root, run_id, fr["frame"])
        except api.NotFound:
            continue
    return snap


def script_json(value) -> str:
    """JSON safe to place inside a <script> element.

    `<`, `>` and `&` become `\\u003c`, `\\u003e` and `\\u0026` — the same string
    to JavaScript, but nothing an HTML parser could take for `</script>`, a
    comment opener or an entity; U+2028/U+2029 are escaped for older parsers.
    """
    from pinnforge.ui.server import _finite

    text = json.dumps(_finite(value), allow_nan=False, default=str)
    for ch, esc in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                    ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(ch, esc)
    return text


def render(root: Path, run_id: str) -> str:
    snap = snapshot(root, run_id)
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    body = index.split("<body>", 1)[1].split("</body>", 1)[0]
    body = body.replace('<script src="app.js" defer></script>', "")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    data = script_json(snap)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>PINNForge — {html.escape(run_id)}</title>\n<style>\n{css}\n</style>\n"
        f"</head>\n<body>\n{body}\n<script>window.__PF_RUN__ = {script_json(run_id)};\n"
        f"window.__PF_SNAPSHOT__ = {data};\n{SHIM}</script>\n<script>\n{js}\n</script>\n"
        "</body>\n</html>\n"
    )
