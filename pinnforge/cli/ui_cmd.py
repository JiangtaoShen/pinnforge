"""`pinnforge ui` — open (or start) the live dashboard."""

from __future__ import annotations

import os
import signal

from pinnforge import oscompat, paths, settings


def cmd_ui(args) -> int:
    from pinnforge.ui import server

    root = paths.project_root()
    if args.stop:
        print("dashboard stopped" if server.stop(root) else "no dashboard running")
        return 0
    if args.serve:
        host = args.host or str(settings.get(root, "ui.host", "127.0.0.1"))
        port = int(args.port or settings.get(root, "ui.port", 8765))
        # SIGTERM from `pinnforge ui --stop` ends the loop like Ctrl-C (on Windows
        # it arrives as Ctrl-C itself).
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        oscompat.listen_for_stop()
        server.serve(root, host, port)
        return 0
    url = server.ensure_running(root, args.host, args.port)
    print(f"dashboard: {url}")
    info = server.running(root) or {}
    if info.get("host") in server.LOOPBACK and os.environ.get("SSH_CONNECTION"):
        print(f"  over SSH: ssh -L {info['port']}:127.0.0.1:{info['port']} <this host>, "
              "then open the URL above on your machine")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    return 0


def register(sub) -> None:
    p = sub.add_parser("ui", help="the live dashboard (starts it if needed)")
    p.add_argument("--host", default=None, help="interface to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="default 8765")
    p.add_argument("--open", action="store_true", help="open it in a browser")
    p.add_argument("--stop", action="store_true", help="stop this workspace's dashboard")
    p.add_argument("--serve", action="store_true", help="serve in the foreground")
    p.set_defaults(func=cmd_ui)
