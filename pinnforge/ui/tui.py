"""`pinnforge run watch` — the dashboard's essentials in a terminal.

For the machine you reach over SSH without a tunnel: the run's best score
against its anchor, every block's budget and best, the latest evaluations and
what the agent is doing, refreshed every two seconds. `q` quits.

Much of what it prints was written by agents, so control characters are
stripped first: a transcript must not be able to drive the terminal (retitle
it, rewrite the clipboard, hide lines) with escape sequences.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from pinnforge.ui import api

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _plain(text: str) -> str:
    return _CONTROL.sub("", text)


def _fmt(x) -> str:
    if x is None:
        return "—"
    return f"{x:.4g}" if abs(x) >= 1e-3 else f"{x:.2e}"


def _bar(frac: float, width: int = 20) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "░" * (width - n)


def lines(root: Path, run_id: str, width: int = 100) -> list[tuple[str, str]]:
    """(style, text) rows: style is '' / 'head' / 'good' / 'dim' / 'bad'."""
    d = api.run_detail(root, run_id)
    out: list[tuple[str, str]] = []
    live = "live" if d["live"] else "not running"
    agent = d["config"]["agent"]
    out.append(("head", f"PINNForge · {d['id']} · {d['task']} · {d['status']} ({live}) · "
                        f"{agent.get('runtime')} / {agent.get('model') or 'default'} · "
                        f"GPU {d['config']['gpus'] or 'CPU'}"))
    ratio = d["anchor"] / d["best"] if d["best"] and d["anchor"] else None
    better = f"  ×{ratio:.1f} vs anchor" if ratio and ratio > 1 else ""
    spent = sum(b["spent"] or 0 for b in d["blocks"] if b["id"] != "b00")
    out.append(("good" if better else "", f"best {_fmt(d['best'])} ({d['best_block'] or '—'})   "
                                         f"anchor {_fmt(d['anchor'])}{better}   "
                                         f"evals {len(d['evals'])}   GPU {spent / 60:.0f} min"))
    out.append(("", ""))
    out.append(("head", "blocks"))
    for b in d["blocks"]:
        if b["id"] == "b00":
            continue
        frac = (b["spent"] or 0) / b["budget"] if b["budget"] else 0
        style = {"done": "good", "failed": "bad", "running": ""}.get(b["status"], "dim")
        out.append((style, f" {b['id']:<4} {b['status']:<11} {_bar(frac)} "
                           f"{b['spent'] or 0:>5.0f}/{b['budget'] or 0:<5.0f} "
                           f"{b['evals']:>3} evals  best {_fmt(b['best'])}"))
    out.append(("", ""))
    out.append(("head", "latest evaluations"))
    for e in d["evals"][-6:][::-1]:
        err = f"  ✕ {e['error'][:60]}" if e["error"] else ""
        out.append(("bad" if err else "", f" {_fmt(e['rRMSE']):>10}  {e['block']:<4} "
                                          f"{e['candidate'][:40]:<40}{err}"))
    act = api.activity(root, run_id, None, None)
    if act["events"]:
        out.append(("", ""))
        out.append(("head", f"agent ({act['block']})"))
        marks = {"tool": "$", "text": "›", "info": "•"}
        for ev in act["events"][-8:]:
            out.append(("dim" if ev["kind"] == "info" else "",
                        f" {marks.get(ev['kind'], '·')} {_plain(ev['text'])}"[:width]))
    return [(style, _plain(text)) for style, text in out]


def watch(root: Path, run_id: str) -> int:
    try:
        import curses
    except ImportError:  # Windows' Python ships without it
        return _watch_plain(root, run_id)

    def loop(scr) -> None:
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        styles = {"head": curses.color_pair(1) | curses.A_BOLD, "good": curses.color_pair(2),
                  "bad": curses.color_pair(3), "dim": curses.A_DIM, "": 0}
        scr.timeout(2000)
        while True:
            h, w = scr.getmaxyx()
            scr.erase()
            try:
                rows = lines(root, run_id, w - 1)
            except api.NotFound:
                rows = [("bad", f"no run {run_id}")]
            for y, (style, text) in enumerate(rows[: h - 2]):
                scr.addnstr(y, 0, text, w - 1, styles.get(style, 0))
            scr.addnstr(h - 1, 0, f"q quit · refreshed {time.strftime('%H:%M:%S')}", w - 1,
                        curses.A_DIM)
            scr.refresh()
            if scr.getch() in (ord("q"), ord("Q"), 27):
                return

    curses.wrapper(loop)
    return 0


def _watch_plain(root: Path, run_id: str) -> int:
    """`watch` without curses: the same lines, redrawn every two seconds; q or Ctrl-C quits."""
    import os
    import shutil

    keys = None
    if os.name == "nt":
        import msvcrt as keys

        os.system("")  # lets the console take the escape codes below
    try:
        while True:
            width, height = shutil.get_terminal_size()
            try:
                rows = lines(root, run_id, width - 1)
            except api.NotFound:
                rows = [("bad", f"no run {run_id}")]
            text = "\n".join(t[: width - 1] for _, t in rows[: height - 2])
            print(f"\x1b[H\x1b[2J{text}\n\nq quit · refreshed {time.strftime('%H:%M:%S')}",
                  end="", flush=True)
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if keys is not None and keys.kbhit() and keys.getwch() in ("q", "Q", "\x1b"):
                    return 0
                time.sleep(0.1)
    except KeyboardInterrupt:
        return 0
    finally:
        print()
