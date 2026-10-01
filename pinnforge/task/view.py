"""`view.json` — what the dashboard draws in a run's predicted-field window.

The window itself is fixed: same place, same size, for every task. What goes
in it is the task's call, because tasks differ in exactly the ways that
decide it: a 1-D steady problem is a line, an (x, t) problem a space-time map,
a (t, x, y) flow a map with a time slider, a 3-D field a slice; and the output
heads differ too — one scalar, a velocity pair best seen as a speed, several
quantities side by side. The task author writes this file alongside the
package, since it knows the problem; the framework lays the panels out in the
window and draws them.

It is declarative and small on purpose. Nothing in it runs: the dashboard
reads it from a run directory a block can write to, so it is validated
against the task's own coordinates and output heads and anything else is
ignored. And it only ever shows predictions — derived quantities are computed
from the predicted fields, never from the reference.

    {
      "axes":   {"x": "x", "y": "t"},      # grid coordinates on the screen axes;
                                           # no "y" draws each panel as a line
      "time":   "t",                       # optional: a coordinate on a slider
      "slice":  {"axis": "z", "at": 0.5},  # optional: a coordinate held fixed
      "panels": [                          # 1-4, laid out to fill the window
        {"title": "speed |V|", "show": {"magnitude": ["u", "v"]},
         "colors": "sequential"},
        {"title": "v", "show": "v", "colors": "diverging"}
      ]
    }

Every grid coordinate (the first INPUT_COLUMNS columns of eval_grid.csv, at
most three) is used exactly once: on an axis, on the time slider, or as the
slice. `show` is an output head, `{"magnitude": [heads]}` or `{"abs": head}`.
`colors` is "auto", "sequential" (one hue, low to high) or "diverging" (two
hues around zero).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

VIEW_FILE = "view.json"
MAX_BYTES = 16_384
MAX_PANELS = 4
MAX_TITLE = 40
COLORS = ("auto", "sequential", "diverging")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")


def load(path: Path) -> object | None:
    """The parsed file, or None if it is missing, oversized or not JSON."""
    try:
        raw = Path(path).read_bytes()[: MAX_BYTES + 1]
    except OSError:
        return None
    if len(raw) > MAX_BYTES:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _show(show, fields: list[str]) -> tuple[dict | None, str | None]:
    """A panel's `show`, normalised to {"op", "of"}, or why it is not valid."""
    if isinstance(show, str):
        return ({"op": "field", "of": [show]}, None) if show in fields else (
            None, f"{show!r} is not an output head ({', '.join(fields)})")
    if isinstance(show, dict) and len(show) == 1:
        (op, arg), = show.items()
        if op == "abs" and isinstance(arg, str) and arg in fields:
            return {"op": "abs", "of": [arg]}, None
        if op == "magnitude" and isinstance(arg, list) and 2 <= len(arg) <= 3 \
                and all(isinstance(a, str) and a in fields for a in arg) \
                and len(set(arg)) == len(arg):
            return {"op": "magnitude", "of": list(arg)}, None
    return None, ('`show` must be an output head, {"magnitude": [2-3 heads]} '
                  'or {"abs": head}')


def check(spec, coords: list[str], fields: list[str]) -> tuple[dict | None, list[str]]:
    """Validate a view against the task's grid coordinates and output heads.

    Returns the normalised view (None if anything is wrong) and the problems.
    """
    problems: list[str] = []
    if not isinstance(spec, dict):
        return None, ["view.json must hold a JSON object"]
    unknown = sorted(set(spec) - {"axes", "time", "slice", "panels"})
    if unknown:
        problems.append(f"unknown keys: {', '.join(unknown)}")
    coords = [c for c in coords if isinstance(c, str)][:3]

    axes = spec.get("axes")
    if not isinstance(axes, dict) or not isinstance(axes.get("x"), str) \
            or set(axes) - {"x", "y"}:
        problems.append('`axes` must be {"x": <coordinate>} or {"x": ..., "y": ...}')
        axes = {}
    used = [axes.get("x"), axes.get("y")]
    time = spec.get("time")
    if time is not None and not isinstance(time, str):
        problems.append("`time` must name a coordinate")
        time = None
    used.append(time)
    sl = spec.get("slice")
    at = 0.5
    if sl is not None:
        if not isinstance(sl, dict) or not isinstance(sl.get("axis"), str) \
                or set(sl) - {"axis", "at"}:
            problems.append('`slice` must be {"axis": <coordinate>, "at": 0..1}')
            sl = None
        else:
            raw = sl.get("at", 0.5)
            ok = isinstance(raw, (int, float)) and not isinstance(raw, bool) \
                and math.isfinite(raw) and 0 <= raw <= 1
            if not ok:
                problems.append("`slice.at` must be a number from 0 to 1")
            at = float(raw) if ok else 0.5
    used.append(sl.get("axis") if isinstance(sl, dict) else None)
    named = [u for u in used if u is not None]
    for u in named:
        if u not in coords:
            problems.append(f"{u!r} is not a grid coordinate ({', '.join(coords)})")
    if len(named) != len(set(named)):
        problems.append("a coordinate is used twice")
    missing = [c for c in coords if c not in named]
    if missing:
        problems.append(f"coordinate(s) {', '.join(missing)} are on no axis, slider or slice")

    panels = spec.get("panels")
    out_panels = []
    if not isinstance(panels, list) or not 1 <= len(panels) <= MAX_PANELS:
        problems.append(f"`panels` must list 1-{MAX_PANELS} panels")
        panels = []
    for k, p in enumerate(panels, 1):
        if not isinstance(p, dict) or set(p) - {"title", "show", "colors"}:
            problems.append(f"panel {k}: keys are title, show, colors")
            continue
        show, why = _show(p.get("show"), fields)
        if why:
            problems.append(f"panel {k}: {why}")
        title = p.get("title", "")
        if not isinstance(title, str) or len(title) > MAX_TITLE or any(ord(c) < 32 for c in title):
            problems.append(f"panel {k}: `title` must be plain text, at most {MAX_TITLE} chars")
            title = ""
        colors = p.get("colors", "auto")
        if colors not in COLORS:
            problems.append(f"panel {k}: `colors` must be one of {', '.join(COLORS)}")
        if show:
            out_panels.append({"title": title or _label(show), "show": show,
                               "colors": colors if colors in COLORS else "auto"})
    if problems:
        return None, problems
    view = {"axes": {k: axes[k] for k in ("x", "y") if k in axes}, "panels": out_panels}
    if time:
        view["time"] = time
    if sl:
        view["slice"] = {"axis": sl["axis"], "at": at}
    return view, []


def _label(show: dict) -> str:
    if show["op"] == "magnitude":
        return "|" + ", ".join(show["of"]) + "|"
    if show["op"] == "abs":
        return f"|{show['of'][0]}|"
    return show["of"][0]


def task_axes(task_dir: Path) -> tuple[list[str], list[str]]:
    """(grid coordinates, output heads) of a task package, read without running it."""
    import ast

    task_dir = Path(task_dir)
    coords: list[str] = []
    fields: list[str] = []
    try:
        src = (task_dir / "eval.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("INPUT_COLUMNS", "FIELDS"):
                        values[t.id] = ast.literal_eval(node.value)
        n_in = int(values.get("INPUT_COLUMNS", 0))
        fields = [str(f) for f in values.get("FIELDS", ())]
        with (task_dir / "eval_grid.csv").open(encoding="utf-8") as fh:
            header = [h.strip() for h in fh.readline().split(",")]
        coords = header[: min(n_in, 3)]
    except (OSError, SyntaxError, ValueError, TypeError):
        pass
    return coords, fields
