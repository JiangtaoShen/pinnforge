"""Operator settings, layered.

A run's configuration is assembled from, lowest precedence first:

1. built-in defaults (`config.py`);
2. the user file, `~/.config/pinnforge/config.yaml` — this machine, every
   workspace;
3. the workspace file, `<workspace>/pinnforge.yaml` — this project;
4. command-line flags;
5. `--set a.b=c` overrides, which win over everything, as they always have.

A key that is absent from a file, or `null` in it, is unset there and the
layer below decides: a new workspace's `pinnforge.yaml` leaves `agent`,
`sandbox.gpus`, `sandbox.wall_budget_s` and `private_dir` null, so the user
file's choices hold in every workspace until the workspace names its own.

The result is frozen into `runs/<id>/run.yaml` when a run is created, so
editing either file later never changes a run that already exists — the
same rule that makes a run's `task/` a copy.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_FILE = "pinnforge.yaml"


def user_dir() -> Path:
    """`$XDG_CONFIG_HOME/pinnforge`, else `~/.config/pinnforge`."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "pinnforge"


def read_yaml(path: Path) -> Any:
    """A YAML file's contents; a file that does not parse is a ValueError naming it.

    A typo in a settings file is the operator's to fix, so it must arrive as
    the file and the line rather than as a traceback from deep inside PyYAML.
    """
    path = Path(path)
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not UTF-8 text (byte {e.start})") from None
    except yaml.YAMLError as e:
        problem = getattr(e, "problem", None) or str(e).splitlines()[0]
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise ValueError(f"{path}: invalid YAML — {problem}{where}") from None


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = read_yaml(path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping, not {type(data).__name__}")
    return data


def user() -> dict[str, Any]:
    return _load(user_dir() / "config.yaml")


def workspace(root: Path) -> dict[str, Any]:
    return _load(Path(root) / WORKSPACE_FILE)


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` onto a copy of `base`; nested mappings merge, everything else replaces.

    None in `over` (YAML `null`) means "unset here": it leaves `base`'s value
    in place rather than erasing it, at any depth, and never appears in the
    result.
    """
    out = dict(base)
    for key, value in over.items():
        if value is None:
            continue
        if isinstance(value, dict):
            under = out.get(key)
            out[key] = deep_merge(under if isinstance(under, dict) else {}, value)
        else:
            out[key] = value
    return out


def merged(root: Path | None) -> dict[str, Any]:
    """User file overlaid by the workspace file, with every null left out."""
    layers = deep_merge({}, user())
    if root is not None:
        layers = deep_merge(layers, workspace(root))
    return layers


def get(root: Path | None, dotted: str, default: Any = None) -> Any:
    """One setting by dotted key, e.g. `get(root, "ui.port", 8765)`."""
    node: Any = merged(root)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def origin(root: Path | None, dotted: str) -> str | None:
    """Which file a setting comes from: the workspace file, the user file, or neither."""

    def has(data: dict[str, Any]) -> bool:
        node: Any = data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return node is not None

    if root is not None and has(workspace(root)):
        return WORKSPACE_FILE
    if has(user()):
        return str(user_dir() / "config.yaml")
    return None


def _flow(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, width=10_000).strip()
    return text.removesuffix("...").strip()


def _entry_indent(lines: list[str], start: int, end: int) -> int | None:
    """The indentation of the first entry in `lines[start:end]` — blanks and comments skipped."""
    for line in lines[start:end]:
        body = line.strip()
        if body and not body.startswith("#"):
            return len(line) - len(line.lstrip())
    return None


def _set_lines(lines: list[str], keys: list[str], value: Any) -> list[str] | None:
    """`lines` with `keys` set to `value` in place, or None if the shape defeats it.

    A key is looked for only at its parent's own child indentation — the
    indentation of the parent's first entry — so `sandbox.gpus` never lands
    on a `gpus:` nested deeper inside `sandbox`, and a key that is missing is
    added at that same indentation, however many spaces the file uses.
    """
    lines = list(lines)
    start, end, parent = 0, len(lines), None
    want, step = 0, 2  # the indentation a key at this depth has; the file's indent unit
    for depth, key in enumerate(keys):
        hit = None
        for i in range(start, end):
            m = re.match(r"^(\s*)" + re.escape(key) + r":(.*?)(\s+#.*)?$", lines[i].rstrip("\n"))
            if m and len(m.group(1)) == want:
                hit = (i, m)
                break
        if hit is None:
            # Not there: add it under the deepest key that is.
            at = end if parent is None else parent + 1
            if at and not lines[at - 1].endswith("\n"):
                lines[at - 1] += "\n"
            new = [f"{' ' * (want + step * n)}{k}:\n" for n, k in enumerate(keys[depth:-1])]
            new.append(f"{' ' * (want + step * (len(keys) - 1 - depth))}{keys[-1]}: "
                       f"{_flow(value)}\n")
            return lines[:at] + new + lines[at:]
        i, m = hit
        indent, parent = len(m.group(1)), i
        if depth == len(keys) - 1:
            head = f"{m.group(1)}{key}: {_flow(value)}"
            if comment := (m.group(3) or "").strip():
                column = lines[i].index(comment)  # keep the comment where it stood
                head = head.ljust(column - 1) + " " + comment
            lines[i] = head + "\n"
            return lines
        if m.group(2).strip():
            return None  # `key: value` where a mapping was expected
        start = i + 1
        for j in range(start, end):  # the block ends where the indentation does
            line = lines[j]
            if line.strip() and not line.lstrip().startswith("#") \
                    and len(line) - len(line.lstrip()) <= indent:
                end = j
                break
        child = _entry_indent(lines, start, end)
        if child is not None:
            step = child - indent
        want = indent + step
    return None


def _with(data: dict[str, Any], keys: list[str], value: Any) -> dict[str, Any]:
    """A copy of `data` with only `keys` set to `value` (parents made mappings)."""
    out = copy.deepcopy(data)
    node = out
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value
    return out


def write(path: Path, dotted: str, value: Any) -> None:
    """Set one setting in a settings file, keeping the file's comments.

    `pinnforge.yaml` is written for people — every key carries a comment — so
    the value is replaced on the key's own line, or a line is added under its
    parent. The edit is kept only if the whole file then parses to exactly
    what it held before with that one key changed; a file whose shape defeats
    that is re-dumped (or created).
    """
    path = Path(path)
    keys = dotted.split(".")
    expected = _with(_load(path), keys, value)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    edited = _set_lines(lines, keys, value) if lines else None
    if edited is not None:
        try:
            got = yaml.safe_load("".join(edited))
        except yaml.YAMLError:
            got = None
        if got == expected:
            path.write_text("".join(edited), encoding="utf-8")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")
