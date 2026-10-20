"""What an agent is doing right now, read out of its harness's own log.

Each harness writes its transcript differently — Claude Code and Cursor emit
stream-JSON, Codex and Pi emit JSON events of their own, OpenCode writes
logfmt — so each gets a small reader that turns lines into events of three
kinds:

    text   something the agent said
    tool   something it ran or touched (a command, a file it read or wrote)
    info   session boundaries: started (and on which model), finished, cost

Everything shown passes through `redact()` first — before it is clipped, so
a key cut in half by the clip is still masked. A transcript can contain
whatever an agent printed, and the dashboard can be opened by anyone with the
port — including, on the same machine, a block. Keys and tokens are masked.

A transcript is also something an agent can write to directly, so a reader
never trusts its shape: a line that is not what the harness would emit is
skipped, not an error.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pinnforge.ui import safe

MAX_TEXT = 400
READ_CAP = 256 * 1024  # never read more than this per request

_SECRETS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}"),
]


def redact(text: str) -> str:
    text = str(text)
    for pat in _SECRETS:
        text = pat.sub((lambda m: m.group(1) + "•••") if pat.groups else "•••", text)
    return text


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _tool_text(name: str, args: dict) -> str:
    if not isinstance(args, dict):
        return name
    for key in ("command", "cmd", "file_path", "path", "pattern", "url", "description"):
        if args.get(key):
            return f"{name}: {args[key]}"
    return name


def _content(obj: dict) -> list:
    msg = obj.get("message")
    parts = msg.get("content") if isinstance(msg, dict) else None
    return parts if isinstance(parts, list) else []


def _claude(obj: dict) -> list[dict]:
    kind = obj.get("type")
    if kind == "system" and obj.get("subtype") == "init":
        return [{"kind": "info", "text": f"session started · {obj.get('model', '?')}"}]
    if kind == "result":
        cost = obj.get("total_cost_usd")
        extra = (f" · ${cost:.2f}"
                 if isinstance(cost, (int, float)) and not isinstance(cost, bool) else "")
        return [{"kind": "info",
                 "text": f"session finished · {obj.get('num_turns', '?')} turns{extra}"}]
    out = []
    if kind == "assistant":
        for part in _content(obj):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if part.get("type") == "text" and isinstance(text, str) and text.strip():
                out.append({"kind": "text", "text": text})
            elif part.get("type") == "tool_use":
                out.append({"kind": "tool",
                            "text": _tool_text(str(part.get("name", "tool")),
                                               part.get("input", {}))})
    return out


def _codex(obj: dict) -> list[dict]:
    item = obj.get("item") if isinstance(obj.get("item"), dict) else obj
    t = str(item.get("type", obj.get("type", "")))
    if "command" in item and item.get("command"):
        cmd = item["command"]
        return [{"kind": "tool",
                 "text": " ".join(map(str, cmd)) if isinstance(cmd, list) else str(cmd)}]
    for key in ("text", "message", "msg"):
        val = item.get(key)
        if isinstance(val, str) and val.strip() and "delta" not in t:
            return [{"kind": "text", "text": val}]
    if t in ("thread.started", "session.created"):
        return [{"kind": "info", "text": "session started"}]
    return []


def _pi(obj: dict) -> list[dict]:
    """Pi's `--mode json` events.

    A reply is read once, from its `message_end` (`turn_end` and `agent_end`
    repeat it, and the adapter keeps both out of the log); a tool call from
    `tool_execution_start`, which carries its arguments; the end from
    `agent_settled`, after which Pi does nothing more on its own.
    """
    kind = obj.get("type")
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    if kind == "session":
        return [{"kind": "info", "text": "session started"}]
    if kind == "tool_execution_start":
        return [{"kind": "tool", "text": _tool_text(str(obj.get("toolName", "tool")),
                                                    obj.get("args", {}))}]
    if kind == "message_end" and msg.get("role") == "assistant":
        out = [{"kind": "text", "text": part["text"]} for part in msg.get("content") or []
               if isinstance(part, dict) and part.get("type") == "text"
               and isinstance(part.get("text"), str) and part["text"].strip()]
        if msg.get("stopReason") in ("error", "aborted"):
            why = msg.get("errorMessage") or msg["stopReason"]
            out.append({"kind": "info", "text": f"reply failed · {why}"})
        return out
    if kind == "auto_retry_start":
        return [{"kind": "info", "text": f"retrying · {obj.get('errorMessage', '')}"}]
    if kind == "compaction_start":
        return [{"kind": "info", "text": "compacting the context"}]
    if kind == "agent_settled":
        return [{"kind": "info",
                 "text": "session aborted" if obj.get("aborted") else "session finished"}]
    return []


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RESET = "\x1b[0m"


def _opencode(lines: list[str]) -> list[dict]:
    """OpenCode's default output: logfmt logs, markers, text and command output.

    `→ Read file` and `$ command` lines are tool calls; `> build · model` names
    the agent and model; a bare ANSI reset separates segments. Lines after a
    `$` are that command's output until the next segment, and are skipped.
    """
    out: list[dict] = []
    in_output = False
    for raw in lines:
        if raw.startswith("timestamp="):
            if " level=ERROR " in raw:
                out.append({"kind": "info", "text": raw.split("message=", 1)[-1]})
            continue
        marked = raw.startswith(_RESET)
        text = _ANSI.sub("", raw).strip()
        if marked and not text:
            in_output = False
            continue
        if text.startswith("→ "):
            out.append({"kind": "tool", "text": text[2:]})
            in_output = False
        elif text.startswith("$ "):
            out.append({"kind": "tool", "text": text[2:]})
            in_output = True
        elif text.startswith("> ") and "·" in text:
            out.append({"kind": "info", "text": f"session · {text[2:]}"})
        elif text and not in_output:
            out.append({"kind": "text", "text": text})
    return out


def parse(runtime: str, lines: list[str], base: str | None = None) -> list[dict]:
    reader = {"claude_code": _claude, "cursor_agent": _claude, "replay": _claude,
              "codex": _codex, "pi": _pi}.get(runtime)
    events: list[dict] = []
    if reader is None:
        events = _opencode(lines)
        lines = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            try:
                events += reader(obj)
            except Exception:  # an agent can write any line; one odd line is skipped
                continue

    def short(text: str) -> str:
        # paths inside the run read better relative to it, as the agent sees them
        return text.replace(base.rstrip("/") + "/", "") if base else text

    # redact first: a clip could cut a key short enough to escape its pattern
    return [{"kind": e["kind"], "text": _clip(redact(short(str(e["text"]))))} for e in events]


def tail(path: Path, after: int | None, runtime: str,
         base: str | None = None, root_real: str | None = None) -> tuple[list[dict], int]:
    """Events from complete lines appended to `path` since byte offset `after`.

    `after=None` reads the last stretch of the log. Returns the events and the
    offset to ask from next time. The log is opened like every other file the
    dashboard reads (`safe.open_regular`): not through a link, not a FIFO,
    not a hard link, and only inside `root_real` and outside the answers.
    """
    fd = safe.open_regular(path, root_real)
    if fd is None:
        return [], after or 0
    with os.fdopen(fd, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        want = max(0, size - READ_CAP) if after is None else min(max(0, after), size)
        start = max(want, size - READ_CAP)
        fh.seek(start)
        data = fh.read(min(size - start, READ_CAP))
    if start != (after or 0) and start > 0:
        # entered mid-log: the first line is a fragment
        nl = data.find(b"\n")
        data, start = (data[nl + 1 :], start + nl + 1) if nl >= 0 else (b"", size)
    last = data.rfind(b"\n")
    if last < 0:
        return [], start
    lines = data[:last].decode("utf-8", errors="replace").splitlines()
    return parse(runtime, lines, base)[-200:], start + last + 1
