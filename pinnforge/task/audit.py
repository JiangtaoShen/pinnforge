"""What an authoring agent reached for, and what a harness kept of the ones that saw the answers.

The author agent writes the package a block will be handed, and must never
see the answers. Agents run as the operator's own user, so nothing walls them
off; `audit()` reads what the agent actually asked for — every tool call in
its transcript, and the output that came back — and says whether any of it
touched:

* a task's private directory — any task's, in this workspace or another —
  the configured external private directory, or the copy of this task's
  answers sealed away while the author works;
* the operator's reference data, wherever it came from;
* a harness's session store, which holds the transcripts of the agents that
  did see the answers;
* another task's authoring record;
* the answers by what they are called (`ref_values.npy`), or `.npy` files
  searched for across directories that hold answers.

Paths are judged by where they lead: every path-like token of a call is
resolved against the agent's working directory, so `cd private;`,
`../ldc/private;python`, `glob('**/ref_values.npy')` and `ls ../*/priv*`
count, while macOS's `/private/tmp`, a workspace that happens to live under a
directory named `private`, or the data's file name in a web search do not. A
plain listing that shows `private/` among its entries is not a reach for it;
a recursive grep that printed lines out of one is.

On Windows a path is compared the way Windows reads it: `\\` and `/` alike,
case-blind (`PRIVATE` is `private`), a name's trailing dots dropped; Git
Bash's `/c/Users` is `C:/Users`, and the `\\\\?\\` prefix, an admin share of
this machine and an NTFS stream suffix lead where the plain path does. The
home directory is found however a shell spells it — `~`, `$HOME`,
`$env:USERPROFILE`, `%USERPROFILE%` — and PowerShell's and cmd.exe's ways
of searching and copying a tree count as `find` and `cp -r` do.

The public record of a hit is its category only (`Hit.category`); the call
that made it (`Hit.detail`) can name the operator's data, so it is logged
privately by the caller.

`sweep_sessions()` moves what a harness kept of an agent's session — the
transcript of an agent that saw the answers — out of the harness's store and
into the task's private directory.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import platform
import re
import shutil
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from pinnforge import paths, private

_WINDOWS = os.name == "nt"

PRIVATE = "a private directory"
SEALED = "the sealed answers"
OPERATOR = "the operator's data"
SESSIONS = "a harness session store"
AUTHORING = "another task's authoring record"
HUNT = "a search for the answers"
STRAY = "created private/"

AUTHORING_DIR = ".authoring"
PRIVATE_NAME = "private"
# What a task package is made of; the operator's data is never matched by one
# of these names, however it is called.
_PACKAGE_NAMES = {"problem.md", "baseline.py", "eval.py", "eval_grid.csv", "_forgekit.py"}


@dataclass(frozen=True)
class Hit:
    category: str  # what may be published
    detail: str  # the call itself: private


@dataclass
class Call:
    """One tool call, reduced to what can lead somewhere."""

    tool: str
    shell: str = ""  # a command line, or code, read for paths
    paths: list[str] = field(default_factory=list)  # explicit path arguments
    globs: list[tuple[str, str]] = field(default_factory=list)  # (base directory, pattern)
    write: bool = False  # writes `paths`; what it writes is the agent's own text
    search: bool = False  # a content search: the files it lists matched its pattern
    cwd: str = ""  # where it ran, if not the agent's working directory
    result: str = ""  # what came back
    raw: str = ""  # the call as the transcript has it, for the private log
    inert: bool = False  # touches no files (a web search): only its URL can matter

    def describe(self) -> str:
        if self.shell:
            return f"{self.tool}: {self.shell.strip()}"
        bits = list(self.paths) + [f"{b}/{g}" if b else g for b, g in self.globs]
        return f"{self.tool}: {' '.join(bits)}" if bits else (self.raw or self.tool)


# ─────────────────────────────── transcripts ───────────────────────────────

_SKIP = {"todowrite", "todoread", "websearch", "webfetch", "task", "agent", "exitplanmode",
         "killshell", "bashoutput", "updatetodos", "todo", "searchweb", "fetch"}
_WRITE_TOOLS = {"write", "edit", "multiedit", "notebookedit", "delete", "deletefile", "create",
                "createfile", "strreplace", "strreplacebasededit", "applypatch", "writefile",
                "editfile", "searchreplace", "patch"}
_SEARCH_TOOLS = {"grep", "ripgrep", "rg", "search", "codebasesearch", "grepsearch"}
_GLOB_TOOLS = {"glob", "globsearch", "filesearch", "listfiles", "find"}
_PATH_KEYS = ("file_path", "notebook_path", "path", "paths", "filePath", "file", "files",
              "directory", "dir", "target_directory", "targetDirectory", "target", "absolutePath",
              "relative_workspace_path", "target_file", "targetFile")
_GLOB_KEYS = ("glob", "globPattern", "glob_pattern", "include", "includePattern")
_SHELL_KEYS = ("command", "cmd", "script", "code")
_BASE_KEYS = ("path", "targetDirectory", "target_directory", "directory", "dir", "cwd", "workdir")


def _strs(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if isinstance(x, (str, int, float))]
    return []


def _flat(v) -> str:
    """Every string inside a JSON value, one per line — a tool's output, however nested."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return "\n".join(_flat(x) for x in v.values())
    if isinstance(v, list):
        return "\n".join(_flat(x) for x in v)
    return ""


def _norm_tool(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def from_args(tool: str, args) -> Call:
    """A call from a tool name and its JSON arguments."""
    low = _norm_tool(tool)
    call = Call(tool=str(tool), raw=f"{tool} {json.dumps(args, default=str)[:2000]}")
    if low in _SKIP or low.startswith("todo"):
        call.inert = True
        return call
    if not isinstance(args, dict):
        return call
    call.write = low in _WRITE_TOOLS
    call.search = low in _SEARCH_TOOLS
    base = next((str(args[k]) for k in _BASE_KEYS if isinstance(args.get(k), str)), "")
    if low in ("shell", "bash", "exec", "run", "terminal", "runterminalcmd", "localshell",
               "powershell", "pwsh"):
        call.cwd = str(args.get("cwd") or args.get("workdir") or args.get("working_directory")
                       or "")
    for key, value in args.items():
        if key in _SHELL_KEYS and value:
            call.shell += " " + (value if isinstance(value, str) else " ".join(_strs(value)))
        elif key in _PATH_KEYS:
            call.paths += _strs(value)
        elif key in _GLOB_KEYS or (key == "pattern" and low in _GLOB_TOOLS):
            call.globs += [(base, g) for g in _strs(value)]
    return call


def _claude(lines: list[str]) -> list[Call]:
    """Claude Code (and anything speaking its stream-json): tool_use, and tool_result."""
    calls: list[Call] = []
    by_id: dict[str, Call] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        content = (obj.get("message") or {}).get("content") if isinstance(
            obj.get("message"), dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                call = from_args(part.get("name", "tool"), part.get("input") or {})
                calls.append(call)
                if part.get("id"):
                    by_id[str(part["id"])] = call
            elif part.get("type") == "tool_result":
                call = by_id.get(str(part.get("tool_use_id")))
                if call is not None:
                    call.result += "\n" + _flat(part.get("content"))
    return calls


def _cursor(lines: list[str]) -> list[Call]:
    """Cursor Agent's stream-json: top-level `tool_call` events, started and completed."""
    calls = _claude(lines)  # its assistant messages, if a version emits Claude-style ones
    by_id: dict[str, Call] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "tool_call":
            continue
        tc = obj.get("tool_call")
        if not isinstance(tc, dict):
            continue
        for key, body in tc.items():
            if not isinstance(body, dict):
                continue
            name = key[: -len("ToolCall")] if key.endswith("ToolCall") else key
            cid = str(obj.get("call_id") or obj.get("callId") or id(body))
            call = by_id.get(cid)
            if call is None:
                args = body.get("args") if isinstance(body.get("args"), dict) else {
                    k: v for k, v in body.items() if k != "result"}
                call = from_args(name, args)
                by_id[cid] = call
                calls.append(call)
            if "result" in body:
                call.result += "\n" + _flat(body.get("result"))
    return calls


def _codex(lines: list[str]) -> list[Call]:
    """Codex's JSON events: `item.*` (current) and `msg` (older) shapes."""
    calls: list[Call] = []
    by_id: dict[str, Call] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        item = obj.get("item") if isinstance(obj.get("item"), dict) else (
            obj.get("msg") if isinstance(obj.get("msg"), dict) else obj)
        kind = str(item.get("type", ""))
        cid = str(item.get("id") or item.get("call_id") or "")
        call = by_id.get(cid) if cid else None
        if item.get("command"):
            if call is None:
                cmd = item["command"]
                call = Call(tool="shell", shell=" ".join(_strs(cmd)) if isinstance(cmd, list)
                            else str(cmd), cwd=str(item.get("cwd") or ""),
                            raw=json.dumps(item, default=str)[:2000])
                calls.append(call)
                if cid:
                    by_id[cid] = call
        elif kind in ("file_change", "patch_apply_begin", "patch_apply_end") or "changes" in item:
            changes = item.get("changes")
            files = list(changes) if isinstance(changes, dict) else [
                c.get("path") for c in changes or [] if isinstance(c, dict)]
            calls.append(Call(tool="apply_patch", paths=[str(f) for f in files if f],
                              write=True))
            continue
        elif kind in ("mcp_tool_call", "function_call", "custom_tool_call"):
            args = item.get("arguments")
            if isinstance(args, str):
                with contextlib.suppress(json.JSONDecodeError):
                    args = json.loads(args)
            calls.append(from_args(str(item.get("tool") or item.get("name") or "tool"),
                                   args or {}))
            continue
        elif kind in ("web_search", "web_search_call"):
            calls.append(Call(tool="web_search", raw=json.dumps(item, default=str)[:2000],
                              inert=True))
            continue
        if call is not None:
            out = [item.get(k) for k in ("aggregated_output", "output", "stdout", "stderr",
                                         "formatted_output")]
            call.result += "\n" + "\n".join(o for o in out if isinstance(o, str))
    return calls


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_RESET = "\x1b[0m"
_OC_MARK = re.compile(r"^(→|←|✱|✗|\$|%|⚙|◇|◈|•|⇆|\|)\s+(.*)$")
_OC_SEARCH = re.compile(r'^(Grep|Glob)\s+"(.*)"(?:\s+in\s+(.+?))?(?:\s+·.*)?$')


def _opencode(lines: list[str]) -> list[Call]:
    """OpenCode's default output: marker lines are tool calls, `$` output is their result.

    `→ Read path`, `← Write path`, `✱ Grep "pattern" in path · N matches`,
    `✱ Glob "pattern" in path`, `$ command` (followed by its output until the
    next segment), `✗ Read path failed` (an attempt is still an attempt).
    """
    calls: list[Call] = []
    current: Call | None = None
    for raw in lines:
        if raw.startswith("timestamp="):
            continue
        reset = raw.startswith(_RESET)
        text = _ANSI.sub("", raw).strip()
        if reset and not text:
            current = None
            continue
        m = _OC_MARK.match(text)
        if not m:
            if current is not None:
                current.result += "\n" + text
            continue
        mark, rest = m.groups()
        current = None
        if mark == "$":
            current = Call(tool="bash", shell=rest, raw=text)
            calls.append(current)
            continue
        if mark == "✗" and rest.endswith(" failed"):
            rest = rest[: -len(" failed")]
        s = _OC_SEARCH.match(rest)
        if s:
            tool, pattern, where = s.group(1), s.group(2), (s.group(3) or "").strip()
            if tool == "Grep":
                calls.append(Call(tool=tool, paths=[where or "."], search=True, raw=text))
            else:
                calls.append(Call(tool=tool, globs=[(where, pattern)], raw=text))
            continue
        word, _, arg = rest.partition(" ")
        arg = re.sub(r"\s+\[[^\]]*\]\s*$", "", arg.strip())
        low = _norm_tool(word)
        if low in _SKIP:
            calls.append(Call(tool=word, raw=text, inert=True))
            continue
        if low in ("read", "list", "ls", "view"):
            calls.append(Call(tool=word, paths=[arg] if arg else [], raw=text))
        elif low in _WRITE_TOOLS:
            calls.append(Call(tool=word, paths=[arg] if arg else [], write=True, raw=text))
        else:
            calls.append(Call(tool=word or mark, shell=arg or rest, raw=text))
    return calls


def tool_calls(runtime: str, lines: list[str]) -> list[Call]:
    """Every tool call in a harness transcript, with its output where the harness logs it."""
    if runtime == "cursor_agent":
        return _cursor(lines)
    if runtime == "codex":
        return _codex(lines)
    if runtime == "opencode":
        return _opencode(lines)
    return _claude(lines)  # claude_code, replay


# ─────────────────────────────── where things lead ───────────────────────────────


def _abs(p) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(str(p)))))


# Windows spellings of a place that the plain path also names: the `\\?\` and
# `\\.\` prefixes, an admin share of this machine (`\\localhost\C$\x` is `C:\x`),
# an NTFS stream suffix (`dir::$INDEX_ALLOCATION` is `dir`).
_EXTENDED = re.compile(r"[\\/]{2}[?.][\\/](unc[\\/])?", re.IGNORECASE)
_ADMIN_SHARE = re.compile(r"//([^/]+)/([a-z])\$(?=/|$)")
_THIS_HOST = {"localhost", "127.0.0.1", platform.node().casefold()} - {""}
_STREAM = re.compile(r":(?:\$i30)?:\$(?:index_allocation|data)\b", re.IGNORECASE)
_MSYS = re.compile(r"/(?:cygdrive/|mnt/)?([a-z])(?=/)", re.IGNORECASE)
_DRIVE = re.compile(r"[a-z]:[\\/]", re.IGNORECASE)


def _fold(name: str) -> str:
    """A name as the filesystem compares it: case-blind on Windows."""
    return name.casefold() if _WINDOWS else name


def _canon(p: str) -> str:
    """A path as comparisons see it: itself on POSIX.

    On Windows `/` for `\\`, case-folded (NTFS names `Private` and `PRIVATE`
    the same directory), and without the spellings `_EXTENDED`, `_ADMIN_SHARE`
    and `_STREAM` describe. Idempotent.
    """
    if not _WINDOWS:
        return p
    p = _STREAM.sub("", str(p).replace("\\", "/").casefold())
    if m := _EXTENDED.match(p):
        p = ("//" if m.group(1) else "") + p[m.end():]
    if (m := _ADMIN_SHARE.match(p)) and m.group(1) in _THIS_HOST:
        p = m.group(2) + ":" + (p[m.end():] or "/")
    return p


def _native(token: str) -> str:
    """A path token as Windows itself would name it; unchanged on POSIX.

    Git Bash's `/c/Users` (Cygwin's `/cygdrive/c/...`, WSL's `/mnt/c/...`) is
    `C:/Users`, and `\\\\?\\C:\\x` is `C:\\x`. A bare `/c` is left alone: far
    more often it is a cmd.exe switch (`cmd /c`) than a drive.
    """
    if not _WINDOWS:
        return token
    if m := _EXTENDED.match(token):
        token = ("\\\\" if m.group(1) else "") + token[m.end():]
    if m := _MSYS.match(token):
        token = m.group(1).upper() + ":" + token[m.end():]
    return token


def _is_abs(token: str) -> bool:
    """Does `token` lead somewhere whatever directory the command ran in?

    On Windows `\\x` counts: it depends on the drive, not the directory.
    """
    if _WINDOWS:
        return token.startswith(("/", "\\")) or bool(_DRIVE.match(token))
    return token.startswith("/")


def _has_sep(token: str) -> bool:
    return "/" in token or (_WINDOWS and "\\" in token)


def _parts(p: str) -> tuple[str, ...]:
    """A path's components as comparisons see them (see `_canon`).

    On Windows the first is the anchor (`c:\\`, `\\\\server\\share\\`), and a
    name's trailing dots and spaces are dropped: Windows opens `private.\\x`
    as `private\\x`.
    """
    if not _WINDOWS:
        return PurePosixPath(p).parts
    parts = PureWindowsPath(_canon(p)).parts
    return tuple(x.rstrip(". ") if x.strip(".") else x for x in parts)


def _is_anchor(part: str) -> bool:
    """Is `part`, of `_parts`, a path's root — where `..` stops?"""
    if _WINDOWS:
        return part.endswith("\\") or (len(part) == 2 and part[1] == ":")
    return part.startswith("/")


def _forms(p: Path) -> set[str]:
    out = {_canon(str(p))}
    with contextlib.suppress(OSError, ValueError, RuntimeError):
        out.add(_canon(os.path.realpath(p)))
    return out


def _under(p: str, root: str) -> bool:
    p, root = _canon(p), _canon(root)
    return p == root or p.startswith(root.rstrip("/") + "/")


def _seq(parts: tuple[str, ...], want: tuple[str, ...]) -> bool:
    n = len(want)
    return any(parts[i : i + n] == want for i in range(len(parts) - n + 1))


def _glob_match(pat: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """Component-wise glob match; `**` spans any number of components."""
    if not pat:
        return not path
    head = pat[0]
    if head == "**":
        return any(_glob_match(pat[1:], path[i:]) for i in range(len(path) + 1))
    return bool(path) and fnmatch.fnmatchcase(path[0], head) and _glob_match(pat[1:], path[1:])


def harness_stores(home: Path | None = None) -> dict[str, Path]:
    """Where each harness keeps its sessions — transcripts, file backups, snapshots."""
    home = Path(home or Path.home())
    claude = Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude")
    codex = Path(os.environ.get("CODEX_HOME") or home / ".codex")
    data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return {
        "claude_projects": claude / "projects",
        "claude_file_history": claude / "file-history",
        "claude_session_env": claude / "session-env",
        "claude_todos": claude / "todos",
        "codex_sessions": codex / "sessions",
        "codex_archived": codex / "archived_sessions",
        "opencode": data / "opencode",
        "cursor_chats": home / ".cursor" / "chats",
    }


class Scope:
    """What, for one task's author, counts as the answers."""

    def __init__(self, root: Path, task: str, *, cwd: Path | None = None,
                 private_home: Path | None = None, operator: str = "",
                 home: Path | None = None):
        self.root = _abs(root)
        self.task = task
        self.tasks = self.root / "tasks"
        self.task_dir = self.tasks / task
        self.cwd = _abs(cwd or self.task_dir)
        self.sealed = self.root / paths.STATE_DIR / "sealed"
        conf = private.configured(self.root)
        self.configured = _abs(conf) if conf else None
        self.private_home = _abs(private_home or private.locate(self.task_dir))
        self.home = Path(home or Path.home())
        self.stores = [_abs(p) for p in harness_stores(self.home).values()]
        self.operator_path: Path | None = None
        self.operator_url = ""
        name = ""
        if operator:
            if "://" in operator and not operator.startswith("file://"):
                self.operator_url = operator
                name = Path(urllib.parse.urlparse(operator).path).name
            else:
                self.operator_path = _abs(paths.local_path(operator))
                name = self.operator_path.name
        self.operator_name = name if len(name) >= 3 and name not in _PACKAGE_NAMES else ""
        self._roots = self._build_roots()
        self.candidates = self._build_candidates()

    def _build_roots(self) -> list[tuple[str, str]]:
        roots = [(self.sealed, SEALED), (self.private_home, PRIVATE)]
        if self.configured is not None:
            roots.append((self.configured, PRIVATE))
        roots += [(s, SESSIONS) for s in self.stores]
        if self.operator_path is not None:
            roots.append((self.operator_path, OPERATOR))
        out = []
        for path, cat in roots:
            out += [(form, cat) for form in _forms(path)]
        return out

    def _build_candidates(self) -> list[tuple[tuple[str, ...], str, bool, Path]]:
        """Answer paths a glob or a recursive search could reach.

        Each is (its `_parts`, category, is_dir, the path itself).
        """
        homes: list[tuple[Path, str]] = []
        names = {"x"}
        with contextlib.suppress(OSError):
            names |= {t.name for t in self.tasks.iterdir() if t.is_dir()}
        # this task's own private/ is sealed while its author works
        names = {n for n in names if _fold(n) != _fold(self.task)}
        homes += [(self.tasks / n / PRIVATE_NAME, PRIVATE) for n in sorted(names)]
        homes += [(self.sealed / n / PRIVATE_NAME, SEALED) for n in sorted(names | {self.task})]
        if self.configured is not None:
            homes += [(self.configured / n, PRIVATE) for n in sorted(names | {self.task})]
        if not _under(str(self.private_home), str(self.task_dir)):
            homes.append((self.private_home, PRIVATE))
        found: list[tuple[Path, str, bool]] = []
        for home, cat in homes:
            found.append((home, cat, True))
            for leaf in ("ref_values.npy", "reference/levels/L1.npy", "reference/solve.py"):
                found.append((home / leaf, cat, False))
        for store in self.stores:
            found.append((store, SESSIONS, True))
            found.append((store / "x" / "session.jsonl", SESSIONS, False))
        if self.operator_path is not None:
            found.append((self.operator_path, OPERATOR, False))
        return [(_parts(str(p)), cat, is_dir, p) for p, cat, is_dir in found]

    # ── single paths ──

    def classify(self, p: Path) -> str | None:
        """What an absolute path leads to, of the answers; None for anything else."""
        task = _fold(self.task)
        for form in _forms(p):
            for root, cat in self._roots:
                if _under(form, root):
                    return cat
            parts = _parts(form)  # case-folded on Windows, as the names below already are
            if _seq(parts, (paths.STATE_DIR, "sealed")):
                return SEALED
            for i in range(len(parts) - 2):
                if parts[i] != "tasks":
                    continue
                if parts[i + 2] == PRIVATE_NAME:
                    return PRIVATE
                if parts[i + 2] == AUTHORING_DIR and parts[i + 1] != task:
                    return AUTHORING
        return None

    def classify_relative(self, token: str) -> str | None:
        """A relative path judged by its components, wherever the command had gone first."""
        parts = _parts(token)
        if PRIVATE_NAME in parts:
            return PRIVATE
        if _seq(parts, (paths.STATE_DIR, "sealed")):
            return SEALED
        task = _fold(self.task)
        for i, part in enumerate(parts):
            if part == AUTHORING_DIR and i > 0 and parts[i - 1] not in (task, ".", ".."):
                return AUTHORING
        return None

    def glob_hits(self, pattern: tuple[str, ...], dirs: bool = False) -> str | None:
        for parts, cat, is_dir, _ in self.candidates:
            if is_dir and not dirs:
                continue
            if _glob_match(pattern, parts):
                return cat
        return None


# ─────────────────────────────── reading a call ───────────────────────────────

_SPLIT = re.compile(r"""[\s'"`;&|()<>{}=,:]+""")
# Windows: a word as `_SPLIT` would leave it, except that a drive's colon
# (`C:\`, `d:/`, `\\?\C:\`) stays in its path; a `file:///C:/` URL is its path.
_WORD_NT = re.compile(
    r"""(?:file:/{2,3}(?=[a-z]:[\\/]))?"""
    r"""((?:[\\/]{2}[?.][\\/])?(?:[a-z]:(?=[\\/]))?[^\s'"`;&|()<>{}=,:]+)""", re.IGNORECASE)
_GLOB_CHARS = re.compile(r"[*?\[]")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_HEREDOC_WRITER = re.compile(r"(?:^|[\s;&|(])(?:cat|tee|apply_patch)\b")
# The one line of a task's eval.py that names the answers' file, as a template
# states it: an author rewriting eval.py writes it, and is not looking for them.
_CANONICAL_REF = re.compile(r"""REF_VALUES\s*=\s*\\?["']ref_values\.npy\\?["']""")
# PowerShell's `Get-ChildItem -Recurse` (or any abbreviation, or -Depth); cmd.exe's
# `dir /s`, `findstr /s`, `forfiles /s`, `where /r`; .NET's `AllDirectories`. Unlike a
# POSIX shell's, their glob is not expanded first: `dir /s ..\*.npy` searches all of `..`.
_RECURSIVE_NT = re.compile(
    r"\b(?:get-childitem|gci)\b[^|;&\n]*\s-(?:r|depth)"
    r"|\b(?:dir|findstr|forfiles)\b[^|;&\n]*(?:\s|/\w)/s\b|\bwhere\b[^|;&\n]*\s/r\b"
    r"|\ballDirectories\b"
    # on Windows `ls` and `dir` are PowerShell's own names for Get-ChildItem
    + (r"|\b(?:ls|dir)\b[^|;&\n]*\s-r" if _WINDOWS else ""), re.IGNORECASE)
_RECURSIVE = re.compile(
    r"\b(?:find|fd|fdfind|locate|plocate|mlocate|rg|ag|ack|tree|du|rglob|iglob|walk|copytree)\b"
    r"|\bglob\s*\(|\bglob\.glob\b|recursive\s*=\s*True"
    r"|\bgrep\b[^|;&\n]*\s-[A-Za-z]*[rR]|\bgrep\b[^|;&\n]*--recursive"
    r"|\bls\b[^|;&\n]*\s-[A-Za-z]*R"
    rf"|(?i:{_RECURSIVE_NT.pattern})"
)
# A command that copies or moves whole directories: `cp -r`, `cp -a`, rsync, mv,
# an archive. (`cp` without -r skips directories, and so skips a private/.) On
# Windows: robocopy, xcopy, `Copy-Item -Recurse`, `Move-Item`, `Compress-Archive`.
_COPY_TREE = re.compile(
    r"\bcp\s+(?:[^\n]*\s)?-[A-Za-z]*[rRa][A-Za-z]*\b|\bcp\s+(?:[^\n]*\s)?--(?:recursive|archive)\b"
    r"|\bscp\s+(?:[^\n]*\s)?-[A-Za-z]*r|\b(?:rsync|mv|ditto)\b|\b(?:tar|zip|cpio)\b|copytree\s*\("
    r"|(?i:\b(?:robocopy|xcopy|move-item|move|compress-archive)\b"
    r"|\b(?:copy-item|cpi|copy)\b[^\n]*\s-r)")
_ARCHIVERS = re.compile(r"\b(?:tar|zip|cpio)\b|(?i:\bcompress-archive\b)")
_SEGMENTS = re.compile(r"&&|\|\||[;|&\n]")
_ANSWER_NAME = re.compile(r"ref_values", re.IGNORECASE)
_LOOKUP = re.compile(r"\b(?:find|locate|plocate|mlocate|fd)\b")
_ANSWER_FILE = re.compile(r"ref_values(?:\.\w+|\*|\?)|[/\\]ref_values", re.IGNORECASE)
_NAME_FLAGS = ("-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "-g", "--glob",
               "--include", "-glob")
_PS_NAME_FLAGS = ("-filter", "-include")  # PowerShell's, matched in any case
# A command whose output lines are `path:line:text` of files that matched
_CONTENT_SEARCH = re.compile(r"\b(?:grep|rg|ag|ack)\b|(?i:\b(?:findstr|select-string|sls)\b)")
# How a command names the home directory, beyond a leading `~`: `$HOME`, `${HOME}`
# (POSIX shells, PowerShell), `$env:USERPROFILE` (PowerShell), `%USERPROFILE%`
# (cmd.exe), `$USERPROFILE` (Git Bash). Read on every platform: where one means
# nothing, spelling it out can only add a finding.
_HOME_VAR = re.compile(
    r"\$\{HOME\}|\$HOME"
    r"|(?i:%HOMEDRIVE%%HOMEPATH%|%(?:USERPROFILE|HOMEPATH|HOME)%"
    r"|\$env:HOMEDRIVE\$env:HOMEPATH|\$\{env:(?:USERPROFILE|HOMEPATH|HOME)\}"
    r"|\$env:(?:USERPROFILE|HOMEPATH|HOME)\b|\$\{USERPROFILE\}|\$USERPROFILE\b|\$home\b)")
_HOME_MARK = "\0home\0"  # holds a home's place while a command is split into words


def _strip_written_heredocs(cmd: str) -> str:
    """A command with the bodies of heredocs it *writes* (`cat > f <<EOF`) taken out.

    What an agent writes into a file is its own text; what it pipes into
    `python -` is code it runs, and stays.
    """
    if "<<" not in cmd:
        return cmd
    lines = cmd.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC.search(line)
        if not m:
            i += 1
            continue
        delim, body, j = m.group(2), [], i + 1
        while j < len(lines) and lines[j].strip() != delim:
            body.append(lines[j])
            j += 1
        before = line[: m.start()]
        writes = _HEREDOC_WRITER.search(before) and (">" in line or re.search(
            r"\b(?:tee|apply_patch)\b", before))
        if not writes:
            out.extend(body)
        if j < len(lines):
            out.append(lines[j])
        i = j + 1
    return "\n".join(out)


def _split(text: str) -> list[str]:
    """A command's words, as far as paths go (see `_SPLIT`, `_WORD_NT`)."""
    if not _WINDOWS:
        return _SPLIT.split(text)
    return [m.group(1) for m in _WORD_NT.finditer(_STREAM.sub("", text))]


def _expand(token: str, home: Path) -> str:
    """A path token with its home spelled out, named as the platform names it (`_native`)."""
    if token == "~" or token.startswith(("~/", "~\\") if _WINDOWS else "~/"):
        token = str(home) + token[1:]
    return _native(_HOME_VAR.sub(lambda _: str(home), token))


def _tokens(text: str, home: Path) -> list[str]:
    out = []
    # The home is marked before the split: `${HOME}` and `$env:USERPROFILE` hold
    # characters that split, and a home with a space in it must stay one word.
    text = _HOME_VAR.sub(_HOME_MARK, text)
    for t in _split(text):
        t = t.strip("[]")
        if t not in (".", "..") and not t.endswith(("/.", "\\.") if _WINDOWS else "/."):
            t = t.rstrip(".")
        if not t or (t.startswith("-") and not t.startswith("-/")):
            continue
        out.append(_expand(t.replace(_HOME_MARK, str(home)), home))
    return out


def _name_patterns(text: str, home: Path) -> list[str]:
    """What a recursive search looks for by name: globs, `-name X`, and `.endswith('.npy')`."""
    raw = [t for t in _split(text) if t]
    pats = []
    for i, t in enumerate(raw):
        if (t in _NAME_FLAGS or t.lower() in _PS_NAME_FLAGS) and i + 1 < len(raw):
            pats.append(raw[i + 1])
        elif _GLOB_CHARS.search(t) and not _has_sep(t):
            pats.append(t)
        elif re.fullmatch(r"\.(?:npy|npz|mat|h5|hdf5|csv|json|txt)", t):
            pats.append("*" + t)
    return [_expand(p, home) for p in pats]


class _Reader:
    def __init__(self, scope: Scope):
        self.scope = scope
        self._memo: dict[tuple[str, str], str | None] = {}

    def hits(self, call: Call) -> list[Hit]:
        scope = self.scope
        base = _abs(Path(scope.cwd) / _native(call.cwd)) if call.cwd else scope.cwd
        found: list[tuple[str, str]] = []

        def flag(cat: str | None, why: str = "") -> None:
            if cat:
                found.append((cat, why))

        raw_text = call.raw + " " + call.shell
        if scope.operator_url and scope.operator_url in raw_text:
            flag(OPERATOR)
        if call.inert:
            return [Hit(cat, call.describe()[:400]) for cat, _ in found[:1]]

        shell = _strip_written_heredocs(call.shell) if call.shell else ""
        shell = _CANONICAL_REF.sub(" ", shell)
        tokens = [t for p in call.paths for t in [_expand(p.strip(), scope.home)] if t]
        shell_tokens = _tokens(shell, scope.home) if shell else []
        recursive = bool(_RECURSIVE.search(shell)) if shell else False

        for t in tokens + shell_tokens:
            explicit = t in tokens
            if _GLOB_CHARS.search(t):
                flag(self._glob(base, t, dirs=False))
                if _ANSWER_NAME.search(t):
                    flag(HUNT)
                continue
            # The answers' file name is a search target only where something
            # searches for it: `find -name`, `locate`, a recursive glob or grep,
            # or a read of a path by that name. Anywhere else it reaches nothing
            # a path of its own would not (an author saving a stand-in
            # reference in a temp directory to try its metric is not looking).
            looked_up = (recursive or _LOOKUP.search(shell)) and _ANSWER_NAME.search(t)
            read_by_name = explicit and not call.write and _ANSWER_FILE.search(t)
            if looked_up or read_by_name:
                flag(HUNT)
            flag(self._path(base, t))

        for segment in _SEGMENTS.split(shell) if shell else []:
            flag(self._copied(base, segment))

        for gbase, pattern in call.globs:
            root = _abs(Path(base) / _expand(gbase, scope.home)) if gbase else base
            pattern = _expand(pattern, scope.home)
            if call.search and not _has_sep(pattern):
                pattern = "**/" + pattern  # a content search's file filter applies at any depth
            flag(self._glob(root, pattern, dirs=False))
            if _ANSWER_NAME.search(pattern):
                flag(HUNT)

        if recursive:
            flag(self._recursive(base, shell))

        if call.result and call.result.strip():
            flag(*self._result(base, call))

        seen, out = set(), []
        for cat, why in found:
            if cat in seen:
                continue
            seen.add(cat)
            out.append(Hit(cat, (call.describe()[:400] + (f"  ⇒ {why[:300]}" if why else ""))))
        return out

    # ── one token ──

    def _path(self, base: Path, token: str) -> str | None:
        key = (str(base), token)
        if key not in self._memo:
            self._memo[key] = self._path_uncached(base, token)
        return self._memo[key]

    def _copied(self, base: Path, segment: str) -> str | None:
        """A whole directory copied or moved, one that has some task's answers in it."""
        if not _COPY_TREE.search(segment):
            return None
        toks = _tokens(segment, self.scope.home)
        if not _ARCHIVERS.search(segment) and len(toks) > 2:
            toks = toks[:-1]  # cp/mv/rsync: the last argument is where it goes
        for t in toks:
            cat = (self._glob(base, t, dirs=True) if _GLOB_CHARS.search(t)
                   else self._holds_answers(base, t))
            if cat:
                return cat
        return None

    def _holds_answers(self, base: Path, token: str) -> str | None:
        """A directory copied whole that has some task's answers inside it."""
        p = _abs(base / token)  # an absolute token replaces `base` (on Windows, keeps its drive)
        if not p.is_dir():
            return None
        for _, cat, is_dir, answers in self.scope.candidates:
            if is_dir and _under(str(answers), str(p)) and answers.exists():
                return cat
        return None

    def _path_uncached(self, base: Path, token: str) -> str | None:
        scope = self.scope
        rel = not _is_abs(token)
        p = _abs(base / token)  # an absolute token replaces `base` (on Windows, keeps its drive)
        cat = scope.classify(p)
        if cat is None and rel:
            cat = scope.classify_relative(token)
        if cat is None and scope.operator_name and _fold(scope.operator_name) in _parts(token):
            # The data's file name, as a path anywhere — except a file of that
            # name the author made in its own task directory.
            own = _under(str(p), str(scope.task_dir)) and (p.exists() or p.is_symlink())
            if not own:
                cat = OPERATOR
        return cat

    def _glob(self, base: Path, token: str, *, dirs: bool) -> str | None:
        scope = self.scope
        parts = _parts(os.path.join(base, token))  # an absolute token replaces `base`
        norm: list[str] = []
        for part in parts:
            if part == "..":
                if norm and not _is_anchor(norm[-1]):
                    norm.pop()
            elif part not in (".", ""):
                norm.append(part)
        pattern = tuple(norm)
        static = []
        for part in pattern:
            if _GLOB_CHARS.search(part):
                break
            static.append(part)
        static_path = Path(*static) if static else Path("/")
        if len(static) < len(pattern):
            cat = scope.classify(static_path)
            if cat:
                return cat
        own = _under(str(static_path), str(scope.task_dir))
        for part in pattern[len(static):]:
            letters = re.sub(r"\[[^\]]*\]", "", part)
            letters = re.sub(r"[^A-Za-z]", "", letters)
            if (not own and letters and fnmatch.fnmatchcase(PRIVATE_NAME, part)
                    and _subsequence(letters.lower(), PRIVATE_NAME)):
                return PRIVATE  # `priv*`, `p*e`: a pattern only private/ fits in spirit
        return scope.glob_hits(pattern, dirs=dirs)

    def _recursive(self, base: Path, shell: str) -> str | None:
        """A recursive search rooted above some answers, for files the answers could be."""
        scope = self.scope
        pats = _name_patterns(shell, scope.home)
        roots = [base]
        one_word = _WINDOWS and bool(_RECURSIVE_NT.search(shell))
        for t in _tokens(shell, scope.home):
            if _GLOB_CHARS.search(t):
                # `dir /s ..\*.npy`: the tree's root and the names searched for, in one word
                head, _, tail = t.replace("\\", "/").rpartition("/")
                if one_word and head and not _GLOB_CHARS.search(head):
                    roots.append(_abs(base / head))
                    pats.append(tail)
                continue
            if t in (".", "..", "/", "~") or _has_sep(t) or t.startswith(str(scope.home)):
                roots.append(_abs(base / t))  # an absolute token replaces `base`
        if not pats:
            return None
        for root in roots:
            here = _parts(str(root))
            for pat in pats:
                full = here + _parts(pat) if _has_sep(pat) else (*here, "**", _fold(pat))
                cat = scope.glob_hits(full, dirs=False)
                if cat:
                    return HUNT if cat in (PRIVATE, SEALED) else cat
        return None

    def _result(self, base: Path, call: Call) -> tuple[str | None, str]:
        """Lines of a call's output that came out of the answers.

        `path:content` (grep, rg) always counts; a bare path only for a content
        search, whose listed files matched what it looked for. A listing
        (`ls -R`'s `dir:` headers, `find`'s paths) does not.
        """
        search = call.search or bool(_CONTENT_SEARCH.search(call.shell))
        for line in call.result.splitlines()[:20000]:
            line = line.strip()
            if not line:
                continue
            first = line.split()[0]
            # the colon that ends the path, not a Windows drive's (`C:\x\notes.md:3:...`)
            drive = 2 if _WINDOWS and _DRIVE.match(first) else 0
            path, sep, rest = first[drive:].partition(":")
            path = first[:drive] + path
            content = (rest + line[len(first):]).strip() if sep else ""
            if sep and not content:
                continue  # `dir:` — a listing's header
            if not sep and not search:
                continue
            if not _has_sep(path) and _fold(path) != PRIVATE_NAME:
                continue
            cat = self._path(base, _native(path))
            if cat in (PRIVATE, SEALED, SESSIONS, OPERATOR, AUTHORING):
                return cat, line
        return None, ""


def _subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def audit(runtime: str, lines: list[str], scope: Scope) -> list[Hit]:
    """Every reach for the answers in an agent's transcript."""
    reader = _Reader(scope)
    out: list[Hit] = []
    seen = set()
    for call in tool_calls(runtime, lines):
        for hit in reader.hits(call):
            if (hit.category, hit.detail) not in seen:
                seen.add((hit.category, hit.detail))
                out.append(hit)
    return out


def categories(hits: list[Hit]) -> list[str]:
    """What may be said publicly about a list of hits: their categories, once each."""
    return sorted({h.category for h in hits})


# ─────────────────────────────── harness session stores ───────────────────────────────


def claude_slug(cwd: str) -> str:
    """Claude Code's project directory name for a working directory."""
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _long(p) -> str:
    """`p` as Windows reaches it past MAX_PATH (260 characters); `str(p)` on POSIX.

    A Claude Code session directory is named after the whole working
    directory, and is moved into a task's private directory: twice a deep
    path, which Windows, unless long paths are enabled machine-wide, neither
    lists nor moves under its plain name.
    """
    s = str(p)
    if not _WINDOWS or s.startswith(("\\\\?\\", "\\\\.\\")):
        return s
    s = os.path.abspath(s)
    return "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s


def _plain(p) -> str:
    """`p` without the `\\\\?\\` prefix `_long` gives it, as a person would write it."""
    s = str(p)
    if s.startswith("\\\\?\\UNC\\"):
        return "\\\\" + s[8:]
    return s[4:] if s.startswith("\\\\?\\") else s


def _unique(parent: Path, name: str) -> Path:
    target = parent / name
    n = 2
    while target.exists() or target.is_symlink():
        target = parent / f"{name}-{n}"
        n += 1
    return target


def _names_dir(path: Path, needles: list[bytes]) -> bool:
    keep = max(len(n) for n in needles)
    tail = b""
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                block = tail + chunk
                if any(n in block for n in needles):
                    return True
                tail = block[-keep:]
    except OSError:
        return False
    return False


def _session_ids(log: Path | None) -> set[str]:
    ids: set[str] = set()
    if log is None or not Path(log).is_file():
        return ids
    with contextlib.suppress(OSError):
        for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
            if '"session_id"' not in line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get("session_id"), str):
                    sid = obj["session_id"]
                    if re.fullmatch(r"[A-Za-z0-9_-]{6,80}", sid):
                        ids.add(sid)
    return ids


def sweep_sessions(runtime: str, workdir: Path, since: float, dest: Path, *,
                   log: Path | None = None, home: Path | None = None) -> tuple[list[str], str]:
    """Move what a harness kept of the sessions run in `workdir` into `dest`.

    For an agent that saw the answers, the harness's own session store holds
    its whole transcript — solver arrays, the operator's path — where any
    later agent of the same user can read it. Returns what was moved, and a
    warning for what could not be moved safely (empty if nothing).

    * Claude Code: `~/.claude/projects/<cwd slug>/` (the slug is the absolute
      working directory with every non-alphanumeric character replaced by
      `-`), plus the file backups and environment it keeps per session;
    * Codex: `~/.codex/sessions/**/rollout-*.jsonl` written since `since`
      that name `workdir`;
    * OpenCode and Cursor Agent keep sessions in databases that moving files
      out of would corrupt; they are left, with a warning saying where.

    Nothing outside what the stage's own sessions wrote is touched. A file
    that could not be moved (Windows refuses to move one another process has
    open) is said in the warning, never passed over.
    """
    workdir = Path(workdir)
    shown = harness_stores(home)
    stores = {k: Path(_long(v)) for k, v in shown.items()}
    dest = Path(_long(dest))
    cwds = sorted({str(_abs(workdir)), _plain(os.path.realpath(workdir))})
    moved: list[str] = []
    failed: list[str] = []

    def move(src: Path, sub: str) -> None:
        try:
            target_dir = dest / sub
            target_dir.mkdir(parents=True, exist_ok=True)
            target = _unique(target_dir, src.name)
            shutil.move(str(src), str(target))
        except OSError as e:
            failed.append(type(e).__name__)
            return
        moved.append(f"{_plain(src)} -> {_plain(target)}")

    def left(store: Path) -> str:
        if not failed:
            return ""
        return (f"{len(failed)} of the files the harness kept of this agent, which saw the "
                f"answers, could not be moved out of {store} ({', '.join(sorted(set(failed)))}); "
                "delete the sessions run in the task's private reference directory there "
                "before any other agent runs as this user")

    if runtime == "claude_code":
        projects = stores["claude_projects"]
        sessions = _session_ids(log)
        slugs = {_fold(claude_slug(c)) for c in cwds}  # Windows: names are case-blind
        found: list[Path] = []
        with contextlib.suppress(OSError):
            found = sorted(projects.iterdir()) if projects.is_dir() else []
        for d in found:
            name = _fold(d.name)
            long_match = any(len(s) > 200 and name.startswith(s[:200]) for s in slugs)
            if d.is_dir() and (name in slugs or long_match):
                with contextlib.suppress(OSError):
                    sessions |= {f.stem for f in d.glob("*.jsonl")}
                move(d, "claude_code/projects")
        for sid in sorted(sessions):
            for key in ("claude_file_history", "claude_session_env"):
                p = stores[key] / sid
                if p.exists():
                    move(p, f"claude_code/{p.parent.name}")
            todos = stores["claude_todos"]
            if todos.is_dir():
                for p in sorted(todos.glob(f"{sid}*")):
                    move(p, "claude_code/todos")
        return moved, left(shown["claude_projects"].parent)
    if runtime == "codex":
        needles = []
        for c in cwds:
            # as written, JSON-escaped (a Windows path's `\`), and JSON-escaped with its
            # non-ASCII characters left as they are (a user name in any script)
            needles += [c.encode(), json.dumps(c)[1:-1].encode(),
                        json.dumps(c, ensure_ascii=False)[1:-1].encode()]
        for key in ("codex_sessions", "codex_archived"):
            root = stores[key]
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("rollout-*.jsonl")):
                try:
                    recent = f.stat().st_mtime >= since - 5.0
                except OSError:
                    continue
                if recent and _names_dir(f, needles):
                    move(f, f"codex/{key}")
        return moved, left(shown["codex_sessions"].parent)
    if runtime == "opencode":
        sid = sorted(_opencode_ids(log))
        which = f" (session {', '.join(sid)})" if sid else ""
        return moved, (f"OpenCode keeps the session of this agent, which saw the answers{which}, "
                       f"in its own database under {shown['opencode']}; moving it would corrupt "
                       "that store, so it was left — delete the session there before any other "
                       "agent runs as this user")
    if runtime == "cursor_agent":
        return moved, (f"Cursor Agent keeps the chat of this agent, which saw the answers, under "
                       f"{shown['cursor_chats']}; it was left — delete it there before any "
                       "other agent runs as this user")
    return moved, ""


def _opencode_ids(log: Path | None) -> set[str]:
    if log is None or not Path(log).is_file():
        return set()
    with contextlib.suppress(OSError):
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        return set(re.findall(r"\b(ses_[A-Za-z0-9]{6,})\b", text))
    return set()
