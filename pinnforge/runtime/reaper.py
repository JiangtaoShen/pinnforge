"""Run a harness CLI: keep its log to what is read, and stop what its tools leave running.

`python -m pinnforge.runtime.reaper [--drop-type TYPE]… -- CMD …`

Some harnesses start each tool command in a session of its own — Pi's bash
tool does — so the agent's process group, which the framework signals, waits
on and records in the run lock, does not hold the commands the agent runs:
an `eval.py` and its training worker. Pi stops the commands it is waiting on
when it gets SIGTERM or SIGHUP, but not on SIGINT, and nothing stops one it
started in the background (`nohup … &`) once the call that started it has
returned. Left alone, that worker keeps the GPU lock and keeps charging a
block that has ended.

This process sits between the framework and the CLI, in the agent's group,
so the framework sees it as the agent: the same pid it records, the same
group it signals. On Linux it makes itself the subreaper of everything below
it (PR_SET_CHILD_SUBREAPER), so a process whose parent exits comes back to it
instead of to init, and puts the CLI in a group of its own, told to stop
when this process dies (PR_SET_PDEATHSIG). Stopped, it stops the tools
first: SIGTERM to the group of every process below it, so `eval.py` settles
its budget to the seconds it ran and frees the lock — Pi itself would SIGKILL
them, and the block would stay charged the whole training wall — then the
CLI (SIGKILL if it will not go), then whatever is left. When the CLI exits on
its own, what it left running gets SIGTERM and LEFT_GRACE_S to settle before
SIGKILL, and this exits with the CLI's status.
Elsewhere there is no subreaper, and it only relays (on Windows the agent's
Job Object holds every descendant).

`--drop-type TYPE` keeps the CLI's JSON lines of that `type` out of the log.
Pi streams every token it receives as an event of its own and repeats each
finished message and tool output two more times, so a block's log grew by a
megabyte a minute, nearly all of it nothing a reader uses — and the
dashboard, which reads the end of a log, found only token deltas there. The
session itself is kept whole by the harness.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PR_SET_PDEATHSIG = 1
PR_SET_CHILD_SUBREAPER = 36
# A stop, all told (tools, CLI, what is left, 10 s), stays under the 15 s a
# PiHandle gives a stopped agent before the framework SIGKILLs this process.
# A tool gets longer than an evaluation's worker gets from it (5 s) to unwind.
TOOL_GRACE_S = 6.0
CLI_GRACE_S = 2.0
TERM_GRACE_S = 1.0  # after a stop, for what came since: the tools had their turn
KILL_GRACE_S = 1.0
# After the CLI exits on its own, nothing is waiting on a deadline: what it
# left running settles as long as the orchestrator gave such groups before.
LEFT_GRACE_S = 10.0
USAGE = "usage: python -m pinnforge.runtime.reaper [--drop-type TYPE]… -- CMD [ARGS…]"


def _prctl():
    try:
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


def _become_subreaper() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    prctl = _prctl()
    return prctl is not None and prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0


def _die_with(parent: int):
    """For the CLI, before it runs: SIGTERM it when `parent` dies, even by SIGKILL."""

    def arm() -> None:
        prctl = _prctl()
        if prctl is not None:
            prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != parent:  # died before the request was in place
            os._exit(1)

    return arm


def _table() -> dict[int, tuple[int, int, str]]:
    """pid -> (parent, group, state) for every process /proc shows."""
    out = {}
    with contextlib.suppress(OSError):
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                raw = (d / "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            fields = raw[raw.rfind(")") + 2 :].split()  # the name may hold ") "
            try:
                out[int(d.name)] = (int(fields[1]), int(fields[2]), fields[0])
            except (IndexError, ValueError):
                continue
    return out


def _below(me: int, table: dict[int, tuple[int, int, str]]) -> list[int]:
    """Every live process descended from `me`."""
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _, state) in table.items():
        if state not in ("Z", "X"):
            kids.setdefault(ppid, []).append(pid)
    out, todo = [], list(kids.get(me, []))
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo += kids.get(pid, [])
    return out


class Reaper:
    """The CLI, and everything below this process, on Linux."""

    def __init__(self, cli: subprocess.Popen):
        self.cli = cli
        self.me = os.getpid()
        self.own = os.getpgid(0)
        self.status: int | None = None

    def collect(self) -> bool:
        """Collect every child that has exited, the CLI's status included. False once none."""
        while True:
            try:
                pid, st = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return False
            if pid == 0:
                return True
            if pid == self.cli.pid:
                self.status = self.cli.returncode = os.waitstatus_to_exitcode(st)

    def _groups(self, spare_cli: bool) -> set[int]:
        table = _table()
        spared = {self.own} | ({self.cli.pid} if spare_cli else set())
        return {table[pid][1] for pid in _below(self.me, table)
                if pid in table and table[pid][1] not in spared}

    def _settle(self, groups: set[int], grace: float) -> set[int]:
        """Wait up to `grace` for every member of `groups` to exit; the groups still alive."""
        deadline = time.monotonic() + grace
        while True:
            self.collect()
            alive = {g for _, g, state in _table().values() if g in groups
                     and state not in ("Z", "X")}
            if not alive or time.monotonic() >= deadline:
                return alive
            time.sleep(0.05)

    def stop_tools(self) -> int:
        """SIGTERM the tools' groups — not the CLI's — and give them TOOL_GRACE_S."""
        groups = self._groups(spare_cli=True)
        for g in groups:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(g, signal.SIGTERM)
        self._settle(groups, TOOL_GRACE_S)
        return len(groups)

    def stop_cli(self) -> None:
        """SIGTERM the CLI; SIGKILL its group if it is still there after CLI_GRACE_S."""
        for sig, grace in ((signal.SIGTERM, CLI_GRACE_S), (signal.SIGKILL, KILL_GRACE_S)):
            self.collect()
            if self.status is not None:
                return  # gone: its pid may be someone else's by now
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if sig == signal.SIGTERM:
                    os.kill(self.cli.pid, sig)
                else:
                    os.killpg(self.cli.pid, sig)  # it leads a group of its own
            deadline = time.monotonic() + grace
            while self.status is None and time.monotonic() < deadline:
                self.collect()
                time.sleep(0.05)

    def sweep(self, term_grace: float) -> int:
        """Stop everything still below this process. Returns how many groups were signalled."""
        hit: set[int] = set()
        for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, KILL_GRACE_S)):
            groups = self._groups(spare_cli=False)
            if not groups:
                break
            for g in groups:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(g, sig)
            hit |= groups
            if not self._settle(groups, grace):
                break
        deadline = time.monotonic() + KILL_GRACE_S  # one SIGKILL could not end is not waited on
        while self.collect() and _below(self.me, _table()) and time.monotonic() < deadline:
            time.sleep(0.05)
        return len(hit)


def pump(src, dst, drop: tuple[bytes, ...]) -> None:
    """Copy lines from `src` to `dst` as they come, leaving out those starting with `drop`."""
    for line in iter(src.readline, b""):
        if drop and line.startswith(drop):
            continue
        with contextlib.suppress(OSError, ValueError):
            dst.write(line)
            dst.flush()


def _parse(argv: list[str]) -> tuple[list[str], list[str]] | None:
    types: list[str] = []
    while argv and argv[0] != "--":
        if argv[0] == "--drop-type" and len(argv) > 1:
            types.append(argv[1])
            argv = argv[2:]
        else:
            return None
    return types, argv[1:]


def main(argv: list[str] | None = None) -> int:
    parsed = _parse(list(sys.argv[1:] if argv is None else argv))
    if parsed is None or not parsed[1]:
        print(USAGE, file=sys.stderr)
        return 2
    types, cmd = parsed
    from pinnforge import oscompat  # an npm shim (`pi.cmd`) is resolved on Windows

    drop = tuple(f'{{"type":"{t}"'.encode() for t in types)
    reap = _become_subreaper()
    extra = {"process_group": 0, "preexec_fn": _die_with(os.getpid())} if reap else {}
    try:
        proc = subprocess.Popen(oscompat.resolve_command(cmd),
                                stdout=subprocess.PIPE if drop else None, **extra)
    except OSError as e:
        print(f"pinnforge: cannot start {cmd[0]}: {e}", file=sys.stderr)
        return 127
    copier = None
    if drop:
        copier = threading.Thread(target=pump, args=(proc.stdout, sys.stdout.buffer, drop),
                                  daemon=True)
        copier.start()
    asked = []

    def relay(signum, frame):
        asked.append(signum)
        if not reap:  # the CLI gets its orderly stop: for Pi that is SIGTERM
            with contextlib.suppress(OSError):
                proc.terminate()

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), relay)
    if not reap:
        status = oscompat.wait(proc)  # in slices on Windows, so `relay` gets its turn
    else:
        r = Reaper(proc)
        stopping = False
        while r.status is None:
            if asked and not stopping:
                stopping = True
                r.stop_tools()
                r.stop_cli()
            if not r.collect():
                break
            if r.status is None:
                time.sleep(0.05)
        status = 1 if r.status is None else r.status
        left = r.sweep(TERM_GRACE_S if stopping else LEFT_GRACE_S)
        if left:
            print(f"pinnforge: stopped {left} process group(s) the agent's tools left running",
                  file=sys.stderr, flush=True)
    if copier is not None:
        # The CLI's output ends with it; a process past SIGKILL still holding
        # the pipe is not waited on past the stop's own time.
        copier.join(timeout=2)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
