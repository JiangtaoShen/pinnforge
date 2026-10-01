"""The orchestrator as a process: one per run, detachable, and stoppable.

Three things a run used to lack, and what replaces them:

* **Ownership.** Nothing stopped two orchestrators driving one run — a
  `run resume` while the first was still alive put two agents on the same
  workspace. Now the driving process holds `runs/<id>/.lock` (an advisory
  `flock`, released by the kernel when the process dies, so it cannot go
  stale) and records its pid and its agent's process group in it.

* **Stopping.** `run stop` used to edit `state.json` and nothing else; the
  live orchestrator overwrote it at its next checkpoint and carried on. Now
  it signals the process that holds the lock, which unwinds exactly as a
  Ctrl-C does — the agent's group interrupted, the segment's cost booked,
  the integrity verdict written — and waits for the lock to come free.

* **Signals.** SIGTERM and SIGHUP (a closed terminal, `tmux kill-session`)
  used to kill the orchestrator outright: no cleanup, the agent's process
  group orphaned on the GPU, the state left saying `running`. They now raise
  `RunStopped`, a `KeyboardInterrupt`, so they take the path Ctrl-C always
  took. A signal that lands in a *critical section* — between starting an
  agent and being ready to stop it, or while a stop is being cleaned up — is
  deferred to the section's end rather than tearing it in half. A second
  signal never raises: it kills the recorded agent group outright, so an
  impatient operator shortens the cleanup instead of aborting it.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

LOCK_FILE = ".lock"
DETACH_FILE = ".detach.json"
LEASE_DIR = Path("/tmp/pinnforge-leases")
# A tmux socket younger than this may belong to a server still starting up.
SOCKET_GRACE_S = 60.0

logger = logging.getLogger(__name__)


class RunLocked(RuntimeError):
    """Another live process is driving this run."""


class RunStopped(KeyboardInterrupt):
    """A stop signal, delivered where Ctrl-C would have been."""

    def __init__(self, signum: int):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


# ─────────────────────────────── the lock ───────────────────────────────


def _open_regular(path: Path, flags: int = os.O_RDONLY, mode: int = 0o644) -> int:
    """Open `path` without following a link or waiting on a FIFO; regular files only.

    The lease directory is world-writable and `runs/<id>/.lock` sits in the
    directory a block works in, so either may hold something planted: a link
    to follow somewhere else, or a FIFO whose `open()` would hang the caller
    until somebody writes to it.
    """
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_fd(fd: int) -> dict:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 1 << 16)
        info = json.loads(data.decode("utf-8") or "{}")
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _read_json(path: Path) -> dict:
    """A small JSON object from `path`, or {} — whatever is, or is not, there."""
    try:
        fd = _open_regular(path)
    except OSError:
        return {}
    try:
        return _read_fd(fd)
    finally:
        os.close(fd)


def _as_pid(value) -> int | None:
    """A recorded pid or pgid, if it is one that may be signalled.

    The lock file is writable by the block it describes, so what it says is
    checked before anything is sent to it: `kill(-1, …)` signals every
    process the user owns, and 0 or 1 are not a driver or an agent either.
    """
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 1 < n < 2**31 else None


class RunLock:
    """`runs/<id>/.lock`, held for as long as this process drives the run."""

    def __init__(self, run: Path):
        self.run = Path(run)
        self.path = self.run / LOCK_FILE
        self.fd: int | None = None
        self.info: dict = {}

    def acquire(self) -> RunLock:
        try:
            fd = _open_regular(self.path, os.O_RDWR | os.O_CREAT)
        except OSError as e:
            raise RunLocked(
                f"cannot use {self.path} as the run's lock ({e}); remove it and try again"
            ) from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            h = _read_json(self.path)
            raise RunLocked(
                f"{self.run.name} is already being driven by pid {h.get('pid', '?')} "
                f"on {h.get('host', '?')} since {h.get('started', '?')} — "
                f"`pinnforge run stop {self.run.name}` to stop it"
            ) from None
        previous = _read_fd(fd)
        self.fd = fd
        # The previous driver is gone (its lock was free), but the agent it
        # started runs in a session of its own and need not have gone with it.
        # Starting another on the same workspace would put two agents on one
        # block, both charging its budget.
        self._reap_orphan(previous)
        self.info = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "agent_pgid": None,
        }
        self._write()
        return self

    def _reap_orphan(self, previous: dict) -> None:
        pgid = _as_pid(previous.get("agent_pgid"))
        if pgid is None or pgid == os.getpgid(0):
            return
        host = previous.get("host")
        if host and host != socket.gethostname():
            return  # another machine's process group; not ours to signal
        if not group_alive(pgid):
            return
        if group_in(pgid, self.run) is False:
            return  # the number has since been reused by something else
        logger.warning(
            "the agent a previous driver of %s started (process group %s) is still "
            "running; stopping it before this driver starts another",
            self.run.name,
            pgid,
        )
        if not terminate_group(pgid):
            logger.error("process group %s survived SIGKILL", pgid)

    def update(self, **fields) -> None:
        self.info.update(fields)
        self._write()

    def _write(self) -> None:
        if self.fd is None:
            return
        data = (json.dumps(self.info) + "\n").encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, data)

    def release(self) -> None:
        if self.fd is None:
            return
        with contextlib.suppress(OSError):
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.ftruncate(self.fd, 0)
        os.close(self.fd)
        self.fd = None

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def holder(run: Path) -> dict | None:
    """Who is driving `run` right now, or None if nobody is."""
    path = Path(run) / LOCK_FILE
    try:
        fd = _open_regular(path)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return _read_fd(fd) or {"pid": None}
    except OSError:
        # A filesystem without flock (some NFS): fall back to the recorded pid,
        # which says nothing about a process on another machine.
        info = _read_fd(fd)
        if info.get("host") and info["host"] != socket.gethostname():
            return info
        return info if _pid_alive(info.get("pid")) else None
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _pid_alive(pid) -> bool:
    pid = _as_pid(pid)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ─────────────────────────────── process groups ───────────────────────────────


def _group_members(pgid: int) -> list[tuple[int, str]] | None:
    """(pid, state) of every process in group `pgid`; None where /proc is not available."""
    proc = Path("/proc")
    if not (proc / "self" / "stat").is_file():
        return None
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # the command name may hold spaces and parentheses; fields follow the last ")"
        fields = raw[raw.rfind(")") + 2 :].split()
        try:
            if int(fields[2]) == pgid:
                out.append((int(d.name), fields[0]))
        except (IndexError, ValueError):
            continue
    return out


def group_alive(pgid) -> bool:
    """Does process group `pgid` still have a live (not merely unreaped) member?"""
    pgid = _as_pid(pgid)
    if pgid is None:
        return False
    members = _group_members(pgid)
    if members is not None:
        return any(state not in ("Z", "X") for _, state in members)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_in(pgid, run: Path) -> bool | None:
    """Is group `pgid` working in `run`? None where that cannot be told (no /proc).

    An agent is started with the run directory as its working directory, and
    everything it starts inherits it — so a group none of whose members works
    anywhere under the run is not this run's agent, whatever number it has.
    """
    pgid = _as_pid(pgid)
    if pgid is None:
        return False
    members = _group_members(pgid)
    if members is None:
        return None
    root = Path(run).resolve()
    for pid, _ in members:
        try:
            cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            continue
        if cwd == root or root in cwd.parents:
            return True
    return False


def terminate_group(pgid, grace: float = 10.0) -> bool:
    """SIGTERM the group, SIGKILL what is left after `grace`. True once it is gone.

    TERM first, because the processes in an agent's group that matter most —
    `eval.py` and its training worker — reconcile the block's budget and free
    the GPU lock on the way out, and a KILL takes that chance away.
    """
    pgid = _as_pid(pgid)
    if pgid is None or pgid == os.getpgid(0):
        return True
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        if not group_alive(pgid):
            return True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            if not group_alive(pgid):
                return True
            time.sleep(0.05)
    return not group_alive(pgid)


# ─────────────────────────────── signals ───────────────────────────────

_sig = {"depth": 0, "deferred": None, "count": 0, "escalate": None}


def install_signal_handlers(escalate=None) -> None:
    """Route SIGTERM, SIGHUP and SIGINT through `RunStopped`.

    `escalate()` is called on the second signal instead of raising.
    """
    _sig.update(depth=0, deferred=None, count=0, escalate=escalate)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _handler)


def _handler(signum, frame) -> None:
    _sig["count"] += 1
    if _sig["count"] > 1:
        if _sig["escalate"] is not None:
            with contextlib.suppress(Exception):
                _sig["escalate"]()
        return
    if _sig["depth"] > 0:
        _sig["deferred"] = signum
        return
    raise RunStopped(signum)


@contextlib.contextmanager
def critical():
    """A section no stop signal may tear in half; one that arrives waits for its end."""
    _sig["depth"] += 1
    try:
        yield
    finally:
        _sig["depth"] -= 1
        if _sig["depth"] == 0 and _sig["deferred"] is not None:
            signum, _sig["deferred"] = _sig["deferred"], None
            raise RunStopped(signum)


def kill_group(pgid) -> None:
    pgid = _as_pid(pgid)
    if pgid is None or pgid == os.getpgid(0):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


# ─────────────────────────────── stopping ───────────────────────────────


def stop(run: Path, force: bool = False, wait_s: float = 90.0) -> str:
    """Stop whoever drives `run`, the way Ctrl-C would. Returns what happened."""
    h = holder(run)
    if h is None:
        return "not running"
    pid = h.get("pid")
    host = h.get("host")
    if host and host != socket.gethostname():
        # A pid means nothing on another machine — on a shared filesystem
        # signalling it here would hit whatever this host runs under that number.
        return (f"refused: {Path(run).name} is driven from {host} (pid {pid}), not from "
                f"{socket.gethostname()}; stop it on {host}")
    if not _pid_alive(pid):
        return "not running"
    pid = int(pid)
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + (10.0 if force else wait_s)
    while time.time() < deadline:
        if holder(run) is None:
            return "stopped"
        time.sleep(0.25)
    if not force:
        return f"still stopping after {wait_s:.0f}s (pid {pid}) — `run stop --force` to kill it"
    info = holder(run) or h
    if group_in(info.get("agent_pgid"), run) is not False:
        kill_group(info.get("agent_pgid"))
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)
    return "killed"


# ─────────────────────────────── detaching ───────────────────────────────


def session_name(root: Path, run_id: str, kind: str = "run") -> str:
    """Unique across workspaces: `ldc_1` in two workspaces must not collide."""
    h = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:6]
    return f"pf-{h}-{kind}-{run_id}".replace(".", "_")


def _sweep_sockets(prefix: str) -> None:
    """Remove this workspace's dead tmux sockets.

    Each detached run gets a tmux server of its own; the server exits with
    the run but leaves its socket file in `/tmp/tmux-<uid>/`, one per run
    forever. A socket nothing answers on is such a leftover — unless it is
    new: a server another `run start` is bringing up right now has bound its
    socket but may not be listening yet, and refuses connections until it is.
    """
    base = Path(os.environ.get("TMUX_TMPDIR") or "/tmp") / f"tmux-{os.getuid()}"
    for p in base.glob(f"{prefix}*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        if not stat.S_ISSOCK(st.st_mode) or time.time() - st.st_mtime < SOCKET_GRACE_S:
            continue
        s = socket.socket(socket.AF_UNIX)
        try:
            s.connect(str(p))
        except ConnectionRefusedError:
            with contextlib.suppress(OSError):
                p.unlink()
        except OSError:
            pass
        finally:
            s.close()


def detach(cmd: list[str], *, cwd: Path, log: Path, name: str, record: Path) -> dict:
    """Start `cmd` so it outlives this terminal, and say how to find it again.

    With tmux, on a tmux server of its own: a fresh server inherits *this*
    environment — API keys, PATH, CUDA_VISIBLE_DEVICES — whereas a session
    added to a long-running server gets that server's environment from
    whenever it was started. Without tmux, a new session with output in `log`.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    tmux = shutil.which("tmux")
    if tmux:
        import shlex

        # No `| tee` here: a killed session would take the pipe's reader down
        # first and turn the orchestrator's cleanup into a broken pipe. The
        # process writes its own log (`--log`) beside what the pane shows.
        line = " ".join(shlex.quote(c) for c in cmd)
        _sweep_sockets(name[: name.index("-", 3) + 1])  # pf-<workspace hash>-
        subprocess.run(
            [tmux, "-L", name, "new-session", "-d", "-s", name, "-c", str(cwd),
             "-x", "220", "-y", "50", line],
            check=True,
            env={**os.environ, "TMUX": ""},
        )
        info = {"mode": "tmux", "socket": name, "session": name, "log": str(log)}
    else:
        with log.open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        info = {"mode": "background", "pid": proc.pid, "log": str(log)}
    record.write_text(json.dumps(info) + "\n", encoding="utf-8")
    return info


def attach(record: Path) -> int:
    """Put the terminal on a detached run: its tmux session, or its log."""
    info = _read_json(record)
    if info.get("mode") == "tmux" and shutil.which("tmux"):
        probe = subprocess.run(
            ["tmux", "-L", info["socket"], "has-session", "-t", info["session"]],
            capture_output=True, check=False,
        )
        if probe.returncode == 0:
            os.execvp("tmux", ["tmux", "-L", info["socket"], "attach", "-t", info["session"]])
    log = info.get("log")
    if log and Path(log).is_file():
        os.execvp("tail", ["tail", "-n", "60", "-f", log])
    print("nothing to attach to: the run is not detached here, or has finished")
    return 1


def self_command(*args: str) -> list[str]:
    """This same pinnforge, as a command line another process can run."""
    return [sys.executable, "-m", "pinnforge", *args]


def anchor_status(run: Path, scored: bool) -> str:
    """What to call b00, which has no block state of its own.

    Its directory exists, and its budget is pre-charged, from the moment the
    measurement starts — so neither says it is done; a scored record does.
    """
    if scored:
        return "done"
    return "measuring" if holder(run) is not None else "missing"


# ─────────────────────────────── GPU leases ───────────────────────────────


def _lease_path(gpu: int, pid: int | None = None) -> Path:
    # One file per holder: two runs sharing a card (`--share`) must not
    # overwrite each other's claim, or the first to finish would erase the
    # other's and the card would look free while it is still in use.
    return LEASE_DIR / f"gpu{gpu}.{pid or os.getpid()}.json"


def holders(gpu: int, sweep: bool = True) -> list[dict]:
    """The live runs on this machine that have claimed `gpu`; stale claims are swept."""
    out = []
    for p in sorted(LEASE_DIR.glob(f"gpu{gpu}.*.json")):
        info = _read_json(p)  # a link, a FIFO or a non-object reads as nothing
        if _pid_alive(info.get("pid")):
            out.append(info)
        elif sweep:
            with contextlib.suppress(OSError):
                p.unlink()
    return out


def leases(pool: list[int]) -> list[tuple[int, dict]]:
    """Which of `pool` another live run on this machine has claimed."""
    return [(g, info) for g in pool for info in holders(g) if info.get("pid") != os.getpid()]


def claim(pool: list[int], run: Path) -> None:
    """Record that this process's run uses `pool` — advisory, for `run start`'s warning.

    Two runs on one card do not break anything: `eval.py`'s per-GPU lock
    queues their evaluations. They do halve each other's throughput, so a
    new run is told before it joins a busy card.
    """
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        LEASE_DIR.chmod(0o1777)
    data = (json.dumps({"pid": os.getpid(), "run": str(run)}) + "\n").encode()
    for g in pool:
        # Written whole and renamed into place: a reader sweeping stale claims
        # must never see a half-written one and take it for a dead holder's.
        dest = _lease_path(g)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dest)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()


def release_claim(pool: list[int]) -> None:
    for g in pool:
        with contextlib.suppress(OSError):
            _lease_path(g).unlink()
