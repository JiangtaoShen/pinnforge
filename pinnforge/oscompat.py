"""What pinnforge needs from the operating system, on POSIX and on Windows.

Each function here stands for the POSIX call the rest of the code was written
against, and on Linux and macOS it *is* that call. Windows has none of them,
and some of its near-equivalents do something else entirely:

* **Files.** `os.open` opens in text mode — `os.write` turns `\\n` into
  `\\r\\n`, `os.read` stops at a Ctrl-Z — and has neither `O_NOFOLLOW` nor
  `O_NONBLOCK`. `open_file` opens in binary mode, refuses a final link itself,
  and has no FIFO to fear.

* **Locks.** `flock` becomes `LockFileEx` on one byte far past any data:
  Windows locks are mandatory, and a lock on the data would stop a reader
  from reading the very file whose lock it is asking about. A blocking wait
  polls, so Ctrl-C still reaches a process queued on a lock.

* **Process groups.** A group is a Job Object held by the process that
  started it and killed when that process exits (`KILL_ON_JOB_CLOSE`). On
  POSIX an agent's group outlives a driver that crashes, and the next driver
  has to find it and stop it; on Windows the kernel ends it with its driver.
  So a group this process did not start never needs signalling from here,
  and is never signalled: it reads as gone.

* **Stopping a process.** `os.kill(pid, sig)` is `TerminateProcess` — no
  cleanup — for every signal but the console's, and `os.kill(pid, 0)` kills
  the process it was meant to ask about. `pid_alive` asks without killing;
  `request_stop` sets an event the target waits on (`listen_for_stop`),
  which reaches its main thread as Ctrl-C, so it unwinds the way it would
  on SIGTERM.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path

WINDOWS = os.name == "nt"

SIGKILL = getattr(signal, "SIGKILL", 9)
# What a process here treats as "stop": the terminal's, the session's and the
# operator's signals, and on Windows the console's Ctrl-Break.
STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK")
    if hasattr(signal, name)
)

O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
O_BINARY = getattr(os, "O_BINARY", 0)
_O_WINDOWS = getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)

if WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _BASIC_ACCOUNTING(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                    ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", wintypes.DWORD),
                    ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD),
                    ("TotalTerminatedProcesses", wintypes.DWORD)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                             wintypes.DWORD)
    _k32.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                               wintypes.DWORD, wintypes.LPDWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, wintypes.LPDWORD)
    _k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _k32.CreateEventW.restype = wintypes.HANDLE
    _k32.CreateEventW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL,
                                  wintypes.LPCWSTR)
    _k32.OpenEventW.restype = wintypes.HANDLE
    _k32.OpenEventW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    _k32.SetEvent.argtypes = (wintypes.HANDLE,)
    _k32.ResetEvent.argtypes = (wintypes.HANDLE,)
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _k32.LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))
    _k32.UnlockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))
    _ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)

    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5
    _ERROR_LOCK_VIOLATION = 33
    _ERROR_NOT_LOCKED = 158
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _EVENT_MODIFY_STATE = 0x0002
    _INFINITE = 0xFFFFFFFF
    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectBasicAccountingInformation = 1
    _JobObjectExtendedLimitInformation = 9
    _CREATE_SUSPENDED = 0x4
    _CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    # The locked byte: past the end of anything pinnforge writes, so the lock
    # never covers data a reader wants.
    _LOCK_OFFSET_HIGH = 0x40000000


# ─────────────────────────────── files ───────────────────────────────


def is_link(path) -> bool:
    """Is `path` a symbolic link — or, on Windows, a junction, which acts as one?"""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return getattr(st, "st_reparse_tag", 0) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)


def open_file(path, flags: int, mode: int = 0o777) -> int:
    """`os.open(path, flags | O_NOFOLLOW, mode)`, never in text mode.

    Windows cannot refuse a final link as it opens, so the link is refused
    just before: a race only an attacker able to create links could run,
    which on Windows takes privileges a block does not have.
    """
    if WINDOWS:
        if is_link(path):
            raise OSError(errno.ELOOP, "refusing to open a link", str(path))
        return os.open(path, flags | _O_WINDOWS, mode)
    return os.open(path, flags | O_NOFOLLOW, mode)


def replace(src, dst) -> None:
    """`os.replace(src, dst)`, which on Windows fails while anyone has `dst` open.

    A reader holds a file for milliseconds — the dashboard polling a state
    file — so on Windows the rename is retried for a couple of seconds before
    the error is let through.
    """
    if not WINDOWS:
        os.replace(src, dst)
        return
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def shared_tmp() -> Path:
    """The machine's shared scratch directory: `/tmp` on POSIX, the temp directory on Windows."""
    return Path(tempfile.gettempdir()) if WINDOWS else Path("/tmp")


def link_dir(target: Path, link: Path) -> None:
    """Make `link` point at the directory `target`.

    A symbolic link where the system allows one; on Windows without the
    privilege for it (no Developer Mode), a junction, which needs none.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if not WINDOWS:
            raise
        import _winapi

        _winapi.CreateJunction(str(Path(target).resolve()), str(link))


def link_file(target: Path, link: Path) -> None:
    """Make `link` the file `target`: a symbolic link, or on Windows without the
    privilege for one, a hard link (same volume), which also needs none."""
    try:
        link.symlink_to(target)
    except OSError:
        if not WINDOWS:
            raise
        os.link(target, link)


def read_link(path) -> Path:
    """Where the link (or junction) at `path` points, without Windows' `\\\\?\\` prefix."""
    raw = os.readlink(path)
    if WINDOWS and raw.startswith("\\\\?\\"):
        raw = raw[4:]
    return Path(raw)


# ─────────────────────────────── programs ───────────────────────────────

# The script an npm shim (`claude.cmd`, `codex.cmd`, …) hands to node, in both
# the old (`%~dp0`) and the current (`%dp0%`) form of npm's cmd-shim.
_NPM_SHIM_SCRIPT = re.compile(r'"%(?:dp0%|~dp0)\\([^"]+)"\s+%\*', re.IGNORECASE)
_BATCH_UNSAFE = "\r\n%^&|<>!\""


def resolve_command(argv: list[str]) -> list[str]:
    """`argv` with its program found the way a shell would find it.

    Unchanged on POSIX. On Windows `CreateProcess` looks for `name.exe` only,
    so the `name.cmd` shim npm installs for a CLI is never found; and a batch
    file runs under cmd.exe, which reads its arguments again — a newline ends
    the command, `%` expands, `^` disappears — and stops at 8191 characters,
    so a prompt does not survive it. An npm shim is therefore run as what it
    wraps, node and the package's script; any other batch file only with
    arguments it cannot mangle.
    """
    argv = list(argv)
    if not WINDOWS or not argv:
        return argv
    found = shutil.which(argv[0])
    if found is None:
        return argv
    shim = Path(found)
    if shim.suffix.lower() not in (".cmd", ".bat"):
        return [found, *argv[1:]]
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if m := _NPM_SHIM_SCRIPT.search(text):
        script = shim.parent / m.group(1)
        local_node = shim.parent / "node.exe"
        node = str(local_node) if local_node.is_file() else shutil.which("node")
        if node and script.is_file():
            return [node, str(script), *argv[1:]]
    if any(c in a for a in argv[1:] for c in _BATCH_UNSAFE):
        raise OSError(f"{found} is a batch file, and cmd.exe would rewrite this command's "
                      f"arguments on the way through; install a native build of {shim.stem}")
    return [found, *argv[1:]]


# ─────────────────────────────── locks ───────────────────────────────


def _fileno(f) -> int:
    return f if isinstance(f, int) else f.fileno()


def lock(f, *, block: bool = True) -> None:
    """An exclusive lock on the open file `f` (a descriptor or a file object).

    Held until `unlock` or until the file is closed — or its process dies, so
    it cannot go stale. Raises `BlockingIOError` if another holds it and
    `block` is false.
    """
    if not WINDOWS:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        return
    handle = msvcrt.get_osfhandle(_fileno(f))
    flags = _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY
    while True:
        ov = _OVERLAPPED(Offset=0, OffsetHigh=_LOCK_OFFSET_HIGH)
        if _k32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(ov)):
            return
        err = ctypes.get_last_error()
        if err != _ERROR_LOCK_VIOLATION:
            raise ctypes.WinError(err)
        if not block:
            raise BlockingIOError(errno.EWOULDBLOCK, "the file is locked by another process")
        time.sleep(0.05)


def unlock(f) -> None:
    if not WINDOWS:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_UN)
        return
    handle = msvcrt.get_osfhandle(_fileno(f))
    ov = _OVERLAPPED(Offset=0, OffsetHigh=_LOCK_OFFSET_HIGH)
    if not _k32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(ov)):
        err = ctypes.get_last_error()
        if err != _ERROR_NOT_LOCKED:
            raise ctypes.WinError(err)


# ─────────────────────────────── processes ───────────────────────────────


def _as_pid(pid) -> int | None:
    if isinstance(pid, bool):
        return None
    try:
        n = int(pid)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 0 < n < 2**31 else None


def pid_alive(pid) -> bool:
    """Is there a process `pid`? Asked without signalling it."""
    pid = _as_pid(pid)
    if pid is None:
        return False
    if not WINDOWS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED  # there, but not ours
    try:
        code = wintypes.DWORD()
        return bool(_k32.GetExitCodeProcess(h, ctypes.byref(code))) and \
            code.value == _STILL_ACTIVE
    finally:
        _k32.CloseHandle(h)


def kill(pid) -> None:
    """Kill process `pid` outright (SIGKILL). `ProcessLookupError` if it is gone."""
    pid = _as_pid(pid)
    if pid is None:
        raise ProcessLookupError(errno.ESRCH, "no such process")
    if not WINDOWS:
        os.kill(pid, signal.SIGKILL)
        return
    h = _k32.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if not h:
        raise ProcessLookupError(errno.ESRCH, "no such process")
    try:
        if not _k32.TerminateProcess(h, 1) and pid_alive(pid):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _k32.CloseHandle(h)


def request_stop(pid) -> bool:
    """Ask pinnforge process `pid` to stop the way Ctrl-C would. False if it cannot be asked.

    SIGTERM on POSIX. On Windows, the event `pid` listens on
    (`listen_for_stop`): a process that does not listen cannot be asked,
    only killed.
    """
    pid = _as_pid(pid)
    if pid is None:
        return False
    if not WINDOWS:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
        return True
    h = _k32.OpenEventW(_EVENT_MODIFY_STATE, False, _stop_event_name(pid))
    if not h:
        return False
    try:
        return bool(_k32.SetEvent(h))
    finally:
        _k32.CloseHandle(h)


def _stop_event_name(pid: int) -> str:
    return f"Local\\pinnforge-stop-{pid}"


_listening: dict = {"handle": None}


def listen_for_stop() -> None:
    """Let `request_stop` reach this process (Windows; POSIX has SIGTERM for it).

    Each request is delivered to the main thread as Ctrl-C — the one signal
    that also wakes a main thread asleep in `time.sleep` — so it takes the
    path Ctrl-C takes. A second request delivers a second one.
    """
    if not WINDOWS or _listening["handle"] is not None:
        return
    h = _k32.CreateEventW(None, True, False, _stop_event_name(os.getpid()))
    if not h:
        return
    _listening["handle"] = h

    def wait() -> None:
        while _k32.WaitForSingleObject(h, _INFINITE) == 0:
            _k32.ResetEvent(h)
            signal.raise_signal(signal.SIGINT)

    threading.Thread(target=wait, name="pinnforge-stop", daemon=True).start()


# ─────────────────────────────── process groups ───────────────────────────────

# pid of a group's leader -> the Job Object holding the group (Windows).
_jobs: dict[int, int] = {}


def _new_job() -> int | None:
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation,
                                        ctypes.byref(info), ctypes.sizeof(info)):
        _k32.CloseHandle(job)
        return None
    return job


def popen_group(cmd, **kwargs) -> subprocess.Popen:
    """`Popen(cmd, start_new_session=True, **kwargs)`: `cmd` leads a process group of its own.

    On Windows the group is a Job Object this process holds; the child is
    started suspended and released only once it is in the job, so nothing it
    starts can slip out of it. The group dies with this process.
    """
    if not WINDOWS:
        return subprocess.Popen(cmd, start_new_session=True, **kwargs)
    flags = kwargs.pop("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, creationflags=flags | _CREATE_SUSPENDED, **kwargs)
    try:
        job = _new_job()
        if job is not None and not _k32.AssignProcessToJobObject(job, int(proc._handle)):
            _k32.CloseHandle(job)
            job = None
        if job is not None:
            old = _jobs.pop(proc.pid, None)
            if old is not None:
                _k32.CloseHandle(old)
            _jobs[proc.pid] = job
    finally:
        _ntdll.NtResumeProcess(int(proc._handle))
    return proc


def popen_detached(cmd, **kwargs) -> subprocess.Popen:
    """Start `cmd` so that it outlives this process and this terminal.

    POSIX: a session of its own. Windows: a console of its own with no
    window — so whatever it starts shares that console instead of each
    opening one — out of any job this process is in, where the job allows.
    """
    if not WINDOWS:
        return subprocess.Popen(cmd, start_new_session=True, **kwargs)
    flags = (kwargs.pop("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
             | subprocess.CREATE_NO_WINDOW)
    try:
        return subprocess.Popen(cmd, creationflags=flags | _CREATE_BREAKAWAY_FROM_JOB, **kwargs)
    except PermissionError:
        return subprocess.Popen(cmd, creationflags=flags, **kwargs)


def own_pgid() -> int | None:
    """This process's group (`os.getpgid(0)`); None on Windows, where it has none to guard."""
    return None if WINDOWS else os.getpgid(0)


def getpgid(pid: int) -> int:
    """`os.getpgid(pid)`. On Windows, `pid` if it leads a group this process started."""
    if not WINDOWS:
        return os.getpgid(pid)
    if pid in _jobs:
        return pid
    raise ProcessLookupError(errno.ESRCH, "not a process group this process started")


def _job_active(job: int) -> int:
    info = _BASIC_ACCOUNTING()
    if not _k32.QueryInformationJobObject(job, _JobObjectBasicAccountingInformation,
                                          ctypes.byref(info), ctypes.sizeof(info), None):
        return 0
    return int(info.ActiveProcesses)


def group_alive(pgid) -> bool:
    """Does process group `pgid` have a live member? (`os.killpg(pgid, 0)` succeeding.)"""
    pgid = _as_pid(pgid)
    if pgid is None:
        return False
    if not WINDOWS:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    job = _jobs.get(pgid)
    return job is not None and _job_active(job) > 0


def killpg(pgid, sig: int) -> None:
    """`os.killpg(pgid, sig)`: `ProcessLookupError` once nothing is left to signal.

    On Windows: `SIGKILL` terminates the job; any other stop signal is the
    console's Ctrl-Break, which reaches every member sharing this console
    (and a Python one as SIGBREAK, which pinnforge's processes unwind on).
    """
    if not WINDOWS:
        os.killpg(pgid, sig)
        return
    pgid = _as_pid(pgid)
    if pgid is None or not group_alive(pgid):
        raise ProcessLookupError(errno.ESRCH, "no such process group")
    if sig == 0:
        return
    if sig == SIGKILL:
        _k32.TerminateJobObject(_jobs[pgid], 1)
        return
    with contextlib.suppress(OSError):
        os.kill(pgid, signal.CTRL_BREAK_EVENT)


def forget_group(pgid) -> None:
    """Drop this process's hold on group `pgid` (Windows: close its job, ending what is left)."""
    if WINDOWS and (job := _jobs.pop(_as_pid(pgid) or 0, None)) is not None:
        _k32.CloseHandle(job)
