"""The operating-system layer: the POSIX contract each call keeps, on Windows too.

Everything here runs in real processes. The contract is the one the rest of
the code was written against — an exclusive lock that never stops a reader,
a liveness check that does not kill, a group killed whole, a stop that
arrives as Ctrl-C — and Windows keeps it by other means (see `oscompat`).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from pinnforge import oscompat

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows semantics")


def _until(cond, timeout=20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _pid_in(path: Path) -> int:
    assert _until(lambda: path.is_file() and path.read_text(encoding="utf-8").strip()), path
    return int(path.read_text(encoding="utf-8"))


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


# ─────────────────────────────── files and locks ───────────────────────────────


def test_a_lock_is_exclusive_but_never_stops_a_reader(tmp_path):
    """Windows locks are mandatory: the locked byte lies past anything written."""
    f = tmp_path / "run.lock"
    fd = oscompat.open_file(f, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.write(fd, b'{"pid": 1}\n')
        oscompat.lock(fd, block=False)
        probe = _python(f"""
            import os
            from pinnforge import oscompat
            fd = oscompat.open_file({str(f)!r}, os.O_RDONLY)
            print(os.read(fd, 100))
            try:
                oscompat.lock(fd, block=False)
                print("locked")
            except BlockingIOError:
                print("held")
        """)
        out = subprocess.run(probe, capture_output=True, text=True, timeout=60).stdout.split()
        assert out == ["b'{\"pid\":", "1}\\n'", "held"], out
        oscompat.unlock(fd)
        out = subprocess.run(probe, capture_output=True, text=True, timeout=60).stdout.split()
        assert out[-1] == "locked", "a released lock is anybody's"
    finally:
        os.close(fd)


def test_a_file_opened_here_is_never_in_text_mode(tmp_path):
    fd = oscompat.open_file(tmp_path / "x", os.O_WRONLY | os.O_CREAT, 0o644)
    os.write(fd, b"a\nb\x1ac")
    os.close(fd)
    assert (tmp_path / "x").read_bytes() == b"a\nb\x1ac"
    fd = oscompat.open_file(tmp_path / "x", os.O_RDONLY)
    try:
        assert os.read(fd, 100) == b"a\nb\x1ac", "no \\r added, no stop at Ctrl-Z"
    finally:
        os.close(fd)


def test_a_linked_directory_is_refused_by_open_file(tmp_path):
    """A junction needs no privilege on Windows, so it is the link a block can plant."""
    (tmp_path / "answers").mkdir()
    oscompat.link_dir(tmp_path / "answers", tmp_path / "lock")
    assert oscompat.is_link(tmp_path / "lock")
    assert oscompat.read_link(tmp_path / "lock").resolve() == (tmp_path / "answers").resolve()
    with pytest.raises(OSError):
        oscompat.open_file(tmp_path / "lock", os.O_RDONLY)


def test_a_replace_waits_out_a_reader(tmp_path):
    """On Windows a file being read cannot be renamed over; the reader is waited out."""
    dest, src = tmp_path / "state.json", tmp_path / "state.json.tmp"
    dest.write_text("old", encoding="utf-8")
    src.write_text("new", encoding="utf-8")
    reader = dest.open("rb")
    threading.Timer(0.3, reader.close).start()
    oscompat.replace(src, dest)
    assert dest.read_text(encoding="utf-8") == "new"


# ─────────────────────────────── processes ───────────────────────────────


def test_asking_whether_a_process_lives_does_not_kill_it():
    """`os.kill(pid, 0)` is TerminateProcess on Windows: the question killed the answer."""
    proc = subprocess.Popen(_python("import time; time.sleep(30)"))
    try:
        assert oscompat.pid_alive(proc.pid)
        time.sleep(0.2)
        assert proc.poll() is None, "asked, not killed"
        assert not oscompat.pid_alive(2**31 - 3)
    finally:
        proc.kill()
        proc.wait()
    assert not oscompat.pid_alive(proc.pid)


def test_a_group_is_killed_whole(tmp_path):
    """The worker under the agent is the one holding the card; it goes with the group."""
    pidfile = tmp_path / "worker.pid"
    leader = oscompat.popen_group(_python(f"""
        import subprocess, sys, time
        w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pidfile)!r}, "w").write(str(w.pid))
        time.sleep(60)
    """))
    try:
        worker = _pid_in(pidfile)
        pgid = oscompat.getpgid(leader.pid)
        assert oscompat.group_alive(pgid)
        oscompat.killpg(pgid, oscompat.SIGKILL)
        leader.wait(10)
        assert _until(lambda: not oscompat.group_alive(pgid), 10)
        with pytest.raises(ProcessLookupError):
            oscompat.killpg(pgid, 0)
        assert _until(lambda: not oscompat.pid_alive(worker), 10), "the worker was orphaned"
    finally:
        leader.kill()
        leader.wait()
        oscompat.forget_group(leader.pid)


@windows_only
def test_a_group_dies_with_the_process_that_started_it(tmp_path):
    """No orphan to reap on Windows: the kernel ends the group with its owner."""
    pidfile = tmp_path / "agent.pid"
    owner = subprocess.Popen(_python(f"""
        import sys, time
        from pinnforge import oscompat
        agent = oscompat.popen_group([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pidfile)!r}, "w").write(str(agent.pid))
        time.sleep(60)
    """))
    try:
        agent = _pid_in(pidfile)
        assert oscompat.pid_alive(agent)
        owner.kill()  # nothing to catch: TerminateProcess
        owner.wait()
        assert _until(lambda: not oscompat.pid_alive(agent), 10), "the agent outlived its driver"
    finally:
        owner.kill()
        owner.wait()


def test_a_stop_request_arrives_as_ctrl_c(tmp_path):
    """`run stop` from another terminal: SIGTERM on POSIX, an event on Windows."""
    pidfile, done = tmp_path / "driver.pid", tmp_path / "unwound"
    proc = subprocess.Popen(_python(f"""
        import os, signal, time
        from pinnforge import oscompat
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        oscompat.listen_for_stop()
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            open({str(done)!r}, "w").write("yes")
    """))
    try:
        driver = _pid_in(pidfile)  # the interpreter's own pid, as a driver records it
        assert oscompat.request_stop(driver)
        proc.wait(20)
        assert done.read_text(encoding="utf-8") == "yes", "it unwound as Ctrl-C does"
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("how", ["wait", "communicate"])
def test_a_wait_on_a_child_gives_way_to_a_stop_request(how):
    """On Windows `Popen.wait` held a `run stop` off until the agent it waited on exited."""
    out = subprocess.run(_python(f"""
        import os, signal, subprocess, sys, threading, time
        from pinnforge import oscompat
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        oscompat.listen_for_stop()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 stdout=subprocess.PIPE)
        threading.Timer(0.5, oscompat.request_stop, (os.getpid(),)).start()
        t0 = time.monotonic()
        try:
            oscompat.{how}(child, 60)
        except KeyboardInterrupt:
            print(round(time.monotonic() - t0, 1))
        child.kill()
        child.wait()
    """), capture_output=True, text=True, timeout=60)
    assert out.stdout.strip(), out.stderr
    assert float(out.stdout) < 5, f"the stop waited {out.stdout.strip()} s for the child"


@windows_only
def test_a_stop_reaches_a_group_member_in_a_console_of_its_own(tmp_path):
    """Harnesses hide their tools' windows, which gives each its own console."""
    pidfile, done = tmp_path / "eval.pid", tmp_path / "unwound"
    proc = oscompat.popen_group(_python(f"""
        import os, signal, time
        from pinnforge import oscompat
        signal.signal(signal.SIGINT, lambda *_: open({str(done)!r}, "w").write("yes"))
        oscompat.listen_for_stop()
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        time.sleep(60)
    """), creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        _pid_in(pidfile)
        oscompat.killpg(proc.pid, signal.SIGTERM)
        assert _until(done.is_file, 10), "the group's stop never reached it"
    finally:
        with contextlib.suppress(ProcessLookupError):
            oscompat.killpg(proc.pid, oscompat.SIGKILL)
        proc.wait()
        oscompat.forget_group(proc.pid)


@windows_only
def test_a_group_member_sharing_this_console_is_stopped_once(tmp_path):
    """The Ctrl-Break reaches it; a stop request as well could cut its unwinding short."""
    pidfile, got = tmp_path / "member.pid", tmp_path / "got"
    proc = oscompat.popen_group(_python(f"""
        import os, signal, time
        from pinnforge import oscompat
        def note(signum, frame):
            with open({str(got)!r}, "a") as f:
                f.write(signal.Signals(signum).name + "\\n")
        signal.signal(signal.SIGINT, note)
        signal.signal(signal.SIGBREAK, note)
        oscompat.listen_for_stop()
        open({str(pidfile)!r}, "w").write(str(os.getpid()))
        for _ in range(600):  # SIGBREAK does not cut a sleep short, as SIGINT does
            time.sleep(0.1)
    """))
    try:
        _pid_in(pidfile)
        oscompat.killpg(proc.pid, signal.SIGTERM)
        assert _until(got.is_file, 10), "the Ctrl-Break never reached it"
        time.sleep(1.0)
        assert got.read_text(encoding="utf-8").split() == ["SIGBREAK"]
    finally:
        with contextlib.suppress(ProcessLookupError):
            oscompat.killpg(proc.pid, oscompat.SIGKILL)
        proc.wait()
        oscompat.forget_group(proc.pid)


@windows_only
def test_a_process_that_does_not_listen_cannot_be_asked():
    proc = subprocess.Popen(_python("import time; time.sleep(30)"))
    try:
        time.sleep(0.5)
        assert not oscompat.request_stop(proc.pid)
        assert proc.poll() is None, "and is not killed for being asked"
    finally:
        proc.kill()
        proc.wait()


# ─────────────────────────────── programs ───────────────────────────────


NPM_SHIM = "\r\n".join([
    "@ECHO off", "GOTO start", ":find_dp0", "SET dp0=%~dp0", "EXIT /b", ":start", "SETLOCAL",
    "CALL :find_dp0", "", 'IF EXIST "%dp0%\\node.exe" (', '  SET "_prog=%dp0%\\node.exe"',
    ") ELSE (", '  SET "_prog=node"', "  SET PATHEXT=%PATHEXT:;.JS;=;%", ")", "",
    'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  '
    '"%dp0%\\node_modules\\@fake\\agent\\cli.js" %*', "",
])


@windows_only
@pytest.mark.skipif(not __import__("shutil").which("node"), reason="needs node")
def test_an_npm_shim_runs_as_node_and_its_script(tmp_path, monkeypatch):
    """Through cmd.exe a prompt ends at its first newline; through node it arrives whole."""
    script = tmp_path / "node_modules" / "@fake" / "agent" / "cli.js"
    script.parent.mkdir(parents=True)
    script.write_text("process.stdout.write(JSON.stringify(process.argv.slice(2)))",
                      encoding="utf-8")
    (tmp_path / "fakeagent.cmd").write_text(NPM_SHIM, encoding="utf-8")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    prompt = 'one\ntwo: 100% ^done & | <x> "quoted" !bang'
    argv = oscompat.resolve_command(["fakeagent", "-p", prompt, "--settings", '{"a": false}'])
    assert Path(argv[1]) == script
    out = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=60)
    import json

    assert json.loads(out.stdout) == ["-p", prompt, "--settings", '{"a": false}']


@windows_only
def test_another_batch_file_is_refused_arguments_it_would_rewrite(tmp_path, monkeypatch):
    (tmp_path / "tool.cmd").write_text("@echo %*\r\n", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert Path(oscompat.resolve_command(["tool", "--version"])[0]).name.lower() == "tool.cmd"
    with pytest.raises(OSError, match="batch file"):
        oscompat.resolve_command(["tool", "line one\nline two"])


def test_a_program_elsewhere_is_left_as_it_is():
    assert oscompat.resolve_command([]) == []
    if os.name != "nt":
        assert oscompat.resolve_command(["claude", "-p", "x"]) == ["claude", "-p", "x"]
