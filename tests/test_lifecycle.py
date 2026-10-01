"""One driver per run, stop signals that unwind cleanly, and detached launches."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from pinnforge.run import lifecycle


def test_a_run_has_one_driver(tmp_path):
    first = lifecycle.RunLock(tmp_path).acquire()
    try:
        assert lifecycle.holder(tmp_path)["pid"] == os.getpid()
        with pytest.raises(lifecycle.RunLocked, match="already being driven"):
            lifecycle.RunLock(tmp_path).acquire()
    finally:
        first.release()
    assert lifecycle.holder(tmp_path) is None, "a released lock is nobody's"


def test_the_lock_records_the_agent_group(tmp_path):
    lock = lifecycle.RunLock(tmp_path).acquire()
    try:
        lock.update(agent_pgid=4242)
        assert lifecycle.holder(tmp_path)["agent_pgid"] == 4242
    finally:
        lock.release()


def test_a_signal_inside_a_critical_section_waits_for_its_end():
    lifecycle.install_signal_handlers()
    try:
        reached = []
        with pytest.raises(lifecycle.RunStopped), lifecycle.critical():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            reached.append("end of section")
        assert reached == ["end of section"], "the section was not torn in half"
    finally:
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.default_int_handler if s == signal.SIGINT else signal.SIG_DFL)


def test_a_second_signal_escalates_instead_of_raising():
    escalated = []
    lifecycle.install_signal_handlers(escalate=lambda: escalated.append(True))
    try:
        with pytest.raises(lifecycle.RunStopped):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        # cleanup is running now; a second signal must not abort it
        with lifecycle.critical():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        assert escalated == [True]
    finally:
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.default_int_handler if s == signal.SIGINT else signal.SIG_DFL)


DRIVER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from pinnforge.run import lifecycle
    run = Path(sys.argv[1])
    lock = lifecycle.RunLock(run).acquire()
    lifecycle.install_signal_handlers()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        (run / "cleaned").write_text("yes")
    finally:
        lock.release()
""")


def test_stop_unwinds_the_driver_like_ctrl_c(tmp_path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(tmp_path)])
    try:
        for _ in range(100):
            if lifecycle.holder(tmp_path):
                break
            time.sleep(0.05)
        assert lifecycle.stop(tmp_path, wait_s=10) == "stopped"
        assert (tmp_path / "cleaned").read_text() == "yes", "the cleanup path ran"
    finally:
        proc.kill()
        proc.wait()
    assert lifecycle.stop(tmp_path) == "not running"


def test_detach_without_tmux_logs_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: None)
    log = tmp_path / "out.log"
    info = lifecycle.detach([sys.executable, "-c", "print('hello from the background')"],
                            cwd=tmp_path, log=log, name="x", record=tmp_path / "d.json")
    assert info["mode"] == "background"
    for _ in range(100):
        if "hello" in log.read_text(encoding="utf-8"):
            break
        time.sleep(0.05)
    assert "hello from the background" in log.read_text(encoding="utf-8")


def test_session_names_do_not_collide_across_workspaces(tmp_path):
    a = lifecycle.session_name(tmp_path / "one", "ldc_1")
    b = lifecycle.session_name(tmp_path / "two", "ldc_1")
    assert a != b and a.endswith("ldc_1")


def _other_holder():
    """A live process that is not this one, to stand in for another run's driver."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_sharing_a_card_does_not_erase_the_other_runs_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "LEASE_DIR", tmp_path / "leases")
    other = _other_holder()
    try:
        lifecycle.LEASE_DIR.mkdir()
        lifecycle._lease_path(0, other.pid).write_text(
            f'{{"pid": {other.pid}, "run": "runs/first"}}\n', encoding="utf-8"
        )
        assert [info["run"] for _, info in lifecycle.leases([0, 1])] == ["runs/first"]

        lifecycle.claim([0], tmp_path / "second")  # joined with --share
        assert [info["run"] for _, info in lifecycle.leases([0])] == ["runs/first"], (
            "a run never counts its own claim as someone else's"
        )
        lifecycle.release_claim([0])
        assert [info["run"] for _, info in lifecycle.leases([0])] == ["runs/first"], (
            "the second run finishing must leave the first run's claim in place"
        )
    finally:
        other.kill()
        other.wait()
    assert lifecycle.leases([0]) == [], "a dead holder's claim is not a claim"
    assert not list(lifecycle.LEASE_DIR.iterdir()), "and it is swept"


def test_the_demo_takes_only_free_cards(tmp_path, monkeypatch, capsys):
    from pinnforge.cli import demo_cmd

    monkeypatch.setattr(demo_cmd.gpus, "workspace_pool", lambda root: [0, 1])
    held = {}
    monkeypatch.setattr(lifecycle, "leases",
                        lambda pool: [(g, held[g]) for g in pool if g in held])

    assert demo_cmd._free_cards(tmp_path, share=False) == "", "all free: the workspace's choice"
    held[0] = {"run": "runs/ldc_1", "pid": 1}
    assert demo_cmd._free_cards(tmp_path, share=False) == "1"
    held[1] = {"run": "runs/ks_1", "pid": 2}
    assert demo_cmd._free_cards(tmp_path, share=False) is None
    assert "--share" in capsys.readouterr().out
    assert demo_cmd._free_cards(tmp_path, share=True) == "", "--share keeps the whole pool"


def test_the_anchor_is_done_only_once_it_is_scored(tmp_path):
    assert lifecycle.anchor_status(tmp_path, scored=True) == "done"
    assert lifecycle.anchor_status(tmp_path, scored=False) == "missing"
    lock = lifecycle.RunLock(tmp_path).acquire()
    try:
        assert lifecycle.anchor_status(tmp_path, scored=False) == "measuring", (
            "a pre-charged budget and an empty record mean the measurement is under way"
        )
    finally:
        lock.release()


def test_a_finished_runs_tmux_socket_is_swept(tmp_path, monkeypatch):
    import socket

    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path))
    base = tmp_path / f"tmux-{os.getuid()}"
    base.mkdir()

    def sock(name, listen):
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(base / name))
        if listen:
            s.listen()
        else:
            s.close()
        return s

    sock("pf-abc123-run-old_1", listen=False)
    live = sock("pf-abc123-run-live_1", listen=True)
    sock("pf-fff000-run-other_1", listen=False)
    long_ago = time.time() - 3600
    for name in ("pf-abc123-run-old_1", "pf-abc123-run-live_1", "pf-fff000-run-other_1"):
        os.utime(base / name, (long_ago, long_ago))
    # bound a moment ago by a `run start` whose tmux server is not listening yet
    sock("pf-abc123-run-starting_1", listen=False)
    try:
        lifecycle._sweep_sockets("pf-abc123-")
        left = sorted(p.name for p in base.iterdir())
    finally:
        live.close()
    assert left == ["pf-abc123-run-live_1", "pf-abc123-run-starting_1",
                    "pf-fff000-run-other_1"], (
        "only this workspace's dead sockets go; a live run, one still starting and "
        "other workspaces are untouched"
    )


# ─────────────────── what a crashed driver leaves behind ───────────────────

linux_only = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs /proc")


def _orphan_agent(cwd):
    """An agent left running by a driver that died: its own session, working in `cwd`."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                            cwd=str(cwd), start_new_session=True)
    # reap it the moment it dies, as init would for a real orphan
    import threading

    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _crashed_driver_lock(run, pgid, host=None):
    """The lock file as a driver killed with SIGKILL leaves it: info written, flock gone."""
    import json
    import socket

    (run / lifecycle.LOCK_FILE).write_text(json.dumps({
        "pid": 999999, "host": host or socket.gethostname(),
        "started": "earlier", "agent_pgid": pgid,
    }) + "\n", encoding="utf-8")


@linux_only
def test_a_new_driver_stops_the_agent_a_crashed_one_left_running(tmp_path):
    """Two agents on one workspace: what resume used to start after a driver crash."""
    agent = _orphan_agent(tmp_path)
    try:
        _crashed_driver_lock(tmp_path, agent.pid)
        lock = lifecycle.RunLock(tmp_path).acquire()
        try:
            assert not lifecycle.group_alive(agent.pid), "the orphaned agent was stopped"
            assert lock.info["agent_pgid"] is None
        finally:
            lock.release()
    finally:
        agent.kill()


@linux_only
def test_a_reused_group_number_is_left_alone(tmp_path):
    """A recorded pgid now naming a group that works elsewhere is not this run's agent."""
    elsewhere = tmp_path / "elsewhere"
    run = tmp_path / "run"
    elsewhere.mkdir()
    run.mkdir()
    stranger = _orphan_agent(elsewhere)
    try:
        _crashed_driver_lock(run, stranger.pid)
        lifecycle.RunLock(run).acquire().release()
        assert lifecycle.group_alive(stranger.pid), "an unrelated group must survive"
    finally:
        stranger.kill()


def test_another_hosts_agent_is_not_signalled(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(lifecycle, "terminate_group", lambda pgid, grace=10.0: killed.append(pgid))
    monkeypatch.setattr(lifecycle, "group_alive", lambda pgid: True)
    _crashed_driver_lock(tmp_path, 4242, host="some-other-host")
    lifecycle.RunLock(tmp_path).acquire().release()
    assert killed == []


def test_stop_refuses_a_run_driven_from_another_host(tmp_path, monkeypatch):
    """On a shared filesystem the lock's pid is another machine's number."""
    lock = lifecycle.RunLock(tmp_path).acquire()
    try:
        lock.update(host="node-elsewhere", pid=os.getpid())
        signalled = []
        monkeypatch.setattr(lifecycle.os, "kill", lambda pid, sig: signalled.append(pid))
        outcome = lifecycle.stop(tmp_path)
        assert outcome.startswith("refused") and "node-elsewhere" in outcome
        assert signalled == []
    finally:
        lock.release()


def test_a_planted_lock_file_neither_hangs_nor_is_followed(tmp_path):
    """`.lock` sits in the block's working directory; a FIFO there used to hang open()."""
    os.mkfifo(tmp_path / lifecycle.LOCK_FILE)
    assert lifecycle.holder(tmp_path) is None
    with pytest.raises(lifecycle.RunLocked, match="not a regular file"):
        lifecycle.RunLock(tmp_path).acquire()
    (tmp_path / lifecycle.LOCK_FILE).unlink()
    (tmp_path / lifecycle.LOCK_FILE).symlink_to(tmp_path / "target")
    with pytest.raises(lifecycle.RunLocked):
        lifecycle.RunLock(tmp_path).acquire()
    assert not (tmp_path / "target").exists()


def test_a_recorded_pid_must_be_one_that_may_be_signalled():
    """kill(-1, ...) is every process the user owns; 0 is the caller's own group."""
    for bad in (-1, 0, 1, None, "x", [], True, 2**80):
        assert lifecycle._as_pid(bad) is None, bad
        assert not lifecycle._pid_alive(bad)
    assert lifecycle._as_pid(os.getpid()) == os.getpid()
    assert lifecycle._as_pid(str(os.getpid())) == os.getpid()


def test_leases_tolerate_whatever_is_in_the_shared_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "LEASE_DIR", tmp_path / "leases")
    lifecycle.LEASE_DIR.mkdir()
    (lifecycle.LEASE_DIR / "gpu0.1.json").write_text("[1, 2, 3]\n", encoding="utf-8")
    (lifecycle.LEASE_DIR / "gpu0.2.json").write_text("not json", encoding="utf-8")
    os.mkfifo(lifecycle.LEASE_DIR / "gpu0.3.json")
    (lifecycle.LEASE_DIR / "gpu0.4.json").symlink_to(tmp_path / "nowhere")
    assert lifecycle.holders(0) == [], "none of these is a claim, and none may hang the reader"

    lifecycle.claim([0], tmp_path / "run")
    try:
        mine = lifecycle.holders(0)
        assert [i["run"] for i in mine] == [str(tmp_path / "run")]
        assert not [p for p in lifecycle.LEASE_DIR.iterdir() if p.name.endswith(".tmp")], (
            "a claim is written whole and renamed into place"
        )
    finally:
        lifecycle.release_claim([0])
