"""The shared eval kit's device handling, lock and budget — without training anything."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pinnforge import oscompat
from pinnforge.task import evalkit

LITERAL = "TRAIN_TIME = 60.0"


@pytest.fixture
def kit(tmp_path, monkeypatch):
    """A run with one candidate, and an eval kit whose worker and lock are recorded."""
    run = tmp_path / "toy_1"
    (run / "task").mkdir(parents=True)
    (run / "task" / "eval.py").write_text("", encoding="utf-8")
    (run / "blocks" / "b01").mkdir(parents=True)
    cand = run / "blocks" / "b01" / "b01_x.py"
    cand.write_text(f"{LITERAL}\nFORGE_EVAL_TOKEN = None\n", encoding="utf-8")
    seen = {}

    def worker(eval_py, mode, argv, env, wall, **kw):
        seen["env"] = env
        seen["keep_fds"] = kw.get("keep_fds")
        return {"error": "stub: nothing trained"}, 1.0

    monkeypatch.setattr(evalkit, "_run_worker", worker)
    real_open = evalkit._open_lock

    def open_lock(path):  # record which device lock, but take a private one
        seen.setdefault("lock", path)
        return real_open(str(tmp_path / "device.lock"))

    monkeypatch.setattr(evalkit, "_open_lock", open_lock)
    g = {"__file__": str(run / "task" / "eval.py"), "__doc__": "",
         "TRAIN_TIME_LITERAL": LITERAL, "SMOKE_TRAIN_TIME": "TRAIN_TIME = 3.00",
         "TRAIN_WALL_S": 60.0, "SMOKE_WALL_S": 3.0, "EVAL_WALL_S": 120.0, "BUDGET": 600.0}

    def evaluate(*flags):
        monkeypatch.setattr(sys, "argv", ["eval.py", str(cand), *flags])
        assert evalkit.run(g) == 0
        log = run / "blocks" / "b01" / "evals.jsonl"
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        return rec, seen

    return evaluate


def test_a_gpu_run_trains_on_its_card_and_queues_on_its_lock(kit, monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    rec, seen = kit("--gpu", "7")
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert seen["lock"] == os.path.join(oscompat.shared_tmp(), "pinnforge_gpu7.lock")
    assert rec["gpu"] == "7"


def test_a_cpu_run_hides_every_card_and_leaves_card_0s_queue_alone(kit, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    rec, seen = kit("--gpu", "0")
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "", "a hidden card cannot be trained on"
    assert seen["lock"] == os.path.join(oscompat.shared_tmp(), "pinnforge_cpu.lock"), (
        "a CPU run must not wait behind a GPU run's evaluations on card 0"
    )
    assert rec["gpu"] == "cpu", "the record names what was used"


# ─────────────────────────── the device lock ───────────────────────────

def test_another_users_lock_file_is_usable_without_write_access(tmp_path):
    """`open(path, "w")` failed on a lock file someone else created first."""
    lock = tmp_path / "pinnforge_gpu0.lock"
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o444)  # as unwritable as another user's file under protected_regular
    fd = evalkit._open_lock(str(lock))
    try:
        oscompat.lock(fd, block=False)  # flock needs no write access
    finally:
        os.close(fd)


def test_a_new_lock_file_is_left_usable_by_everyone(tmp_path):
    old = os.umask(0o077)
    try:
        os.close(evalkit._open_lock(str(tmp_path / "pinnforge_gpu1.lock")))
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "pinnforge_gpu1.lock").stat().st_mode) == 0o666


def test_a_planted_lock_file_is_refused_not_followed_or_waited_on(tmp_path, symlink):
    if hasattr(os, "mkfifo"):  # Windows has no FIFOs to plant
        fifo = tmp_path / "fifo.lock"
        os.mkfifo(fifo)
        with pytest.raises(OSError):
            evalkit._open_lock(str(fifo))  # a FIFO would block a plain open forever
    link = symlink(tmp_path / "link.lock", tmp_path / "elsewhere")
    with pytest.raises(OSError):
        evalkit._open_lock(str(link))
    assert not (tmp_path / "elsewhere").exists()


def _toy(tmp_path):
    run = tmp_path / "toy_1"
    (run / "task").mkdir(parents=True)
    (run / "task" / "eval.py").write_text("", encoding="utf-8")
    (run / "blocks" / "b01").mkdir(parents=True)
    cand = run / "blocks" / "b01" / "b01_x.py"
    cand.write_text(f"{LITERAL}\nFORGE_EVAL_TOKEN = None\n", encoding="utf-8")
    g = {"__file__": str(run / "task" / "eval.py"), "__doc__": "",
         "TRAIN_TIME_LITERAL": LITERAL, "SMOKE_TRAIN_TIME": "TRAIN_TIME = 3.00",
         "TRAIN_WALL_S": 60.0, "SMOKE_WALL_S": 3.0, "EVAL_WALL_S": 120.0, "BUDGET": 600.0}
    return run, cand, g


def test_a_lock_that_cannot_be_opened_costs_the_block_nothing(tmp_path, monkeypatch):
    """The lock used to be opened after the training wall had been charged."""
    run, cand, g = _toy(tmp_path)

    def refuse(path):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(evalkit, "_open_lock", refuse)
    monkeypatch.setattr(sys, "argv", ["eval.py", str(cand), "--gpu", "0"])
    assert evalkit.run(g) == 2
    budget = run / "blocks" / "b01" / ".budget"
    assert not budget.exists() or float(budget.read_text(encoding="utf-8") or 0) == 0.0


def test_a_crash_before_training_refunds_the_pre_charge(tmp_path, monkeypatch):
    run, cand, g = _toy(tmp_path)
    monkeypatch.setattr(evalkit, "_open_lock",
                        lambda path: os.open(tmp_path / "dev.lock", os.O_RDWR | os.O_CREAT))

    def boom(*a, **kw):
        raise RuntimeError("the worker could not be started")

    monkeypatch.setattr(evalkit, "_run_worker", boom)
    monkeypatch.setattr(sys, "argv", ["eval.py", str(cand), "--gpu", "0"])
    with pytest.raises(RuntimeError):
        evalkit.run(g)
    assert float((run / "blocks" / "b01" / ".budget").read_text(encoding="utf-8")) == 0.0
    assert signal.getsignal(signal.SIGTERM) is not evalkit._on_signal, "handlers restored"


def test_a_one_column_grid_is_predicted_on(tmp_path):
    """np.loadtxt hands back 1-D for one column; the rows-and-columns slice then raised."""
    grid = tmp_path / "grid.csv"
    grid.write_text("x\n0.0\n0.5\n1.0\n", encoding="utf-8")
    assert evalkit.prediction_inputs({"GRID_CSV": grid, "INPUT_COLUMNS": 1}).shape == (3, 1)
    seen = {}

    def interior(g):
        seen["ndim"] = g.ndim
        return g[:, 0] > 0.2

    X = evalkit.prediction_inputs({"GRID_CSV": grid, "INPUT_COLUMNS": 1, "predict_rows": interior})
    assert X.shape == (2, 1) and seen["ndim"] == 2, "predict_rows sees rows of columns too"


# ─────────────── a stopped evaluation, end to end, in real processes ───────────────

# A task whose training worker only says where it is and waits: what is under
# test is the parent — the lock it hands down, the signals it handles, the
# budget it settles — not training.
FAKE_EVAL = '''
import os, sys, time
from pathlib import Path

BUDGET = 600.0
TRAIN_TIME_LITERAL = "TRAIN_TIME = 60.0"
SMOKE_TRAIN_TIME = "TRAIN_TIME = 3.00"
TRAIN_WALL_S = 60.0
SMOKE_WALL_S = 3.0
EVAL_WALL_S = 30.0

if __name__ == "__main__":
    import _forgekit

    if sys.argv[1:2] == ["--worker-train"]:
        Path(os.environ["PF_TEST_WORKER"]).write_text(str(os.getpid()))
        time.sleep(120)
        raise SystemExit(1)
    real = _forgekit._open_lock
    _forgekit._open_lock = lambda path: real(os.environ["PF_TEST_LOCK"])
    raise SystemExit(_forgekit.main(globals()))
'''

linux_only = pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")
# Linux watches processes through /proc; Windows asks the process table.
live_only = pytest.mark.skipif(os.name != "nt" and not Path("/proc/self/fd").is_dir(),
                               reason="needs /proc")
# What stops an evaluation: SIGTERM, or on Windows its group's Ctrl-Break.
STOP = getattr(signal, "SIGBREAK", signal.SIGTERM) if os.name == "nt" else signal.SIGTERM


@pytest.fixture
def live(tmp_path):
    run = tmp_path / "toy_1"
    (run / "task").mkdir(parents=True)
    (run / "task" / "eval.py").write_text(FAKE_EVAL, encoding="utf-8")
    shutil.copy2(evalkit.__file__, run / "task" / "_forgekit.py")
    (run / "blocks" / "b01").mkdir(parents=True)
    (run / "blocks" / "b01" / "b01_x.py").write_text(
        f"{LITERAL}\nFORGE_EVAL_TOKEN = None\n", encoding="utf-8"
    )
    lock, pidfile = tmp_path / "dev.lock", tmp_path / "worker.pid"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PF_TEST_LOCK": str(lock),
           "PF_TEST_WORKER": str(pidfile)}
    procs = []

    def start(**kw):
        # a group of its own: on Windows a stop reaches a process only through its group
        p = oscompat.popen_group(
            [sys.executable, "task/eval.py", "blocks/b01/b01_x.py", "--gpu", "0"],
            cwd=run, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", **kw,
        )
        procs.append(p)
        return p

    class Live:
        pass

    lv = Live()
    lv.start, lv.lock, lv.pidfile = start, lock, pidfile
    lv.budget = run / "blocks" / "b01" / ".budget"
    yield lv
    for p in procs:
        p.kill()
        p.wait()
    if pidfile.is_file():
        with contextlib.suppress(ProcessLookupError, ValueError):
            oscompat.kill(int(pidfile.read_text(encoding="utf-8")))


def _until(cond, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _stop(proc) -> None:
    """The stop signal an orchestrator sends an evaluation."""
    if os.name == "nt":
        oscompat.killpg(proc.pid, signal.SIGTERM)  # the group's Ctrl-Break
    else:
        proc.send_signal(signal.SIGTERM)


def _running(pid: int) -> bool:
    if os.name == "nt":
        return oscompat.pid_alive(pid)
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def _lock_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        oscompat.lock(fd, block=False)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


@live_only
def test_a_stopped_eval_stops_its_worker_and_charges_only_what_it_took(live):
    proc = live.start()
    assert _until(lambda: live.pidfile.is_file() and live.pidfile.read_text(encoding="utf-8")), (
        "the worker never started")
    worker = int(live.pidfile.read_text(encoding="utf-8"))
    assert float(live.budget.read_text(encoding="utf-8")) == 60.0, (
        "the training wall is pre-charged")
    time.sleep(1.0)
    _stop(proc)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 128 + STOP, err
    assert not _running(worker), "the training worker must not outlive its evaluation"
    spent = float(live.budget.read_text(encoding="utf-8"))
    assert 0.5 <= spent < 30, f"charged {spent} s for about a second of training"
    assert _lock_free(live.lock)


@pytest.mark.skipif(os.name != "nt", reason="Windows consoles")
def test_an_eval_in_a_console_of_its_own_is_stopped_all_the_same(live):
    """Harnesses hide their tools' windows, which gives each its own console.

    The group's Ctrl-Break never reached an `eval.py` there; it trained on to
    its wall and was killed with the group, charged all of it.
    """
    proc = live.start(creationflags=subprocess.CREATE_NO_WINDOW)
    assert _until(lambda: live.pidfile.is_file() and live.pidfile.read_text(encoding="utf-8")), (
        "the worker never started")
    worker = int(live.pidfile.read_text(encoding="utf-8"))
    time.sleep(1.0)
    _stop(proc)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 128 + STOP, err
    assert not _running(worker), "the training worker must not outlive its evaluation"
    spent = float(live.budget.read_text(encoding="utf-8"))
    assert 0.5 <= spent < 30, f"charged {spent} s for about a second of training"
    assert _lock_free(live.lock)


@live_only
def test_the_worker_holds_the_device_lock_and_dies_with_its_eval(live):
    """Killing eval.py alone used to free the card while its worker trained on.

    On Linux the lock is handed down to the worker; on Windows, where a lock
    belongs to the process that took it, the worker cannot outlive eval.py.
    """
    proc = live.start()
    assert _until(lambda: live.pidfile.is_file() and live.pidfile.read_text(encoding="utf-8")), (
        "the worker never started")
    worker = int(live.pidfile.read_text(encoding="utf-8"))
    if os.name != "nt":
        held = set()
        for n in os.listdir(f"/proc/{worker}/fd"):
            try:
                held.add(os.readlink(f"/proc/{worker}/fd/{n}"))
            except OSError:
                continue
        assert str(live.lock) in held, "the device lock is handed down to the worker"
    assert not _lock_free(live.lock)
    proc.kill()  # nothing to catch: only the kernel can take the worker down now
    proc.wait()
    assert _until(lambda: not _running(worker), timeout=10), "the worker outlived eval.py"
    assert _until(lambda: _lock_free(live.lock), timeout=10)


@live_only
def test_an_eval_stopped_while_waiting_for_the_card_is_refunded(live):
    holder = os.open(live.lock, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        oscompat.lock(holder)  # another evaluation has the card
        proc = live.start()
        assert _until(lambda: live.budget.is_file()
                      and live.budget.read_text(encoding="utf-8") == "60.0")
        _stop(proc)
        proc.communicate(timeout=30)
    finally:
        os.close(holder)
    assert proc.returncode == 128 + STOP
    assert float(live.budget.read_text(encoding="utf-8")) == 0.0, (
        "a run that never got the card costs nothing")
    assert not live.pidfile.exists()
