"""The evaluation machinery every task's `eval.py` shares.

Budget accounting, the per-GPU lock, the train/eval worker split, the
restricted unpickler, the score-queue client and the record format used to be
copied into each task's `eval.py` — about 420 of its 480 lines — with only
the metric and a few constants differing. Copies drift: one task's scoring
worker stopped filing the block id with its requests, so its per-block cap
quietly became one cap for the whole run.

So a task now keeps only what is its own — constants, the metric, which grid
rows and fields it predicts — and hands the rest to this module:

    if __name__ == "__main__":
        import _forgekit
        raise SystemExit(_forgekit.main(globals()))

This file is **vendored**, not imported from the installed package:
`registry.install` copies it into every run as `task/_forgekit.py`, where it
is frozen with the rest of `task/` and hashed by the integrity check like the
rest of it. An upgrade of the tool therefore cannot change how a run that
already exists evaluates, and a block cannot change it without that showing.
It imports nothing outside the standard library at module level, so the
score daemon — which imports a task's `eval.py` for its metric — never needs
it, and neither does the training worker until the candidate has run.

What a task module provides (read from its globals):

    TRAIN_TIME_LITERAL  frozen-header literal, e.g. "TRAIN_TIME = 150.0"
    SMOKE_TRAIN_TIME    same-length smoke replacement
    TRAIN_WALL_S        hard wall on a GPU run (train or --diag)
    SMOKE_WALL_S        wall on the CPU smoke
    EVAL_WALL_S         wall on the scoring worker
    BUDGET              per-block GPU seconds (FORGE_WALL_BUDGET, else a default)
    GRID_CSV            the public prediction grid
    INPUT_COLUMNS       leading grid columns handed to predict_fn
    FIELDS              predicted fields filed for scoring, in order
    predict_rows(grid)  optional: the rows to predict (default: all)
    EXTRA_ENV           optional: extra environment for every worker
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

PROTOCOL = 1  # score-queue protocol; must match pinnforge.scoring.PROTOCOL
QUEUE = ".scoreq"
POLL_S = 0.5

_WINDOWS = os.name == "nt"

# ────────────────────────── the operating system ──────────────────────────
#
# Stand-ins for `fcntl.flock` and process groups on Windows, inline because
# this file may import nothing outside the standard library. On Windows a
# lock is `LockFileEx` on one byte far past any data (its locks are
# mandatory, and must not stop anyone reading the file), and a worker lives
# in a Job Object that dies with this process — what a process group and
# PR_SET_PDEATHSIG give on Linux.

if _WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    class _JOB_LIMITS(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                    ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))
    _k32.UnlockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.DWORD, ctypes.POINTER(_OVERLAPPED))
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                             wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)


def _lock(f) -> None:
    """Block until this process holds the exclusive lock on open file `f`."""
    if not _WINDOWS:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)
        return
    handle = msvcrt.get_osfhandle(f if isinstance(f, int) else f.fileno())
    while True:  # polled, so a stop signal is still taken while queued
        ov = _OVERLAPPED(Offset=0, OffsetHigh=0x40000000)
        if _k32.LockFileEx(handle, 0x3, 0, 1, 0, ctypes.byref(ov)):  # exclusive, don't wait
            return
        if ctypes.get_last_error() != 33:  # ERROR_LOCK_VIOLATION: someone else holds it
            raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(0.05)


def _unlock(f) -> None:
    if not _WINDOWS:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_UN)
        return
    handle = msvcrt.get_osfhandle(f if isinstance(f, int) else f.fileno())
    ov = _OVERLAPPED(Offset=0, OffsetHigh=0x40000000)
    _k32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(ov))


def _new_job():
    """A Job Object whose processes are killed when its last handle closes (Windows)."""
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = _JOB_LIMITS(LimitFlags=0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _k32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        _k32.CloseHandle(job)
        return None
    return job


def _device_lock_path(slot: str) -> str:
    """The machine-wide lock for one card (or the CPU): `/tmp` on POSIX."""
    base = tempfile.gettempdir() if _WINDOWS else "/tmp"
    return os.path.join(base, f"pinnforge_{'cpu' if slot == 'cpu' else 'gpu' + slot}.lock")

# ────────────────────────── score-queue client ──────────────────────────


def submit_score(root: Path, meta: dict, pred_path: Path) -> dict:
    """File a score request and block for the verdict.

    The same wire protocol as `pinnforge.scoring.submit_score`, kept here so a
    run's evaluation does not depend on which pinnforge is installed in its
    environment. A timeout reports the scorer as unavailable rather than
    raising, so the block still gets a record.
    """
    wait_s = float(os.environ.get("FORGE_SCORE_WAIT_S", "90"))
    q = Path(root) / QUEUE
    req, res = q / "req", q / "res"
    req.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex
    payload, request, answer = req / f"{rid}.npz", req / f"{rid}.json", res / f"{rid}.json"
    payload.write_bytes(Path(pred_path).read_bytes())
    request.write_text(
        json.dumps({**meta, "pred": payload.name, "protocol": PROTOCOL}), encoding="utf-8"
    )
    deadline = time.time() + wait_s
    try:
        while time.time() < deadline:
            if answer.is_file():
                return json.loads(answer.read_text(encoding="utf-8"))
            time.sleep(POLL_S)
        return {
            "rRMSE": None,
            "error": "scorer unavailable — start `pinnforge scored` "
            "(`pinnforge run start` serves it in-process)",
        }
    finally:
        for f in (payload, request, answer):
            with suppress(OSError):
                f.unlink()


def _paths(g: dict) -> tuple[Path, Path]:
    here = Path(g["__file__"]).resolve().parent  # task/
    return here, here.parent  # run root


def _task_name(root: Path) -> str:
    """Task this run is of — `runs/ldc_1` -> `ldc`. Env wins when set."""
    if name := os.environ.get("FORGE_TASK"):
        return name
    return re.sub(r"_\d+$", "", root.name)


def _submit(root: Path, pred_path: Path) -> dict:
    meta = {
        "task": _task_name(root),
        "run": os.environ.get("FORGE_RUN_ID", root.name),
        "block": os.environ.get("FORGE_BLOCK", ""),
        "candidate": os.environ.get("FORGE_CANDIDATE", ""),
        "seed": os.environ.get("FORGE_SEED", "0"),
        "smoke": os.environ.get("FORGE_SMOKE", "0") == "1",
    }
    return submit_score(root, meta, pred_path)


# ────────────────────────── worker: train ──────────────────────────


def worker_train(code_path: str, seed: str, params_out: str) -> int:
    """Child process: exec candidate, run train(), pickle host params."""
    try:
        import jax

        code = Path(code_path).read_text(encoding="utf-8")
        import types

        mod = types.ModuleType("_forge_candidate__")
        sys.modules["_forge_candidate__"] = mod
        exec(compile(code, "<candidate>", "exec"), mod.__dict__)

        params, step_count = mod.__dict__["train"](jax.random.PRNGKey(int(seed)), None)
        np_params = jax.tree_util.tree_map(jax.device_get, params)
        with open(params_out, "wb") as f:
            pickle.dump(np_params, f)
        print(json.dumps({
            "step_count": int(step_count),
            "device_platform": jax.devices()[0].platform,
        }))
        return 0
    except BaseException:
        print(json.dumps({"error": traceback.format_exc()[-2000:]}))
        return 1


# ────────────────────────── worker: eval ───────────────────────────

_PICKLE_ALLOWED = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"),
    ("builtins", "bytes"), ("builtins", "bytearray"), ("builtins", "object"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("flax.core.frozen_dict", "FrozenDict"),
    *(("numpy", n) for n in (
        "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool_",
    )),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _PICKLE_ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def load_params(params_path: str):
    with open(params_path, "rb") as f:
        return _RestrictedUnpickler(io.BytesIO(f.read())).load()


def prediction_inputs(g: dict):
    """The grid rows a candidate predicts at, and the columns it is handed."""
    import numpy as np

    # ndmin=2: a one-column grid (a 1-D problem) loads as 1-D otherwise, and
    # both `predict_rows` and the column slice below expect rows of columns.
    grid = np.loadtxt(g["GRID_CSV"], delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    rows = g["predict_rows"](grid) if callable(g.get("predict_rows")) else slice(None)
    return grid[rows][:, : int(g["INPUT_COLUMNS"])]


def worker_eval(g: dict, code_path: str, params_path: str) -> int:
    """Child process: restricted-unpickle params, exec candidate, predict on
    the public grid and file the predictions with the score service. This
    process has just executed candidate code, so it is exactly the one that
    must not be able to read the answers."""
    try:
        import jax.numpy as jnp
        import numpy as np

        params = load_params(params_path)

        code = Path(code_path).read_text(encoding="utf-8")
        import types

        mod = types.ModuleType("_forge_eval_candidate__")
        sys.modules["_forge_eval_candidate__"] = mod
        exec(compile(code, "<candidate>", "exec"), mod.__dict__)
        predict_fn = mod.__dict__["predict_fn"]

        data_X = prediction_inputs(g)
        preds = predict_fn(params, jnp.asarray(data_X.astype(np.float32)))

        n = data_X.shape[0]
        fields = {}
        for name in g["FIELDS"]:
            arr = np.asarray(preds[name], dtype=np.float64).reshape(-1)
            if arr.shape != (n,):
                print(json.dumps({"error": f"predict_fn returned bad shape {name}{arr.shape}"}))
                return 1
            fields[name] = arr

        pred_path = Path(params_path).with_name("predictions.npz")
        np.savez(pred_path, **fields)
        _, root = _paths(g)
        print(json.dumps(_submit(root, pred_path)))
        return 0
    except BaseException:
        print(json.dumps({"error": traceback.format_exc()[-2000:]}))
        return 1


# ────────────────────────── orchestrating parent ───────────────────

# How long a worker gets to exit on SIGTERM before its group is killed.
WORKER_GRACE_S = 5.0
# On Linux a worker runs in a process group of its own — so it and anything
# it starts can be stopped as a unit — and is told to die with this process
# (PR_SET_PDEATHSIG), so a kill nobody could catch still takes it down.
# On Windows a Job Object does both. Elsewhere it stays in this process's
# group, where a group kill reaches it.
_ISOLATE = sys.platform.startswith("linux")
# SIGBREAK: the console's Ctrl-Break, which is how a group is asked to stop on Windows.
_STOP_SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK")
                      if hasattr(signal, name))
_SIGKILL = getattr(signal, "SIGKILL", 9)


class Stopped(BaseException):
    """A stop signal, raised where the evaluation was when it arrived."""

    def __init__(self, signum: int):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


# Without `pthread_sigmask` (Windows) a held signal is parked here instead.
_held = {"depth": 0, "signum": None}


def _on_signal(signum, frame) -> None:
    if _held["depth"] > 0:
        _held["signum"] = signum
        return
    raise Stopped(signum)


def _install_handlers() -> dict:
    """SIGTERM, SIGHUP and SIGINT unwind the evaluation instead of ending it.

    Killed outright, this process left its training worker running on the
    card and the block's `.budget` charged the whole training wall for a run
    that may never have started. Unwinding stops the worker first and then
    settles the charge to the seconds actually spent. Returns the handlers
    it replaced.
    """
    old = {}
    for s in _STOP_SIGNALS:
        try:
            old[s] = signal.signal(s, _on_signal)
        except ValueError:  # not the main thread: nothing to install here
            break
    return old


def _restore_handlers(old: dict) -> None:
    for s, h in old.items():
        with suppress(ValueError, TypeError):
            signal.signal(s, h)


class _signals_held:
    """Hold stop signals off for a few lines that must not be torn in half.

    The budget counter is truncated and rewritten under its lock; a signal
    between the two left it empty, which reads as nothing spent.
    """

    def __enter__(self):
        if hasattr(signal, "pthread_sigmask"):
            self.old = signal.pthread_sigmask(signal.SIG_BLOCK, _STOP_SIGNALS)
        else:
            _held["depth"] += 1
        return self

    def __exit__(self, *exc) -> None:
        if hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_SETMASK, self.old)
            return
        _held["depth"] -= 1
        if _held["depth"] == 0 and _held["signum"] is not None:
            signum, _held["signum"] = _held["signum"], None
            raise Stopped(signum)


def _log_wall_spent(log_path: Path) -> float:
    """Wall-seconds already recorded in evals.jsonl (non-smoke records)."""
    spent = 0.0
    if log_path.is_file():
        for ln in log_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not r.get("smoke"):
                spent += r.get("wall_s") or r.get("train_s") or 0.0
    return spent


def _budget_mutate(block_dir: Path, fn) -> float:
    """File-locked read-modify-write of the block's wall-seconds counter."""
    block_dir.mkdir(parents=True, exist_ok=True)
    with open(block_dir / ".budget", "a+", encoding="utf-8") as cf:
        _lock(cf)
        cf.seek(0)
        txt = cf.read().strip()
        try:
            spent = float(txt)
        except ValueError:
            spent = 0.0
        new = fn(spent)
        cf.seek(0)
        cf.truncate()
        cf.write(f"{new:.1f}")
        return new


def _open_lock(path: str) -> int:
    """Open a per-device lock file — another user's included — without writing to it.

    `/tmp` is shared by everyone on the machine. The first user to evaluate on
    a card creates its lock file; `open(path, "w")` by a second user then
    failed on it (another user's file, and `fs.protected_regular`), and did
    so after their budget had been charged. `flock` needs no write access, so
    an existing file is opened read-only; a missing one is created
    exclusively and made world-writable, so whoever comes next can do the
    same. Never through a link, and never anything but a regular file.
    (Windows has no `O_NOFOLLOW`: a link there is refused before the open.)
    """
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_BINARY", 0))
    for _ in range(5):
        if _WINDOWS and os.path.islink(path):
            raise OSError(f"{path} is a link")
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            try:
                fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                continue  # created by someone else in between; open theirs
            with suppress(OSError, AttributeError):  # no fchmod on Windows
                os.fchmod(fd, 0o666)  # the umask had its say at creation
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(f"{path} is not a regular file")
        except BaseException:
            os.close(fd)
            raise
        return fd
    raise OSError(f"could not open {path}")


def _last_json_line(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def _die_with_parent():
    """A `preexec_fn` asking the kernel to SIGKILL the child when this process dies."""
    if not _ISOLATE:
        return None
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent, kill = os.getpid(), int(signal.SIGKILL)

    def arm() -> None:
        prctl(1, kill)  # PR_SET_PDEATHSIG
        if os.getppid() != parent:  # the parent died before the request was made
            os.kill(os.getpid(), kill)

    return arm


def _exited(proc: subprocess.Popen) -> bool:
    """Has `proc` exited? Asked without reaping it, so its group id stays reserved."""
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True
    except (AttributeError, OSError):
        return proc.poll() is not None


def _signal_worker(proc: subprocess.Popen, sig: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        if _ISOLATE:
            os.killpg(proc.pid, sig)
        else:
            os.kill(proc.pid, sig)


def _stop_worker(proc: subprocess.Popen, grace: float = WORKER_GRACE_S) -> None:
    """TERM the worker, KILL it — and anything it started — after `grace`.

    The group is killed while the worker is still unreaped: until then its
    number cannot have been handed to anybody else — which is also why a
    worker already reaped is not signalled at all. On Windows, where a TERM
    would be a kill anyway, the worker's job is ended at once.
    """
    if proc.returncode is not None:
        return
    job = getattr(proc, "_forge_job", None)
    if job is not None:
        _k32.TerminateJobObject(job, 1)
    elif _WINDOWS:
        with suppress(OSError):
            proc.kill()
    else:
        if grace > 0 and not _exited(proc):
            _signal_worker(proc, signal.SIGTERM)
            deadline = time.time() + grace
            while time.time() < deadline and not _exited(proc):
                time.sleep(0.05)
        _signal_worker(proc, _SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def _spawn(cmd: list[str], env: dict, wall: float, keep_fds=(), clock=None):
    """Run a worker to completion under `wall`. -> (returncode, stdout, stderr, timed_out)

    `keep_fds` stay open in the worker: the device lock is passed this way,
    so it is held for as long as either process lives — killing this one
    alone used to free the card while its worker went on training on it.
    Windows cannot pass a descriptor, and its locks belong to the process
    that took them; there the worker cannot outlive this process instead.
    `clock["t0"]` is set the moment the worker exists, for the budget.
    """
    kwargs = {}
    if _ISOLATE:
        kwargs["process_group"] = 0
        if (arm := _die_with_parent()) is not None:
            kwargs["preexec_fn"] = arm
    if not _WINDOWS:
        kwargs["pass_fds"] = tuple(keep_fds)
    if clock is not None:
        clock["t0"] = time.time()
    # errors="replace": a worker's stderr in another encoding must not cost the record
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", env=env, **kwargs,
    )
    if _WINDOWS and (job := _new_job()) is not None:
        if _k32.AssignProcessToJobObject(job, int(proc._handle)):
            proc._forge_job = job
        else:
            _k32.CloseHandle(job)
    try:
        try:
            try:
                out, err = proc.communicate(timeout=wall)
            except subprocess.TimeoutExpired:
                _stop_worker(proc, grace=0)
                out, err = proc.communicate()
                return proc.returncode, out or "", err or "", True
        except BaseException:
            # A second signal must not cut this short: the worker would outlive it.
            with _signals_held():
                _stop_worker(proc)
            raise
        return proc.returncode, out or "", err or "", False
    finally:
        if (job := getattr(proc, "_forge_job", None)) is not None:
            _k32.CloseHandle(job)  # and with it whatever the worker left running


def _run_worker(eval_py: Path, mode: str, argv: list[str], env: dict, wall: float,
                *, keep_fds=(), clock=None):
    t0 = time.time()
    env = {**env, "FORGE_T0": str(t0)}  # wall anchor for the candidate's deadline
    code, out, err, timed_out = _spawn(
        [sys.executable, str(eval_py), mode, *argv], env, wall, keep_fds, clock
    )
    if timed_out:
        return (
            {"error": f"killed at the {wall:.0f}s wall (hung or runaway process)"},
            time.time() - t0,
        )
    payload = _last_json_line(out)
    if code != 0 and "error" not in payload:
        payload = {"error": (err[-2000:] or f"exit code {code}")}
    return payload, time.time() - t0


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_record(log_path: Path, record: dict) -> None:
    with _signals_held(), open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class _Charge:
    """The block's pre-charge for one GPU run, settled exactly once.

    Settled to the seconds the run actually took on the normal path; on any
    other — a stop signal, a crash — by `settle_actual()` in a `finally`,
    with the time since the worker started, or nothing if it never did.
    Either way the full training wall no longer stays charged for a run that
    was interrupted, or that never got the card.
    """

    def __init__(self, block_dir: Path, est: float):
        self.block_dir, self.est = block_dir, est
        self.charged = False
        self.settled = False
        self.clock: dict = {}

    def settle(self, seconds: float) -> float:
        with _signals_held():
            spent = _budget_mutate(self.block_dir,
                                   lambda s: max(0.0, s - self.est + seconds))
            self.settled = True
        return spent

    def settle_actual(self) -> float | None:
        if not self.charged or self.settled:
            return None
        t0 = self.clock.get("t0")
        took = time.time() - t0 if t0 is not None else 0.0
        self.settle(took)
        return took


def run(g: dict) -> int:
    """The evaluation itself: one candidate, one record."""
    _, root = _paths(g)
    eval_py = Path(g["__file__"]).resolve()
    literal, smoke_literal = g["TRAIN_TIME_LITERAL"], g["SMOKE_TRAIN_TIME"]
    train_wall, smoke_wall, eval_wall = g["TRAIN_WALL_S"], g["SMOKE_WALL_S"], g["EVAL_WALL_S"]
    budget = g["BUDGET"]
    extra_env = {k: str(v) for k, v in (g.get("EXTRA_ENV") or {}).items()}

    ap = argparse.ArgumentParser(
        description=g.get("__doc__"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("candidate", help="path to a candidate .py under blocks/bNN/")
    ap.add_argument("--gpu", default="0", help="GPU id to train on (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU dress rehearsal: short TRAIN_TIME, free")
    ap.add_argument("--diag", action="store_true",
                    help="run an arbitrary diagnostic script on GPU, charged by the second")
    args = ap.parse_args()
    if args.smoke and args.diag:
        print("error: --smoke and --diag are mutually exclusive", file=sys.stderr)
        return 2

    cand = Path(args.candidate).resolve()
    if not cand.is_file():
        print(f"error: no such file: {cand}", file=sys.stderr)
        return 2

    # block inference: candidate must live under blocks/bNN/
    try:
        rel = cand.relative_to(root / "blocks")
        block = rel.parts[0]
        if not (block.startswith("b") and block[1:].isdigit()):
            raise ValueError(block)
    except (ValueError, IndexError):
        print(f"error: candidate must live under {root}/blocks/bNN/ — got {cand}", file=sys.stderr)
        return 2
    block_dir = root / "blocks" / block
    log_path = block_dir / "evals.jsonl"

    source = cand.read_text(encoding="utf-8")
    if not args.diag and (source.count(literal) != 1 or "FORGE_EVAL_TOKEN" not in source):
        print(f"error: candidate must contain the frozen-header literal "
              f"`{literal}` exactly once (found {source.count(literal)}) "
              f"and the eval-gate block (`FORGE_EVAL_TOKEN`). "
              f"Keep baseline.py's PDE-CONSTANTS header byte-identical.", file=sys.stderr)
        return 2

    base_env = {**os.environ, **extra_env}
    # CPU mode — the orchestrator sets JAX_PLATFORMS=cpu for a run with no
    # card: train with every card hidden, queue on a CPU lock rather than card
    # 0's (which a GPU run on this machine may be using), and record "cpu"
    # instead of a card that was never touched.
    on_cpu = os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"
    slot = "cpu" if on_cpu else args.gpu
    devices = "" if on_cpu else args.gpu
    lock_path = _device_lock_path(slot)

    old_handlers = _install_handlers()
    lock_fd = None
    charge = _Charge(block_dir, train_wall)
    try:
        # The lock file is opened before anything is charged: a card whose
        # lock cannot be opened must not cost the block a training wall.
        if not args.smoke:
            try:
                lock_fd = _open_lock(lock_path)
            except OSError as e:
                print(f"error: cannot open the device lock {lock_path}: {e}", file=sys.stderr)
                return 2

            # budget: pre-charge the wall cap under the file lock (evals.jsonl
            # lags a running eval by its whole training time, so concurrent runs
            # settle on the counter), then reconcile to actual seconds afterwards
            log_spent = _log_wall_spent(log_path)
            refused = []

            def _claim(spent: float) -> float:
                spent = max(spent, log_spent)
                if spent >= budget:
                    refused.append(spent)
                    return spent
                return spent + charge.est

            with _signals_held():
                _budget_mutate(block_dir, _claim)
                charge.charged = not refused
            if refused:
                print(f"error: block {block} has spent {refused[0]:.0f}/{budget:.0f} "
                      f"wall-seconds — budget exhausted. Write your summary "
                      f"(blocks/kb2/{block}.md) and finish.", file=sys.stderr)
                return 2

        if args.diag:
            return _diag(cand, root, block, slot, devices, base_env, train_wall, budget,
                         lock_fd, charge, log_path)
        return _evaluate(args, g, eval_py, cand, root, block, block_dir, log_path, source,
                         literal, smoke_literal, train_wall, smoke_wall, eval_wall, budget,
                         slot, devices, base_env, lock_fd, charge)
    finally:
        took = charge.settle_actual()
        if took is not None:
            print(f"eval.py: the run did not finish; the block is charged the {took:.0f} s "
                  f"it took, not the {charge.est:.0f} s wall", file=sys.stderr)
        if lock_fd is not None:
            os.close(lock_fd)
        _restore_handlers(old_handlers)


def _diag(cand, root, block, slot, devices, base_env, train_wall, budget, lock_fd, charge,
          log_path) -> int:
    env = dict(base_env)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["FORGE_EVAL_TOKEN"] = "task/eval.py"
    env["CUDA_VISIBLE_DEVICES"] = devices
    _lock(lock_fd)
    t0 = time.time()
    env["FORGE_T0"] = str(t0)
    code, out, err, timed_out = _spawn([sys.executable, str(cand)], env, train_wall,
                                       keep_fds=(lock_fd,), clock=charge.clock)
    if timed_out:
        error = f"killed at the {train_wall:.0f}s wall (hung or runaway process)"
    else:
        error = None if code == 0 else f"exit code {code}"
    wall_s = time.time() - t0
    _unlock(lock_fd)
    spent = charge.settle(wall_s)
    if out:
        print(out[-8000:])
    if err:
        print(err[-3000:], file=sys.stderr)
    record = {
        "ts": _now(),
        "block": block,
        "candidate": cand.relative_to(root).as_posix(),
        "seed": None, "gpu": slot,
        "smoke": False, "diag": True,
        "rRMSE": None, "step_count": None,
        "train_s": None, "wall_s": round(wall_s, 1),
        "params": None, "error": error,
        "wall_budget": f"{spent:.0f}/{budget:.0f}",
    }
    _append_record(log_path, record)
    print(json.dumps(record))
    return 0


def _evaluate(args, g, eval_py, cand, root, block, block_dir, log_path, source, literal,
              smoke_literal, train_wall, smoke_wall, eval_wall, budget, slot, devices,
              base_env, lock_fd, charge) -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="forge_eval_"))
    try:
        params_tmp = tmpdir / "params.pkl"
        env = dict(base_env)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # frozen-header eval gate: without this token a candidate's train()
        # self-limits to CPU/10 s, so eval.py is the only full-fidelity path
        env["FORGE_EVAL_TOKEN"] = "task/eval.py"
        keep = ()
        if args.smoke:
            env["JAX_PLATFORMS"] = "cpu"
            env.pop("CUDA_VISIBLE_DEVICES", None)
            code_for_train = tmpdir / "candidate_smoke.py"
            code_for_train.write_text(source.replace(literal, smoke_literal), encoding="utf-8")
            wall = smoke_wall
        else:
            env["CUDA_VISIBLE_DEVICES"] = devices
            code_for_train = cand
            wall = train_wall
            # blocking per-GPU lock: concurrent evals on the same GPU serialize;
            # the worker holds it too, for as long as it lives
            _lock(lock_fd)
            keep = (lock_fd,)

        train_out, train_s = _run_worker(
            eval_py, "--worker-train", [str(code_for_train), str(args.seed), str(params_tmp)],
            env, wall, keep_fds=keep, clock=charge.clock,
        )
        if lock_fd is not None:
            _unlock(lock_fd)

        # a smoke is free: its "spent" is a read-only peek at the counter
        spent = _budget_mutate(block_dir, lambda s: s) if args.smoke else charge.settle(train_s)

        record = {
            "ts": _now(),
            "block": block,
            "candidate": cand.relative_to(root).as_posix(),
            "seed": args.seed,
            "gpu": None if args.smoke else slot,
            "smoke": bool(args.smoke),
            "rRMSE": None,
            "step_count": train_out.get("step_count"),
            "train_s": round(train_s, 1),
            "wall_s": 0.0 if args.smoke else round(train_s, 1),
            "params": None,
            "error": train_out.get("error"),
        }

        if "error" not in train_out and params_tmp.is_file():
            if not args.smoke:
                suffix = f"_s{args.seed}" if args.seed != 0 else ""
                params_final = cand.with_name(cand.stem + suffix + ".pkl")
                params_final.write_bytes(params_tmp.read_bytes())
                record["params"] = params_final.relative_to(root).as_posix()
            eval_env = dict(base_env)
            eval_env["JAX_PLATFORMS"] = "cpu"
            # Context the score request is filed under — the daemon's ledger
            # and its per-block cap key on these.
            eval_env["FORGE_TASK"] = _task_name(root)
            eval_env["FORGE_RUN_ID"] = root.name
            eval_env["FORGE_BLOCK"] = block
            eval_env["FORGE_CANDIDATE"] = cand.relative_to(root).as_posix()
            eval_env["FORGE_SEED"] = str(args.seed)
            eval_env["FORGE_SMOKE"] = "1" if args.smoke else "0"
            eval_out, _ = _run_worker(
                eval_py, "--worker-eval", [str(cand), str(params_tmp)], eval_env, eval_wall
            )
            record["rRMSE"] = eval_out.get("rRMSE")
            if "error" in eval_out:
                record["error"] = eval_out["error"]

        record["wall_budget"] = f"{spent:.0f}/{budget:.0f}"

        block_dir.mkdir(parents=True, exist_ok=True)
        _append_record(log_path, record)
        print(json.dumps(record))
        return 0
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def main(g: dict) -> int:
    """Entry point for a task's `eval.py`: a worker mode, or the evaluation."""
    argv = sys.argv
    if len(argv) >= 2 and argv[1] == "--worker-train":
        return worker_train(*argv[2:5])
    if len(argv) >= 2 and argv[1] == "--worker-eval":
        return worker_eval(g, *argv[2:4])
    try:
        return run(g)
    except Stopped as e:
        print(f"eval.py: {e} — the evaluation was stopped", file=sys.stderr)
        return 128 + e.signum
