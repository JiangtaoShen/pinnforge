"""Opening a file a block may have planted something at.

Everything the dashboard reads lives somewhere a block can write — its run
directory — and a block can put more than a file there: a symlink to the
answers, a hard link to them (which no path check sees through), a FIFO that
hangs whoever opens it. So every read here:

* checks the path, after every link, lies inside the workspace and nowhere
  near `private/`, `.scoreq/` or `.pinnforge/`;
* opens without following a final link (`O_NOFOLLOW`) and without waiting
  on a FIFO (`O_NONBLOCK`);
* then asks the descriptor what it actually got: a regular file with a
  single link, at an allowed path (on Linux, `/proc/self/fd` says where the
  open really landed — a directory swapped for a link mid-check is caught).
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

FORBIDDEN = ("private", ".scoreq", ".pinnforge")

NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
NONBLOCK = getattr(os, "O_NONBLOCK", 0)
CLOEXEC = getattr(os, "O_CLOEXEC", 0)
READ_FLAGS = os.O_RDONLY | NOFOLLOW | NONBLOCK | CLOEXEC


def _inside(real: str, root_real: str | None) -> bool:
    if root_real is not None:
        base = root_real.rstrip(os.sep) or os.sep
        prefix = base if base.endswith(os.sep) else base + os.sep
        if not real.startswith(prefix):
            return False
        real = real[len(prefix):]
    elif sys.platform == "darwin" and real.startswith("/private/"):
        # macOS keeps /tmp, /var and /etc in /private, the system's directory
        # and never a task's answers: every resolved path there begins with it
        real = real[len("/private"):]
    return not any(part in FORBIDDEN for part in Path(real).parts)


def allowed(path, root_real: str | None) -> bool:
    """Inside the workspace, and nowhere near the answers — after every link."""
    try:
        return _inside(os.path.realpath(path), root_real)
    except (OSError, ValueError):
        return False


def _landed_inside(fd: int, root_real: str | None) -> bool:
    """Where the descriptor really points, checked again (Linux only)."""
    try:
        real = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return True  # no /proc: the check before the open stands
    if not real.startswith(os.sep):
        return False  # not a path at all (a pipe, a socket)
    return _inside(real, root_real)


def open_regular(path, root_real: str | None = None, *, check_path: bool = True,
                 single_link: bool = True) -> int | None:
    """A read-only descriptor on the regular file at `path`, or None.

    None for a link, a FIFO, a device, a directory, a file with more than
    one hard link (when `single_link`), or — when `check_path` — anything
    that resolves outside the workspace or into a forbidden directory.
    """
    if check_path and not allowed(path, root_real):
        return None
    try:
        fd = os.open(path, READ_FLAGS)
    except (OSError, ValueError):  # ValueError: an embedded NUL
        return None
    try:
        st = os.fstat(fd)
        ok = stat.S_ISREG(st.st_mode) and (not single_link or st.st_nlink == 1)
        if ok and check_path:
            ok = _landed_inside(fd, root_real)
    except OSError:
        ok = False
    if not ok:
        os.close(fd)
        return None
    return fd


def read_bytes(path, cap: int, root_real: str | None = None, *, check_path: bool = True,
               single_link: bool = True) -> bytes | None:
    """At most `cap` bytes of the regular file at `path`; None if it is anything else."""
    fd = open_regular(path, root_real, check_path=check_path, single_link=single_link)
    if fd is None:
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            return fh.read(max(0, int(cap)))
    except OSError:
        return None


def is_regular(path, root_real: str | None = None) -> bool:
    """Would `read_bytes` read it?"""
    fd = open_regular(path, root_real)
    if fd is None:
        return False
    os.close(fd)
    return True
