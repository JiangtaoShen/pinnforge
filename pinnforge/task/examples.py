"""Task packages that ship with PINNForge, or are one command away.

* **demo** — a multi-scale Poisson problem with an analytic solution. Its
  package lives inside the wheel, and its reference is computed from the
  formula at install time, so a fresh workspace can run end to end with no
  download and no secret. It is also the template `task new` adapts: small,
  valid, and written against the shared evaluation machinery.
* **ldc, ks, cyl, naca** — the benchmark tasks. Their public files and
  paper-protocol references are published as release assets, listed with
  their sha256 in `examples/catalog.json`; `pinnforge examples add <name>`
  fetches, verifies and unpacks one into `tasks/<name>/`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CATALOG = EXAMPLES_DIR / "catalog.json"
DEMO = "demo"
DEMO_FILES = ("problem.md", "baseline.py", "eval.py", "view.json")
GRID_N = 101  # the demo scores on a uniform 101 × 101 grid, edges included


def template_dir() -> Path:
    """The package `task new` adapts when no other template is named."""
    return EXAMPLES_DIR / DEMO


def demo_solution(x, y):
    """u = sin(πx) sin(πy) + 0.1 sin(8πx) sin(8πy) — the demo's exact answer."""
    import numpy as np

    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.1 * np.sin(8 * np.pi * x) * np.sin(
        8 * np.pi * y
    )


def install_demo(tasks_dir: Path, name: str = DEMO) -> Path:
    """Write the demo package to `tasks/<name>/`, reference included."""
    import numpy as np

    dest = Path(tasks_dir) / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in DEMO_FILES:
        shutil.copy2(EXAMPLES_DIR / DEMO / f, dest / f)
    t = np.linspace(0.0, 1.0, GRID_N)
    x, y = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    np.savetxt(
        dest / "eval_grid.csv", np.stack([x, y], axis=1), delimiter=",", header="x,y",
        comments="", fmt="%.10g",
    )
    # The reference is read back from the written grid, so the two agree to
    # the last digit the score service will see.
    grid = np.loadtxt(dest / "eval_grid.csv", delimiter=",", skiprows=1, dtype=np.float64)
    (dest / "private").mkdir(exist_ok=True)
    ref = demo_solution(grid[:, 0], grid[:, 1]).reshape(-1, 1).astype(np.float64)
    np.save(dest / "private" / "ref_values.npy", ref)
    return dest


# ─────────────────────────── the benchmark catalog ───────────────────────────


def catalog() -> list[dict]:
    if not CATALOG.is_file():
        return []
    return json.loads(CATALOG.read_text(encoding="utf-8")).get("examples", [])


def entry(name: str) -> dict:
    for e in catalog():
        if e.get("name") == name:
            return e
    known = ", ".join([DEMO, *(e["name"] for e in catalog())])
    raise KeyError(f"unknown example {name!r}. Available: {known}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    if "://" not in url:
        shutil.copyfile(Path(url).expanduser(), dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "pinnforge"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def fetch(name: str, tasks_dir: Path, source: str | None = None, force: bool = False) -> Path:
    """Download, verify and unpack one benchmark task into `tasks/<name>/`.

    `source` overrides the catalog URL — a local archive or a mirror. The
    sha256 from the catalog is checked either way: an archive that does not
    hash to what this release was built with is not this release's task.
    """
    e = entry(name)
    dest = Path(tasks_dir) / name
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists (pass --force to replace it)")
    url = source or e["url"]
    with tempfile.TemporaryDirectory(prefix=f"pinnforge-{name}-") as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        try:
            _download(url, archive)
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException,
                OSError) as err:
            why = getattr(err, "reason", None) or err
            if isinstance(err, urllib.error.HTTPError):
                why = f"HTTP {err.code} {err.reason}"
            raise ValueError(
                f"{name}: could not fetch {url} ({why}). If you have the archive — "
                f"{name}.tar.gz from the release, or a mirror's copy — install it with "
                f"`pinnforge examples add {name} --source <path-or-url>`; its sha256 is "
                "checked against the catalog either way"
            ) from None
        got = _sha256(archive)
        if got != e["sha256"]:
            raise ValueError(
                f"{name}: archive sha256 {got[:12]}… does not match the catalog "
                f"({e['sha256'][:12]}…) — refusing to install it"
            )
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(unpacked, filter="data")
        root = unpacked / name
        if not (root / "problem.md").is_file():
            raise ValueError(f"{name}: archive does not hold a {name}/ task package")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(root, dest)
    return dest


def pack(task_dir: Path, out_dir: Path) -> tuple[Path, str]:
    """Build the release archive for one task: public files plus `private/`.

    The anchor cache is left out — it measures this machine's GPU, not the task.
    """
    task_dir = Path(task_dir)
    out = Path(out_dir) / f"{task_dir.name}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if len(parts) > 1 and parts[1] in (".anchor", ".authoring", "__pycache__"):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(out, "w:gz") as tar:
        for f in sorted(task_dir.rglob("*")):
            rel = f.relative_to(task_dir.parent)
            if f.is_file():
                tar.add(f, arcname=str(rel), filter=keep, recursive=False)
    return out, _sha256(out)
