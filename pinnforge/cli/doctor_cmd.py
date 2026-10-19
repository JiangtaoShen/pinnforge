"""`pinnforge doctor` — every prerequisite of a run, checked in one place.

Each check reports OK, WARN or FAIL, and every WARN or FAIL names the command
that fixes it. The order is the order things are needed in: the tool, the
workspace, its environment, the hardware, the agent, the tasks, and the
extras (tmux, the dashboard port, the corpus). `--json` prints the same
findings for a harness plugin to read.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from pinnforge import __version__, bindings, gpus, paths, private, settings, workspace
from pinnforge.runtime import registry


@dataclass
class Finding:
    level: str  # ok | warn | fail
    name: str
    detail: str
    fix: str = ""


PROBE = r"""
import json, importlib.metadata as m
out = {}
try:
    import jax
    devs = jax.devices()
    out.update(jax=jax.__version__, platform=devs[0].platform, devices=len(devs))
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:300]
vers = {}
for p in ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "jaxopt", "evosax",
          "ml_collections", "jaxtyping", "pinnforge"):
    try:
        vers[p] = m.version(p)
    except m.PackageNotFoundError:
        pass
out["versions"] = vers
print(json.dumps(out))
"""


def _pins() -> dict[str, str]:
    s = workspace.stack()
    out = {}
    for req in [*s["common"], *s["cuda12"]]:
        if "==" in req and "@" not in req:
            name, ver = req.split("==")
            out[name.split("[")[0]] = ver
    return out


def _probe(python: Path, timeout: float = 180.0) -> dict:
    try:
        out = subprocess.run(
            [str(python), "-c", PROBE], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"the environment did not import JAX within {timeout:.0f}s"}
    except OSError as e:
        return {"error": str(e)}
    lines = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:300]}
    return json.loads(lines[-1])


def _port_state(host: str, port: int) -> str:
    """free | ours | taken"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) != 0:
            return "free"
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://{host}:{port}/api/ping", timeout=2) as r:
            if json.loads(r.read().decode()).get("app") == "pinnforge":
                return "ours"
    except Exception:
        pass
    return "taken"


def _ambient(root: Path) -> list[Path]:
    """Instruction files a block's CLI would pick up from above its run directory."""
    names = ("CLAUDE.md", "AGENTS.md", ".cursorrules")
    found = []
    for base in [root, *root.parents]:
        for n in names:
            if (base / n).is_file():
                found.append(base / n)
        if (base / ".cursor" / "rules").is_dir():  # Cursor reads every rule in it
            found.append(base / ".cursor" / "rules")
    return found


def _settings_files(root: Path | None) -> list[Finding]:
    """A FAIL for each settings file that does not parse; every later check reads them."""
    out = []
    readers = [("user", settings.user, settings.user_dir() / "config.yaml")]
    if root is not None:
        readers.append(("workspace", lambda: settings.workspace(root),
                        Path(root) / settings.WORKSPACE_FILE))
    for _, read, path in readers:
        try:
            read()
        except ValueError as e:
            out.append(Finding("fail", "settings", str(e),
                               f"fix {path} — it must be a YAML mapping (key: value lines)"))
    return out


def checks(ping: bool = False) -> list[Finding]:
    out: list[Finding] = []
    add = out.append
    add(Finding("ok", "tool", f"pinnforge {__version__} (python {sys.version.split()[0]})"))

    try:
        root = paths.project_root()
    except paths.ProjectNotFound:
        add(Finding("fail", "workspace", "no workspace here or above",
                    "pinnforge init <dir>   (then cd into it)"))
        broken = _settings_files(None)
        out += broken
        if not broken:
            _agent_checks(out, None, ping)
        return out
    marked = paths.is_workspace(root)
    kind = "workspace" if marked else "source checkout"
    add(Finding("ok", "workspace", f"{root} ({kind}, stack {workspace.stack_of(root) or '—'})"))
    # A settings file that does not parse is reported once, here; the checks
    # that read the settings (GPUs, agent, private dir, dashboard) are skipped.
    broken = _settings_files(root)
    out += broken
    readable = not broken

    # ── environment ──
    python = paths.venv_bin(root / ".venv")
    if not python.is_file():
        add(Finding("fail", "environment", f"no {root / '.venv'}",
                    "pinnforge workspace sync" if marked else "uv sync"))
    else:
        probe = _probe(python)
        vers = probe.get("versions", {})
        if "error" in probe:
            add(Finding("fail", "environment", f"JAX does not import: {probe['error']}",
                        "pinnforge workspace sync" if marked else "uv sync"))
        else:
            add(Finding("ok", "environment",
                        f"jax {probe['jax']} on {probe['platform']} ×{probe['devices']}"))
            if probe["platform"] == "cpu" and readable and _pool_or(root, None, []):
                add(Finding("warn", "environment",
                            "GPUs are visible but JAX runs on the CPU",
                            "pinnforge workspace sync --stack cuda12"))
        drift = {p: (v, vers.get(p)) for p, v in _pins().items()
                 if vers.get(p) and vers[p] != v}
        if drift:
            shown = ", ".join(f"{p} {got} (pinned {want})" for p, (want, got) in drift.items())
            add(Finding("warn", "stack pins", f"moved: {shown}",
                        "pinnforge workspace sync   (restores the frozen stack)"))
        venv_pf = vers.get("pinnforge")
        if venv_pf and venv_pf != __version__:
            add(Finding("warn", "versions",
                        f"environment has pinnforge {venv_pf}, the tool is {__version__}",
                        "pinnforge workspace upgrade"))

    # ── hardware ──
    inv = gpus.inventory()
    pool = gpus.resolve(gpus.AUTO, inv=inv)
    if readable:
        _gpu_checks(out, root, inv)
        pool = _pool_or(root, inv, pool)
        if cards := gpus.windows_cards():
            add(Finding("warn", "gpus",
                        f"{', '.join(cards)}: JAX has no CUDA build for Windows, so runs here "
                        "train on the CPU",
                        "run PINNForge inside WSL2 to train on the card"))
        _agent_checks(out, root, ping)
    _task_checks(out, root, pool, inv)

    if readable:
        pdir = private.configured(root)
        if pdir is not None and not pdir.is_dir():
            src = (f"${private.ENV}" if os.environ.get(private.ENV)
                   else settings.origin(root, "private_dir") or "pinnforge.yaml")
            add(Finding("fail", "private_dir", f"{pdir} does not exist",
                        f"create it, or unset private_dir in {src}"))

    # ── extras ──
    if sys.platform == "win32":
        add(Finding("ok", "detach", "detached runs go to the background, with a log "
                    "(`pinnforge run attach` follows it)"))
    elif shutil.which("tmux"):
        add(Finding("ok", "tmux", "runs detach into tmux sessions"))
    else:
        add(Finding("warn", "tmux", "not installed — detached runs fall back to a log file",
                    "install tmux (e.g. `sudo apt install tmux`)"))
    if not shutil.which("git"):
        # the frozen stack pins one package to a git commit, and a workspace
        # installs PINNForge itself from its repository
        add(Finding("warn", "git", "not installed — `pinnforge workspace sync` needs it",
                    "winget install Git.Git" if sys.platform == "win32"
                    else "install git (e.g. `sudo apt install git`)"))
    elif sys.platform == "win32" and _git_bash() is None:
        add(Finding("warn", "git bash", "not found — Claude Code on Windows runs its shell in it",
                    "install Git for Windows, or set CLAUDE_CODE_GIT_BASH_PATH"))
    if readable:
        _dashboard_check(out, root)
    kb1 = paths.kb1_dir(root)
    n = len([p for p in kb1.glob("*.md") if p.name != "INDEX.md"]) if kb1.is_dir() else 0
    if n >= 100:
        add(Finding("ok", "kb1", f"{n} paper notes ({kb1})"))
    else:
        add(Finding("warn", "kb1", f"only {n} paper notes in {kb1}",
                    "reinstall pinnforge, or remove a partial kb1/ override"))
    for f in _ambient(root):
        add(Finding("warn", "instructions", f"{f} will be read by every block under {root}",
                    "move it out of the workspace's ancestry"))
    return out


def _pool_or(root: Path, inv: list[gpus.Gpu] | None, fallback: list[int]) -> list[int]:
    """The workspace's pool, or `fallback` when `sandbox.gpus` does not parse."""
    try:
        return gpus.workspace_pool(root, inv)
    except ValueError:
        return fallback


def _git_bash() -> Path | None:
    """Git for Windows' bash — not `System32\\bash.exe`, which is WSL's."""
    if configured := os.environ.get("CLAUDE_CODE_GIT_BASH_PATH"):
        return Path(configured) if Path(configured).is_file() else None
    git = shutil.which("git")
    if git is None:
        return None
    for base in Path(git).resolve().parents:  # …\Git\cmd\git.exe, …\Git\mingw64\bin\git.exe
        if (base / "bin" / "bash.exe").is_file():
            return base / "bin" / "bash.exe"
    return None


def _gpu_checks(out: list[Finding], root: Path, inv: list[gpus.Gpu]) -> None:
    add = out.append
    spec = settings.get(root, "sandbox.gpus", "auto")
    where = settings.origin(root, "sandbox.gpus") or "defaults"
    try:
        shown = f"sandbox.gpus: {gpus.render(spec)} in {where}"
        pool = gpus.resolve(spec, inv=inv)
    except ValueError as e:
        add(Finding("fail", "gpus", f"{e} (in {where})", "pinnforge setup gpus"))
        return
    if lost := gpus.missing(pool, inv):
        have = ", ".join(str(g.index) for g in inv) or "none"
        add(Finding("fail", "gpus", f"{shown} names GPU {', '.join(map(str, lost))}, which "
                    f"this machine does not have (it has: {have})", "pinnforge setup gpus"))
    elif pool:
        add(Finding("ok", "gpus", f"{gpus.describe(pool, inv)} ({shown})"))
    elif gpus.render(spec) == "[]":
        add(Finding("ok", "gpus", f"CPU only, as chosen ({shown})"
                    + (" — cards are present: `pinnforge setup gpus` to use them" if inv else "")))
    elif inv:
        add(Finding("warn", "gpus", "GPUs exist but none is in the pool "
                    "(CUDA_VISIBLE_DEVICES excludes them)",
                    "unset CUDA_VISIBLE_DEVICES, or choose cards: pinnforge setup gpus"))
    elif not gpus.windows_cards():  # a Windows card has had its own finding
        add(Finding("warn", "gpus", "no NVIDIA GPU visible — runs train on the CPU "
                    "(fine for the demo, slow for real tasks)"))


def _task_checks(out: list[Finding], root: Path, pool: list[int], inv: list[gpus.Gpu]) -> None:
    add = out.append
    from pinnforge.run import anchor
    from pinnforge.task import authoring, contract
    from pinnforge.task import registry as tasks

    hw = gpus.hw_class(pool[:1], inv)
    for t in tasks.list_tasks(root):
        state = authoring.status(t.path)
        if state and not state.get("finalized"):
            add(Finding("warn", f"task {t.name}", f"still being authored "
                        f"(stage {state.get('stage')}: {state.get('status')})",
                        f"pinnforge task status {t.name}"))
            continue
        rep = contract.validate(t.path)
        if not rep.ok:
            names = ", ".join(n for n, _, _ in rep.failures[:3])
            add(Finding("fail", f"task {t.name}", f"fails the contract: {names}",
                        f"pinnforge task validate {t.name}"))
            continue
        meta = anchor.read_meta(t.name, root)
        if meta is None:
            add(Finding("warn", f"task {t.name}", "valid; no anchor yet "
                        "(measured on the first `run start`, or now)",
                        f"pinnforge task anchor {t.name} --measure"))
            continue
        why = anchor.staleness(meta, t.path, t.name, root, hw)
        if why:
            add(Finding("warn", f"task {t.name}", f"anchor is stale: {why}",
                        f"pinnforge task anchor {t.name} --rekey   (or --rebuild)"))
        else:
            add(Finding("ok", f"task {t.name}", f"ready — anchor rRMSE {meta.get('rRMSE'):.4g}"))


def _dashboard_check(out: list[Finding], root: Path) -> None:
    add = out.append
    from pinnforge.ui import server

    info = server.running(root)
    port = int(settings.get(root, "ui.port", 8765))
    if info is not None:
        add(Finding("ok", "dashboard", f"{server.url_of(info)} (running)"))
    elif _port_state("127.0.0.1", port) == "free":
        add(Finding("ok", "dashboard", f"starts on port {port} with `pinnforge ui`"))
    else:
        add(Finding("ok", "dashboard", f"port {port} is taken; `pinnforge ui` will use the "
                    "next free one"))


def _agent_checks(out: list[Finding], root: Path | None, ping: bool) -> None:
    add = out.append
    try:
        agent = bindings.resolve(root)
    except (KeyError, ValueError) as e:
        add(Finding("fail", "agent", str(e).strip('"'), "pinnforge setup"))
        return
    try:
        rt = registry.get_runtime(agent.runtime)
        model = registry.resolve_model(rt, agent.model)
    except (KeyError, ValueError) as e:
        add(Finding("fail", "agent", str(e).strip('"'), "pinnforge setup"))
        return
    binary = agent.command or rt.default_command
    source = f"binding {bindings.default_name()}" if bindings.default_name() else "built-in default"
    if not shutil.which(binary):
        add(Finding("fail", "agent", f"{agent.runtime} / {model}: `{binary}` is not on PATH",
                    "install it and sign in, or `pinnforge setup` to pick another"))
        return
    add(Finding("ok", "agent", f"{agent.runtime} / {model} ({source})"))
    if ping:
        from pinnforge.cli.agents_cmd import ping as do_ping

        ok, detail, _ = do_ping(agent)
        add(Finding("ok" if ok else "fail", "agent ping", detail,
                    "" if ok else "pinnforge setup   (credentials, provider, model id)"))


def cmd_doctor(args) -> int:
    found = checks(ping=args.ping)
    if args.json:
        print(json.dumps([asdict(f) for f in found], indent=2))
    else:
        marks = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
        print("PINNForge doctor")
        for f in found:
            print(f"  {marks[f.level]} {f.name:<14} {f.detail}")
            if f.fix and f.level != "ok":
                print(f"  {'':<6} {'':<14} → {f.fix}")
        fails = sum(1 for f in found if f.level == "fail")
        warns = sum(1 for f in found if f.level == "warn")
        verdict = "ready" if not fails else "not ready"
        print(f"\n{verdict}: {fails} problem(s), {warns} warning(s)")
    return 1 if any(f.level == "fail" for f in found) else 0


def register(sub) -> None:
    p = sub.add_parser("doctor", help="check every prerequisite, with the fix for each problem")
    p.add_argument("--ping", action="store_true", help="also prove the agent binding answers")
    p.add_argument("--json", action="store_true", help="machine-readable findings")
    p.set_defaults(func=cmd_doctor)

