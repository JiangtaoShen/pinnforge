"""`pinnforge agents …` — which harnesses are installed, and do they work."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from pinnforge import bindings, oscompat
from pinnforge.config import AgentConfig
from pinnforge.runtime import registry


def cmd_list(args) -> int:
    print(f"{'runtime':<14} {'command':<14} {'default model':<16} installed")
    for row in registry.detect_available():
        mark = row["path"] or "—"
        # A harness that names no default requires one per run; say so rather
        # than printing a blank column that reads like missing information.
        model = row["default_model"] or "(pass --model)"
        print(f"{row['runtime']:<14} {row['command']:<14} {model:<16} {mark}")
    return 0


# One prompt, identical for every runtime, so the replies are comparable.
PING_PROMPT = "Reply with just the word: ok"


def ping(agent: AgentConfig, timeout: float = 120.0) -> tuple[bool, str, str]:
    """Send one prompt through the binding exactly as a block would be started.

    `--version` proves a binary is installed, which is not the question a run
    needs answered: the credentials, the provider config and the model id have
    to work too, and each of those fails at the first block instead — an hour
    in, with a run directory and a measured anchor already paid for.

    The ping goes through the runtime adapter itself — the same argv, model
    flag and environment handling a block gets — rather than a hand-written
    command line per harness. The old hand-written forms dropped `--model`
    for two of the four harnesses, so a typo in the model id pinged fine and
    failed at the first block anyway.

    Returns (ok, detail, the model id the harness reports having run).
    """
    try:
        runtime = registry.get_runtime(agent.runtime)
        model = registry.resolve_model(runtime, agent.model)
    except (KeyError, ValueError) as e:
        return False, str(e).strip('"'), ""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pinnforge-ping-",
                                     ignore_cleanup_errors=True) as tmp:
        log = Path(tmp) / "ping.log"
        try:
            handle = runtime.start(
                block_id="ping",
                cwd=Path(tmp),
                prompt=PING_PROMPT,
                model=model,
                log_path=log,
                env={"PINNFORGE_ROLE": "ping"},
                command=agent.command or None,
                runtime_options=agent.runtime_options,
            )
        except (OSError, ValueError) as e:
            return False, f"could not start {agent.runtime}: {e}", ""
        rc = handle.wait(timeout=timeout)
        if rc is None:
            handle.stop()
            return False, f"no reply within {timeout:g}s", ""
        handle.close()
        took = time.monotonic() - started
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        resolved = ""
        extract = getattr(runtime, "extract_usage", None)
        if extract is not None:
            try:
                resolved = extract(log).model
            except (OSError, ValueError):
                resolved = ""
    if rc != 0:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return False, f"exit {rc} in {took:.1f}s: {(lines[-1] if lines else '')[:140]}", ""
    if not text.strip():
        return False, f"exit 0 in {took:.1f}s but said nothing", ""
    ran = f", ran {resolved}" if resolved and resolved != model else ""
    return True, f"{took:.1f}s{ran}", resolved


def cmd_doctor(args) -> int:
    """Check each harness is on PATH and answers `--version`.

    A missing CLI is the single most common reason a run dies on its first
    block, and it costs nothing to find out beforehand. `--ping` goes further
    and spends one exchange proving the model binding works too.
    """
    rows = registry.detect_available()
    if args.runtime:
        rows = [r for r in rows if r["runtime"] == registry.canonical(args.runtime)]
        if not rows:
            print(f"unknown runtime: {args.runtime}")
            return 2
    failures = 0
    for row in rows:
        name, cmd, path = row["runtime"], row["command"], row["path"]
        if not path:
            print(f"  [MISS] {name:<14} {cmd} not on PATH")
            failures += 1
            continue
        try:
            out = subprocess.run(
                oscompat.resolve_command([cmd, "--version"]), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=args.timeout
            )
            version = (out.stdout or out.stderr).strip().splitlines()
            version = version[0] if version else "(no version output)"
            print(f"  [OK]   {name:<14} {version}")
            if args.ping:
                model = args.model or row["default_model"]
                ok, detail, _ = ping(AgentConfig(runtime=name, model=model), args.ping_timeout)
                mark = "[PING]" if ok else "[FAIL]"
                print(f"  {mark} {name:<14} {model or '(no model)'} — {detail}")
                failures += 0 if ok else 1
        except subprocess.TimeoutExpired:
            print(f"  [SLOW] {name:<14} {cmd} --version timed out after {args.timeout}s")
            failures += 1
        except OSError as e:
            print(f"  [FAIL] {name:<14} {e}")
            failures += 1
    print(f"{len(rows)} runtime(s), {failures} problem(s)")
    return 0 if failures == 0 else 1


def cmd_bindings(args) -> int:
    data = bindings.load()
    if not data["bindings"]:
        print("no agent bindings yet — create one with `pinnforge setup`")
        return 0
    default = data.get("default")
    print(f"   {'name':<18} {'runtime':<13} model")
    for name, b in sorted(data["bindings"].items()):
        mark = "*" if name == default else " "
        print(f" {mark} {name:<18} {b.get('runtime', ''):<13} {b.get('model', '')}")
    print(f"\n* default   (file: {bindings.path()})")
    return 0


def cmd_remove(args) -> int:
    for name in args.names:
        bindings.remove(name)
        print(f"removed binding {name}")
    return 0


def cmd_default(args) -> int:
    bindings.get(args.name)  # raises with the known names if it is not one
    data = bindings.load()
    data["default"] = args.name
    bindings.save(data)
    print(f"default agent binding: {args.name}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("agents", help="coding-agent harnesses and bindings")
    s = p.add_subparsers(dest="agents_cmd", required=True)

    s.add_parser("list", help="list known runtimes").set_defaults(func=cmd_list)

    doc = s.add_parser("doctor", help="check the harness CLIs are usable")
    doc.add_argument("runtime", nargs="?", default=None)
    doc.add_argument("--timeout", type=int, default=30)
    doc.add_argument(
        "--ping",
        action="store_true",
        help="spend one exchange proving the model binding works, not just the CLI",
    )
    doc.add_argument("--model", default="", help="model to ping with (default: the runtime's)")
    doc.add_argument("--ping-timeout", type=float, default=120.0)
    doc.set_defaults(func=cmd_doctor)

    s.add_parser("bindings", help="list the agent bindings `pinnforge setup` made").set_defaults(
        func=cmd_bindings
    )
    rm = s.add_parser("remove", help="delete agent bindings")
    rm.add_argument("names", nargs="+")
    rm.set_defaults(func=cmd_remove)
    df = s.add_parser("default", help="make a binding the default")
    df.add_argument("name")
    df.set_defaults(func=cmd_default)
