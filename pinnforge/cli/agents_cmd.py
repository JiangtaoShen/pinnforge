"""`pinnforge agents …` — which harnesses are installed, and do they work."""

from __future__ import annotations

import subprocess
import time

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

# How each CLI is put into non-interactive mode for a single exchange. The
# argv is the adapter's, minus everything a block needs and a ping does not.
PING_ARGV = {
    "claude_code": lambda cmd, model: [cmd, "-p", PING_PROMPT, "--model", model],
    "codex": lambda cmd, model: [cmd, "exec", PING_PROMPT],
    "cursor_agent": lambda cmd, model: [cmd, "--print", PING_PROMPT],
    "opencode": lambda cmd, model: [cmd, "run", "--model", model, PING_PROMPT],
}


def _ping(runtime: str, command: str, model: str, timeout: float) -> tuple[bool, str]:
    """Send one prompt through the CLI and its model. Exit 0 and output, or why not.

    `--version` proves a binary is installed, which is not the question a run
    needs answered: the credentials, the provider config and the model id have
    to work too, and each of those fails at the first block instead — an hour
    in, with a run directory and a measured anchor already paid for.
    """
    build = PING_ARGV.get(runtime)
    if build is None:
        return False, f"no non-interactive form known for {runtime!r}"
    if not model:
        return False, "no model to ping with; pass --model"
    started = time.monotonic()
    try:
        out = subprocess.run(
            build(command, model), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"no reply within {timeout:g}s"
    except OSError as e:
        return False, f"could not run {command}: {e}"
    took = time.monotonic() - started
    reply = (out.stdout or "").strip()
    if out.returncode != 0:
        err = (out.stderr or "").strip().splitlines()
        return False, f"exit {out.returncode} in {took:.1f}s: {(err[0] if err else '')[:110]}"
    if not reply:
        return False, f"exit 0 in {took:.1f}s but said nothing"
    first = reply.splitlines()[0]
    return True, f"{took:.1f}s, replied {first[:40]!r}"


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
                [cmd, "--version"], capture_output=True, text=True, timeout=args.timeout
            )
            version = (out.stdout or out.stderr).strip().splitlines()
            version = version[0] if version else "(no version output)"
            print(f"  [OK]   {name:<14} {version}")
            if args.ping:
                model = args.model or row["default_model"]
                ok, detail = _ping(name, cmd, model, args.ping_timeout)
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


def register(sub) -> None:
    p = sub.add_parser("agents", help="coding-agent harnesses")
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
