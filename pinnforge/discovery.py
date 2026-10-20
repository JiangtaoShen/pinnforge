"""The discovery film: a finished run, told by an agent, played by a fixed player.

A run is seen two ways. While it runs, the dashboard shows it live. Once it is
over, it is made into a film, and that is this module:

1. `package` writes `runs/<id>/discovery/`, the run's complete record as the
   film will use it. It holds the facts (`facts.json`, from `ui/story.py`), a
   first script extracted mechanically from the block summaries (`draft.json`),
   the summaries, every block's agent transcript, the problem statement and
   the corpus index. All of it is read through the dashboard's API, so it holds
   nothing of the answers. The package also carries the brief, the script's
   checker and a worked example: the paradigm every film follows.
2. `direct` hands the package to the run's own agent. The agent reads it and
   writes `script.json`: for each block a headline, its line of attack, what
   it found, the papers it drew on, a label for each candidate, and for the
   run one closing sentence.
3. `render` checks the script (`film_check`), lays it over the facts and
   writes `runs/<id>/discovery.html`: the fixed player (`ui/static/replay.js`)
   with everything it needs inside, playable from disk.

The agent writes words, never the numbers on screen: every score, time and
cost comes from the facts. If the script fails its check, or no agent could
write one, the film is rendered from the draft and the result says so.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pinnforge import film_check, oscompat, paths
from pinnforge.config import AgentConfig, RunConfig

logger = logging.getLogger(__name__)

PACKAGE = "discovery"
FILM = "discovery.html"
SCRIPT = "script.json"
TRANSCRIPT_CAP = 4 * 1024 * 1024  # per agent log
TIMEOUT_S = 45 * 60

PROMPT = ("You are making the discovery film of run {run}. Read {brief} and follow it. "
          "Work only inside {pkg}; you are done when its check passes.")


@dataclass
class Film:
    path: Path                      # runs/<id>/discovery.html
    scripted_by: str                # "agent", or "draft" when the agent's script was not used
    problems: list[str] = field(default_factory=list)


def _api(root: Path):
    from pinnforge.ui import api

    api.ROOT_REAL = str(Path(root).resolve())
    return api


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _dump(path: Path, value) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


# ─────────────────────────────── the package ───────────────────────────────


def draft(facts: dict) -> dict:
    """A script read straight off the block summaries: the agent's starting point."""
    return {
        "version": film_check.VERSION,
        "logline": "",
        "blocks": {b["id"]: {
            "headline": b["headline"],
            "thrust": b["thrust"],
            "tags": b["tags"],
            "findings": b["findings"][:film_check.MAX["findings"]],
            "cited": [c["id"] for c in b["cited"]][:film_check.MAX["cited"]],
            "candidates": {},
        } for b in facts["blocks"]},
    }


def _transcript(api, run: Path, block: str, runtime: str) -> str:
    """Everything a block's agent did, across its sessions, one event a line."""
    from pinnforge.ui import feeds

    logs = run / "logs"
    if not logs.is_dir() or logs.is_symlink():
        return ""
    out = []
    for log in sorted(logs.glob(f"{block}.*.log"), key=api._log_index):
        data = api._read_bytes(log, TRANSCRIPT_CAP) if not log.is_symlink() else None
        if data is None:
            continue
        out.append(f"## {log.name}\n")
        lines = data.decode("utf-8", errors="replace").splitlines()
        events = feeds.parse(runtime, lines, str(run.resolve()))
        out += [f"- [{e['kind']}] {e['text']}" for e in events]
        out.append("")
    return "\n".join(out)


def package(root: Path, run: Path) -> Path:
    """Write `runs/<id>/discovery/`: the run's record, the brief, the checker, the example."""
    from pinnforge import agents
    from pinnforge.ui import story

    api = _api(root)
    run = Path(run)
    pkg = run / PACKAGE
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    facts = story.build(root, run.name)
    _dump(pkg / "facts.json", facts)
    _dump(pkg / "draft.json", draft(facts))
    runtime = facts["agent"]["runtime"]
    for b in facts["blocks"]:
        with contextlib.suppress(api.NotFound):
            summary = api.summary_md(root, run.name, b["id"])["markdown"]
            _write(pkg / "summaries" / f"{b['id']}.md", summary)
        text = _transcript(api, run, b["id"], runtime)
        if text:
            _write(pkg / "activity" / f"{b['id']}.md", text)
    for src, dest in ((run / "task" / "problem.md", pkg / "problem.md"),
                      (run / "kb1" / "INDEX.md", pkg / "kb1" / "INDEX.md")):
        text = api._read_text(src)
        if text is not None:
            _write(dest, text)
    shutil.copyfile(Path(film_check.__file__), pkg / "check.py")
    (pkg / "example").mkdir()
    shutil.copyfile(paths.prompts_dir() / "discovery_example.json", pkg / "example" / SCRIPT)
    limits = {f"max_{k}": str(v) for k, v in film_check.MAX.items()}
    python = agents._path(sys.executable)
    python = f'"{python}"' if " " in python else python  # a path with a space, quoted for any shell
    _write(pkg / "BRIEF.md", agents._render(
        "discovery.md", run=run.name, pkg=agents._path(pkg), python=python,
        blocks=", ".join(f"`{b['id']}`" for b in facts["blocks"]), **limits))
    return pkg


# ─────────────────────────────── the agent ───────────────────────────────


def direct(root: Path, run: Path, agent: AgentConfig, timeout_s: float = TIMEOUT_S,
           on_start=None):
    """The run's agent reads the package and writes its `script.json`."""
    from pinnforge import agents

    pkg = Path(run) / PACKAGE
    (pkg / SCRIPT).unlink(missing_ok=True)
    prompt = PROMPT.format(run=Path(run).name, brief=agents._path(pkg / "BRIEF.md"),
                           pkg=agents._path(pkg))
    return agents._drive(
        prompt, cwd=pkg, agent=agent, log_path=pkg / "agent.log", label=f"film-{Path(run).name}",
        env={"PINNFORGE_ROOT": str(root), "PINNFORGE_RUN": str(run)}, role="film",
        timeout_s=timeout_s, on_start=on_start)


# ─────────────────────────────── the film ───────────────────────────────


def lay(facts: dict, script: dict) -> dict:
    """The facts with the script's words laid over them: what the player plays."""
    from pinnforge.ui import story

    film = copy.deepcopy(facts)
    papers = {e["id"]: e for e in film["kb1"]["entries"]}
    for b in film["blocks"]:
        s = script["blocks"].get(b["id"])
        if s is None:
            continue
        b["headline"] = s["headline"] or b["headline"]
        b["thrust"], b["tags"], b["findings"] = s["thrust"], s["tags"], s["findings"]
        b["notes"] = story.notes(b["id"], s["findings"])
        b["cited"] = [{"id": c, "slot": papers[c]["s"], "t": papers[c]["t"], "y": papers[c]["y"]}
                      for c in s["cited"]]
        for r in b["roster"]:
            r["label"] = s["candidates"].get(r["id"], "")
    story.tally(film["blocks"])
    film["logline"] = script.get("logline", "")
    return film


def render(root: Path, run: Path, script: dict | None = None) -> Film:
    """Write `runs/<id>/discovery.html` from the package's facts and a script.

    `script` defaults to the package's `script.json`; a script that is missing
    or fails its check gives way to the draft.
    """
    from pinnforge.ui import report

    run = Path(run)
    pkg = run / PACKAGE
    facts = json.loads((pkg / "facts.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if script is None:
        try:
            script = json.loads((pkg / SCRIPT).read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(f"no {SCRIPT} was written")
        except ValueError as e:
            problems.append(f"{SCRIPT} is not valid JSON: {e}")
    checked, errors = film_check.validate(script, facts) if script is not None else ({}, [])
    problems += errors
    scripted_by = "agent"
    if problems:
        scripted_by = "draft"
        checked, _ = film_check.validate(draft(facts), facts)
    html = report.render(root, run.name, film=lay(facts, checked))
    dest = run / FILM
    tmp = dest.with_name(f".{FILM}.tmp")
    tmp.write_text(html, encoding="utf-8")
    oscompat.replace(tmp, dest)
    return Film(dest, scripted_by, problems)


def make(root: Path, run: Path, agent: AgentConfig | None = None,
         timeout_s: float = TIMEOUT_S, on_start=None) -> Film:
    """Package the run, have its agent write the script, render the film.

    `on_start(pgid)` is told the agent's process group (see `agents._drive`).
    """
    run = Path(run)
    agent = agent or RunConfig.load(paths.config_path(run)).agent
    package(root, run)
    outcome = direct(root, run, agent, timeout_s, on_start=on_start)
    film = render(root, run)
    if not outcome.ok:
        film.problems.insert(0, "the agent timed out" if outcome.timed_out
                             else f"the agent exited with {outcome.returncode}")
    logger.info("discovery film: %s (script: %s, %.0f s)", film.path, film.scripted_by,
                outcome.duration_s)
    for p in film.problems:
        logger.warning("discovery film: %s", p)
    return film
