"""The discovery film's script: what an agent may write, checked.

A discovery film is one fixed player (`ui/static/replay.js`) playing one run.
Every number it shows — scores, times, costs, which candidate beat which —
comes from the run's records (`facts.json`, built by `ui/story.py`). What an
agent adds is the telling: for each block a headline, its line of attack, the
tags that name what it changed, what it found, which papers it drew on, and a
short label for each candidate; for the run, one closing sentence. That is the
script, `script.json`, and this module is its contract.

It is stdlib-only and vendored, as `check.py`, into the package the agent
works in, so the agent can check its own script the way it will be checked:

    python check.py            # reads script.json and facts.json beside it

`validate(script, facts)` returns the script as the player will use it and the
list of what is wrong with it; a script with errors is not used.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VERSION = 1
MAX = {
    "logline": 220,
    "headline": 110,
    "thrust": 520,
    "tag": 28,
    "tags": 6,
    "finding": 240,
    "findings": 4,
    "cited": 8,
    "label": 64,
}
BLOCK_KEYS = {"headline", "thrust", "tags", "findings", "cited", "candidates"}
TOP_KEYS = {"version", "logline", "blocks"}
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _text(value, where: str, cap: int, errors: list[str], required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        errors.append(f"{where}: must be a string")
        return ""
    text = re.sub(r"\s+", " ", CONTROL.sub(" ", value)).strip()
    if required and not text:
        errors.append(f"{where}: must not be empty")
    if len(text) > cap:
        errors.append(f"{where}: {len(text)} characters, at most {cap}")
    return text


def _texts(value, where: str, cap: int, most: int, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{where}: must be a list of strings")
        return []
    if len(value) > most:
        errors.append(f"{where}: {len(value)} items, at most {most}")
    out = [_text(v, f"{where}[{i}]", cap, errors, required=True) for i, v in enumerate(value)]
    return [t for t in out if t]


def validate(script, facts: dict) -> tuple[dict, list[str]]:
    """The script as the player uses it, and what is wrong with it."""
    errors: list[str] = []
    if not isinstance(script, dict):
        return {}, ["the script must be a JSON object"]
    for key in sorted(set(script) - TOP_KEYS):
        errors.append(f"unknown key {key!r} (allowed: {', '.join(sorted(TOP_KEYS))})")
    if script.get("version") != VERSION:
        errors.append(f"version: must be {VERSION}")
    out = {"version": VERSION,
           "logline": _text(script.get("logline"), "logline", MAX["logline"], errors),
           "blocks": {}}
    papers = {e["id"] for e in facts.get("kb1", {}).get("entries", [])}
    blocks = {b["id"]: b for b in facts.get("blocks", [])}
    given = script.get("blocks")
    if not isinstance(given, dict):
        errors.append("blocks: must be an object keyed by block id")
        return out, errors
    for bid in sorted(set(given) - set(blocks)):
        errors.append(f"blocks.{bid}: no such block in facts.json")
    for bid, fact in blocks.items():
        where = f"blocks.{bid}"
        b = given.get(bid)
        if not isinstance(b, dict):
            errors.append(f"{where}: missing (every block in facts.json needs its entry)")
            continue
        for key in sorted(set(b) - BLOCK_KEYS):
            allowed = ", ".join(sorted(BLOCK_KEYS))
            errors.append(f"{where}: unknown key {key!r} (allowed: {allowed})")
        cited = b.get("cited") or []
        if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
            errors.append(f"{where}.cited: must be a list of paper ids such as \"T-012\"")
            cited = []
        if len(cited) > MAX["cited"]:
            errors.append(f"{where}.cited: {len(cited)} papers, at most {MAX['cited']}")
        for c in cited:
            if c not in papers:
                errors.append(f"{where}.cited: {c!r} is not a paper in facts.json kb1.entries")
        roster = {r["id"] for r in fact.get("roster", [])}
        labels = b.get("candidates") or {}
        if not isinstance(labels, dict):
            errors.append(f"{where}.candidates: must be an object, candidate id -> label")
            labels = {}
        for cid in sorted(set(labels) - roster):
            errors.append(f"{where}.candidates: {cid!r} is not one of this block's candidates")
        out["blocks"][bid] = {
            "headline": _text(b.get("headline"), f"{where}.headline", MAX["headline"], errors,
                              required=True),
            "thrust": _text(b.get("thrust"), f"{where}.thrust", MAX["thrust"], errors),
            "tags": _texts(b.get("tags"), f"{where}.tags", MAX["tag"], MAX["tags"], errors),
            "findings": _texts(b.get("findings"), f"{where}.findings", MAX["finding"],
                               MAX["findings"], errors),
            "cited": list(dict.fromkeys(c for c in cited if c in papers)),
            "candidates": {cid: _text(v, f"{where}.candidates.{cid}", MAX["label"], errors)
                           for cid, v in labels.items() if cid in roster},
        }
    return out, errors


def main() -> int:
    here = Path(__file__).resolve().parent
    try:
        facts = json.loads((here / "facts.json").read_text(encoding="utf-8"))
        script = json.loads((here / "script.json").read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        print(f"missing: {e.filename}")
        return 2
    except ValueError as e:
        print(f"script.json is not valid JSON: {e}")
        return 2
    _, errors = validate(script, facts)
    for e in errors:
        print(e)
    print("ok: the script is ready" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
