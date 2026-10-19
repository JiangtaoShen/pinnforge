"""A run told as a story: what the discovery replay plays back.

Each block becomes one beat — what the agent set out to do, the candidates it
wrote and how each scored, what it found, which papers of the theory corpus it
reached for, and what it spent — in the order the run lived it. Everything
here is read through `api`, the dashboard's own read-only layer, from records
the dashboard already shows: block summaries (redacted), evaluation records,
the usage ledger, the agents' activity, the run's copy of the corpus index and
the task's problem statement. So the story carries nothing the dashboard would
not: no reference values, no error field — the replay's field panel shows the
blocks' own predictions, and how much each champion changed them.

The narrative comes from the summary each block is required to write
(`block.md` §5): its title line is the beat's headline, *Thrust & rationale*
its line of attack, the *Evals* table the change each candidate made, and the
*Findings* the notes it adds to the practice knowledge base. Nothing is
generated: a block that wrote little is told in few words.
"""

from __future__ import annotations

import re
from pathlib import Path

from pinnforge.ui import api

TRAIN_TIME = re.compile(r"TRAIN_TIME\s*=\s*([0-9]+(?:\.[0-9]+)?)")
WALL_PHRASE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s\s+of\s+wall", re.I)
KB1_FILE = re.compile(r"\b(\d{3})_(?:19|20)\d{2}_")
KB1_REF = re.compile(r"\b(?:kb1\s*[#:/-]?\s*|T-)(\d{1,3})\b", re.I)
CANDIDATE = re.compile(r"\b(b\d{2,3}_[A-Za-z0-9_.-]+?)(?:\.py)?\b")
MAX_NOTE = 320
MAX_THRUST = 600


# ─────────────────────────────── text ───────────────────────────────


def _plain(md: str) -> str:
    """Markdown prose as one line of text: no emphasis marks, links or code fences."""
    t = re.sub(r"`([^`]*)`", r"\1", md)
    t = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _sections(md: str) -> tuple[str, dict[str, str]]:
    """The `# title` line and each `## heading`'s body, by lower-cased heading."""
    title, out, cur, buf = "", {}, None, []
    for line in md.splitlines():
        if not title and line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = line[3:].strip().lower(), []
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return title, out


def _section(sections: dict[str, str], *names: str) -> str:
    for key, body in sections.items():
        if any(key.startswith(n) for n in names):
            return body
    return ""


def _table(body: str) -> list[dict[str, str]]:
    """A Markdown table's rows, keyed by lower-cased header."""
    rows = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []
    cells = lambda ln: [c.strip() for c in ln.strip().strip("|").split("|")]  # noqa: E731
    head = [c.lower() for c in cells(rows[0])]
    out = []
    for ln in rows[1:]:
        if re.fullmatch(r"\|?[\s:|-]+\|?", ln):
            continue
        out.append(dict(zip(head, cells(ln), strict=False)))
    return out


def _points(body: str) -> list[str]:
    """The items of a findings section: its bullets, else its sentences."""
    bullets, cur = [], None
    for line in body.splitlines():
        m = re.match(r"\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if m:
            if cur:
                bullets.append(cur)
            cur = m.group(1)
        elif cur is not None and line.strip():
            cur += " " + line.strip()
        elif cur is not None:
            bullets.append(cur)
            cur = None
    if cur:
        bullets.append(cur)
    items = bullets or re.split(r"(?<=[.!?])\s+(?=[A-Z(`*])", _plain(body))
    return [p for p in (_plain(i)[:MAX_NOTE] for i in items) if len(p) > 3]


# ─────────────────────────────── the task ───────────────────────────────


def _problem(text: str | None) -> dict:
    """What the PDE panel shows, read from `problem.md`."""
    if not text:
        return {"title": "", "name": "", "equations": [], "facts": [], "train_s": None}
    title, sections = _sections(text)
    name = re.split(r"\s+[—–-]\s+", title, maxsplit=1)[-1].strip() if title else ""
    equations = []
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        before = text[:m.start()].rstrip().splitlines()
        label = before[-1].strip() if before else ""
        label = label.rstrip(":") if label.endswith(":") and len(label) < 90 else ""
        equations.append({"name": label, "tex": re.sub(r"\s+", " ", m.group(1)).strip()})
    facts = []
    for key, label in (("domain", "Domain"), ("boundary", "BC"), ("initial", "IC")):
        body = _section(sections, key)
        first = next((ln for ln in body.splitlines() if ln.strip()), "")
        if first:
            facts.append([label, _plain(first)[:160]])
    train = TRAIN_TIME.search(text) or WALL_PHRASE.search(text)
    return {"title": title, "name": name or title, "equations": equations[:4],
            "facts": facts, "train_s": float(train.group(1)) if train else None}


def _kb1(text: str | None) -> list[dict]:
    """The run's copy of the theory corpus, one entry per paper: slot, id, year, title."""
    out = []
    for m in re.finditer(r"^- \*\*(\d{3})_(\d{4})_[^*]*\*\*\s*\((\d{4})\)\s*[—–-]\s*(.+)$",
                         text or "", re.M):
        n = int(m.group(1))
        out.append({"s": n, "id": f"T-{n:03d}", "y": int(m.group(3)),
                    "t": m.group(4).strip()[:120]})
    return out


# ─────────────────────────────── the blocks ───────────────────────────────


def _roster(evals: list[dict], entry: float | None) -> tuple[list[dict], dict]:
    """Each candidate a block scored, first-evaluated first, with what it did to the champion."""
    order, best, failed = [], {}, set()
    for e in evals:
        name = re.sub(r"\.py$", "", e["candidate"] or "")
        if not name or e.get("diag"):
            continue
        if name not in order:
            order.append(name)
        r = e.get("rRMSE")
        if r is None:
            failed.add(name)
        elif name not in best or r < best[name]:
            best[name] = r
    scored = [best[c] for c in order if c in best]
    top = min(scored, default=None)
    roster, outcomes = [], {}
    for c in order:
        r = best.get(c)
        if r is None:
            st = "failed" if c in failed else "smoke_only"
        elif entry is None or r < entry:
            st = "winner" if r == top else "improved"
        else:
            st = "regressed"
        roster.append({"id": c, "r": r, "st": st})
        outcomes[st] = outcomes.get(st, 0) + 1
    return roster, outcomes


def _cited(texts: list[str], kb1: dict[int, dict]) -> list[dict]:
    seen: list[int] = []
    for text in texts:
        for m in list(KB1_FILE.finditer(text)) + list(KB1_REF.finditer(text)):
            n = int(m.group(1))
            if n in kb1 and n not in seen:
                seen.append(n)
    return [{"id": kb1[n]["id"], "slot": n, "t": kb1[n]["t"], "y": kb1[n]["y"]} for n in seen]


def notes(block: str, findings: list[str]) -> list[dict]:
    """A block's findings as the notes it leaves in the practice knowledge base."""
    return [{"id": f"P-{block}-{k + 1:02d}", "k": "finding", "t": t}
            for k, t in enumerate(findings)]


def tally(blocks: list[dict]) -> None:
    """Each block's running totals: notes written, and distinct papers drawn on, so far."""
    notes_cum, theory = 0, set()
    for b in blocks:
        notes_cum += len(b["notes"])
        theory.update(c["slot"] for c in b["cited"])
        b["notes_cum"], b["theory_cum"] = notes_cum, len(theory)


def _smoke_counts(root: Path, run_id: str, block: str) -> tuple[int, int]:
    """(smoke evaluations, distinct candidates smoked) — what `run_detail` leaves out."""
    run = api._run_dir(root, run_id)
    recs = api._records(run, block)
    smokes = [r for r in recs if r.get("smoke")]
    names = {api._str(r.get("candidate"), cap=4096) for r in recs if not r.get("diag")}
    return len(smokes), len({n for n in names if n})


def build(root: Path, run_id: str) -> dict:
    """The story of run `run_id`, from its records as the dashboard reads them."""
    detail = api.run_detail(root, run_id)
    run = api._run_dir(root, run_id)
    task = _problem(api._read_text(run / "task" / "problem.md"))
    kb1_list = _kb1(api._read_text(run / "kb1" / "INDEX.md"))
    kb1 = {e["s"]: e for e in kb1_list}
    frames = [f for f in detail["frames"] if f["run_best"]]

    champion = None  # the run's best so far; b00, the control, sets the first
    gpu_cum = tok_cum = 0.0
    blocks = []
    for b in detail["blocks"]:
        bid = b["id"]
        evals = [e for e in detail["evals"] if e["block"] == bid]
        md = ""
        if b["summary"]:
            try:
                md = api.summary_md(root, run_id, bid)["markdown"]
            except api.NotFound:
                md = ""
        title, sections = _sections(md)
        thrust = _plain(_section(sections, "thrust"))[:MAX_THRUST]
        findings = _points(_section(sections, "finding"))
        table = _table(_section(sections, "eval"))
        tags = []
        for row in table:
            change = _plain(row.get("change vs parent") or row.get("change") or "")
            if 0 < len(change) <= 28 and change.lower() not in (t.lower() for t in tags):
                tags.append(change)
        activity = api.activity(root, run_id, bid, None) if bid != "b00" else {"events": []}
        texts = [title, thrust, *findings] + [str(e.get("text", "")) for e in activity["events"]]
        cited = _cited(texts, kb1)
        fork = next(iter(CANDIDATE.findall(thrust)), None)
        if fork is None and re.search(r"\bbaseline(\.py)?\b", thrust, re.I):
            fork = "baseline"
        entry = champion
        roster, outcomes = _roster(evals, entry)
        if b["best"] is not None and (champion is None or b["best"] < champion):
            champion = b["best"]
        gpu_cum += (b["spent"] or 0.0) / 3600.0
        tok_cum += b["tokens"] or 0
        smoke, n_cand = _smoke_counts(root, run_id, bid)
        frame = None
        for f in frames:
            if f["block"] and f["block"] <= bid:
                frame = f["frame"]
        headline = title or (f"{bid} — round 0: the standard PINN, measured as it is"
                             if bid == "b00" else f"{bid}")
        blocks.append({
            "id": bid, "status": b["status"], "headline": headline, "thrust": thrust,
            "fork": fork, "findings": findings[:6], "tags": tags[:6], "roster": roster,
            "outcomes": outcomes, "entry": entry, "exit": champion, "best": b["best"],
            "cited": cited, "notes": notes(bid, findings), "frame": frame,
            "cost": {"spent_s": b["spent"], "gpu_cum_h": round(gpu_cum, 4),
                     "tokens": b["tokens"], "tok_cum": tok_cum, "tools": b["tool_uses"],
                     "model": b["model"], "duration_ms": b["duration_ms"],
                     "started": b["started"], "completed": b["completed"]},
            "counts": {"cand": n_cand, "full": b["evals"], "smoke": smoke},
        })

    tally(blocks)
    baseline, final = detail.get("anchor"), detail.get("best")
    if final is None or (baseline is not None and final > baseline):
        final = baseline
    budget = detail["config"]["wall_budget_s"] or 0.0
    n_work = sum(1 for b in blocks if b["id"] != "b00")
    return {
        "run": detail["id"], "task": detail["task"], "status": detail["status"],
        "title": task["title"], "name": task["name"], "equations": task["equations"],
        "facts": task["facts"], "train_s": detail["config"]["train_s"] or task["train_s"],
        "cpu": not detail["config"]["gpus"],
        "agent": detail["config"]["agent"], "baseline": baseline, "final": final,
        "factor": (baseline / final) if baseline and final else None,
        "budget_h": round(budget * n_work / 3600.0, 3),
        "kb1": {"n": len(kb1_list), "entries": kb1_list},
        "grid": detail["grid"], "view": detail["view"],
        "blocks": blocks,
    }
