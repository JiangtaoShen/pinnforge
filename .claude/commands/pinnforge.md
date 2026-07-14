---
description: Run N serial PINNForge blocks. Usage /pinnforge <n_blocks>
argument-hint: "<n_blocks>"
---

Run **$ARGUMENTS** PINNForge block(s) (default 1 if empty), strictly one
after another. You are a mechanical dispatcher: all science happens
inside block subagents. Work from the project root (the directory
containing `CLAUDE.md`); `<ROOT>` below means its absolute path.

## Per block

1. Next id: `ls blocks/` → `bNN` = highest existing + 1 (zero-padded,
   e.g. `b03`; consider only `bNN/` directories — ignore the
   `run_usage.jsonl` / `run_summary.md` bookkeeping files). Exception — if
   the highest existing `blocks/bMM/` fails the step-3 done check —
   no `kb/kb2/bMM.md`, or wall-time budget unspent (b00 is exempt) —
   that block died mid-run: re-dispatch `bMM` instead of a new id,
   appending to the prompt: "Workspace blocks/bMM/ already exists from
   a crashed run. Inspect its evals.jsonl, keep what is useful, spend
   only the remaining budget, then write (or update) kb/kb2/bMM.md."
2. Dispatch ONE subagent and wait for it:

   ```
   Agent(subagent_type="general-purpose",
         model="opus",   # Opus 4.8 — pinned; do NOT omit (see Rules)
         description="PINNForge block bNN",
         prompt="You are PINNForge block bNN. Read
         <ROOT>/block.md and execute the full block
         protocol autonomously until done. Your block id is bNN; your
         workspace is <ROOT>/blocks/bNN/. Reminder: run every
         evaluation in the foreground (block.md §4).")
   # substitute <ROOT> with the absolute project-root path
   ```

   Immediately also start a 20-min stall watchdog in the background
   (Bash `run_in_background`), so a silently stopped block wakes you.
   Age comes from python mtimes (`find -newermt` false-fires here); the
   `blocks/$B` dir entry floors the age while the workspace is empty:

   ```bash
   B=bNN; mkdir -p blocks/$B
   while true; do sleep 1200
     [ -f kb/kb2/$B.md ] && exit 0   # block wrote its summary
     age=$(python3 -c "import os,glob,time; fs=[f for f in glob.glob('blocks/$B/*')+['blocks/$B','kb/kb2/$B.md'] if os.path.exists(f)]; print(int((time.time()-max(map(os.path.getmtime,fs)))/60))")
     [ "$age" -lt 30 ] || { echo "STALL: blocks/$B quiet for ${age} min"; exit 1; }
   done
   ```

   On a STALL wake-up: if the block agent has paused, resume it (step
   3); if it is still working, restart the watchdog. TaskStop the
   watchdog once the block is verified done.

3. Verify — the block is done only when ALL hold: `kb/kb2/bNN.md`
   exists AND the wall-time budget is spent — `blocks/bNN/.budget` ≥
   `FORGE_WALL_BUDGET` minus the task's per-run wall (task/problem.md;
   failed runs' time counts) AND `evals.jsonl` has ≥1 non-smoke,
   non-diag line. Otherwise the block paused mid-run:
   - Resume the SAME agent (SendMessage to its agentId): "Resume block
     bNN: <X>/3000 wall-seconds spent, summary <missing|present>. Run
     evals in the foreground only (block.md §4), spend the remaining
     budget, then write kb/kb2/bNN.md." Resume again if it pauses
     again.
   - If the agent is gone (cannot be resumed), dispatch ONE fresh
     continuation subagent (same model pin) with the step-1 crashed-run
     prompt — or, when only the summary is missing after a spent
     budget, the no-eval repair prompt: "Block bNN finished without
     writing kb/kb2/bNN.md. Read block.md §5, read blocks/bNN/ (code +
     evals.jsonl), and write the summary faithfully. Do not run any
     evaluations." Never write it yourself.
   - A hard environment failure (GPU/driver down) is the one excuse:
     stop the run and report to the user instead of resume-looping.
4. Record + relay. The block's completion notification carries a
   `<usage>` block — read `subagent_tokens`, `tool_uses`, `duration_ms`
   from it and append ONE line to the run ledger (substitute the three
   numbers; model is the pinned `claude-opus-4-8`; notifications are
   per-segment, NOT cumulative — for a block that paused and resumed,
   sum each field across all its notifications):

   ```bash
   .venv/bin/python -c "import json,datetime; open('blocks/run_usage.jsonl','a').write(json.dumps({'block':'bNN','model':'claude-opus-4-8','duration_ms':DURATION_MS,'tokens':TOKENS,'tool_uses':TOOL_USES,'completed_at':datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})+chr(10))"
   ```

   Then relay one line to the user: `bNN done — best <rRMSE> (<file>),
   wall <spent>/3000 s`, and continue with the next block.

## After the last block

First write the run summary to `blocks/run_summary.md` — per-block model,
duration, tokens, tool-uses, eval wall-seconds, evals and best rRMSE,
plus totals — from the ledger (`blocks/run_usage.jsonl`) joined with
each `evals.jsonl`.
This regenerates the whole table each run, so it always reflects every
block recorded so far:

```bash
.venv/bin/python - <<'PY'
import json, pathlib, datetime
USAGE = pathlib.Path('blocks/run_usage.jsonl'); OUT = pathlib.Path('blocks/run_summary.md')
def fmt_dur(ms):
    if not ms: return '—'
    s = round(ms/1000); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'
def fmt_tok(t): return f'{t:,}' if t else '—'
usage = {}
if USAGE.exists():
    for line in USAGE.read_text().splitlines():
        if line.strip():
            r = json.loads(line); usage[r['block']] = r     # last append wins
rows, tot_ms, tot_tok, tot_tools, tot_wall = [], 0, 0, 0, 0.0
for ev in sorted(pathlib.Path('blocks').glob('b*/evals.jsonl')):   # every block, ledgered or not (b00!)
    b = ev.parent.name; u = usage.get(b, {}); used, diags, wall, best = 0, 0, 0.0, None
    if ev.exists():
        non = [r for r in (json.loads(x) for x in ev.read_text().splitlines() if x.strip()) if not r.get('smoke')]
        wall = sum((r.get('wall_s') or r.get('train_s') or 0) for r in non)
        scores = [r for r in non if not r.get('diag')]
        used = len(scores); diags = len(non) - used
        fin = [r['rRMSE'] for r in scores if r.get('rRMSE')]; best = min(fin) if fin else None
    ms = u.get('duration_ms') or 0; tok = u.get('tokens') or 0; tools = u.get('tool_uses') or 0
    tot_ms += ms; tot_tok += tok; tot_tools += tools; tot_wall += wall
    rows.append((b, u.get('model', '—'), ms, tok, tools, wall, used, diags, best))
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
L = ['# PINNForge — run summary', '', f'_Generated {now} · {len(rows)} block(s)_', '',
     '| Block | Model | Duration | Tokens | Tool uses | Wall s | Evals | Diags | Best rRMSE |', '|---|---|--:|--:|--:|--:|--:|--:|--:|']
for b, model, ms, tok, tools, wall, used, diags, best in rows:
    L.append(f'| {b} | {model} | {fmt_dur(ms)} | {fmt_tok(tok)} | {tools or "—"} | {wall:.0f} | {used} | {diags} | {format(best, ".5g") if best is not None else "—"} |')
L += [f'| **total** | | **{fmt_dur(tot_ms)}** | **{fmt_tok(tot_tok)}** | **{tot_tools}** | **{tot_wall:.0f}** | | | |', '']
OUT.write_text('\n'.join(L) + '\n')
print(f'wrote {OUT} ({len(rows)} blocks, total {fmt_dur(tot_ms)}, {fmt_tok(tot_tok)} tokens)')
PY
```

Then report to the user: per-block best table (block, thrust one-liner
from each kb2 header, best rRMSE) and the overall best:

```bash
.venv/bin/python -c "
import json, pathlib
best = (None, None)
for f in sorted(pathlib.Path('blocks').glob('b*/evals.jsonl')):
    scores = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    real = [r['rRMSE'] for r in scores if not r.get('smoke') and r.get('rRMSE')]
    bb = min(real) if real else None
    print(f.parent.name, 'evals', sum(1 for r in scores if not r.get('smoke') and not r.get('diag')), 'best', bb)
    if bb is not None and (best[0] is None or bb < best[0]): best = (bb, f.parent.name)
print('OVERALL BEST', best[0], 'in', best[1])"
```

## Rules

- **Model pin: always dispatch block and step-3-repair subagents
  with `model="opus"` (Opus 4.8).** Never omit it — without the pin the
  subagent silently inherits the session's current model, so runs would
  not be reproducible. The `opus` alias currently resolves to Opus 4.8.
- Strictly serial: never two subagents alive at once; block N is
  verified done before block N+1 is dispatched.
- Never do block work yourself: no reading `kb/kb1/`, no writing
  candidates or summaries, no running `task/eval.py` — step-3 repairs
  run in subagents too.
- A block whose evals all failed is still done once its budget is
  spent and its summary exists — do not re-run it.
- Never touch anything in `task/`, `kb/kb1/`, existing `kb/kb2/*.md`,
  or other blocks' directories.
