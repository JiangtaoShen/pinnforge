---
name: swap-task
description: Swap in a new PINNForge task package and rebuild the b00 baseline node, leaving the project ready to run from scratch. Trigger: "swap the task to <name> (from <folder>)".
---

Install a new problem into `task/` and rebuild the baseline node.
The framework touches `task/` only through the **Task contract** in
CLAUDE.md — read it first. Work from the project root.

**Canonical template — the current `task/` package** (`problem.md`,
`baseline.py`, `eval.py`, plus its data files). Build the new package by
**adapting these files**, not by writing from scratch: keep their
structure, wording, and machinery; change only what the new problem
genuinely requires. Everything that can stay identical stays identical.

1. Locate the source package the user names (bare names: look under
   `/home/jiangtao/reference/`) — it supplies the **physics and data**
   (PDE, domain, reference field). It rarely ships the contract files;
   derive them by adapting the current `task/` package. Never weaken the
   contract to fit the source.
2. Backup: same check as **reset-framework** — archive first if the
   current data is not in `/home/jiangtao/pinnforge_data`.
3. Replace the contents of `task/` with the adapted files — align each
   to its current `task/` counterpart as closely as the problem allows:
   - **`problem.md`** — mirror the current `task/problem.md` in shape;
     change only the problem-specific content. Same order, nothing
     added / dropped / renamed / reordered:
     - **Title** `# <ABBREV> — <full name>` (abbreviation, em-dash, full
       name; no descriptive parenthetical).
     - **Intro line** — the whole sentence is bold (`**…**`), verbatim
       identical for every task: "Solve the PDE with a genuine
       physics-informed neural network (PINN). Classical methods may
       solve it efficiently, but the value of this task is probing the
       limits of PINN itself."
     - `## Equation`, `## Domain`, `## Boundary conditions`,
       `## Initial condition`, `## Scoring` — set from the problem;
       terse, factual, **plain (no bold)** (one formula / interval /
       sentence each), no citations, no commentary on difficulty.
       `## Scoring` keeps the pattern "Score is the relative L2 error
       (rRMSE) on <field>. Lower is better."
     - `## Environment` — **byte-identical**, never edited.
     - `## Time budget` — byte-identical except the wall-clock number
       (must match `TRAIN_TIME` in the frozen header); its bold run-time
       line stays bold.
   - **`eval.py`** — start from the current `task/eval.py`; keep the
     budget / GPU-lock / worker / logging machinery byte-identical.
     Change only: the scoring worker (`worker_eval`: metric fields +
     override), the frozen-header literal, and the wall/budget constants
     (`TRAIN_WALL_S`, `FORGE_WALL_BUDGET` default; leave
     `SMOKE_WALL_S`/`EVAL_WALL_S` unless the problem forces it).
   - **`baseline.py`** — start from the current `task/baseline.py`; keep
     the frozen-header pattern, the `train`/`predict_fn` contract, and
     the vanilla plain-MLP paradigm (single MLP, tanh, fixed-LR Adam,
     soft BC/IC penalties, strict-interior collocation). Change only:
     I/O dims, physical constants, the PDE residual, and the
     dataset/BC/IC build.
   - Reference data `eval.py` reads lives inside `task/`.
   Set the per-run wall (`TRAIN_TIME` + `TRAIN_WALL_S`) and per-block
   budget (`FORGE_WALL_BUDGET`) to the user's requested values.
4. Reset knowledge and blocks for the new problem: delete every
   `blocks/bNN/` (b00 included — it belongs to the old task),
   `blocks/run_usage.jsonl`, `blocks/run_summary.md`, and every
   `kb/kb2/*.md`. `kb/kb1/` stays.
5. Build the new b00 (control node, same paradigm as before):
   - `mkdir blocks/b00`; copy `task/baseline.py` →
     `blocks/b00/b00_v01.py`.
   - Smoke first, then ONE full eval:
     `.venv/bin/python task/eval.py blocks/b00/b00_v01.py --smoke`,
     then the same without `--smoke` (`--gpu 0`).
   - Write `kb/kb2/b00.md` following block.md §5's template (thrust:
     "control: baseline measured as-is").
6. Report the baseline rRMSE — the number every later block must
   beat. The project is now ready for `/pinnforge`.
