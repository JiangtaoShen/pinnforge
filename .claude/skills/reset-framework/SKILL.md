---
name: reset-framework
description: Reset PINNForge to its b00 state so a fresh run can start — archives the current data first if not yet backed up. Trigger phrase: "reset the framework".
---

Reset the project to the initial-node state. Work from the project
root. Destructive — follow the order exactly.

1. Backup check: the newest archive in `/home/jiangtao/pinnforge_data`
   must be newer than every file under `blocks/`, `kb1/` and `task/`.
   If there is no archive or any other file is newer, run the
   **archive-data** skill first.
2. Delete run data, keep the initial node:
   - `blocks/`: every `bNN/` except `b00/` (leave `blocks/kb2/` alone);
     also `run_usage.jsonl` and `run_summary.md`.
   - `blocks/kb2/`: every summary except `b00.md`.
   - `task/` and `kb1/` stay untouched.
3. Verify: the quick-status snippet in CLAUDE.md must show only b00.
   Report what was deleted and the archive that covers it.
