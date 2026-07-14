---
name: archive-data
description: Archive a finished /pinnforge run — pack blocks/ + kb/ + task/ + run_summary.md into /home/jiangtao/pinnforge_data. Trigger phrase: "archive the run data".
---

Archive the current PINNForge run. Work from the project root.

1. Refresh `blocks/run_summary.md` with the generator script in
   `.claude/commands/pinnforge.md` (After the last block).
2. Destination: `pinnforge_results_$(date +%F).tar.gz` under
   `/home/jiangtao/pinnforge_data/`; on collision append `_run2`,
   `_run3`, …
3. Pack — 3 folders + 1 file, the summary also at archive root:

   ```bash
   cp blocks/run_summary.md run_summary.md
   tar czf /home/jiangtao/pinnforge_data/<name> \
       --exclude='__pycache__' blocks kb task run_summary.md
   rm run_summary.md
   ```

4. Verify (`tar -tzf … | head`, file size) and report the path.
