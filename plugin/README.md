# PINNForge plugin

Drive PINNForge from your own coding agent: describe a PDE in conversation and
get a ready task; start, watch and diagnose runs. Skills-first, no MCP — the
capability is guidance plus `pinnforge` shell calls. It orchestrates nothing
itself: runs are still driven by the `pinnforge` process, and blocks are
unchanged.

## What is in here

| path | role |
|---|---|
| `skills/pinnforge-quickstart/` | what it is, install, workspace, demo, the fast path to a run |
| `skills/creating-a-pinnforge-task/` | `task new` from a described PDE, references, the stages, anchors |
| `skills/running-pinnforge-blocks/` | start, watch (dashboard), status, stop, resume, audit, GPUs |
| `skills/setting-up-harnesses/` | `pinnforge setup`, bindings, pings, providers |
| `agents/pinnforge-task-author.md` | subagent: conversation → intake → `task new --detach` → ready; never starts a run |
| `agents/pinnforge-run-doctor.md` | subagent: read-only diagnosis of a stuck or failing run |
| `hooks/session-start.py` | finds the CLI (PATH or workspace `.venv`), says which skill/subagent to use |
| `cursor/pinnforge.mdc` | the same workflow as a Cursor rule |
| `opencode/pinnforge.md` | the same workflow as an OpenCode subagent |
| `AGENTS.md` | the same workflow for any CLI that reads `AGENTS.md` |

## Install

**No install:** `pinnforge chat` opens Claude Code with this plugin loaded for
that session (`claude --plugin-dir`), from the copy inside the installed package.

**Claude Code:**

```
/plugin marketplace add JiangtaoShen/pinnforge
/plugin install pinnforge@pinnforge-marketplace
```

**Codex, Cursor and OpenCode — per project, never globally.** Install into the
project directory you drive PINNForge *from*: your own project, which is
neither a PINNForge workspace nor a directory above one. From a checkout
(`plugin/`; `pinnforge chat` prints the installed package's own copy):

```bash
# Codex: the instructions, and the skills
cp plugin/AGENTS.md . && mkdir -p .agents/skills && cp -r plugin/skills/. .agents/skills/
# Cursor: the rule
mkdir -p .cursor/rules && cp plugin/cursor/pinnforge.mdc .cursor/rules/
# OpenCode: the subagent
mkdir -p .opencode/agent && cp plugin/opencode/pinnforge.md .opencode/agent/
```

Why not globally (`codex plugin add`, `~/.config/opencode/agent/`, Cursor user
rules): blocks are started by the very same CLI, with a workspace's
`runs/<id>/` as their working directory. A user-level install is loaded by
every process that CLI starts, blocks included, and these harnesses also read
configuration from parent directories — so the operator's instructions would
reach every block, and a block on one harness would stop being the same block
as on another. Only Claude Code blocks are started with the plugin switched
off (`--settings`), which is why the Claude Code install above may be global.

## The one placement rule

**Never copy `AGENTS.md`, `pinnforge.mdc` or any other instruction file into a
PINNForge workspace or any directory above it.** Blocks run with `runs/<id>/`
as their working directory, and several CLIs read instruction files from every
ancestor directory, so every block would read it — and a block on one harness
would stop being the same block as on another, the one property the framework
is built to preserve. This was measured, not assumed: an `AGENTS.md` at a
project root was read by an OpenCode agent working in `runs/ldc_9/`, which
quoted a sentinel token out of it. `pinnforge doctor` warns about any it finds.

The framework guards the other direction too. Every agent it starts carries
`PINNFORGE_RUN` (blocks) or `PINNFORGE_ROLE` (author, reference and ping
agents), and the session hook stays silent when it sees either; Claude Code
blocks are additionally started with this plugin disabled.
