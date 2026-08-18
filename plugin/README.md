# PINNForge plugin

A skills-first plugin for authoring and running PINNForge tasks **from your own
harness**. It teaches the `pinnforge` CLI workflow and checks the CLI is
installed at session start. It does not orchestrate anything: runs are still
driven by the `pinnforge` process, and blocks are unchanged.

## What is in here

| Path | Role |
|---|---|
| `skills/pinnforge-quickstart/` | what it is, install, the fast path to a launched run |
| `skills/running-pinnforge-blocks/` | start, status, log, show, resume, audit, stop |
| `skills/creating-a-pinnforge-task/` | the three contract files and the three gates |
| `skills/setting-up-harnesses/` | picking, authenticating and proving a coding-agent CLI |
| `hooks/session-start.py` | detects the CLI, injects which-skill-to-reach-for context |
| `cursor/pinnforge.mdc` | the same workflow as a Cursor rule |
| `opencode/pinnforge.md` | the same workflow as an OpenCode subagent |
| `AGENTS.md` | the same workflow for any CLI that reads `AGENTS.md` |

## Install

**Claude Code** — add the marketplace at the repo root, then install:

```
/plugin marketplace add JiangtaoShen/pinnforge
/plugin install pinnforge@pinnforge-marketplace
```

**Codex** — the same layout, driven by `.codex-plugin/plugin.json` and
`hooks/hooks-codex.json`.

**Cursor** — copy the rule into the project you drive PINNForge from:

```bash
mkdir -p .cursor/rules && cp plugin/cursor/pinnforge.mdc .cursor/rules/
```

**OpenCode** — install the subagent globally:

```bash
mkdir -p ~/.config/opencode/agent && cp plugin/opencode/pinnforge.md ~/.config/opencode/agent/
```

## The one placement rule

**Never copy `AGENTS.md`, `pinnforge.mdc` or any other instruction file to the
root of a PINNForge checkout.**

Blocks are started with `runs/<id>/` as their working directory, and several
CLIs walk up the directory tree looking for instruction files. A file at the
project root is an ancestor of every run directory, so every block would read
it. That would make a block on one harness different from the same block on
another, which is the single property the framework is built to preserve, and
which `tests/test_charter.py` exists to pin.

This is measured, not assumed: an `AGENTS.md` placed at a project root was read
by an OpenCode agent whose working directory was `runs/ldc_9/`, and the agent
quoted a sentinel token out of it.

The SessionStart hook carries the same guard from the other side. The
orchestrator sets `PINNFORGE_RUN` for every block process, and the hook exits
without injecting anything when it sees that variable, so a block never receives
operator context even if the plugin is installed on the harness driving it.
