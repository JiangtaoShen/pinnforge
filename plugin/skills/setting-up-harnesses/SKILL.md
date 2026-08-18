---
name: setting-up-harnesses
description: Pick, install and authenticate the coding-agent CLI that drives PINNForge blocks — claude_code, codex, cursor_agent or opencode — and prove the binding works with `pinnforge agents doctor --ping` before spending a run on it. Use whenever a run dies on its first block, `agents doctor` reports MISS or FAIL, a model id is rejected, a provider needs configuring (OpenCode with DeepSeek or any OpenAI-compatible endpoint), or the user asks which harness to use or how to point PINNForge at a different model.
---

# Setting up a harness

PINNForge does not talk to a model. It starts a coding-agent CLI as a
subprocess, waits, and reads the filesystem. Every adapter delivers the **same
block**: identical charter, prompt, working directory and environment. Only the
argv that selects headless mode differs.

| `agent.runtime` | CLI | default model |
|---|---|---|
| `claude_code` (default) | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | none; pass `--model` |

```bash
pinnforge agents list                                     # what is installed
pinnforge agents doctor --ping --model <id>               # does the binding answer
```

## The ping is the point

`--version` proves a binary exists, which is not the question a run needs
answered. Credentials, provider config and the model id all have to work too,
and each of them otherwise fails at the first block: an hour in, with a run
directory and a measured anchor already paid for. A good result looks like:

```
[OK]   opencode       1.18.18
[PING] opencode       deepseek/deepseek-chat — 2.6s, replied 'ok'
```

## Pin an exact id, never an alias

`opus` meant `claude-opus-4-8` for a year and `claude-opus-5` after. A run
pinned to the alias is not pinned. Whatever was asked for, the id the harness
*actually ran* is read back out of its log into `run_usage.jsonl`, `state.json`
and the summary; `run status` shows both when they differ.

## OpenCode with a provider it does not ship

OpenCode fronts many providers, which is why this repo gives it no default
model: any id picked here would be a guess, and a guess that runs is worse than
one that does not, because the ledger records it as though it were chosen.

Configure the provider in `~/.config/opencode/opencode.json` (outside any
PINNForge project) and keep the key in the environment:

```json
{
  "provider": {
    "deepseek": {
      "options": {
        "baseURL": "https://api.deepseek.com",
        "apiKey": "{env:DEEPSEEK_API_KEY}"
      }
    }
  }
}
```

```bash
export DEEPSEEK_API_KEY=...
pinnforge agents doctor --ping --model deepseek/deepseek-chat
pinnforge run start -t ldc --runtime opencode --model deepseek/deepseek-chat -n 2
```

## Troubleshooting matrix

| symptom | cause | fix |
|---|---|---|
| `runtime 'opencode' states no default model` | no `--model` | pass `--model provider/id`; the refusal is deliberate |
| OpenCode hangs at `init`, no output, no exit | a killed run left its session DB wedged | remove `~/.local/share/opencode/opencode.db-{wal,shm}` while no opencode is running |
| OpenCode hangs after adding a provider | an `"npm"` field in the provider block makes it fetch a package | drop `npm`; built-in providers need only `baseURL` + `apiKey` |
| `opencode run --format json` produces nothing | broken in 1.18.x | do not use it; the adapter parses the `--print-logs` stream instead |
| `[MISS] <runtime> ... not on PATH` | CLI not installed | install it, or pick another `--runtime` |
| ping returns exit 0 but empty output | auth expired | re-authenticate that CLI directly |

## A harness this repo has never heard of

Point `agent.runtime` at `module.path:ClassName`. Anything implementing the
`AgentRuntime` protocol in `pinnforge/runtime/base.py` works: `start`,
`extract_session_id`, `extract_usage`, `default_model`, `default_command`.

## Do not put instruction files at a PINNForge project root

Blocks run with `runs/<id>/` as their working directory, and some CLIs walk up
the tree for `AGENTS.md` or `.cursor/rules/`. A file like that at the project
root is an ancestor of every run directory, so it is read by every block and
silently changes what a block is. Keep operator-facing instructions inside
`plugin/`, or in the project you are driving PINNForge *from*, never at the root
of the PINNForge checkout itself.
