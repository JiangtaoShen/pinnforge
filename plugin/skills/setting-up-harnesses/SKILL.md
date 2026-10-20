---
name: setting-up-harnesses
description: Choose, install, authenticate and prove the coding-agent CLI that drives PINNForge rounds — claude_code, codex, cursor_agent or opencode — with `pinnforge setup` (named bindings pinned to exact model ids) and `pinnforge doctor --ping`. Use whenever a run dies on its first round, doctor reports a missing or failing agent, a model id is rejected, a provider needs configuring (OpenCode with DeepSeek or any OpenAI-compatible endpoint), or the user asks which agent or model to use.
---

# Setting up the agent that runs the rounds

PINNForge does not talk to a model. It starts a coding-agent CLI headlessly,
waits, and reads the filesystem. Every harness gets the **same round**:
identical charter, prompt, working directory and environment.

| runtime | CLI | default model |
|---|---|---|
| `claude_code` | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | none — name one (`provider/id`) |

## `pinnforge setup`

Once per machine. It finds the installed CLIs, asks for an exact model id for
each one worth using, **pings it through the very adapter a round is started
with**, and saves a named binding in `~/.config/pinnforge/agents.yaml`:

```bash
pinnforge setup                                    # the wizard
pinnforge setup agent --name opus --runtime claude_code --model claude-opus-4-8 --default
pinnforge agents bindings                          # list; * marks the default
pinnforge agents default deepseek                  # switch the default
```

In a workspace on a GPU machine the wizard also offers to change which cards
the workspace's runs use; `pinnforge setup gpus 0,1` (or `1`, `all`, `none`)
does only that, and `--user` sets the default for every workspace here (a
workspace whose `pinnforge.yaml` leaves `sandbox.gpus` null, as `init` writes
it unless cards were chosen, follows that default).

Runs use the default binding unless given `--agent <name>` (or `--runtime` and
`--model`). A workspace can pin its own with `agent: <name>` in `pinnforge.yaml`.

## The ping is the point

`--version` proves a binary exists. Credentials, the provider config and the
model id all have to work too, and otherwise fail at the first round — an hour
in, with a measured anchor already paid for.

```bash
pinnforge doctor --ping          # the default binding, end to end
pinnforge agents doctor --ping   # every installed CLI
```

## Pin an exact id, never an alias

`opus` meant `claude-opus-4-8` for a year and `claude-opus-5` after. The id the
harness *actually ran* is read back from its log into the run's ledger, state
and summary; `run status` shows both when they differ.

## OpenCode with a provider it does not ship

Configure the provider in `~/.config/opencode/opencode.json` (outside any
workspace) and keep the key in the environment:

```json
{ "provider": { "deepseek": { "options": {
    "baseURL": "https://api.deepseek.com", "apiKey": "{env:DEEPSEEK_API_KEY}" } } } }
```

```bash
export DEEPSEEK_API_KEY=...
pinnforge setup agent --name deepseek --runtime opencode --model deepseek/deepseek-chat
```

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `runtime 'opencode' states no default model` | no model named | bind one: `setup agent … --model provider/id` |
| OpenCode hangs at start, no output | a killed run left its session DB wedged | remove `~/.local/share/opencode/opencode.db-{wal,shm}` while no opencode runs |
| OpenCode hangs after adding a provider | an `"npm"` field in the provider block | drop `npm`; built-in providers need only `baseURL` + `apiKey` |
| ping: exit 0 but empty | expired auth | re-authenticate that CLI directly |
| doctor: agent CLI not on PATH | not installed | install and sign in, or bind another runtime |

## A harness PINNForge has never heard of

Set the runtime to `module.path:ClassName` — anything implementing the
`AgentRuntime` protocol in `pinnforge/runtime/base.py`.

## Never put instruction files above a workspace's runs

Rounds run in `runs/<id>/`, and several CLIs read `CLAUDE.md`, `AGENTS.md` or
`.cursor/rules/` from every ancestor directory. One at the workspace root (or
above it) is read by every round and silently changes what a round is.
`pinnforge doctor` warns about any it finds.
