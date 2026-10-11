# Using hotpath from AI coding agents

hotpath has two agent-facing pieces. Both call the same CLI; neither contains a second implementation of the toolkit.

## MCP server

```
pip install .            # installs `hotpath` and `hotpath-mcp`
hotpath-mcp              # or: hotpath mcp
```

Transport is local stdio (newline-delimited JSON-RPC 2.0). Run it from the directory where your `hotpath.local.json` and `results/` live; file arguments must stay inside that directory.

Generic client configuration (the shape most MCP clients accept):

```json
{
  "mcpServers": {
    "hotpath": { "command": "hotpath-mcp", "args": [] }
  }
}
```

If the client cannot find the command, use the absolute path to `hotpath-mcp` inside your virtual environment, or `"command": "python", "args": ["-m", "hotpath", "mcp"]` with that environment's Python.

### Tools

| Tool | Kind | CLI equivalent |
|---|---|---|
| `hotpath_tools` | read-only | `hotpath tools` |
| `hotpath_gguf_summary` | read-only | `hotpath gguf` |
| `hotpath_observe` | read-only (no inference) | `hotpath observe` |
| `hotpath_triage` | read-only | `hotpath triage` |
| `hotpath_diagnose` | read-only | `hotpath diagnose` |
| `hotpath_propose` | writes a spec file; launches nothing | `hotpath propose` |
| `hotpath_compare` | read-only | `hotpath compare` |
| `hotpath_workload` | writes a workload file | `hotpath workload` |
| `hotpath_experiment_plan` | read-only dry run | `hotpath experiment --dry-run` |
| `hotpath_list_results` | read-only | lists `results/local` |
| `hotpath_experiment_run` | **active** | `hotpath experiment` |
| `hotpath_measure` | **active** | `hotpath measure` |
| `hotpath_bench` | **active** | `hotpath bench` |
| `hotpath_kernels` | **active** | `hotpath kernels` |
| `hotpath_profile` | **active** | `hotpath profile` |

### Safety model

- Active tools are **off by default**. The operator enables them by starting the server with `HOTPATH_MCP_ALLOW_ACTIVE=1`, and every active call must also pass `confirm=true`. Agents should only do so after the user agrees.
- `measure` can only target servers in `owned_servers` in your local config; the `--i-own-this-server` override is not exposed.
- Experiments launch their own servers on the spec's port and never touch your regular server.
- Paths: experiment specs, workloads and records must be inside the directory the server was started in, or in a directory listed in `HOTPATH_MCP_EXTRA_ROOTS` (separated by `;` on Windows, `:` elsewhere). Models, server logs and `bench --model` can be anywhere. Runtime binaries, model paths and build directories come from `hotpath.local.json` or the spec, not from tool arguments, so keep your models and runtimes wherever you like.
- Each call runs as a child process with a timeout (300 s read-only, 1800 s active by default, per-call `timeout_s`, capped by `HOTPATH_MCP_MAX_TIMEOUT_S`, default 7200). On expiry the whole process tree is killed.
- Outputs are returned to the agent as text, including local paths from tool output. Do not connect this server to a client you do not trust with that information.

### The agent workflow

1. `hotpath_tools` — is the environment supported (GPU, config, external tools)?
2. Evidence: `hotpath_observe` for a running server, or (active, with agreement) `hotpath_bench` and `hotpath_measure` against a server listed in `owned_servers`.
3. `hotpath_diagnose` with those records — ranked findings; read the confidence level and the "cannot establish" list before anything else.
4. `hotpath_propose` with the diagnosis — a one-variable spec and its dry-run plan; show the plan to the user.
5. With agreement: `hotpath_experiment_run` on the spec (active; interleaved rounds, own port).
6. `hotpath_diagnose` with the experiment record — the keep/reject/inconclusive verdict, the verified cache state of each arm, output identity, output validity (and the level it was checked at) and VRAM delta; explain it to the user in those terms, keeping speed, identity, validity and cache comparability separate.

## Agent skill

`skills/hotpath/SKILL.md` follows the [Agent Skills](https://agentskills.io) format (`name`, `description`, instructions in Markdown). It teaches an agent the install steps, the passive-first workflow, when to ask before active commands, and how to read results.

To use it, copy the `skills/hotpath` directory into your agent's skills directory. For Claude Code that is `.claude/skills/hotpath/` in a project or `~/.claude/skills/hotpath/` for all projects.

## What has and has not been validated

| Item | Status |
|---|---|
| MCP server: `initialize`, `tools/list`, `tools/call` over real stdio, active-tool refusal with and without the env var, path confinement, timeout kill of the process tree, a real `bench` call | **exercised** with a scripted JSON-RPC client on Windows |
| `hotpath_diagnose` and `hotpath_propose` over stdio on real measurement and bench records (findings returned, spec written inside the working directory, out-of-directory record refused) | **exercised** with a scripted JSON-RPC client on Windows |
| **Claude Code** (2.1.x, `claude -p --mcp-config … --strict-mcp-config --allowedTools "mcp__hotpath__*"`) against a real observation record: discovered the server, listed the 15 tools and which are active, ran `hotpath_diagnose`, explained the top finding with its confidence level and the measurement it rests on, ran `hotpath_propose` and summarized the arms, stated the two authorizations an active run needs, then attempted `hotpath_experiment_run` with `confirm=true` and relayed the refusal verbatim (server started without `HOTPATH_MCP_ALLOW_ACTIVE`) | **validated** on Windows, 5 turns |
| **OpenCode** (1.18.x, project `opencode.json` with a `local` MCP entry, `opencode run --format json`) with the same task: discovered the server (tools appear as `hotpath_hotpath_*`), listed capabilities and the five active ones, ran `hotpath_diagnose` and `hotpath_propose`, explained the top finding with its confidence and request count, stated the authorization needed, and relayed the refusal of `hotpath_experiment_run` with `confirm=true` | **validated** on Windows |
| Cursor and other clients | **not validated**; the configuration above is the standard stdio shape |

Both validations used only read-only tools plus one deliberate active attempt; no server or benchmark was launched by either agent. Neither agent could reach an active operation by changing arguments (`confirm=true` without the operator flag is refused) or through another tool (no read-only tool launches anything; file arguments are confined to the working directory).

OpenCode project configuration that was used (`opencode.json` in the working directory):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "hotpath": { "type": "local", "command": ["python", "-m", "hotpath", "mcp"], "enabled": true }
  }
}
```

Claude Code one-shot invocation that was used (an interactive session configured with `claude mcp add hotpath -- hotpath-mcp` works the same way):

```
claude -p "<task>" --mcp-config mcp.json --strict-mcp-config --allowedTools "mcp__hotpath__*"
```
| SKILL.md frontmatter against the Agent Skills field rules (name format, description length) | checked with a script |
| SKILL.md loaded by a specific agent (Claude Code, OpenCode, ...) | **not yet validated** |

Reports from real agent setups are welcome.
