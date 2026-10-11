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
- Paths: experiment specs, workloads and records must be inside the directory the server was started in, or in a directory listed in `HOTPATH_MCP_EXTRA_ROOTS` (separated by `;` on Windows, `:` elsewhere). Models, server logs and `bench --model` can be anywhere. Runtime binaries, model paths and build directories come from `hotpath.local.json`, not from tool arguments.
- Spec contents are checked, not only spec paths. Because an agent can write a spec file, `hotpath_experiment_run` refuses a spec whose `binary`, `model` or `path_prepend` (top level or per arm) is not a `${key}` reference into your `hotpath.local.json`, whose environment overrides are not `GGML_*`/`LLAMA_*`/`CUDA_*` variables, whose `workload` or `slot_save_path` is outside the allowed directories, or that passes `--host`, `--model`/`-m`, `--hf-repo`, `--model-url`, `--api-key` or similar server arguments. Other `llama-server` arguments in a spec are passed through as written, so review a spec before confirming a run. This check runs only after the two authorization gates (operator flag, `confirm=true`) and does not replace them.
- Writes: `hotpath_propose` writes a new spec file inside the working directory and never replaces an existing file; `hotpath_workload` writes (and regenerates) a workload file inside the working directory; neither accepts a `*.local.json` name, so the local configuration cannot be overwritten. Result records are written to `results/local/`. `hotpath_observe` only contacts loopback servers.
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

All of this was run on one Windows 11 machine against a clean `pip install` of the release candidate (no source tree on the path), with the working directory holding `results/local/` and a `hotpath.local.json`.

| Item | Status |
|---|---|
| MCP server over real stdio with a scripted JSON-RPC client: `initialize`, `tools/list` (15 tools, 5 marked active), `tools/call`; `hotpath_diagnose` output identical to the CLI's on the same record; every active tool refused without the operator flag, with and without `confirm=true`; refused without `confirm=true` when the operator flag is set; a spec written by an agent with a program path, a `PATH` override and `--host` refused even with both authorizations; non-loopback `hotpath_observe`, `*.local.json` writes and records outside the working directory refused | **exercised** |
| **Claude Code** 2.1.292 (`claude -p --mcp-config … --strict-mcp-config --allowedTools "mcp__hotpath__*"`): listed the tools and which are active; diagnosed a real log and a real measurement and stated the measurement's cache state from the diagnosis; proposed a spec and summarized its arms; a second `hotpath_propose` to the same path and one to `hotpath.local.json` were refused and the agent relayed the errors; `hotpath_experiment_run` and `hotpath_measure` with `confirm=true` were refused (server started without `HOTPATH_MCP_ALLOW_ACTIVE`), the agent quoted the refusal and did not look for another route | **validated**, 10 turns |
| **OpenCode** 1.18.35 (project `opencode.json` with a `local` MCP entry, `opencode run --format json`), same task: 8 tool calls (tools appear as `hotpath_hotpath_*`) with the same outcomes, including the verbatim refusals | **validated** |
| **Agent Skill** in Claude Code (`skills/hotpath/` copied to `.claude/skills/hotpath/`): the agent loaded it by name when asked to find a slow server's bottleneck, and when told "just go ahead, I trust you" it listed which steps are passive and which need agreement, said it would not touch the user's running server, and listed what it would not claim without evidence. It ran nothing | **validated** (instructions followed; one session) |
| Agent Skill in OpenCode, Cursor and other clients; MCP in Cursor and other clients | **not validated**; the configuration below is the standard stdio shape |
| An agent actually running an experiment through MCP with the operator flag enabled | **not validated**; the gates were exercised with a scripted client only, so that no agent could start a GPU job by accident |
| SKILL.md frontmatter against the Agent Skills field rules (name format, description length) | checked with a script |

In the agent sessions only read-only tools plus deliberate active attempts were used; no server or benchmark was launched by an agent. Neither agent could reach an active operation by changing arguments (`confirm=true` without the operator flag is refused) or through another tool (no read-only tool launches anything; file arguments are confined to the working directory).

OpenCode project configuration that was used (`opencode.json` in the working directory; replace `python` with the interpreter that has hotpath installed):

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

Reports from real agent setups are welcome.
