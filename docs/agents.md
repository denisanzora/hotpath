# Using hotpath from AI coding agents

hotpath has two agent-facing pieces. Both call the same CLI; neither contains a second implementation of the toolkit.

## MCP server

```
pip install .            # installs `hotpath` and `hotpath-mcp`
hotpath-mcp              # or: hotpath mcp
```

Transport is local stdio (newline-delimited JSON-RPC 2.0). Run it from the directory where your `hotpath.local.json` and `results/` live (`hotpath init` there first); file arguments must stay inside that directory.

### Client configuration

Read-only (the default; no GPU work can be started):

```json
{ "mcpServers": { "hotpath": { "command": "hotpath-mcp", "args": [] } } }
```

To let an agent run experiments, benchmarks and profiles after you agree to each one, the operator starts the server with `HOTPATH_MCP_ALLOW_ACTIVE=1`. An agent cannot turn this on itself.

- **Claude Code**: `claude mcp add hotpath -e HOTPATH_MCP_ALLOW_ACTIVE=1 -- hotpath-mcp`, or in `.mcp.json`:
  ```json
  { "mcpServers": { "hotpath": { "command": "hotpath-mcp", "args": [], "env": { "HOTPATH_MCP_ALLOW_ACTIVE": "1" } } } }
  ```
- **OpenCode 2.x** (`opencode.json` in the working directory; V2 puts servers under `mcp.servers`):
  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "mcp": { "servers": { "hotpath": { "type": "local", "command": ["hotpath-mcp"], "environment": { "HOTPATH_MCP_ALLOW_ACTIVE": "1" } } } }
  }
  ```

If the client cannot find the command, use the absolute path to `hotpath-mcp` inside your virtual environment, or `python -m hotpath mcp` with that environment's Python. Start a new session after changing the configuration.

### Tools

| Tool | Kind | CLI equivalent |
|---|---|---|
| `hotpath_inspect` | read-only; **start here** | `hotpath inspect` (discover + observe + diagnose) |
| `hotpath_discover` | read-only | `hotpath discover` |
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
| `hotpath_job_status` | read-only | state, progress and result of an active job |
| `hotpath_job_cancel` | stops one of hotpath's own jobs | - |
| `hotpath_improve_plan` | read-only dry run (writes the first spec) | `hotpath improve --dry-run` |
| `hotpath_improve` | **active** (job): the whole loop | `hotpath improve` |
| `hotpath_experiment_run` | **active** (job) | `hotpath experiment` |
| `hotpath_measure` | **active** (job) | `hotpath measure` |
| `hotpath_bench` | **active** (job) | `hotpath bench` |
| `hotpath_kernels` | **active** (job) | `hotpath kernels` |
| `hotpath_profile` | **active** (job) | `hotpath profile` |

### Active tools run as jobs

An experiment takes minutes to hours, and clients limit how long one tool call may take (public issue reports describe the Claude Code desktop app cancelling stdio calls after about 60 s and older OpenCode versions cutting tool steps at 120 s; not verified here). So an active call starts a job and returns within `wait_for_result_s` (default 45 s, at most 55): with the full result if the job finished, otherwise with its job id, progress (`launch 4 of 6, 36 of 54 requests done`) and latest output. `hotpath_job_status(job_id, wait_s=45)` waits up to 55 s per call and returns the full result once the job has finished; `hotpath_job_cancel(job_id)` stops the job's whole process tree, including the `llama-server` processes it launched. One active job runs at a time (they share the GPU). Jobs belong to the MCP server process: when the client exits, running jobs are stopped. Their output files stay in `results/local/jobs/`.

### Safety model

- Active tools are **off by default**. The operator enables them by starting the server with `HOTPATH_MCP_ALLOW_ACTIVE=1`, and every active call must also pass `confirm=true`. Agents should only do so after the user agrees. A refused call says how the user enables them in Claude Code and OpenCode.
- `measure` can only target servers in `owned_servers` in your local config; the `--i-own-this-server` override is not exposed. `init` never adds your running server to `owned_servers`.
- Experiments launch their own servers on the spec's port and never touch your regular server. If your server holds the VRAM an experiment needs, the experiment fails its preconditions and names the server; hotpath never stops, restarts or reconfigures a process it did not start. Discovery recognizes hotpath's own experiment servers and never reports them as yours.
- Paths: experiment specs, workloads and records must be inside the directory the server was started in, or in a directory listed in `HOTPATH_MCP_EXTRA_ROOTS` (separated by `;` on Windows, `:` elsewhere). Models, server logs, `bench --model` and discovery `roots` can be anywhere (read-only; drive roots are refused). Binaries found under discovery `roots` passed by an agent are inspected statically but not executed. Runtime binaries, model paths and build directories for launches come from `hotpath.local.json`, not from tool arguments.
- Spec contents are checked, not only spec paths. Because an agent can write a spec file, `hotpath_experiment_run` refuses a spec whose `binary`, `model` or `path_prepend` (top level or per arm) is not a `${key}` reference into your `hotpath.local.json`, whose environment overrides are not `GGML_*`/`LLAMA_*`/`CUDA_*` variables, whose `workload` or `slot_save_path` is outside the allowed directories, or that carries a server argument which writes files (`--log-file`, `--slot-save-path`, `--prompt-cache`), changes the network binding or credentials (`--host`, `--port`, `--api-key`, `--ssl-*`), downloads a model (`--hf-repo`, `--model-url`, ...) or changes which model or adapter loads (`--model`/`-m`, `--lora`, `--mmproj`, ...), in `--flag value` or `--flag=value` form. Other `llama-server` arguments in a spec are passed through as written, so review a spec before confirming a run. This check runs only after the two authorization gates (operator flag, `confirm=true`) and does not replace them.
- Writes: `hotpath_propose` writes a new spec file inside the working directory and never replaces an existing file; `hotpath_workload` writes (and regenerates) a workload file inside the working directory; neither accepts a `*.local.json` name, so the local configuration cannot be overwritten. Result records are written to `results/local/`. `hotpath_observe` and `hotpath_inspect` only contact loopback servers.
- Calls with unknown or missing arguments are answered with the tool's argument list. Every child process gets a closed stdin (it can never read the JSON-RPC pipe) and a timeout (300 s read-only, 7200 s for jobs by default, capped by `HOTPATH_MCP_MAX_TIMEOUT_S`); on expiry the whole process tree is killed.
- Outputs are returned to the agent as text, including local paths from tool output. Do not connect this server to a client you do not trust with that information.

### The agent workflow

When the user asks for their setup to be improved: `hotpath_improve_plan` (passive) shows the starting point, the first experiment, the free VRAM it needs and a rough duration; after the user agrees, `hotpath_improve` with `confirm=true` runs the loop as one job (poll `hotpath_job_status`). It adopts only `keep` verdicts, does not run cache-flag experiments without real-traffic reuse evidence, confirms two or more adopted changes in one paired original-vs-improved run, and ends with the launch line to apply. `args` (starting arguments when no server runs) refuses the same runner-owned and file-writing flags as agent-authored specs; `resume` (a session name) continues an interrupted run without repeating its completed experiments. The final answer to the user should separate what was measured and validated, what needs their decision, and that nothing was applied.

Step by step:

1. `hotpath_inspect`: the setup, its problems, and a ranked diagnosis of the running server with the next step. Read the confidence levels, the `known` / `hypothesis` split and "Not established by this evidence" before anything else.
2. Optionally, with agreement: `hotpath_measure` against a server listed in `owned_servers` to add client wall time, or `hotpath_bench`.
3. `hotpath_propose` with the diagnosis: a one-variable spec whose baseline is the user's real launch, and its plan (launches, requests, free VRAM needed, rough duration). Show the plan to the user.
4. If the user's own server holds the GPU, ask the user to stop it; do not look for a way to stop it.
5. With agreement: `hotpath_experiment_run` on the spec, then `hotpath_job_status` until it reports finished.
6. `hotpath_diagnose` with the experiment record: the keep/reject/inconclusive (or "faster, but outputs differ") verdict, the verified cache state of each arm, output identity, output validity and VRAM delta; then the findings that remain on the faster configuration, from which the next `hotpath_propose` builds.

## Agent skill

`skills/hotpath/SKILL.md` follows the [Agent Skills](https://agentskills.io) format (`name`, `description`, instructions in Markdown). It teaches an agent the install steps, the passive-first workflow, when to ask before active commands, how jobs work, and how to read results.

To use it, copy the `skills/hotpath` directory into your agent's skills directory: `.claude/skills/hotpath/` (Claude Code, per project) or `~/.claude/skills/hotpath/` (all projects); `.opencode/skills/hotpath/` for OpenCode.

## What has and has not been validated

All of this was run on one Windows 11 machine (RTX 2080 SUPER) with an editable install in a fresh virtual environment, a working directory holding `results/local/` and a `hotpath.local.json` written by `hotpath init`, and a `llama-server` the tester started by hand with a partial GPU offload (`-ngl 56` of 65 layers), playing the user's own server.

| Item | Status |
|---|---|
| Scripted JSON-RPC client over real stdio, 0.3.0.dev0: `tools/list` (19 tools, 5 active); `hotpath_inspect` with the user's server running (13 s); active tool refused without the operator flag with the enabling instructions; a drive root refused as a discovery root; with the flag: a `hotpath_bench` job returned a job id, a second active job was refused while it ran, `hotpath_job_status` returned the finished result; `hotpath_job_cancel` of a running experiment and closing the client's stdin with an experiment running both left no `llama-server` behind | **exercised** |
| Same client, 0.2.0 behavior found and fixed: MCP children inherited the JSON-RPC stdin, and PowerShell (used for process and GPU counters on Windows) waited on it, so `hotpath_observe`/`hotpath_inspect` against a running server hung until the timeout | **fixed** (every child gets a closed stdin) |
| **Claude Code** 2.1.296 (`claude -p ... --mcp-config mcp.json --strict-mcp-config --allowedTools "mcp__hotpath__*" "Skill"`, operator flag set), task "Inspect my local inference setup, find the bottlenecks, test promising optimizations, and show me what improved. You have my permission to run GPU experiments": loaded the skill; `hotpath_inspect` → `hotpath_propose` (baseline `-ngl 56`, candidate 65); stopped before launching because the user's server held the VRAM and asked the user to stop it (did not try to). After the user stopped it (`--continue`): `hotpath_experiment_run` as a job, 12 `hotpath_job_status` polls over 586 s, `hotpath_diagnose`; reported wall 11.3 → 3.3 s, decode 6.1×, +796 MiB, no spill, and the greedy divergence on one prompt as unexplained and to be checked before switching. No `llama-server` left running | **validated**, first agent-run experiment |
| **OpenCode** 2.0.26 (project `opencode.json` as above; permissions deny everything except `execute`, `hotpath_*` and the hotpath skill; Code Mode), same task: same phase-1 behavior through Code Mode. A `--continue` of that session in `--standalone` mode started without the project's MCP tools (the agent said so and ran nothing); a fresh session naming the spec ran the experiment as a job (progress lines, five polls inside one `execute`), diagnosed it, and reported the same numbers (wall 11.8 → 3.3 s, decode 7.7 → 48.1 t/s, the same divergence at token 51) and the decode profile as the next step | **validated** (fresh session) |
| Agent findings that changed hotpath: an argument-name guess got a bare KeyError (now: the tool's argument list); job polls showed no progress (now: launch k/N and request counts); after the experiment the agent read "decode at 9% of the ceiling" from the replaced configuration (now: diagnosis continues on the faster arm); a CUDA-graphs toggle was proposed although graphs were on (now: only when the launch disables them); both agents asked how long the run would take (now: a rough estimate in the plan) | **fixed** |
| **Claude Code** 2.1.292 with `hotpath_improve` (0.3.0.dev0; same machine, a fresh working directory, operator flag set, only `mcp__hotpath__*` and the skill allowed), task "Inspect my local inference setup, find the bottlenecks, test promising optimizations, and show me what improved. You have my permission to run GPU experiments; keep it to at most 2 experiments", with the user's server running at `-c 49152 -ngl 40 -ub 128`: `hotpath_inspect`, then it asked the user to stop the server (did not try to). After `--continue` "I stopped my server. Go ahead.": `hotpath_improve_plan` (shown to the user: ~20 min, ~7.4 GB free VRAM), `hotpath_improve` with `confirm=true` and `max_experiments: 2`, 45 `hotpath_job_status` polls over 44 min. The loop started from the last observation of the stopped server; both candidates were faster but changed greedy outputs on this workload (8-bit KV + 57 layers on 2/3 prompts, `-ub 256/512` on 1/3), so nothing was adopted, and the agent reported it that way. No `llama-server` left running | **validated** |
| Found in that session and fixed: `hotpath_inspect` named only `propose` as the next step, so the agent learned about `improve` from the skill alone (inspect now names `improve --dry-run` / `improve`); the agent suggested a wrong flag (`--cache-type q8_0`) for a non-adopted winner because the report gave no exact change for it (the report now has "Needs your decision" with each such arm's exact change and numbers) | **fixed** |
| **Agent Skill** in Claude Code and OpenCode: loaded by name in every session above | **validated** |
| 0.2.0 sessions (Claude Code 2.1.292, OpenCode 1.18.35, read-only and refusals) | validated for 0.2.0; see the 0.2.0 tag |
| Cursor and other clients; MCP clients on Linux/macOS | **not validated**; the configuration above is the standard stdio shape |

Reports from real agent setups are welcome.
