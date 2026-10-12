---
name: hotpath
description: Find why local llama.cpp-family GPU inference is slow and test fixes with the hotpath CLI or MCP server. Use when the user wants their local inference setup inspected (running llama-server, model, GPU, launch settings), bottlenecks found from a llama-server log, /metrics or measurements, ranked findings with confidence levels, controlled experiments run (GPU layers, microbatch, CUDA graphs, cache flags, builds, models, request shape), or results verified for cache state and output validity. Not for training, non-llama.cpp runtimes, or non-NVIDIA GPUs.
license: MIT
compatibility: Requires Python 3.10+ and a llama.cpp-family llama-server; GPU features need NVIDIA drivers (nvidia-smi). Validated on Windows 11 with one NVIDIA GPU only.
metadata:
  repository: https://github.com/denisanzora/hotpath
---

# hotpath

hotpath is a CLI (and MCP server) that discovers a local llama.cpp setup, diagnoses it from passive evidence, and wraps `llama-server`, `llama-bench`, `test-backend-ops` and Nsight in a controlled-experiment workflow. Use it; do not re-implement its measurements with ad-hoc scripts.

## Ground rules

1. **Passive first.** `hotpath inspect`, `discover`, `tools`, `gguf`, `observe`, `triage`, `diagnose`, `propose`, `improve --dry-run` and `experiment --dry-run` submit no inference and launch no server (`discover`/`inspect` run `llama-server --version` on installed binaries; `propose` writes a new spec file). Run these before anything active.
2. **Ask before active commands.** `improve` and `experiment` (without `--dry-run`), `measure`, `bench`, `profile` and `kernels` launch servers or benchmarks and occupy the GPU for minutes to hours. Tell the user what will run, roughly how long (the plan prints launches and requests), and that the GPU must be free, and get agreement. Never retry a refused active call with different arguments or look for another route: a refusal means the operator has not enabled it.
3. **Never touch the user's own server.** hotpath never stops, restarts or reconfigures a server it did not start; experiments launch their own server on the spec's port (8099). On a single GPU the user's server usually holds the VRAM an experiment needs: the plan and the precondition error name the running server; ask the user to stop it themselves (you must not), or use `--wait-s`. `measure` sends real requests and clears a prompt-cache slot, so it only targets servers in `owned_servers`; pass `--i-own-this-server` only when the user says the server is theirs to load-test.
4. **Do not publish results.** Records are local (`results/local/`). Server logs and `.nsys-rep` captures contain raw paths; workloads built from captured requests contain the user's prompts.
5. **Do not overclaim, and do not invent.** Report each finding with its confidence (`confirmed`, `likely`, `hypothesis`, `insufficient`), what is `known` versus the `hypothesis`, and its `measured` status. The `impact` bound is the share of request time a change can act on, not a predicted gain. A speedup exists only as an experiment verdict; report medians, the workload size and the verdict's cache-state line.
6. **Specs come from `propose`.** Do not hand-write specs with paths, programs or environment settings; machine-specific values belong in `hotpath.local.json` as `${key}` references. MCP-launched experiments refuse specs that do otherwise.

## Setup

```
pip install git+https://github.com/denisanzora/hotpath     # or: pip install . from a clone; add [schema] for full JSON Schema validation
hotpath init            # discovers the running llama-server (binary, model, launch args, log), installs, models, CUDA DLL fixes; writes hotpath.local.json
hotpath init --root DIR # also search DIR (bounded) when the runtime or models live elsewhere; --binary/--model choose among several
```

`init` never guesses between several candidates; it lists them. It never adds the user's server to `owned_servers`.

## Workflow

**"Inspect my setup, find the bottlenecks, test what helps":** `hotpath improve --dry-run` (MCP: `hotpath_improve_plan`) is passive and shows the first experiment, its free-VRAM need and duration. Show it, get agreement, then `hotpath improve` (MCP: `hotpath_improve` with `confirm=true`, then poll `hotpath_job_status`). It runs one-variable experiments on hotpath's own server, adopts only `keep` verdicts, confirms several adopted changes in one paired original-vs-improved run, and reports the launch line to apply. Report its result line (only a validated original-vs-improved run supports a recommendation), each step's kept/not-kept reason, the "Needs your decision" items with their trade-off, and the apply and revert lines; the user applies them to their own server, so say "recommended", never "applied". Output classes: `token-identical` is the only automatic pass; `diverged-near-tie` means a numerically close decision went the other way, which is not evidence the different output is acceptable; never call a top-k KL a full-distribution comparison. A smaller context is always the user's decision unless they set `max_request_tokens`. Never present a change, subset or combination as safe or as identical unless the report shows a validation run of exactly that set; a step result on three prompts is not a validation. When the report lists sets as "not validated", say there is no evidence either way and offer the `resume` command it gives. If a job is interrupted, continue it with `resume` (the session name from the report) instead of starting over. If it stops on preconditions because the user's server holds the GPU, ask the user to stop that server, then run it again: it starts from the server's last observation. Budget with `max_experiments` / `max_minutes`.

The same loop step by step:

1. `hotpath inspect` (MCP: `hotpath_inspect`): discovery + passive observation of the running server (command line, log, `/metrics`, GPU) + diagnosis, with the exact next step. Say which findings are observed and which inferred; read "Not established by this evidence".
2. If wall time or per-request overhead matter and the server is the user's to load: `hotpath measure --server URL --workload W --i-own-this-server` (active, ask first), then `hotpath diagnose` with the observation and the measurement.
3. `hotpath propose results/local/diagnosis-*.json --out experiments/next.json` (add `--finding ID` and `--workload W` as needed): a one-variable spec whose baseline is the user's real launch (observed command line, or `server_args`), and its plan. Show the plan, the number of launches and requests, and the VRAM precondition.
4. After agreement and with the GPU free: `hotpath experiment experiments/next.json` (MCP: `hotpath_experiment_run`, then poll `hotpath_job_status`).
5. `hotpath diagnose results/local/experiment-<name>-*.json`: per-arm keep / reject / inconclusive from paired per-prompt ratios, cache state, VRAM delta, output identity and validity. Report it as a result for this machine and workload. Read the cache-state line first; diverging outputs need explanation before recommending a change; a validity drop is a rejection whatever the speed.
6. For request shape (schemas, samplers): `hotpath workload workloads/real.json --from-requests captured.jsonl`, then propose/experiment with that workload. For kernels: `hotpath kernels test` before `perf`; `hotpath profile --pp 0 --tg 64` for decode kernel shares.

## MCP

`hotpath mcp` (or `hotpath-mcp`) serves the same commands over stdio. Start with `hotpath_inspect`. Active tools are disabled unless the operator started the server with `HOTPATH_MCP_ALLOW_ACTIVE=1`; each active call also needs `confirm=true`. Active tools run as jobs: the call returns the result when it finishes within `wait_for_result_s` (default 45 s), otherwise a job id; poll `hotpath_job_status(job_id, wait_s=45)` until it reports finished, and stop it with `hotpath_job_cancel` if the user asks. Jobs end when the MCP server exits. File arguments stay inside the working directory; read-only tools only contact loopback servers; `*.local.json` cannot be written. See `docs/agents.md` for the validation status per client.

## Limits

llama.cpp-family servers only; NVIDIA only. Settings given through `LLAMA_ARG_*` environment variables are not visible in a command line. hotpath does not measure logit divergence or perplexity (use `llama-perplexity`) and does not check what the user's application decides from an output. If asked for something outside this, say so.
