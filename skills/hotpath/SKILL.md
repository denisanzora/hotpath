---
name: hotpath
description: Diagnose why local llama.cpp-family GPU inference is slow and test fixes with the hotpath CLI or MCP server. Use when the user wants to find inference bottlenecks from a llama-server log or measurements, get ranked findings with confidence levels, generate and run controlled experiments (runtime flags, builds, models, request shape), verify cache state and output validity, or inspect a llama-server deployment. Not for training, non-llama.cpp runtimes, or non-NVIDIA GPUs.
license: MIT
compatibility: Requires Python 3.10+ and a llama.cpp-family llama-server; GPU features need NVIDIA drivers (nvidia-smi). Validated on Windows 11 with one NVIDIA GPU only.
metadata:
  repository: https://github.com/denisanzora/hotpath
---

# hotpath

hotpath is a CLI (and optional MCP server) that wraps `llama-server`, `llama-bench`, `test-backend-ops` and Nsight with a controlled-experiment workflow. Use it; do not re-implement its measurements with ad-hoc scripts.

## Ground rules

1. **Passive first.** `hotpath tools`, `gguf`, `observe`, `triage`, `diagnose`, `propose` and `experiment --dry-run` do not submit inference or launch servers (`propose` writes a spec file and never overwrites an existing one). Run these before anything active.
2. **Ask before active commands.** `experiment` (without `--dry-run`), `measure`, `bench`, `profile` and `kernels` launch servers or benchmarks and occupy the GPU for minutes to hours. Tell the user what will run, how long it may take, and that the GPU must be otherwise idle, and get agreement first. Never retry a refused active call with different arguments, and never look for another tool to do the same thing: a refusal means the operator has not enabled it.
3. **Never touch the user's day-to-day server.** Experiments launch their own server on the spec's port. `measure` sends real requests to the server you name and clears its prompt-cache slot, so it only targets servers listed in `owned_servers`; do not use it on a server that is serving live traffic, and never pass `--i-own-this-server` unless the user explicitly says the server is theirs to load-test.
4. **Do not publish results.** Records are local (`results/local/`). Server logs and `.nsys-rep` captures contain raw paths; workload files built from captured requests contain the user's prompts. Do not paste or commit them without the user's say-so.
5. **Do not overclaim, and do not invent.** Report medians with spread, the evidence class, the confidence level and any "not comparable" notice. Never present a `hypothesis` as a cause, a single round as a speedup, or a number that is not in a record. If a command fails or a tool is refused, report the error text as it is.
6. **Specs come from `propose`.** Do not hand-write specs containing paths, program names or environment settings; machine-specific values belong in `hotpath.local.json` and are referenced as `${key}`. MCP-launched experiments refuse specs that do otherwise.

## Setup

```
pip install git+https://github.com/denisanzora/hotpath    # or: pip install . from a clone
pip install jsonschema    # optional: full JSON Schema validation of request-shape outputs (hotpath[schema])
hotpath init            # writes hotpath.local.json; fill in model_path, runtime_binary, runtime_bin_dir, build_dir
hotpath tools           # verify GPU, config and external tools
```

On Windows, if a server or `llama-bench` exits immediately with no output, a CUDA runtime DLL is missing: add the CUDA `bin` directory to `path_prepend` in `hotpath.local.json`.

## Workflow

1. `hotpath gguf MODEL.gguf --bandwidth-gbs N` to see bytes streamed per token and the decode ceiling.
2. If the user already runs a server, start passive: `hotpath observe --log LOG` (add `--server URL` if it is up) then `hotpath diagnose results/local/observation-*.json`. This alone gives the prefill/decode split of real traffic, the prefix-reuse rate, decode vs the bandwidth ceiling, CPU-fallback sessions and session drift — all `passive`/`historical`. It cannot give client wall time or per-request overhead; the diagnosis names the one measurement that would. Say which findings are observed and which are inferred.
3. `hotpath workload workloads/w.json --words 450 900 1300 --n-predict 64` for a synthetic workload, or `hotpath workload workloads/real.json --from-requests captured.jsonl` when the user has captured their application's request bodies (the file stays local; never paste its contents into a reply or a record).
4. Collect direct evidence (active, ask first): `hotpath bench --ub <current> --pp 512 --tg 128` for the runtime alone, and `hotpath measure --server URL --workload workloads/w.json --label baseline` against a server the user owns (listed in `owned_servers`).
5. `hotpath diagnose results/local/measurement-baseline-*.json results/local/bench-*.json` (add observation/profile records if present). Report findings with their confidence level (`confirmed`, `likely`, `hypothesis`, `insufficient`), the measurements they cite, and the "cannot establish" list. Do not present a `hypothesis` as a cause.
6. `hotpath propose results/local/diagnosis-*.json --out experiments/next.json` writes a one-variable spec for the recommended finding (or `--finding ID`) and prints the plan; show the plan to the user. Only edit `common_args` if the user's real launch arguments differ (or set `server_args` in `hotpath.local.json`).
7. After approval: `hotpath experiment experiments/next.json --wait-s 600`. Use at least 3 rounds for any claim.
8. `hotpath diagnose results/local/experiment-<name>-*.json` gives the verdict per arm: paired per-prompt ratios against within-prompt noise, VRAM delta, output identity (token ids or content hash), output validity when a schema was involved, and keep / reject / inconclusive with the workload size stated. Report it as a result for this machine and workload, never as a general property of the runtime. Read the verdict's cache-state line first: a prefill or wall-time claim stands only when both arms were verified in the same state (cold by `cache_n` = 0, or warm by design); otherwise say the comparison is inconclusive for those metrics. Say which validation level was used (full schema with `jsonschema`, or required keys only) and never call an output "schema-valid" at the subset level. For cache/checkpoint flags, repeat the production gate the verdict prints (it depends on the real traffic's reuse rate from an `observe --log` record). Diverging outputs need explanation before recommending a change; a validity drop is a rejection whatever the speed. Then diagnose again with the new record to see what remains.
9. For kernel work: `hotpath kernels test --op OP` (correctness vs CPU) before `hotpath kernels perf`, and `hotpath profile --label L --pp 0 --tg 64` for a decode kernel-share profile; `diagnose` ranks a profile's kernel classes.

## MCP

`hotpath mcp` (or `hotpath-mcp`) serves the same commands over stdio. Active tools are disabled unless the operator set `HOTPATH_MCP_ALLOW_ACTIVE=1`, and each active call needs `confirm=true`; passing `confirm=true` without the operator flag is refused by the server, and no read-only tool can launch a server or benchmark. Spec files are checked before an MCP-launched experiment (programs and models must be `${key}` references, environment overrides limited to `GGML_`/`LLAMA_`/`CUDA_`, no file-writing, network or model-changing server arguments), and the tools only talk to loopback servers when read-only, write only inside the working directory, and never touch `*.local.json`. Validated with Claude Code and OpenCode on one machine; see `docs/agents.md` for the exact status per client.

## Limits

llama.cpp-family servers only; NVIDIA only; hotpath does not measure logit divergence or perplexity (use `llama-perplexity`) and does not check what the user's application decides from an output (identity and schema validity are separate from that). If asked for something outside this, say so.
