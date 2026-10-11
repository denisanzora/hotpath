---
name: hotpath
description: Profile and benchmark local llama.cpp-family GPU inference with the hotpath CLI or MCP server. Use when the user wants to find inference bottlenecks, compare runtime flags/builds/models with controlled experiments, check greedy-output equivalence between configurations, or inspect a llama-server deployment. Not for training, non-llama.cpp runtimes, or non-NVIDIA GPUs.
license: MIT
compatibility: Requires Python 3.10+ and a llama.cpp-family llama-server; GPU features need NVIDIA drivers (nvidia-smi). Validated on Windows 11 with one NVIDIA GPU only.
metadata:
  repository: https://github.com/denisanzora/hotpath
---

# hotpath

hotpath is a CLI (and optional MCP server) that wraps `llama-server`, `llama-bench`, `test-backend-ops` and Nsight with a controlled-experiment workflow. Use it; do not re-implement its measurements with ad-hoc scripts.

## Ground rules

1. **Passive first.** `hotpath tools`, `hotpath gguf`, `hotpath observe`, `hotpath triage` and `hotpath experiment --dry-run` do not submit inference or use the GPU for long. Run these before anything active.
2. **Ask before active commands.** `experiment` (without `--dry-run`), `measure`, `bench`, `profile` and `kernels` launch servers or benchmarks and occupy the GPU for minutes to hours. Tell the user what will run, how long it may take, and that the GPU must be otherwise idle, and get agreement first.
3. **Never touch the user's day-to-day server.** Experiments launch their own server on the spec's port. `measure` only targets servers listed in `owned_servers`; never pass `--i-own-this-server` unless the user explicitly says the server is theirs to load-test.
4. **Do not publish results.** Records are local (`results/local/`). Server logs and `.nsys-rep` captures contain raw paths. Do not paste or commit them without the user's say-so.
5. **Do not overclaim.** Report medians with spread, the evidence class, and any "not directly comparable" notice. A single round is a smoke test, not a speedup.

## Setup

```
pip install git+https://github.com/denisanzora/hotpath    # or: pip install . from a clone
hotpath init            # writes hotpath.local.json; fill in model_path, runtime_binary, runtime_bin_dir, build_dir
hotpath tools           # verify GPU, config and external tools
```

On Windows, if a server or `llama-bench` exits immediately with no output, a CUDA runtime DLL is missing: add the CUDA `bin` directory to `path_prepend` in `hotpath.local.json`.

## Workflow

1. `hotpath gguf MODEL.gguf --bandwidth-gbs N` to see bytes streamed per token and the decode ceiling.
2. If a server is already running: `hotpath observe --server URL --log LOG` then `hotpath triage results/local/observation-*.json`. Findings are hypotheses with a proposed experiment; they are not results.
3. `hotpath workload workloads/w.json --words 450 900 1300 --n-predict 64`.
4. Collect direct evidence (active, ask first): `hotpath bench --ub <current> --pp 512 --tg 128` for the runtime alone, and `hotpath measure --server URL --workload workloads/w.json --label baseline` against a server the user owns (listed in `owned_servers`).
5. `hotpath diagnose results/local/measurement-baseline-*.json results/local/bench-*.json` (add observation/profile records if present). Report findings with their confidence level (`confirmed`, `likely`, `hypothesis`, `insufficient`), the measurements they cite, and the "cannot establish" list. Do not present a `hypothesis` as a cause.
6. `hotpath propose results/local/diagnosis-*.json --out experiments/next.json` writes a one-variable spec for the recommended finding (or `--finding ID`) and prints the plan; show the plan to the user. Only edit `common_args` if the user's real launch arguments differ (or set `server_args` in `hotpath.local.json`).
7. After approval: `hotpath experiment experiments/next.json --wait-s 600`. Use at least 3 rounds for any claim.
8. `hotpath diagnose results/local/experiment-<name>-*.json` gives the verdict per arm: wall-time ratio with spread overlap, VRAM delta, greedy-output equivalence, and keep / reject / inconclusive. Diverging outputs need explanation before recommending a change. Then diagnose again with the new record to see what remains.
9. For kernel work: `hotpath kernels test --op OP` (correctness vs CPU) before `hotpath kernels perf`, and `hotpath profile --label L --pp 0 --tg 64` for a decode kernel-share profile; `diagnose` ranks a profile's kernel classes.

## MCP

`hotpath mcp` (or `hotpath-mcp`) serves the same commands over stdio. Active tools are disabled unless the operator set `HOTPATH_MCP_ALLOW_ACTIVE=1`, and each active call needs `confirm=true`. See `docs/agents.md` in the repository.

## Limits

llama.cpp-family servers only; NVIDIA only; hotpath does not measure logit divergence or perplexity (use `llama-perplexity`). If asked for something outside this, say so.
