# hotpath

Find where local GPU inference actually spends its time, change one thing, and check that the change really helped.

hotpath is a small command-line toolkit for people running GGUF models with [llama.cpp](https://github.com/ggml-org/llama.cpp)-family servers (`llama-server`) on NVIDIA GPUs they own. It is written in standard-library Python with no runtime dependencies.

It does not make inference faster by itself. It gives you a disciplined loop around the tools you already have:

1. **Observe** a running deployment without sending it any inference: build and model identity, GPU state, and the request timings in the server's own log.
2. **Triage** that observation: how much of each request is prompt processing vs generation, how decode compares with a memory-bandwidth ceiling for your model and GPU, and which configuration causes are worth testing first. Each finding is labeled with how strong its evidence is and which experiment would settle it.
3. **Experiment** under control: hotpath launches each variant (different flags, environment variables, runtime binary or model file) on its own port, interleaves the variants across rounds so drift and thermals hit every arm, and checks preconditions such as an idle GPU.
4. **Compare** honestly: medians with spread, ratios only between comparable records, an explicit list of reasons two records are not comparable, and a greedy-output equivalence check between arms.

It wraps rather than replaces `llama-bench`, `test-backend-ops`, Nsight Systems and Nsight Compute. `hotpath tools` tells you which are present.

## Why use it

Speed-ups on local inference are easy to misjudge: a different binary also changes memory use, a second run is faster because the cache is warm, another process was using the GPU, or the "faster" build quietly changed the output. hotpath makes each of those visible in the result record instead of leaving them to memory.

## Status and what has been validated

This is an early (0.1) release by a single developer. What has actually been run:

| Component | Validated on |
|---|---|
| OS | Windows 11 |
| Python | 3.10 |
| GPU | one NVIDIA GeForce RTX 2080 SUPER (compute capability 7.5), driver via `nvidia-smi` |
| Runtime | CUDA builds of a llama.cpp-derived `llama-server` / `llama-bench` / `test-backend-ops` |
| Profilers | Nsight Systems 2024.6, Nsight Compute 2025.1 |

| Diagnose → propose → experiment → verdict loop | one real run on the setup above (0.2.0): `bench` + `measure` of a 3-prompt workload → `diagnose` ranked a 194 ms (5%) per-request residual first → `propose` wrote a 3-arm spec (default, `--cache-ram 0`, `--ctx-checkpoints 0`) → `experiment` ran 9 interleaved launches → `diagnose` on the record: both candidates −4.2…4.4% wall, consistent on 3/3 prompts, greedy outputs identical, VRAM unchanged, with the finding's caveat (prefix reuse lost) attached |

Not validated: Linux, macOS, AMD or Intel GPUs, multi-GPU systems, upstream `llama.cpp` releases other than the one above, other runtimes (vLLM, TensorRT-LLM, Ollama, ...). The code avoids Windows-only assumptions where it can (`nvidia-smi`, `lsof`, standard install locations for Nsight on Linux), but those paths are untested; please report what you find. The per-process GPU memory counters (`windows_process_gpu_counters`) are Windows-only and are simply absent elsewhere.

## Requirements

- Python 3.10 or newer.
- An NVIDIA GPU and driver with `nvidia-smi` on `PATH` (GPU facts and the idle-GPU precondition use it; commands that only read files or HTTP endpoints work without it).
- A `llama-server` you can run yourself, and a GGUF model. `hotpath bench` additionally needs `llama-bench`; `hotpath kernels` needs `test-backend-ops` from a llama.cpp build tree; `hotpath profile` needs `llama-bench` and Nsight Systems (`nsys`).

Nsight on Windows requires the "Allow access to the GPU performance counters to all users" setting in the NVIDIA control panel's developer options for most profiling.

## Install

```
# one command, no clone (not on PyPI; installs from GitHub):
pip install git+https://github.com/denisanzora/hotpath

# or from a clone (recommended for development and for using examples/):
git clone https://github.com/denisanzora/hotpath
cd hotpath
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install .
hotpath --version
hotpath tools
```

`pip install -e .` works for development. You can also run it without installing: `python -m hotpath ...` from the repository root.

## Quickstart with your own model and runtime

Work in a directory of your own (results are written relative to the current directory under `results/local/`, which is the default and is gitignored in this repository).

**1. Describe your setup once.**

```
hotpath init              # writes hotpath.local.json
```

Edit it: `model_path`, `runtime_binary` (your `llama-server`), `runtime_bin_dir` (where `llama-bench` lives), `build_dir` (a llama.cpp build tree containing `test-backend-ops`), and, if you profile a server that is already running, `server_url` and `server_log`. Every key is optional and each command says which one it needs. See `hotpath.local.example.json` for all keys. Keep this file out of version control; it holds your paths.

```
hotpath tools             # confirms GPU, config, and which external tools were found
```

**2. Look at your model.** Reads only the GGUF header:

```
hotpath gguf path/to/model.gguf --bandwidth-gbs <your GPU's peak GB/s>
```

This prints tensor types, bytes streamed per decoded token, and the batch-1 decode ceiling for that bandwidth. If your GPU is in the built-in table the bandwidth is picked up automatically in `observe`/`triage`; otherwise pass `--bandwidth-gbs`.

**3. Observe a running server (passive).** Point at a server and, optionally, its stderr log:

```
hotpath observe --server http://127.0.0.1:8080 --log path/to/llama-server.log
hotpath triage results/local/observation-<timestamp>.json
```

Nothing here submits inference; it reads `/health`, `/props`, `/v1/models`, `/slots`, the GPU state, and the log. Start `llama-server` with `--slots` to get slot details.

**4. Make a workload.** Prompts are synthetic and deterministic (seed + word count), so a workload is reproducible and safe to share:

```
hotpath workload workloads/example.json --n-prompts 4 --words 450 900 1300 2000 --n-predict 64
```

`--format chatml` (default) wraps prompts for ChatML-style chat models; `--format raw` is for base models. You can also write a workload file with your own prompts; records then store only prompt hashes unless you pass `--store-prompts`.

**5. Run a controlled experiment.** Copy `examples/experiment-ubatch.json`, adapt `common_args` to your model and GPU, and preview it first:

```
hotpath experiment my-experiment.json --dry-run
hotpath experiment my-experiment.json            # ACTIVE: launches servers
```

A spec names a base launch and a list of arms; arms override or remove command-line arguments and environment variables, or swap the binary or model. `${key}` values are filled from `hotpath.local.json`, so the spec itself contains no paths. The run is `rounds x arms` server launches, interleaved. `preconditions` can require a maximum GPU utilization, minimum free VRAM and free ports, and `--wait-s N` waits for them instead of failing. hotpath only ever starts its own servers on the spec's port (default 8099); it does not touch the server you use day to day.

**6. Compare.**

```
hotpath compare results/local/experiment-<name>-<timestamp>.json
```

An experiment prints its comparison when it finishes; `compare` re-renders it, or compares several measurement/kernel records.

## From "it is slow" to a measured bottleneck: diagnose and propose

`compare` tells you the numbers; `diagnose` says what they mean and what to test next, and `propose` turns that into a spec the experiment runner accepts.

```
hotpath bench --ub 256 --pp 512 --tg 128                      # ACTIVE: the runtime alone, same model
hotpath measure --server URL --workload workloads/w.json --label baseline   # ACTIVE: the served model, your workload
hotpath diagnose results/local/measurement-baseline-*.json results/local/bench-*.json
hotpath propose results/local/diagnosis-*.json --out experiments/next.json   # writes a one-variable experiment and prints its plan
hotpath experiment experiments/next.json                       # ACTIVE
hotpath diagnose results/local/experiment-<name>-*.json        # verdict: keep / reject / inconclusive, with the evidence
```

`diagnose` reads any mix of records (observation, measurement, experiment, bench, profile) and produces ranked findings. Each finding carries the measurements it rests on, a confidence level, why it matters for end-to-end latency, an opportunity bound when the measured shares allow one, the controlled experiment that would confirm or reject it, and the risks of the candidate change. The confidence levels are:

| Level | Meaning |
|---|---|
| `confirmed` | the records contain a controlled comparison that shows it |
| `likely` | direct measurements point at it; no controlled comparison yet |
| `hypothesis` | consistent with the evidence; the named experiment would decide |
| `insufficient` | the records cannot establish it; the finding says what to collect |

What it looks at today: the per-request split into prompt processing, generation and everything else (prefill-bound vs decode-bound vs overhead); the residual per-request overhead and the server flags that affect it; the microbatch size against the prefill share; decode against the memory-bandwidth ceiling for your model and GPU, and whether CUDA graphs are disabled; the served prefill/decode against `llama-bench` on the same model (the gap is server-side work); VRAM headroom and system-memory spill signs; kernel-class shares from a profile; and, for finished experiments, a per-arm verdict that combines the wall-time ratio, whether the p10-p90 spreads overlap, the VRAM delta and the greedy-output equivalence. It ends with the list of things the given records cannot establish.

`propose` keeps the comparison fair by construction: one variable differs between arms; the baseline arm is your current configuration (`server_args` in the local config, or the diagnosed experiment's own arguments, or `--common-args`); binary and model are `${...}` references into `hotpath.local.json`; rounds are interleaved by the runner. It prints the dry-run plan and never launches anything.

Diagnoses are not results. A `hypothesis` finding is an argument for an experiment; the experiment's verdict is what you act on.

## Use it from an AI coding agent

`hotpath mcp` (also installed as `hotpath-mcp`) serves the commands as MCP tools over local stdio, with active tools (experiments, benchmarks, profiling) disabled unless you start it with `HOTPATH_MCP_ALLOW_ACTIVE=1`. `skills/hotpath/SKILL.md` is a portable Agent Skill that teaches an agent the workflow. Both reuse the CLI. The protocol behavior has been exercised with a scripted client; the integration with specific agents has **not** been validated yet. Details and the exact validation status are in [docs/agents.md](docs/agents.md).

## Other commands

| Command | What it does |
|---|---|
| `hotpath diagnose RECORD...` | ranked bottleneck findings with confidence, measurements, opportunity bound, confirming experiment and risks; keep/reject/inconclusive verdicts for finished experiments |
| `hotpath propose DIAGNOSIS --out SPEC` | one-variable experiment spec for a finding (default: the diagnosis's recommended one) plus its dry-run plan; launches nothing |
| `hotpath measure --server URL --workload W --i-own-this-server` | ACTIVE: run a workload against a server you own and record timings (refuses servers not in `owned_servers` unless the flag is given) |
| `hotpath bench` | ACTIVE: `llama-bench` prefill/decode with repetitions; refuses to run while something else is using the GPU |
| `hotpath profile --label L --pp 0 --tg 64` | ACTIVE: Nsight Systems kernel-share profile of a decode-only (`--pp 0`) or prefill-only (`--tg 0`) `llama-bench` run |
| `hotpath kernels test\|perf --op MUL_MAT --params REGEX` | ACTIVE: `test-backend-ops` correctness (NMSE vs CPU) or per-kernel timing for ops matching a filter |

Each command's `--help` lists its options. `--out DIR` changes the results directory; `--local FILE` (or `HOTPATH_LOCAL`) changes the config file.

## Correctness verification

hotpath treats "faster" and "still correct" as separate questions:

- **Greedy output equivalence.** Experiment workloads use temperature 0. `compare` reports, per prompt, whether each arm produced the same token sequence as the first arm and, if not, the index of the first divergence. Identical tokens are strong evidence an optimization did not change behavior. A divergence is not automatically a bug (reordered floating-point sums can flip a near-tie), but it should be understood before you adopt the change.
- **Isolated kernel correctness.** `hotpath kernels test` runs `test-backend-ops` and reports how many cases pass the runtime's own NMSE threshold against the CPU reference, plus failures and unsupported cases.
- **Not included.** hotpath does not compute logit divergence or perplexity. For numerical-difference checks beyond token equality use `llama-perplexity` or your own evaluation, and treat hotpath's equivalence check as a first gate.

## Reading the results

Every JSON record has an `evidence_class`:

| Class | Meaning |
|---|---|
| `passive` | read from logs, config and non-inference endpoints |
| `active` | hotpath sent inference requests to a server |
| `controlled` | arms launched by hotpath under one spec, interleaved, same workload |
| `historical` | a comparison of records collected under conditions hotpath did not control |
| `hypothesis` | a claim that still needs an experiment |

Practical guidance:

- Prefer `controlled` over `historical`. A ratio between two separate `measure` runs is labeled historical for a reason.
- Look at the spread (p10-p90), not only the median. With one round and one repetition there is no spread to speak of; use at least a few rounds before believing a difference of a few percent.
- Read "Not directly comparable" first. If prompt sets, `n_predict` or model files differ, ratios mean something other than what you intended.
- `decode_vs_bandwidth_ceiling` in `triage` is a peak-bandwidth estimate, not a promise. Real decode also reads the KV cache and pays launch overhead.
- Check per-arm VRAM columns: a configuration that is faster because it silently spills to or from system memory is not a clean win.
- `prefill` and `decode` rates are the server's own timings; `wall_ms` is client-side per request. An end-to-end improvement is what matters for users, and the ratio of `wall_ms` is the one that shows it.

## Privacy and what gets stored

- Results go to `results/local/` in the directory you run from. Nothing is uploaded anywhere.
- Records store file **basenames**, hashes of model/binary heads, GPU facts, `GGML_*`/`LLAMA_*`/`CUDA_*` environment variable values (these can include install paths such as `CUDA_PATH`), the OS version string, and a short hash of the hostname. Model, binary, workload and path-like command-line arguments are reduced to basenames. Records do not store prompt text (unless `--store-prompts`), or generated text (token ids only).
- Experiments also write the raw stderr/stdout of each launched server next to the results (`server-*.log`). Those logs contain full paths and are **not** redacted. Do not publish them without reading them.
- Profiler captures (`nsys-*.nsys-rep`, `.sqlite`) are written next to the results and contain raw paths and process details. Treat them as private. Text in records (tool output, command lines) is passed through a path scrubber, but it is best effort: read a record before sharing it.
- Review anything you intend to share. `hotpath.local.json` is for your machine only.

## Limitations

- Only `llama.cpp`-family servers are supported (the HTTP API, log format and `--version` output are what the parsers expect). Log parsing is regex-based and can break when upstream changes its log lines.
- NVIDIA only. The built-in bandwidth table lists published peak memory bandwidth for a handful of GPUs; only the RTX 2080 SUPER entry has been checked against a real run. Use `--bandwidth-gbs` to override.
- The experiment runner launches whole servers per arm, which is slow for large models but isolates arms properly.
- MCP server and agent skill are new and unvalidated in real agents (see docs/agents.md).
- The test suite covers the diagnosis and proposal logic on synthetic record structures (`python -m unittest discover -s tests`); the commands that drive servers and profilers are verified by running the actual workflows (see CONTRIBUTING).
- `diagnose` reasons from the measurements hotpath itself collects. Its rules are relative (shares, ratios, fractions of a ceiling) and were checked on one GPU/model/runtime; a finding is only as good as the records behind it, and it says `insufficient` rather than guess. It cannot see prefix-cache effectiveness (hotpath workloads disable the cache on purpose), request-level CPU costs inside the client, or anything a kernel profile has not been run for.
- `propose` covers the experiment kinds `diagnose` emits as arm lists (microbatch sweep, CUDA-graph toggle, prompt-cache/checkpoint flags); findings whose confirmation is a procedure (a profile, a client-side request-shape comparison) are described, not generated.
- `hotpath profile` and `hotpath kernels` are thin wrappers: they run the underlying tool and summarize its output; `diagnose` ranks a profile's kernel classes but does not interpret individual kernels.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nvidia-smi found=False` | Driver not installed or not on `PATH`. Facts and the idle-GPU precondition need it. |
| `llama-bench found=False` / `test-backend-ops found=False` | Set `runtime_bin_dir` / `build_dir` in `hotpath.local.json`, or put them on `PATH`. |
| `nsys found=False` | Install Nsight Systems or the CUDA toolkit, or put `nsys` on `PATH`. |
| Server exits immediately in an experiment | Open `results/local/server-<name>-<arm>-r0.log`. A native Windows build may need its CUDA runtime DLLs on `PATH`; list that directory under `path_prepend` in the config. |
| `preconditions not met` | Something else is using the GPU or the port. Free it, or use `--wait-s`. |
| `refusing to submit inference ... not in owned_servers` | `measure` only targets servers you declare as yours (`owned_servers` or `--i-own-this-server`). |
| Every output in `compare` diverges | Check that `temperature` is 0 and that arms are not reordering sums on purpose (different kernels, batch sizes); one near-tie flip is common and not necessarily a bug. |
| `error: ... (set HOTPATH_DEBUG=1 for a traceback)` | Set that variable to see the full traceback when reporting a bug. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with `hotpath tools` output, OS, GPU, runtime build and the exact command are the most useful contribution right now, along with reports from setups listed above as not validated.

## License

MIT, see [LICENSE](LICENSE). hotpath does not bundle or redistribute llama.cpp, Nsight or any model; those are separate projects with their own licenses.
