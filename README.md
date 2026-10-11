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

This is an early (0.2) release by a single developer. What has actually been run:

| Component | Validated on |
|---|---|
| OS | Windows 11 |
| Python | 3.10 |
| GPU | one NVIDIA GeForce RTX 2080 SUPER (compute capability 7.5), driver via `nvidia-smi` |
| Runtime | CUDA builds of a llama.cpp-derived `llama-server` / `llama-bench` / `test-backend-ops` |
| Profilers | Nsight Systems 2024.6, Nsight Compute 2025.1 |
| Passive `observe --log` → `diagnose` | one real log of 78 sessions / 1,849 requests (0.2.0): prefill/decode split on server time, 8% median prefix reuse, decode at 50% of the bandwidth ceiling, session drift, CPU-speed sessions and an unknown-microbatch session all reported from the log alone |
| Cache modes and schema validation (0.2.0) | five two-arm request-shape runs on real application request bodies (3 prompts x 3 rounds each): `cold` with erase unsupported (HTTP 501) and with erase honored both verified 36/36 cold via `cache_prompt: false`; an earlier erase-only cold run was correctly reported "not cold" (36/36 warm, host prompt cache); `warm` labelled as reuse behavior; `as-is` without erase marked not comparable. Removing the schema was rejected in every mode on full Draft 2020-12 validation (18/18 -> 0/18) despite 14-22% lower wall time. CLI and MCP produced identical verdicts on the same record |
| MCP in real agents | Claude Code 2.1.x and OpenCode 1.18.x, scripted one-shot sessions (details and limits in docs/agents.md) |
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

**Optional: full JSON Schema validation** of outputs in request-shape experiments needs the `jsonschema` package. Everything else (observe, diagnose, propose, experiments, MCP) works without it, and records say which level of validation was actually run.

```
pip install ".[schema]"                                                    # from a clone
pip install "hotpath[schema] @ git+https://github.com/denisanzora/hotpath"  # without cloning
```

**Upgrading from 0.1:** `pip install --upgrade .` (or the git command above). No configuration changes are required. One behavior change matters for existing records: 0.1 stored the generated text of every measured request in `output_text`; 0.2 stores token ids, an output hash and lengths only (see "Privacy"). Old records are not rewritten, so review or delete 0.1 result files before sharing them.

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

## Already running a server? Diagnose from its log, no benchmark needed

```
hotpath observe --log path/to/llama-server.log         # passive: reads the log, no inference
hotpath diagnose results/local/observation-*.json
```

From the server's own log hotpath reads every request's prompt and generation timings, token counts, how the slot was selected (LRU / longest common prefix) and how many tokens were reused, per session, with the launch facts the log printed. `diagnose` then reports, from that alone: whether the workload is prefill- or decode-bound (on the server's prompt+generation time — a log has no client wall time, and hotpath says so instead of inventing an overhead figure), how much of each prompt real traffic actually reuses (the evidence that decides whether cache settings may be touched), decode against the bandwidth ceiling for your model, sessions that ran at CPU speed, drift or regressions between sessions and whether launch facts changed, error lines, and whether the log format was understood at all. Each finding keeps its evidence class (`passive` / `historical`) and the list at the end names the single best next measurement — usually one `hotpath measure` against your own server to add wall time and per-request overhead.

Mixed logs are handled explicitly: sessions below a prefill-rate threshold are treated as CPU fallback and reported, not averaged in; a session that did not print its microbatch is reported as unknown rather than borrowed from an older session; an unrecognized format is a finding, not an empty report.

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

What it looks at today: the per-request split into prompt processing, generation and everything else (prefill-bound vs decode-bound vs overhead); the residual per-request overhead and the server flags that affect it; the microbatch size against the prefill share; decode against the memory-bandwidth ceiling for your model and GPU, and whether CUDA graphs are disabled; the served prefill/decode against `llama-bench` on the same model (the gap is server-side work); VRAM headroom and system-memory spill signs; kernel-class shares from a profile; and, for finished experiments, a per-arm verdict that combines paired per-prompt ratios judged against each prompt's own repetition noise, the verified cache state of both arms, the VRAM delta, output identity and output validity. It ends with the list of things the given records cannot establish.

`propose` keeps the comparison fair by construction: one variable differs between arms; the baseline arm is your current configuration (`server_args` in the local config, or the diagnosed experiment's own arguments, or `--common-args`); binary and model are `${...}` references into `hotpath.local.json`; rounds are interleaved by the runner. It prints the dry-run plan and never launches anything.

Diagnoses are not results. A `hypothesis` finding is an argument for an experiment; the experiment's verdict is what you act on. Verdicts are paired per prompt (each prompt's candidate/baseline ratio against that prompt's own repetition noise), state the workload size (`N prompts x R rounds`, marked preliminary below 3 x 3), judge output identity, output validity and speed separately, and describe this machine and workload only. Experiments that trade prefix reuse for per-request overhead (`--cache-ram`, `--ctx-checkpoints`) carry a production gate: without an `observe --log` record showing how much real traffic reuses, the verdict says applicability is unknown; with high real reuse it says not to apply.

## Measuring your real request shape

Synthetic prompts measure the runtime; your application's requests add sampler settings, JSON schemas or grammars, cache options and generation limits that change the cost. A `requests` workload sends your own captured bodies:

```
hotpath workload workloads/real.json --from-requests captured.jsonl   # JSON array or JSONL of llama-server request bodies
```

Each item is sent as-is to `/v1/chat/completions` or `/completion` (streaming off), in order. Arms change one thing with `request_override` (`{"set": {...}, "remove": [...]}` on top-level fields), e.g. the same requests with and without the JSON schema.

**Cache modes.** A spec's `cache_mode` says what prompt-cache state the comparison is about, and every request's actual state is read back from the server's own `cache_n` (0 = cold):

| `cache_mode` | What hotpath does | What a verdict may claim |
|---|---|---|
| `cold` (default) | `cache_prompt: false` on every request in every arm (this overrides the `cache_prompt` in your request bodies), plus a slot erase before each pass when the server honors it | cold-prefill comparisons, but only if both arms are verified cold: at least 90% of an arm's requests report `cache_n` = 0, **and** the cache-sensitive metrics are then computed only from the requests that individually were cold (see below) |
| `warm` | nothing is cleared; items and repetitions reuse the slot as your application would | comparisons labelled warm-cache: prefill reflects reuse behavior; both arms must reuse about the same number of cached tokens per prompt (within 10%) or the comparison is marked not comparable |
| `as-is` | no intervention | timings are reported; prefill, residual and wall time are marked not comparable unless both arms verifiably ended up in the same state |

**Verification is per request, not per percentage.** A request's state comes only from the server's own `cache_n`: a missing, null, non-numeric or negative value is `unknown`, never cold. For prefill, per-request overhead and wall time, requests whose own state does not match the comparison basis are excluded (and counted in the record and the verdict), a prompt needs verified requests in **both** arms to be compared, and fewer than two such prompts makes the comparison not comparable. One prompt that was warm in only one arm therefore cannot bias the result, even though the arm-level percentage would still pass. Decode throughput and output validity do not depend on cache state and use every request.

Why not slot erase alone: `llama-server` honors slot erase only when started with `--slot-save-path` (otherwise HTTP 501), and even then its host prompt cache (`--cache-ram`) can restore an erased prompt on the next request — measured: with erase honored, all 36 requests of a two-arm run still reused their prompt (`cache_n` > 0), and hotpath's verdict reported the comparison as not cold instead of trusting the erase. Per-request `cache_prompt: false` is what produced verified cold requests (36/36 `cache_n` = 0) with and without erase support. `"slot_save_path": "auto"` (or a directory) in a spec adds `--slot-save-path` to servers hotpath launches, so erase is also issued; by default the launched command line is left as you wrote it. hotpath never restarts or reconfigures a server it did not launch; `measure` against your own server uses the same per-request isolation. If the arms' verified states differ — for example one arm reused a cached prefix — the verdict is `inconclusive` for prefill, residual and wall time and says why; decode and validity remain comparable.

**Output checks.** Per request, records keep timings, token counts, the cache state, an output hash and length, and — when the application's request carried a JSON schema — separate checks against that schema (the workload item's own schema, whatever an arm removes): the output parses as JSON; the schema's required top-level keys are present; and full validation of the whole schema (nested objects, types, enums, arrays and their bounds, string lengths, `additionalProperties`) with the maintained `jsonschema` package (Draft 2020-12) when it is installed (`hotpath tools` shows whether it is). A spec's `schema_validation` chooses the level: `auto` (default: full when available, otherwise required keys, and records say which), `full` (stops before any server is launched, with an install hint, if `jsonschema` is missing — it never downgrades silently), or `required-keys`. Draft 2020-12 treats `format` and `content*` as annotations and a non-local `$ref` cannot be resolved offline: records list such keywords under `schema_unchecked_keywords` instead of calling them checked, and a schema the validator cannot process is reported as "validator could not run", not as valid. Validation errors store the failing keyword and a path in which object keys the schema does not declare are masked as `*`, never output text. The verdict decides validity on full validation only if every judged request in both arms got it, compares rates (not counts), and says when it fell back to required keys. Output identity is a separate check: token ids where the endpoint returns them, otherwise a content hash. Decision-level agreement (what your application does with an output) is not checked by hotpath; a verdict treats identity, validity and speed as different questions.

What is retained where: the **workload file** holds your request bodies (prompts included) and lives in your local `workloads/`, which this repository's `.gitignore` excludes; **records** store body hashes, counts, timings and output hashes, never prompt or output text unless the spec sets `"store_outputs": true`; **server logs** written next to results are raw. Diagnoses never quote prompt or output content.

## Use it from an AI coding agent

`hotpath mcp` (also installed as `hotpath-mcp`) serves the commands as MCP tools over local stdio, with active tools (experiments, benchmarks, profiling) disabled unless you start it with `HOTPATH_MCP_ALLOW_ACTIVE=1`. `skills/hotpath/SKILL.md` is a portable Agent Skill that teaches an agent the workflow. Both reuse the CLI. The protocol behavior has been exercised with a scripted client, and the server has been used end to end from real Claude Code and OpenCode sessions on one Windows machine; other clients are untested. Details and the exact validation status are in [docs/agents.md](docs/agents.md).

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
- Records store file **basenames**, hashes of model/binary heads, GPU facts, `GGML_*`/`LLAMA_*`/`CUDA_*` environment variable values (these can include install paths such as `CUDA_PATH`), the OS version string, and a short hash of the hostname. Model, binary, workload and path-like command-line arguments are reduced to basenames. Records do not store prompt text (unless `--store-prompts`) or generated text (unless a spec sets `"store_outputs": true`). They do store generated **token ids** for hotpath's own workloads (needed for the output-equivalence check); token ids can be decoded with the model's tokenizer, so treat them as content when the prompts are yours. Application-shaped (`requests`) workloads store only a content hash and length of each output, plus validation results (keywords and masked paths, never output text). 0.1 stored generated text by default; see "Upgrading from 0.1".
- Experiments also write the raw stderr/stdout of each launched server next to the results (`server-*.log`). Those logs contain full paths and are **not** redacted. Do not publish them without reading them.
- Profiler captures (`nsys-*.nsys-rep`, `.sqlite`) are written next to the results and contain raw paths and process details. Treat them as private. Text in records (tool output, command lines) is passed through a path scrubber, but it is best effort: read a record before sharing it.
- Review anything you intend to share. `hotpath.local.json` is for your machine only.

## Limitations

- Only `llama.cpp`-family servers are supported (the HTTP API, log format and `--version` output are what the parsers expect). Log parsing is regex-based and can break when upstream changes its log lines.
- NVIDIA only. The built-in bandwidth table lists published peak memory bandwidth for a handful of GPUs; only the RTX 2080 SUPER entry has been checked against a real run. Use `--bandwidth-gbs` to override.
- The experiment runner launches whole servers per arm, which is slow for large models but isolates arms properly.
- MCP server and agent skill have been validated in scripted Claude Code and OpenCode sessions on one machine only; other clients are untested (see docs/agents.md).
- The test suite covers the diagnosis and proposal logic on synthetic record structures (`python -m unittest discover -s tests`); the commands that drive servers and profilers are verified by running the actual workflows (see CONTRIBUTING).
- `diagnose` reasons from the measurements hotpath itself collects. Its rules are relative (shares, ratios, fractions of a ceiling) and were checked on one GPU/model/runtime; a finding is only as good as the records behind it, and it says `insufficient` rather than guess. From a server log alone it cannot see client wall time, per-request overhead, or GPU state during requests; prefix-cache effectiveness comes only from a log's slot-selection lines (hotpath's own workloads disable the cache on purpose); nothing is known about kernels until a profile has been run.
- Log parsing recognizes llama-server's per-request timing, slot-selection and release lines; builds that log differently are reported as a format problem. The threshold that separates CPU-fallback sessions (`log_cpu_mode_prefill_tps_below`, default 10 t/s) is a heuristic you can change in the local config.
- `propose` covers the experiment kinds `diagnose` emits as arm lists (microbatch sweep, CUDA-graph toggle, prompt-cache/checkpoint flags, request shape); findings whose confirmation is a procedure (a profile, a launch check) are described, not generated.
- `requests` workloads send top-level body fields as given; `request_override` edits top-level fields only. Output identity on the chat endpoint is a content-hash comparison. Full schema validation needs the optional `jsonschema` package; schemas using external `$ref`s or formats the validator does not assert are checked only as far as the validator goes. Grammar-constrained (`grammar` GBNF) outputs are not validated.
- Cache state is verified from the server's `cache_n` and nothing else; a server that does not report it yields `unknown` and no prefill-side verdict. `cold` disables reuse entirely (including between consecutive items), so it measures cold processing, not the reuse your application would get; use `warm` for that. Cold mode overrides `cache_prompt` in your request bodies.
- Spec files written by an agent are checked before an MCP-launched experiment: `binary`/`model`/`path_prepend` must be `${key}` references into `hotpath.local.json`, environment overrides are limited to `GGML_`/`LLAMA_`/`CUDA_` variables, and server arguments that move the server off loopback or choose another model (`--host`, `--model`, `--hf-repo`, ...) are refused. Arguments not on that list are passed to `llama-server` as written, so only run specs you would run yourself.
- Agent integration is validated for Claude Code and (see docs/agents.md) OpenCode by scripted runs on one machine; other clients are untested.
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
