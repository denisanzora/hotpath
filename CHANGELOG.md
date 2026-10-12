# Changelog

## Unreleased (0.3.0.dev0)

### Added
- `hotpath discover`: the local setup without configuration, read-only and bounded. Running `llama-server` processes and their launch settings from the command line; loopback `/health`, `/props`, `/metrics`; installations (running, configured, `PATH`, standard locations, `--root` directories to a fixed depth, drive roots refused) with backends, sibling tools, and on Windows the imported DLLs that would not be found plus the CUDA toolkit directory that fixes them, verified with `--version`; GGUF headers with roles and VRAM fit; GPUs, driver CUDA version, profilers, logs, launch scripts. Lists where it looked and what it ran.
- `hotpath init` now discovers: it writes the running server's URL, binary, model, log and launch arguments (and `path_prepend` when a build needs it), each with its source; it lists ambiguous candidates instead of guessing (`--binary`, `--model`, `--server` choose), `--update` adds keys without changing existing ones, `--template` keeps the old placeholder file.
- `hotpath inspect`: discover + observe the running server + diagnose in one passive step, with the exact next commands and whether an experiment fits next to what runs now.
- `observe` and `measure` record the measured server's launch settings from its process command line; `observe` reads `/metrics` (aggregate prefill/decode time and prompt-cache reuse) and picks up the server's `--log-file` and model, or the model from the log's own `load_model` line.
- Diagnosis: a GPU-offload rule (log line, explicit `-ngl`, or the server's VRAM against the model's weights) with a stated memory estimate that chooses the candidate arm; `/metrics`-based phase split and cache reuse; findings ranked by the measured share of request time the change can act on, weighted by confidence; explicit `known`, `hypothesis` and `measured` fields; superseded findings; thin traffic is not read as a reuse pattern.
- The improve step: after an experiment, `diagnose` describes what remains on the candidate the verdict found faster, and `propose` builds the next spec from it.
- `propose` takes its baseline from the diagnosed server's observed command line (or `server_args` when it matches), sets the VRAM precondition from the GGUF header, and the plan states requests, free VRAM needed and a rough duration. Without a workload it writes synthetic prompts sized like the diagnosed real traffic.
- A hardware rule: BF16 matrices on a GPU below compute capability 8.0 (no BF16 tensor cores); a hypothesis until a prefill profile shows the fallback GEMM. Profile records are re-classified from their kernel names.
- MCP: `hotpath_inspect` and `hotpath_discover`; active tools run as jobs (result within 45 s, or a job id), `hotpath_job_status` with progress, `hotpath_job_cancel`; concurrent request handling; running jobs are stopped when the client exits; refusals explain how to enable active tools in Claude Code and OpenCode; unknown or missing arguments get the tool's argument list.

### Changed
- Experiment preconditions that fail on VRAM, load or ports name the running `llama-server` processes and state that hotpath will not stop them.
- Argument overrides in the runner and in `propose` match all spellings of an argument (`-ngl`/`--gpu-layers`/`--n-gpu-layers`, `-ub`/`--ubatch-size`, ...), so exactly one value reaches the server.
- A decode-ceiling gap proposes a CUDA-graphs experiment only when the launch disables graphs, with the current state as the baseline arm; otherwise a decode profile.
- An experiment arm's full command line without `-ub` is reported as the build default, not as unknown.
- GGUF headers are read 2-3x faster (tokenizer arrays are skipped, not decoded).

### Fixed
- MCP child processes inherited the JSON-RPC stdin. On Windows, PowerShell (used for process and GPU counters) waited on it, so `hotpath_observe` against a running server hung until the 300 s timeout; a child could also read the client's messages. Every subprocess hotpath starts now gets a closed stdin. This affects 0.2.0.
- MCP job children import the same hotpath when it runs from a source checkout in another working directory.

### Validation
Windows 11, RTX 2080 SUPER, a llama.cpp-derived CUDA build, one 27B ternary GGUF; a `llama-server` started by hand with `-ngl 56` of 65 layers as the user's server. Discovery, init and inspect found and diagnosed it; the proposed `-ngl 65` experiment refused to start while that server held the VRAM; after it was stopped, three runs (CLI, Claude Code through MCP, OpenCode through MCP) measured wall time per request 5.0 → 1.4 s (small workload) and 11.3–11.8 → 3.3 s (default workload), decode 7.7–8.5 → 48.1 t/s, +796–804 MiB, no shared-memory spill; outputs identical on the small workload and diverging deterministically on one of three prompts at token 51 of 64 on the default workload. On prompts shaped like the server's observed traffic: wall 7.24 → 1.92 s per request, outputs identical 3/3, verdict keep. Details in the README and docs/agents.md.

## 0.2.0

### Added
- `hotpath diagnose`: ranked bottleneck findings from any mix of records (observation, measurement, experiment, bench, profile). Each finding carries the measurements it rests on, a confidence level (`confirmed` / `likely` / `hypothesis` / `insufficient`), why it matters, an opportunity bound where the measured shares allow one, the controlled experiment that would confirm or reject it, and the risks of the candidate change. Finished experiments get a per-arm keep / reject / inconclusive verdict.
- Passive diagnosis from a server log: `hotpath observe --log LOG` now keeps per-request timings, slot selection and reuse, and parse coverage; `diagnose` works from that alone (prefill/decode split on server time, real-traffic prefix reuse, CPU-fallback sessions, session drift, error lines, unreadable-format detection) and names the single best next measurement.
- `hotpath propose`: turns a finding into a one-variable experiment spec for the existing runner (baseline = your configuration; `${key}` references instead of paths) and prints its dry-run plan. Launches nothing; never overwrites an existing file.
- Request-shape experiments: `hotpath workload --from-requests` builds a workload from your own captured llama-server request bodies; arms change one thing with `request_override`.
- Explicit cache modes (`cold`, `warm`, `as-is`) with per-request cache state read from the server's `cache_n`. Cold uses `cache_prompt: false` on every request (slot erase alone is not trusted: a host prompt cache can restore an erased prompt). Prefill, overhead and wall-time comparisons use only requests individually verified in the comparison's state, pair prompts across arms, and are marked not comparable otherwise.
- Output checks at separate levels: valid JSON, required top-level keys, and full JSON Schema (Draft 2020-12) validation with the optional `jsonschema` package (`pip install "hotpath[schema]"`); `schema_validation: full` fails with an install hint instead of downgrading. Unenforced keywords (`format`, non-local `$ref`) are listed.
- MCP tools `hotpath_diagnose` (read-only) and `hotpath_propose`; `hotpath_workload` accepts `from_requests`. Agent Skill and docs updated; validated with Claude Code and OpenCode.
- 75 unit tests (`python -m unittest discover -s tests`).

### Changed
- Experiment verdicts are paired per prompt against within-prompt noise, state the workload size (marked preliminary below 3 prompts × 3 rounds), keep speed, output identity and validity separate, and describe the measured machine and workload only.
- Experiments trading prefix reuse for per-request overhead (`--cache-ram`, `--ctx-checkpoints`) carry a production gate based on the real traffic's reuse rate.
- MCP-launched experiments check the spec's contents: `binary`/`model`/`path_prepend` must be `${key}` references, environment overrides are limited to `GGML_`/`LLAMA_`/`CUDA_`, and server arguments that write files, change the network binding, download a model or change which model loads are refused. `hotpath propose` leaves such arguments (`--port`, `--log-file`, `--slot-save-path`, `--model`, ...) out of the baseline it copies from your config, so an experiment server cannot take over your server's port or log. `hotpath_observe` only contacts loopback servers; `*.local.json` files cannot be written by MCP tools.
- `hotpath tools` reports whether the optional validator is installed.

### Fixed
- Records no longer store generated text by default. 0.1 wrote the full generated text of every measured request to `output_text`, contradicting the documented "token ids only". Text is now opt-in (`"store_outputs": true`). Existing 0.1 result files are not rewritten; review them before sharing.
- A malformed `cache_n` crashed a run mid-way and values such as `-1` or `true` counted as reuse; missing or malformed values are now `unknown`, never cold.
- Summaries no longer crash on records without `cache_n` or timing fields.
- Schema-error paths no longer contain output keys.
- Two records of the same kind written within one second no longer overwrite each other (found when an agent ran two diagnoses back to back).
- Commands no longer abort with `UnicodeEncodeError` after writing their record when the console code page cannot encode a dash in a report.
- `pip install` works: the optional-dependencies table in the candidate's `pyproject.toml` had swallowed the classifiers list, which broke every install.

### Upgrading
`pip install --upgrade .` (or `pip install --upgrade git+https://github.com/denisanzora/hotpath`). No configuration change is required. Add `".[schema]"` for full schema validation.

### Compatibility and limits
Validated on Windows 11, Python 3.10, one NVIDIA GeForce RTX 2080 SUPER, with a llama.cpp-derived CUDA `llama-server`. Linux, macOS, other GPUs and runtimes are untested. Agent integration is validated with Claude Code and OpenCode in scripted sessions on one machine. Decision-level agreement with an application's own checks is not measured. See the README for the full list.

## 0.1.0
Initial public release: observe, triage, gguf, workload, measure, experiment, compare, bench, profile, kernels, MCP server and Agent Skill.
