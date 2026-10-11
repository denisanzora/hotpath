# Changelog

## 0.2.0

### Added
- `hotpath diagnose`: ranked bottleneck findings from any mix of records (observation, measurement, experiment, bench, profile). Each finding carries the measurements it rests on, a confidence level (`confirmed` / `likely` / `hypothesis` / `insufficient`), why it matters, an opportunity bound where the measured shares allow one, the controlled experiment that would confirm or reject it, and the risks of the candidate change. Finished experiments get a per-arm keep / reject / inconclusive verdict.
- Passive diagnosis from a server log: `hotpath observe --log LOG` now keeps per-request timings, slot selection and reuse, and parse coverage; `diagnose` works from that alone (prefill/decode split on server time, real-traffic prefix reuse, CPU-fallback sessions, session drift, error lines, unreadable-format detection) and names the single best next measurement.
- `hotpath propose`: turns a finding into a one-variable experiment spec for the existing runner (baseline = your configuration; `${key}` references instead of paths) and prints its dry-run plan. Launches nothing; never overwrites an existing file.
- Request-shape experiments: `hotpath workload --from-requests` builds a workload from your own captured llama-server request bodies; arms change one thing with `request_override`.
- Explicit cache modes (`cold`, `warm`, `as-is`) with per-request cache state read from the server's `cache_n`. Cold uses `cache_prompt: false` on every request (slot erase alone is not trusted: a host prompt cache can restore an erased prompt). Prefill, overhead and wall-time comparisons use only requests individually verified in the comparison's state, pair prompts across arms, and are marked not comparable otherwise.
- Output checks at separate levels: valid JSON, required top-level keys, and full JSON Schema (Draft 2020-12) validation with the optional `jsonschema` package (`pip install "hotpath[schema]"`); `schema_validation: full` fails with an install hint instead of downgrading. Unenforced keywords (`format`, non-local `$ref`) are listed.
- MCP tools `hotpath_diagnose` (read-only) and `hotpath_propose`; `hotpath_workload` accepts `from_requests`. Agent Skill and docs updated; validated with Claude Code and OpenCode.
- 65 unit tests (`python -m unittest discover -s tests`).

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

### Upgrading
`pip install --upgrade .` (or `pip install --upgrade git+https://github.com/denisanzora/hotpath`). No configuration change is required. Add `".[schema]"` for full schema validation.

### Compatibility and limits
Validated on Windows 11, Python 3.10, one NVIDIA GeForce RTX 2080 SUPER, with a llama.cpp-derived CUDA `llama-server`. Linux, macOS, other GPUs and runtimes are untested. Agent integration is validated with Claude Code and OpenCode in scripted sessions on one machine. Decision-level agreement with an application's own checks is not measured. See the README for the full list.

## 0.1.0
Initial public release: observe, triage, gguf, workload, measure, experiment, compare, bench, profile, kernels, MCP server and Agent Skill.
