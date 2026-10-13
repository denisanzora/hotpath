# Contributing

Thanks for helping. Hotpath is small on purpose: standard-library Python with no required runtime dependencies. The one optional extra is `jsonschema` (`pip install "hotpath[schema]"`) for full JSON Schema validation of outputs; everything must keep working, and say what it did not check, without it.

## Ground rules

- **Keep the core dependency-free** unless there is a strong reason. `dependencies = []` is a feature; new capabilities that need a library go behind an optional extra, and an explicit request for that capability must fail with an install hint instead of silently doing less.
- **Never commit measurements from your machine, model files, logs, profiler captures, prompts, or local paths.** `.gitignore` covers the common cases; check `git status` before you commit. Records written by hotpath reduce paths to basenames, but server logs written next to them are raw.
- **Do not claim support you have not run.** If you add Linux/macOS/AMD/other-runtime behavior, say in the PR what hardware, OS and runtime version you exercised it on, and update the "Validated" table in the README.
- **Evidence labels matter.** Every record carries an evidence class (`passive`, `active`, `historical`, `controlled`, `hypothesis`). New features should not blur them, and comparisons must keep refusing to compute ratios between non-comparable records.

## Workflow

1. Fork, create a branch, make a focused change.
2. Verify by running the real CLI workflow your change touches against a llama.cpp server you own (see the README quickstart). End-to-end runs are preferred over unit tests here; include the commands you ran and their outcome in the PR description.
3. Run the unit tests: `python -m unittest discover -s tests` (with and without `jsonschema` installed). They cover diagnosis, proposal, cache-state and validation logic, MCP gates and packaging metadata on synthetic inputs (shapes and arithmetic, never performance claims). Source-tree tests cannot see a broken `pyproject.toml`: after touching packaging, install into a fresh virtual environment and run `hotpath --version`.
4. Open a pull request describing what changed and how you verified it.

## Diagnosis rules

Rules in `hotpath/diagnose.py` must be relative (shares of a request, ratios between the user's own measurements, fractions of a ceiling computed from their model and GPU), never thresholds tied to one GPU, model or build. Every finding names the measurement it rests on, states a confidence level (`confirmed`, `likely`, `hypothesis`, `insufficient`), and either proposes a controlled experiment or says what evidence is missing. If `propose` can build the experiment, the finding's `experiment.arms` must be a list of arm dicts in the spec's vocabulary (`args_override`, `env_override`, `env_remove`, `args_remove`); otherwise describe the procedure in prose.

## MCP server and skill

`hotpath/mcp_server.py` must stay a thin layer over the CLI (each tool builds argv and runs `python -m hotpath`). Active tools must stay gated behind `HOTPATH_MCP_ALLOW_ACTIVE=1` and `confirm=true`, and an experiment an agent can start must pass `check_spec_for_agent` (programs and models only as `${key}` references to the operator's config). Read-only tools must not launch processes, reach non-loopback hosts or overwrite files.

## Adding a tool adapter

External tools live in `hotpath/adapters.py` as one class each (locate, install hint, fixed recipe). Add a class and register it in `all_status`.

## Discovery and launch settings

`hotpath/launch.py` is the one place that knows llama-server's argument spellings (`FLAGS`, `BOOLEAN_FLAGS`, `PATH_FLAGS`); add a setting there and every consumer (discovery, observe, diagnose, propose, the runner's alias-safe overrides) sees it. Path-valued flags must be listed in `PATH_FLAGS` so records keep only basenames and specs never inherit a path. `hotpath/discover.py` must stay read-only and bounded: new search locations are explicit lists or user-named roots walked by `walk()` (depth and entry caps, no links); the only program it may run is a found `llama-server --version`, recorded in `executed`, and never for binaries under agent-supplied roots. Every subprocess gets `stdin=subprocess.DEVNULL`. Support for another runtime's process names belongs in `discovery_process_names` and a parser of its own; do not report it as supported until it has been run.

## Reporting bugs

Include `hotpath tools` output, your OS, GPU, runtime build, and the exact command. Do not attach model files or unredacted logs.
