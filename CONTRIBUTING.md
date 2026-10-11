# Contributing

Thanks for helping. Hotpath is small on purpose: standard-library Python, no runtime dependencies.

## Ground rules

- **Keep it dependency-free** unless there is a strong reason. `dependencies = []` is a feature.
- **Never commit measurements from your machine, model files, logs, profiler captures, prompts, or local paths.** `.gitignore` covers the common cases; check `git status` before you commit. Records written by hotpath reduce paths to basenames, but server logs written next to them are raw.
- **Do not claim support you have not run.** If you add Linux/macOS/AMD/other-runtime behavior, say in the PR what hardware, OS and runtime version you exercised it on, and update the "Validated" table in the README.
- **Evidence labels matter.** Every record carries an evidence class (`passive`, `active`, `historical`, `controlled`, `hypothesis`). New features should not blur them, and comparisons must keep refusing to compute ratios between non-comparable records.

## Workflow

1. Fork, create a branch, make a focused change.
2. Verify by running the real CLI workflow your change touches against a llama.cpp server you own (see the README quickstart). End-to-end runs are preferred over unit tests here; include the commands you ran and their outcome in the PR description.
3. Open a pull request describing what changed and how you verified it.

## MCP server and skill

`hotpath/mcp_server.py` must stay a thin layer over the CLI (each tool builds argv and runs `python -m hotpath`). Active tools must stay gated behind `HOTPATH_MCP_ALLOW_ACTIVE=1` and `confirm=true`.

## Adding a tool adapter

External tools live in `hotpath/adapters.py` as one class each (locate, install hint, fixed recipe). Add a class and register it in `all_status`.

## Reporting bugs

Include `hotpath tools` output, your OS, GPU, runtime build, and the exact command. Do not attach model files or unredacted logs.
