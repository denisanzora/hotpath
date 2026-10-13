# hotpath

Find where local GPU inference actually spends its time, change one thing, and check that the change really helped.

hotpath is a small command-line toolkit for people running GGUF models with [llama.cpp](https://github.com/ggml-org/llama.cpp)-family servers (`llama-server`) on NVIDIA GPUs they own. It is written in standard-library Python with no runtime dependencies.

It does not make inference faster by itself. It gives you a disciplined loop around the tools you already have:

0. **Discover** your setup without configuration: the running `llama-server` and how it was launched (from its command line), installed runtimes and whether they can start (missing CUDA DLLs and the directory that fixes them), GGUF models and whether they fit in VRAM, the GPU, driver, profilers and logs. `hotpath init` writes the result to `hotpath.local.json`; `hotpath inspect` goes straight on to a diagnosis.
1. **Observe** a running deployment without sending it any inference: build and model identity, launch settings, GPU state, `/metrics` counters and the request timings in the server's own log.
2. **Triage** that observation: how much of each request is prompt processing vs generation, how decode compares with a memory-bandwidth ceiling for your model and GPU, and which configuration causes are worth testing first. Each finding is labeled with how strong its evidence is and which experiment would settle it.
3. **Experiment** under control: hotpath launches each variant (different flags, environment variables, runtime binary or model file) on its own port, interleaves the variants across rounds so drift and thermals hit every arm, and checks preconditions such as an idle GPU.
4. **Compare** honestly: medians with spread, ratios only between comparable records, an explicit list of reasons two records are not comparable, and a greedy-output equivalence check between arms.

It wraps rather than replaces `llama-bench`, `test-backend-ops`, Nsight Systems and Nsight Compute. `hotpath tools` tells you which are present.

## Why use it

Speed-ups on local inference are easy to misjudge: a different binary also changes memory use, a second run is faster because the cache is warm, another process was using the GPU, or the "faster" build quietly changed the output. hotpath makes each of those visible in the result record instead of leaving them to memory.

## Status and what has been validated

This is an early release by a single developer (0.2.0 released; this branch is 0.3.0.dev0). What has actually been run:

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
| Discovery, `init`, `inspect` (0.3.0.dev0) | the same machine with no configuration: four installs found under a named directory, one native CUDA build reported as not starting (`0xC0000135`, `cudart64_12.dll`/`cublas64_12.dll` unresolved) with the toolkit directory that fixes it, verified by re-running `--version` with it on `PATH`; four GGUF headers with roles and VRAM fit; ambiguous binaries and models listed, not chosen. With a user's `llama-server` running, `init` wrote its URL, binary, model, log and launch arguments with their provenance, and `inspect` read the launch from the process command line, the log and `/metrics` |
| Offload diagnosis → experiment → verdict (0.3.0.dev0) | a `llama-server` started by hand with `-ngl 56` of 65 layers (a deliberately partial offload): `inspect` ranked "9 of 65 layers on the CPU" first (impact bound 97% from measured shares) and proposed `ngl56` vs `ngl65` from the observed command line; the experiment refused to start while that server held the VRAM and named it; after the server was stopped by hand: 3 prompts × 3 rounds, both arms 18/18 cold, wall 5.0 → 1.4 s per request, decode 8.5 → 48.1 t/s, +804 MiB, no shared-memory spill, outputs identical 3/3 (small workload). With the default workload (3 × 450–1300 words) in two agent runs: wall 11.3–11.8 → 3.3 s, decode 7.7–8.0 → 48.1 t/s, and a deterministic greedy divergence on one prompt at token 51 of 64, reported by the verdict as "faster, but outputs differ". With synthetic prompts shaped by `propose` like the server's observed traffic (≈240/320/420 prompt tokens, 48 generated): wall 7.24 → 1.92 s per request (−73.6%), decode 8.2 → 48.3 t/s, prefill 257 → 501 t/s, outputs identical 3/3, verdict keep |
| `improve`, the whole loop (0.3.0.dev0) | the same machine and 27B ternary GGUF, from a fresh virtual environment. **With a user's server running** (`-c 49152 -ngl 40 -ub 128`, a log and `/metrics`, 25 application-like requests: 16-20 s each, 14% median prefix reuse): `improve --dry-run` diagnosed it from its command line, log and `/metrics` and planned an 8-bit-KV + 57-layer arm (no more layers fit with the f16 cache by the estimate); `improve` refused in 23 s with nothing launched, naming the server. After the server was stopped, `improve` started from its last observation with prompts shaped like its traffic: step 1 `kvq8-ngl57` keep (−58.9% wall, outputs identical 3/3), step 2 `-ub 512` keep (−5.9%), step 3 `--cache-ram 0` inconclusive, not adopted; the joint original-vs-improved run: wall 17.3 → 6.6 s per request (−61.5%), prompt processing 126 → 303 t/s, generation 3.4 → 9.0 t/s, +958 MiB, outputs identical 3/3, verdict keep; about 50 min in all. **With nothing running**: a baseline at defaults fitted to the GPU (all layers, `-c 4096`), then `--cache-ram 0`: −4.7% wall, identical outputs, not adopted because nothing showed real traffic's prefix reuse (such steps are now skipped with that reason instead of run). **Smaller context from real traffic** (same observation and workload, one experiment, 3 arms): `ctx4096-ngl65` (context sized from the largest of 25 real requests, 528 tokens; all layers on the GPU, f16 KV) wall 17.2 → 2.0 s per request, outputs identical 3/3, listed under "Needs your decision" because it limits request length; `kvq8-ngl57` adopted (keep, −58.7%, identical 3/3) |
| `improve` with probes and final validation (0.3.0.dev0) | **Correction of the row above:** its "outputs identical 3/3" was evidence from three prompts. Re-run with top-10 probability probes and the final validation on six prompts spanning the real request sizes: the same changes (8-bit KV + 57 layers + `-ub 512`) are 17.2 → 6.6 s per request but change greedy outputs on 2/6 prompts, each at a near-tie (baseline margins 0.016 and 0.0005 nats; first divergence at tokens 10 and 47); each change alone also diverges on 1/6; the smaller-context option (17.2 → 2.1 s) on 2/6. A context-only control showed the context change itself is logprob-identical and worth -4%, the rest is the extra GPU layers. Nothing was recommended automatically; every option is listed with its validated numbers. From a fast launch (`-c 8192 -ngl 99 -ub 64`): `--cache-ram 0` validated alone, token-identical 6/6, -6%, recommended; `-ub 256` (-17%) and `--ctx-checkpoints 0` (-5%) changed outputs on 2/6 and 1/6 and were offered as decisions. Session resume reused all completed experiments (CLI and OpenCode). A wider fidelity study at full offload (10 prompts of 91-4,090 tokens, 128 generated): `-ub 128`, an 8-bit KV cache with flash attention, and flash attention off each changed greedy outputs on 3-5 of 10 prompts, every time at a near-tie (baseline margins 0.011-0.038 nats, never before token 30), with no speed gain at this configuration; the baseline was reproducible. The same positions flipped under different changes (three prompts at identical tokens and margins), so these are fragile points of the model on those prompts rather than effects of one change |
| MCP jobs and agent-run experiments (0.3.0.dev0) | scripted stdio client: active job returned a job id, a second active job refused, `hotpath_job_status` returned the finished result; `hotpath_job_cancel` and closing the client's stdin both left no `llama-server` running. Claude Code 2.1.296 and OpenCode 2.0.26 each ran the experiment above through MCP with the operator flag set (details in docs/agents.md) |

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

## Quickstart: let hotpath find and test the improvements

Work in a directory of your own (results are written relative to the current directory under `results/local/`, which is the default and is gitignored in this repository). Start your `llama-server` as you normally do; adding `--log-file PATH` and `--metrics` gives hotpath the most passive evidence.

```
hotpath init                   # discovers your setup and writes hotpath.local.json (passive)
hotpath improve --dry-run      # passive: diagnoses your server and shows the first experiment, its free-VRAM need and duration
hotpath improve                # ACTIVE: runs the experiments on hotpath's own llama-server and reports what improved
hotpath improve --requests captured.jsonl   # ACTIVE: the same loop measured on your application's own requests (recommended)
```

If you can capture the request bodies your application sends to `llama-server` (a JSONL file, one body per line, optionally `{"class": "<step name>", "body": {...}}`), pass them with `--requests`: every step and the final validation then run your real requests, with their system messages, sampling settings, JSON schemas or grammars and real output lengths, instead of synthetic prompts. See "Improving on your own requests" below.

`improve` runs the whole loop for you: it diagnoses the configuration, tests the top finding that has an executable experiment (one change against your configuration, interleaved rounds, the same workload throughout), adopts the change only on a `keep` verdict, diagnoses again on the adopted configuration, and repeats until no executable finding is left or the budget (`--max-experiments`, default 3; `--max-minutes`, default 60, checked against each step's estimate before it starts) runs out. It ends with the launch line to apply: your own command line with the adopted changes. It never changes your server.

A change is adopted only when all of these hold: the step's verdict is `keep` (faster than the configuration so far beyond each prompt's own repetition noise, in the same direction on most prompts, verified cache state, no launch failure, no request the server rejected); the outputs are identical and reproducible within the candidate itself (see below); no single prompt, and no request class of your application, got slower beyond its noise, so a gain on short prompts or on one kind of request cannot hide a loss elsewhere; the candidate did not spill more into system memory than the baseline and left VRAM headroom while it ran; and it does not change what your server accepts or what your application sends.

Every candidate the run measured ends in exactly one outcome, and the report opens with them:

| Outcome | Meaning |
|---|---|
| Recommended | faster, and the required correctness criteria held in a validation run of exactly this set of changes against your original launch |
| Needs review | faster, but with output differences, a context limit, a change to what your application sends, little VRAM headroom, or launch flags the experiments could not include; only you can accept it |
| Rejected | slower, a regression on some prompt or request class, failed launches or rejected requests, a validity drop, memory spill, or ruled out by real-traffic evidence |
| Inconclusive | no difference beyond the noise, a preliminary run (too few prompts or rounds), or not validated within the budget |

When the evidence does not decide, the report names the next useful measurement (more rounds, your own requests, a resume with a larger budget, the fields your application decides on) instead of a conclusion.

What it will not do on its own:

- **Recommend anything that only a step measured.** Every adopted change must survive a final validation: your original launch against the improved one, interleaved, on a broader workload than the steps used (six prompts from your smallest to your largest real request size, generating your longest observed output). If the validation is not a clean `keep` and more than one change was adopted, each change is then validated alone on the same workload (largest step gain first) while the budget allows, and the first one that passes is recommended; a faster combination that failed on output fidelity is listed under "Needs your decision" with its numbers. If nothing passes, or the budget runs out first, your original launch stays the recommendation. Step results are never multiplied together.
- **Treat a near-tie as harmless.** After an arm's timed requests, hotpath sends every prompt once more (untimed, because asking for probabilities slows generation) with the top-10 log-probabilities, and classifies each prompt: `logprob-identical`, `token-identical` (same tokens, shifted probabilities, with the largest shift reported), `diverged-near-tie` (the outputs differ from a token where the two choices were within 0.1 nats) or `diverged`, and reports how far a divergence carried (tokens generated after it and how much of the rest still lines up). A near-tie is evidence of numerical sensitivity, not of an acceptable output: on real structured requests a near-tie flip has changed numeric score fields of the JSON output (see "What output checks can and cannot establish"). Such arms go to "Needs review" like any other difference. The KL figure covers only the tokens both arms reported in their top 10; it is never a full-distribution comparison. If the baseline's own repeated outputs differ, identity is not judged at all; if the candidate's do, it is `candidate not reproducible`.
- **Shrink your context.** When the model does not fit, hotpath also tries a context sized from your real traffic (per slot, twice the largest observed request, a power of two, at least 4096, times `--parallel`), next to a control arm with the same context and your current GPU layers, so the report can say how much of the gain is the context change itself and how much the extra layers. It is never adopted on its own: the report states the cap, the largest request it saw, what happens to a longer request (an error, or silent truncation with `--context-shift`) and that the largest request a client may send is not in the evidence. It is not proposed at all when your traffic looks like growing conversations. Set `max_request_tokens` in `hotpath.local.json` to make your own limit binding; the arm is then sized from it and adoptable.
- **Adopt a cache flag without real-traffic evidence.** `--cache-ram 0` or `--ctx-checkpoints 0` trade prompt-cache reuse for less per-request overhead; they are tested only when your server's log or `/metrics` shows how much real traffic reuses, and adopted only when that reuse is low.
- **Mix setups.** When the server you started from ran a different model file or `llama-server` build than `hotpath.local.json` points to, `improve` refuses before launching anything.
- **Touch your server.** On one GPU your server holds the memory an experiment needs: `improve` refuses before launching anything and names the server. Stop it yourself and run `hotpath improve` again: it starts from the last observation of that server (its command line, log and `/metrics`), so nothing is lost. `--wait-s N` waits for a free GPU instead.

An interrupted run (a closed terminal, a cancelled agent job) is continued with `hotpath improve --resume improve-YYYYMMDD-HHMMSS`: experiments the session already recorded for the same spec are reused, not run again (the time budget applies to the resumed run; reused experiments cost nothing). A record is reused only when its spec fingerprint (launch arguments, environment, arms, sizes, cache mode and the workload's content) matches the step about to run; a record written for another starting point or workload is measured again.

The default output policy accepts no difference. An operator can state a weaker, explicit criterion in `hotpath.local.json` for runs on application requests: `"output_policy": "schema_valid"` accepts differing outputs when every judged request of the candidate passed full JSON Schema validation (and, when `decision_fields` is set, every one of those fields came out equal); `"output_policy": "decisions_equal"` additionally requires `"decision_fields"` (a list of top-level output fields, or `{"<request class>": [...]}`) and accepts differing outputs only when all of them are equal on every request. Neither says the outputs are equivalent, only that they met the operator's stated criteria; the report names the policy wherever it was used.

With no server running and no log, `improve` first measures a baseline on its own server: `server_args` from the config, `--args "..."`, or llama-server's usual flags with as many layers on the GPU as the GGUF header and free VRAM allow. Without a workload it uses synthetic prompts sized like your observed traffic (prompt and output lengths from the log or `/metrics`), else three default prompts. The report (`results/local/improvement-*.md`, with the same content as JSON next to it) opens with the result, then: what was found (the starting launch and the biggest bottleneck), what was tested and why each change was kept or not, what needs your decision, the validated end-to-end measurement with its output-fidelity class, the exact launch line to apply and the line to revert to, what could not be tested and what is still open. `--dry-run` shows the same "what was found" and the plan: the first experiment's arms, the free VRAM it needs and a rough duration, and the cost of the final validation.

The individual steps are also commands of their own:

```
hotpath inspect           # discover + observe the running server + diagnose (passive): ranked findings and the next step
hotpath propose results/local/diagnosis-<ts>.json --out experiments/next.json   # one-variable spec from the top finding, with its plan
hotpath experiment experiments/next.json                 # ACTIVE: launches hotpath's own servers on port 8099
hotpath diagnose results/local/experiment-<name>-<ts>.json   # verdict, and what remains on the faster configuration
```

`init` finds the running `llama-server` and takes its executable, model, log file and launch arguments from its command line (each value printed with where it came from); it also searches standard install locations, `PATH` and model caches, and any directory you name with `--root DIR`. When several binaries or models qualify it lists them instead of guessing; choose with `--binary`/`--model`/`--server`. `--update` adds newly discovered keys to an existing file without changing any value; `--template` writes the old placeholder file. Your running server is never added to `owned_servers`.

`init` also chooses among several installs when only one of them has a working CUDA backend and an NVIDIA GPU is present, and says which others it found.

`inspect` prints the setup, its problems with fixes, and the top findings, each with what is known, what is a hypothesis, the share of request time it can act on, and the experiment that would verify it. The full reports are written next to it.

On a single GPU your own server usually holds the VRAM an experiment needs. The plan states the free VRAM a launch needs, and the experiment refuses to start without it, naming the server that holds the GPU. hotpath never stops, restarts or reconfigures a server it did not start: stop it yourself, or use `--wait-s N`.

The rest of this section is the manual path and the individual commands.

**1. Configure by hand (optional).** `hotpath init --template` writes placeholders; the keys are `model_path`, `runtime_binary` (your `llama-server`), `runtime_bin_dir` (where `llama-bench` lives), `build_dir` (a llama.cpp build tree containing `test-backend-ops`), `server_url`, `server_log`, `server_args` and `path_prepend`. Every key is optional and each command says which one it needs. See `hotpath.local.example.json` for all keys. Keep this file out of version control; it holds your paths. `hotpath tools` confirms GPU, config and which external tools were found.

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

## What discovery looks at, and what it does not

`hotpath discover` (also run by `init` and `inspect`) is read-only and bounded. It looks at:

- **Running servers:** processes named `llama-server` and their command lines (model, port, `-ngl`, `-c`, `-ub`, `-fa`, KV types, cache flags, log file, ...), and on loopback only `GET /health`, `/props` and `/metrics`. Servers that hotpath itself launched for an experiment are recognized and never treated as yours.
- **Installations:** the running servers' executables, `runtime_binary` from the config, `PATH`, a short list of standard locations (winget, scoop, `~/llama.cpp/build/bin`, `/usr/local/bin`, Homebrew), and directories you name with `--root` (depth 4, an entry cap, no links followed; drive and filesystem roots are refused). Per installation: compute backends present, sibling tools, and on Windows the DLLs its executables import that Windows would not find. A missing CUDA runtime DLL makes `llama-server` exit with no output; discovery names it and the toolkit directory that has it, and confirms the fix by running `--version` with that directory on `PATH`. `--version` is the only program discovery runs (`--no-exec` turns it off).
- **Models:** GGUF headers only (architecture, quantization, layers, size, whether the weights fit in VRAM, projector/embedding roles) next to the running server's model, in the llama.cpp and Hugging Face caches, LM Studio's folders, `~/models`, and directories you name.
- **GPU and tools:** every NVIDIA GPU, the driver's CUDA version (and whether a build needs a newer one), Nsight Systems/Compute, `llama-bench`, `test-backend-ops`, the optional `jsonschema`.
- **Launch scripts** in directories you name that start `llama-server` (batch/shell files, llama-swap style YAML): their command lines as written.

The record lists where it looked and what it executed. What it cannot see is said explicitly: settings given through `LLAMA_ARG_*` environment variables or a launcher's defaults are not in a command line (`unset` means "not on the command line", not "off"), and environment variables of another process are not read.

## Already running a server? Diagnose it passively, no benchmark needed

```
hotpath inspect                                          # finds the running server, its launch, log and /metrics
hotpath observe --log path/to/llama-server.log           # or: just a log file, no server needed
hotpath diagnose results/local/observation-*.json
```

`observe` records the launch settings of the server it observes from the process command line, picks up that server's `--log-file` and model when they are not configured, and reads `/metrics` (aggregate prompt/generation time and prompt-cache reuse since the server started) when the server runs with `--metrics`. Note that at default verbosity current builds do not print launch facts such as `offloaded N/M layers` or `n_ubatch` into the log; the command line is then the evidence, and hotpath labels it so.

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

What it looks at today: how many of the model's layers are on the GPU (the log's `offloaded N/M layers` line, an explicit `-ngl` on the command line, or the server's VRAM against the model's weights), with a stated estimate of whether all layers would fit, and, when no more fit with an f16 KV cache, whether a context sized from your real traffic, or an 8-bit KV cache (with flash attention), frees enough memory for at least two more layers (each proposed as one arm that changes both settings, labelled as such); the per-request split into prompt processing, generation and everything else (prefill-bound vs decode-bound vs overhead), from client wall times, a server log, or `/metrics` totals; the residual per-request overhead and the server flags that affect it; the microbatch size against the prefill share; decode against the memory-bandwidth ceiling for your model and GPU, and whether CUDA graphs are disabled; the served prefill/decode against `llama-bench` on the same model (the gap is server-side work); prompt-cache reuse of real traffic (log or `/metrics`, and only when there is enough traffic to judge); VRAM headroom and system-memory spill signs; BF16 matrices in the model on a GPU without BF16 tensor cores (compute capability below 8.0), which stays a hypothesis until a prefill profile shows the fallback GEMM; kernel-class shares from a profile (re-classified from kernel names, so older records benefit from newer classes); and, for finished experiments, a per-arm verdict that combines paired per-prompt ratios judged against each prompt's own repetition noise, the verified cache state of both arms, the VRAM delta, output identity and output validity. It ends with the list of things the given records cannot establish.

**Ranking and wording.** Problems are ranked by the share of request time the candidate change can act on (for example, prefill + decode for a partial offload, the residual for cache flags), taken from the measured shares and weighted by confidence. The share is an upper bound on what the change can touch, not a predicted gain. Problems whose share is not measurable follow, then context findings. Each finding separates what is `known` (with its source: log, command line, `/metrics`, a measurement) from the `hypothesis` it suggests, lists what `could regress`, names the experiment that would verify it, and says whether it has been `measured`. A finding that another one supersedes (a decode-ceiling gap while part of the model runs on the CPU) is marked so and not recommended.

**The improve step.** After an experiment, `diagnose` reports the verdict for the replaced configuration and then describes what remains on the candidate the verdict found faster (labelled, and with the caveat when the verdict also reports output differences). `propose` on that diagnosis starts the next spec from the faster configuration, so a second change is tested on top of the first instead of against the old baseline.

`propose` keeps the comparison fair by construction: one variable differs between arms; the baseline arm is your current configuration (the arm the last verdict found faster, else the diagnosed server's observed command line, else `server_args` in the local config, else the diagnosed experiment's own arguments; `--common-args` overrides all of them); argument spellings are matched (`-ngl` and `--gpu-layers` are one argument, so exactly one value reaches the server); binary and model are `${...}` references into `hotpath.local.json`; rounds are interleaved by the runner. The plan states the requests it will send, the free VRAM a launch needs (from the GGUF header), and a rough duration from the diagnosed configuration's own rates. Without `--workload` (or `workload` in the config), `propose` writes synthetic prompts sized like the real traffic in the diagnosed log or measurement (p10, median and p90 prompt tokens, cached tokens included, and the median output length), so the experiment measures the request shape you actually serve; with no traffic in the evidence it falls back to three prompts of 450/900/1300 words. Server arguments the runner supplies itself or that would share your own server's resources (`--host`, `--port`, `--model`, `--log-file`, `--slot-save-path`, `--api-key`, ...) are left out of the copied baseline, and the command says which; an experiment server therefore cannot take over your server's port or write to its log. It prints the dry-run plan and never launches anything.

Diagnoses are not results. A `hypothesis` finding is an argument for an experiment; the experiment's verdict is what you act on. Verdicts are paired per prompt (each prompt's candidate/baseline ratio against that prompt's own repetition noise), state the workload size (`N prompts x R rounds`, marked preliminary below 3 x 3), judge output identity, output validity and speed separately, and describe this machine and workload only. Experiments that trade prefix reuse for per-request overhead (`--cache-ram`, `--ctx-checkpoints`) carry a production gate: without an `observe --log` record showing how much real traffic reuses, the verdict says applicability is unknown; with high real reuse it says not to apply.

## Improving on your own requests

`hotpath improve --requests captured.jsonl` (MCP: `hotpath_improve` with `requests`) builds a workload from your application's request bodies (`workloads/<name>.requests.json`; it holds your prompts, keep it local) and runs the whole loop on it:

- **Request classes.** Each request belongs to a class: the label your capture gives it (`class` or `stage`), else one derived from its endpoint, the JSON schema's name (or grammar / free text) and a hash of the system message. Verdicts report wall time, prefill and decode per class, and a change that makes one class slower beyond that class's noise is not adopted, however much the others gain.
- **Steps and validation on real requests.** Each step runs, per class, the median and the largest request (`--step-requests`, default 2); the final validation runs a broader set, per class from the smallest to the largest (`--validation-requests`, default 4), mostly requests the steps never ran, plus each class's largest. Both are lists of request ids kept in the session, so a resumed run measures the same requests and request bodies are never copied out of the workload file.
- **What is compared.** Output identity (content hashes), untimed top-k probes of every greedy request (chat `logprobs`/`top_logprobs`, completion `n_probs`; the probe tokens stay in memory and only the comparison is recorded), JSON validity against each request's own schema, and the output's schema fields one by one (value hashes per declared property, with the property's kind: enum/boolean, number, text, structure). Enum, boolean and number fields are reported as decision candidates; set `decision_fields` to say which fields your application actually decides on.
- **Sampling.** Requests with a temperature above 0 are not judged on identity (their outputs are not expected to repeat); validity and decision fields are their evidence, and without an operator policy such a change needs review.
- **Requests the server rejects** (a request larger than a candidate's context, say) are failed requests of that arm with the error type, and a candidate that cannot serve requests the baseline served is rejected.
- **Cache.** `--cache-mode cold` (default) processes every request in full, so each change is measured on full prompt processing; `--cache-mode warm` replays your requests in your application's order with the prompt cache as they send it. The report says which.

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

`hotpath mcp` (also installed as `hotpath-mcp`) serves the commands as MCP tools over local stdio, with active tools (experiments, benchmarks, profiling) disabled unless you start it with `HOTPATH_MCP_ALLOW_ACTIVE=1`. `skills/hotpath/SKILL.md` is a portable Agent Skill that teaches an agent the workflow. Both reuse the CLI. An agent starts with `hotpath_inspect`; when you ask it to improve your setup it shows `hotpath_improve_plan` (passive) and, with your agreement, runs `hotpath_improve` (the whole loop as one job). Active tools run as jobs that return within 45 s (a result or a job id to poll with `hotpath_job_status`), because some clients are reported to cancel a tool call after about a minute while an experiment takes many. Claude Code and OpenCode have each run an experiment end to end through MCP on one Windows machine; other clients are untested. Details and the exact validation status are in [docs/agents.md](docs/agents.md).

## Other commands

| Command | What it does |
|---|---|
| `hotpath discover [--root DIR]` | the local setup: running servers and their launch settings, installs and whether they start, GGUF models, GPU, tools, logs, problems with fixes (passive) |
| `hotpath init [--binary B] [--model M] [--server URL] [--update]` | write `hotpath.local.json` from discovery, each value with its source |
| `hotpath inspect` | passive, one step: discover, observe the running server (command line, log, `/metrics`, GPU), diagnose, and print the next step |
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
- **Top-k probes.** Untimed probes with the top-10 log-probabilities give, per prompt, the largest log-probability shift before any divergence, the margin of both arms at the first divergent token, a KL over the shared top-k only, and how far a divergence carried.
- **Isolated kernel correctness.** `hotpath kernels test` runs `test-backend-ops` and reports how many cases pass the runtime's own NMSE threshold against the CPU reference, plus failures and unsupported cases.
- **Not included.** hotpath does not compute perplexity or a full-distribution divergence. For those use `llama-perplexity` or your own evaluation.

### What output checks can and cannot establish

hotpath keeps four questions apart and never answers one with another:

| Question | Evidence hotpath has | What it does not show |
|---|---|---|
| Numerical similarity | log-probability shifts and top-k KL before any divergence; the margin at the first divergent token | that a different continuation is acceptable |
| Baseline instability | the baseline's own repetitions and separate launches (identity is not judged when they differ); a control arm of the same configuration in a study | whether a candidate's divergence is "only" instability |
| Output equivalence | identical token ids or content on every compared request, reproducible within each arm | that other prompts behave the same |
| Application-level correctness | JSON validity against the request's schema; the output's schema fields and the operator's `decision_fields` compared one by one | anything your application checks beyond those fields |

A near-tie at the first divergence shows the change moved a numerically close decision. It does not show the resulting output is equivalent: the continuation from that token on is generated from a different prefix, and on structured outputs it can change fields the application reads. hotpath's default therefore accepts identical outputs only; anything else is "Needs review" unless the operator configured, and hotpath checked, a policy.

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
- Runs on your own requests (`--requests`): the workload file under `workloads/` holds your request bodies. Records and session files hold request ids, body hashes, token counts, timings, validation results, per-field value hashes of structured outputs and probe comparisons (divergence positions, margins); never prompt or output text, and never the probe tokens of your outputs (they are compared in memory and dropped). Request class labels (your capture's labels, or JSON schema names) appear in records and reports.
- Secret arguments (`--api-key`, `--hf-token`) and environment variables whose names contain KEY, TOKEN, SECRET or PASSWORD are recorded as `<redacted>`.
- Records store file **basenames**, hashes of model/binary heads, GPU facts, `GGML_*`/`LLAMA_*`/`CUDA_*` environment variable values (these can include install paths such as `CUDA_PATH`), the OS version string, and a short hash of the hostname. Model, binary, workload and path-like command-line arguments are reduced to basenames. Records do not store prompt text (unless `--store-prompts`) or generated text (unless a spec sets `"store_outputs": true`). They do store generated **token ids** for hotpath's own workloads (needed for the output-equivalence check); token ids can be decoded with the model's tokenizer, so treat them as content when the prompts are yours. Application-shaped (`requests`) workloads store only a content hash and length of each output, plus validation results (keywords and masked paths, never output text). 0.1 stored generated text by default; see "Upgrading from 0.1".
- Experiments also write the raw stderr/stdout of each launched server next to the results (`server-*.log`). Those logs contain full paths and are **not** redacted. Do not publish them without reading them.
- Profiler captures (`nsys-*.nsys-rep`, `.sqlite`) are written next to the results and contain raw paths and process details. Treat them as private. Text in records (tool output, command lines) is passed through a path scrubber, but it is best effort: read a record before sharing it.
- Review anything you intend to share. `hotpath.local.json` is for your machine only.

## Limitations

- Only `llama.cpp`-family servers are supported (the HTTP API, log format and `--version` output are what the parsers expect). Log parsing is regex-based and can break when upstream changes its log lines.
- NVIDIA only. The built-in bandwidth table lists published peak memory bandwidth for a handful of GPUs; only the RTX 2080 SUPER entry has been checked against a real run. Use `--bandwidth-gbs` to override.
- The experiment runner launches whole servers per arm, which is slow for large models but isolates arms properly.
- MCP server and agent skill have been validated in Claude Code and OpenCode sessions on one machine only; other clients are untested (see docs/agents.md).
- Discovery was exercised on Windows only. Its Linux and macOS paths (`/proc` and `ps` for processes, `ldd` for libraries, standard install and cache locations) are written but untested. Process command lines of other users may be unreadable; environment variables of a running server (`LLAMA_ARG_*`) are not read at all, so a setting given that way appears as `unset`.
- The offload rule's memory estimate (weights + KV cache at the configured context and types + the compute buffer, against free VRAM) uses the log's buffer sizes when printed and otherwise stated allowances; it only chooses the candidate arm. The experiment's per-arm dedicated and shared VRAM is the evidence.
- The duration estimate in a plan uses the diagnosed configuration's rates for every arm and an assumed model-load time; in the validation run it was 16 min for a run that took 10 (the candidate was faster).
- MCP jobs live as long as the MCP server: they are stopped when the client exits and their state does not survive a restart (their output stays in `results/local/jobs/`). One active job runs at a time.
- The test suite covers the diagnosis and proposal logic on synthetic record structures (`python -m unittest discover -s tests`); the commands that drive servers and profilers are verified by running the actual workflows (see CONTRIBUTING).
- `diagnose` reasons from the measurements hotpath itself collects. Its rules are relative (shares, ratios, fractions of a ceiling) and were checked on one GPU/model/runtime; a finding is only as good as the records behind it, and it says `insufficient` rather than guess. From a server log alone it cannot see client wall time, per-request overhead, or GPU state during requests; prefix-cache effectiveness comes from a log's slot-selection lines or the `/metrics` aggregate (hotpath's own workloads disable the cache on purpose); nothing is known about kernels until a profile has been run.
- Log parsing recognizes llama-server's per-request timing, slot-selection and release lines; builds that log differently are reported as a format problem. The threshold that separates CPU-fallback sessions (`log_cpu_mode_prefill_tps_below`, default 10 t/s) is a heuristic you can change in the local config.
- `propose` covers the experiment kinds `diagnose` emits as arm lists (GPU layers, microbatch sweep, CUDA-graph toggle when graphs are disabled, prompt-cache/checkpoint flags, request shape); findings whose confirmation is a procedure (a profile, a launch check, freeing VRAM) are described, not generated.
- Field-by-field comparison covers the top-level properties a request's JSON schema declares; nested fields count as one `structure` value. The divergence carry measure (`rest_similarity`) is a token-sequence measure, not a semantic one. hotpath cannot tell which fields your application decides on unless `decision_fields` says so; enum/boolean/number fields are only reported as candidates.
- Runs on your own requests are sequential (one request at a time); concurrency with `--parallel` greater than 1 is not exercised.
- `requests` workloads send top-level body fields as given; `request_override` edits top-level fields only. Output identity on the chat endpoint is a content-hash comparison. Full schema validation needs the optional `jsonschema` package; schemas using external `$ref`s or formats the validator does not assert are checked only as far as the validator goes. Grammar-constrained (`grammar` GBNF) outputs are not validated.
- Cache state is verified from the server's `cache_n` and nothing else; a server that does not report it yields `unknown` and no prefill-side verdict. `cold` disables reuse entirely (including between consecutive items), so it measures cold processing, not the reuse your application would get; use `warm` for that. Cold mode overrides `cache_prompt` in your request bodies.
- Spec files written by an agent are checked before an MCP-launched experiment: `binary`/`model`/`path_prepend` must be `${key}` references into `hotpath.local.json`, environment overrides are limited to `GGML_`/`LLAMA_`/`CUDA_` variables, and server arguments that write files, change the network binding, download a model or change which model loads (`--log-file`, `--host`, `--port`, `--model`, `--hf-repo`, `--lora`, ...) are refused. Arguments not on that list are passed to `llama-server` as written, so only run specs you would run yourself.
- `hotpath profile` and `hotpath kernels` are thin wrappers: they run the underlying tool and summarize its output; `diagnose` ranks a profile's kernel classes but does not interpret individual kernels.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `nvidia-smi found=False` | Driver not installed or not on `PATH`. Facts and the idle-GPU precondition need it. |
| `llama-bench found=False` / `test-backend-ops found=False` | Set `runtime_bin_dir` / `build_dir` in `hotpath.local.json`, or put them on `PATH`. |
| `nsys found=False` | Install Nsight Systems or the CUDA toolkit, or put `nsys` on `PATH`. |
| Server exits immediately in an experiment, or `llama-server` prints nothing | Run `hotpath discover --root <its directory>`: it lists the DLLs the build imports that Windows cannot find and the CUDA toolkit directory that has them (verified with `--version`); `hotpath init --update` writes it as `path_prepend`. Otherwise open `results/local/server-<name>-<arm>-r0.log`. |
| `preconditions not met: free VRAM ... Running now: llama-server pid N` | Your own server holds the GPU memory the experiment needs. hotpath will not stop it; stop it yourself (or use `--wait-s` to wait). |
| `preconditions not met` (other) | Something else is using the GPU or the port. Free it, or use `--wait-s`. |
| `init` leaves `runtime_binary` or `model_path` unset | Several candidates were found (they are listed) or none: choose with `--binary`/`--model`, or search with `--root DIR`. |
| `refusing to submit inference ... not in owned_servers` | `measure` only targets servers you declare as yours (`owned_servers` or `--i-own-this-server`). |
| Every output in `compare` diverges | Check that `temperature` is 0 and that arms are not reordering sums on purpose (different kernels, batch sizes); one near-tie flip is common and not necessarily a bug. |
| `error: ... (set HOTPATH_DEBUG=1 for a traceback)` | Set that variable to see the full traceback when reporting a bug. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with `hotpath tools` output, OS, GPU, runtime build and the exact command are the most useful contribution right now, along with reports from setups listed above as not validated.

## License

MIT, see [LICENSE](LICENSE). hotpath does not bundle or redistribute llama.cpp, Nsight or any model; those are separate projects with their own licenses.
