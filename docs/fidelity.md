# What output checks can establish: a study on real requests

hotpath compares a candidate configuration's outputs with the baseline's on every compared request (greedy token
identity, untimed top-10 log-probability probes, JSON validity, and the output's schema fields one by one). This page
records a study of what those checks can and cannot demonstrate, run to decide how hotpath treats outputs that differ
at a "near-tie" (a token where the baseline's and the candidate's choices were within 0.1 nats of each other).

Hypothesis examined: some greedy divergences between configurations are not effects of the change but intrinsically
fragile positions of the model on those prompts, so a near-tie divergence might be harmless.

## Setup

- One NVIDIA RTX 2080 SUPER (8 GB), Windows 11, a llama.cpp-derived CUDA `llama-server` build, a 27B ternary GGUF fully
  offloaded (`-c 8192 -ngl 99 -b 256 -ub 256`), the application's own production launch.
- 24 real request bodies of one application (8 from each of its three request classes, a scoring step, a draft step and a
  critic step, from the smallest to the largest): chat completions with a system message, a JSON schema, temperature 0 and
  a fixed seed; prompts of 842-2,621 tokens, outputs of 71-160 tokens (each stopping at the end of its JSON object).
- One launch per arm, cold prompt cache (every request fully processed), probes in the same launch as the timed requests.
  Every arm is compared with the production launch.

| Arm | What it changes | Numerically |
|---|---|---|
| production again | nothing (a separate launch of the same configuration) | identical computation |
| CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`) | how kernels are launched | identical computation |
| `-ub 128`, `-ub 512 -b 512` | the microbatch, so the tiling and reduction order of prompt processing | mathematically equivalent, different rounding |
| flash attention off | the attention kernel | mathematically equivalent, different rounding |
| 8-bit KV cache (`-ctk q8_0 -ctv q8_0`) | the precision of the stored keys and values | lossy |

## Results

| Arm | Wall time | Requests whose output diverged | at a near-tie | Outputs with a changed enum/number field | any schema field | JSON valid |
|---|---|---|---|---|---|---|
| production again | +0.8% | 0/24 (log-probabilities bit-identical) | - | 0 | 0 | 24/24 |
| CUDA graphs off | +3.3% | 0/24 (log-probabilities bit-identical) | - | 0 | 0 | 24/24 |
| `-ub 128` | +7.1% | 9/24 | 9 | 2 | 9 | 24/24 |
| `-ub 512` | −2.0% | 9/24 | 8 | 2 | 8 | 24/24 |
| flash attention off | +6.2% | 6/24 | 6 | 1 | 6 | 24/24 |
| 8-bit KV cache | +2.4% | 8/24 | 8 | 2 | 8 | 24/24 |

- **Reproducibility.** The same configuration in a separate launch, and the same computation launched without CUDA
  graphs, gave bit-identical top-10 log-probabilities on all 24 requests. The baseline is not noisy; every divergence below
  comes from a change in the computation.
- **Mathematically equivalent changes flip outputs.** A different microbatch or attention kernel changed the greedy output
  of 25-38% of these real requests. At the first divergent token the baseline's margin was 0.0015-0.068 nats; on requests
  whose tokens stayed identical, log-probabilities still moved by a median 0.19-0.28 nats (up to 1.3).
- **Fragile positions are shared.** Seven (request, token) positions diverged under two or more different changes with the
  same baseline margin; one scoring request diverged at the same token (margin 0.0015 nats) under all four computation
  changes. Fragility is a property of the model on that prompt.
- **A lossy change looks the same.** The 8-bit KV cache diverged on 8/24 requests, all at near-ties, at positions shared
  with the equivalent changes. By the near-tie measure it cannot be told apart from a different microbatch.
- **Divergence cascades.** After the first divergent token, between 15% and 97% of the remaining tokens still lined up
  (median 0.74): some continuations re-converge, others become a different answer.
- **Structured outputs change beyond text.** 31 of the 32 divergences changed at least one schema field (structure or
  free text), and in 1-2 of 24 outputs per change an enum or number field changed: for example, two numeric risk scores of a
  critic request. JSON validity never dropped.
- **Length.** Requests with longer outputs (114-160 tokens) diverged three times as often as those with shorter outputs
  (71-109 tokens): 24 against 8 divergences over 72 arm-request pairs each. Prompt length showed no clear effect (18 against 14).

**Cache state.** Replayed warm (the 24 requests in the application's order, prompt cache on, so consecutive requests reuse
the shared system prompt: a median 126 cached tokens, 10% of prompt tokens), the production launch gave the same outputs
as cold on 24/24 requests, and a second warm launch was logprob-identical to the first. At this configuration prefix
reuse did not change any output. `--cache-ram 0` in the same warm replay: −2.8% wall, logprob-identical, and no reuse lost
(the reuse comes from the slot itself, not the host cache), which agrees with the cold measurement `improve` validated
(one launch per arm, so preliminary).

The same pattern appeared in an `improve` run from a partial offload: 8-bit KV + 58 GPU layers diverged on 3 of 6 real
requests at near-ties (margins 0.005-0.047 nats), and on one critic request two numeric risk scores differed.

## What this establishes, and what it does not

| Question | Established by this evidence? |
|---|---|
| Is the baseline reproducible? | yes, at this configuration: bit-identical across launches and execution paths, warm or cold |
| Is a near-tie divergence caused by numerical sensitivity? | yes: equivalent reorderings produce the same kind of divergence at the same positions |
| Does a near-tie mean the candidate is numerically harmless? | **no**: a lossy change produced indistinguishable near-tie divergences |
| Does a near-tie mean the outputs are equivalent? | **no**: continuations diverge, and decision-like fields changed |
| Is the application's behavior preserved? | only as far as validity and the compared fields show; the fields an application acts on can change |

A calibration arm (a neutral perturbation next to the candidate) can therefore mark a divergence as "this request is
fragile here", but not as acceptable. hotpath keeps identical outputs as its only automatic pass. A divergent option is
"Needs review" with its evidence: the margin, how far the divergence carried, and which schema fields changed. An operator
who knows which fields their application decides on can say so (`decision_fields`) and opt into
`"output_policy": "decisions_equal"`, which accepts a difference only when every one of those fields is equal on every
compared request and the outputs pass full schema validation. That policy is a statement about the operator's
application, not a claim of equivalence, and the report names it wherever it is used.

## Limits of the study

One GPU, one runtime build, one model, one application, 24 requests and one launch per arm; greedy decoding only; the
warm replay used requests spread across sizes rather than a contiguous slice of the application's sequence; sequential
requests (no concurrent batching). Field comparisons cover the top-level schema
properties; "remaining tokens that line up" is a token-sequence measure, not a semantic one.
