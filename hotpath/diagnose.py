"""hotpath diagnose: read the evidence hotpath already collects and say what is limiting this workload.

Input: any mix of hotpath records (observation, measurement, experiment, bench, profile, kernels, triage).
Output: a ranked list of findings. Every finding states

  - what was measured (numbers copied from the records, with their evidence class),
  - how confident the diagnosis is:
        confirmed    - the records contain a controlled comparison that shows it
        likely       - direct measurements point at it, but no controlled comparison exists
        hypothesis   - consistent with the evidence; a specific experiment would decide
        insufficient - the records cannot establish it; says what to collect
  - why it matters for end-to-end latency,
  - an opportunity bound when it can be derived from measured shares (never a promised speedup),
  - the controlled experiment that would confirm or reject it (consumable by `hotpath propose`),
  - risks and trade-offs of the candidate change.

Nothing here is specific to one GPU, model or runtime build: thresholds are relative (shares of request
time, fractions of a bandwidth ceiling, ratios between two of the user's own measurements), and
every rule names the measurement it rests on.
"""
import struct
from pathlib import Path

from hotpath import gguf as gguf_mod
from hotpath.records import percentiles

# relative thresholds; each is used in exactly one rule and documented there
OVERHEAD_SHARE_WORTH_NOTING = 0.05      # residual (wall - prefill - decode) share of a request
SMALL_UBATCH = 256                      # below this, batched prefill is usually still gaining from a larger microbatch
BENCH_GAP_WORTH_NOTING = 0.85           # server prefill (or decode) below this fraction of llama-bench's number
DECODE_CEILING_LOW = 0.35               # achieved decode below this fraction of the bandwidth ceiling
DECODE_CEILING_NEAR = 0.65              # above this, kernel-level work is what remains
VRAM_USED_HIGH = 0.92                   # fraction of total VRAM in use
PROCESS_SHARED_MIB_HIGH = 256           # per-process shared (system-memory) GPU allocations
MIN_REQUESTS_FOR_SPLIT = 3


def _med(p):
    return (p or {}).get("median")


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _args_value(args, flag, default=None):
    """Value following a flag in a recorded argv list (several spellings accepted)."""
    if not args:
        return default
    flags = {flag} | {{"--ubatch-size": "-ub", "--batch-size": "-b", "--ctx-size": "-c", "--gpu-layers": "-ngl", "--parallel": "-np"}.get(flag, flag)}
    for i, a in enumerate(args[:-1]):
        if a in flags:
            return args[i + 1]
    return default


# ---- evidence collection -----------------------------------------------------------------------

def collect(records):
    """Flatten records into one evidence set. Nothing is interpreted here."""
    ev = {"arms": {}, "gpu": {}, "model": None, "bench": [], "profiles": [], "log_stats": None, "launch": {},
          "classes": set(), "experiments": [], "sources": [], "server_props": None}
    for rec in records:
        kind = rec.get("kind")
        ev["sources"].append("%s(%s)" % (kind, rec.get("name") or rec.get("label") or rec.get("created")))
        if rec.get("evidence_class"):
            ev["classes"].add(rec["evidence_class"])
        if rec.get("gpu") and not ev["gpu"]:
            ev["gpu"] = rec["gpu"]
        if kind == "experiment":
            ev["experiments"].append(rec)
            for name, arm in rec["arms"].items():
                args = arm.get("args") or []
                lf = arm.get("launch_facts") or []
                ev["arms"]["%s/%s" % (rec["name"], name)] = {
                    "requests": arm.get("requests") or [], "evidence_class": "controlled", "experiment": rec["name"], "arm": name,
                    "settings": {"n_ubatch": _f(_args_value(args, "--ubatch-size")), "n_batch": _f(_args_value(args, "--batch-size")),
                                 "n_ctx": _f(_args_value(args, "--ctx-size")), "n_parallel": _f(_args_value(args, "--parallel"), 1),
                                 "cache_ram": _args_value(args, "--cache-ram"), "ctx_checkpoints": _args_value(args, "--ctx-checkpoints"),
                                 "env_ggml": arm.get("env_ggml") or {}},
                    "vram": {"process_dedicated_mib": [x.get("process_counters_after", {}).get("dedicated_mib") for x in lf],
                             "process_shared_mib": [x.get("process_counters_after", {}).get("shared_mib") for x in lf],
                             "gpu_after": [x.get("gpu_after") for x in lf]},
                    "binary": arm.get("binary") or rec.get("binary"), "failures": arm.get("failures") or [], "args": args}
            if not ev["model"] and rec.get("model") and rec["model"].get("summary"):
                ev["model"] = rec["model"]["summary"]
        elif kind == "measurement":
            props = (rec.get("target") or {}).get("props") or {}
            ev["arms"][rec.get("label") or rec["created"]] = {
                "requests": rec.get("requests") or [], "evidence_class": "active", "settings": {"n_ctx": _f(props.get("n_ctx_per_slot")), "n_parallel": _f(props.get("total_slots"), 1), "env_ggml": {}},
                "vram": {"gpu_after": [s.get("gpu") for s in rec.get("gpu_snapshots") or []],
                         "process_dedicated_mib": [(s.get("process") or {}).get("dedicated_mib") for s in rec.get("gpu_snapshots") or []],
                         "process_shared_mib": [(s.get("process") or {}).get("shared_mib") for s in rec.get("gpu_snapshots") or []]},
                "failures": []}
            ev["server_props"] = ev["server_props"] or props
        elif kind == "observation":
            ev["log_stats"] = (rec.get("log") or {}).get("gpu_like_sessions_stats") or ev["log_stats"]
            ev["launch"] = {"launcher_settings": rec.get("launcher_settings") or {}, "launcher_env": rec.get("launcher_env_documented") or {},
                            "session_table": (rec.get("log") or {}).get("session_table") or [], "gpu_now": rec.get("gpu_now") or {},
                            "server_process": rec.get("server_process") or {}}
            ev["server_props"] = ev["server_props"] or ((rec.get("server") or {}).get("props"))
            if (rec.get("model") or {}).get("summary"):
                ev["model"] = rec["model"]["summary"]
        elif kind == "bench":
            parsed = rec.get("parsed") or []
            for row in parsed if isinstance(parsed, list) else []:
                ev["bench"].append({"n_prompt": row.get("n_prompt"), "n_gen": row.get("n_gen"), "n_ubatch": row.get("n_ubatch"), "n_batch": row.get("n_batch"),
                                    "avg_ts": row.get("avg_ts"), "stddev_ts": row.get("stddev_ts"), "model": row.get("model_filename") or row.get("model_type")})
        elif kind == "profile" and rec.get("kernels"):
            ev["profiles"].append({"label": rec.get("label"), "mode": rec.get("mode") or {}, "by_class": rec.get("time_share_by_class_pct") or {},
                                   "kernels": rec.get("kernels") or [], "kernel_count": rec.get("kernel_count")})
    return ev


def phase_split(requests):
    """Per-request shares of prompt processing, generation and the residual (client wall minus both)."""
    rows = []
    for r in requests:
        w, p, g = _f(r.get("wall_ms")), _f(r.get("prompt_ms")), _f(r.get("predicted_ms"))
        if not w or p is None or g is None or w <= 0:
            continue
        rows.append({"prefill": p / w, "decode": g / w, "residual": max(0.0, w - p - g) / w, "residual_ms": max(0.0, w - p - g),
                     "prompt_n": r.get("prompt_n"), "predicted_n": r.get("predicted_n"), "wall_ms": w})
    if len(rows) < MIN_REQUESTS_FOR_SPLIT:
        return None
    return {"n": len(rows),
            "prefill_share": percentiles([x["prefill"] for x in rows]), "decode_share": percentiles([x["decode"] for x in rows]),
            "residual_share": percentiles([x["residual"] for x in rows]), "residual_ms": percentiles([x["residual_ms"] for x in rows]),
            "prompt_n": percentiles([x["prompt_n"] for x in rows]), "predicted_n": percentiles([x["predicted_n"] for x in rows]),
            "wall_ms": percentiles([x["wall_ms"] for x in rows])}


def _baseline_arm(ev):
    """The arm a diagnosis describes: the first arm of the newest experiment, else the newest measurement."""
    if ev["experiments"]:
        rec = ev["experiments"][-1]
        first = next(iter(rec["arms"]))
        return "%s/%s" % (rec["name"], first), ev["arms"]["%s/%s" % (rec["name"], first)]
    if ev["arms"]:
        k = list(ev["arms"])[-1]
        return k, ev["arms"][k]
    return None, None


def _finding(fid, title, severity, confidence, evidence_class, measurements, why, experiment=None, opportunity=None, risks=None, notes=None):
    f = {"id": fid, "title": title, "severity": severity, "confidence": confidence, "evidence_class": evidence_class,
         "measurements": measurements, "why_it_matters": why}
    if opportunity:
        f["opportunity"] = opportunity
    if experiment:
        f["experiment"] = experiment
    if risks:
        f["risks"] = risks
    if notes:
        f["notes"] = notes
    return f


# ---- rules --------------------------------------------------------------------------------------

def rule_phase(ev, label, arm):
    sp = phase_split(arm["requests"]) if arm else None
    if not sp:
        return None
    pf, dc, rs = _med(sp["prefill_share"]), _med(sp["decode_share"]), _med(sp["residual_share"])
    if pf >= 0.6:
        shape, why = "prefill-bound", "most of each request is prompt processing; generation-side changes cannot move most of the latency"
    elif dc >= 0.6:
        shape, why = "decode-bound", "most of each request is token generation; prompt-side changes cannot move most of the latency"
    else:
        shape, why = "mixed", "prompt processing and generation each take a substantial part of the request; both sides matter"
    return _finding("workload_phase_split", "Workload is %s (prefill %.0f%%, decode %.0f%%, other %.0f%% of request wall time)" % (shape, 100 * pf, 100 * dc, 100 * rs),
                    "info", "confirmed" if sp["n"] >= 6 else "likely", arm["evidence_class"],
                    {"arm": label, "requests": sp["n"], "prompt_tokens_median": _med(sp["prompt_n"]), "generated_tokens_median": _med(sp["predicted_n"]),
                     "wall_ms_median": _med(sp["wall_ms"]), "prefill_share_median": round(pf, 3), "decode_share_median": round(dc, 3), "residual_share_median": round(rs, 3),
                     "shares_p10_p90": {"prefill": [sp["prefill_share"]["p10"], sp["prefill_share"]["p90"]], "decode": [sp["decode_share"]["p10"], sp["decode_share"]["p90"]]}},
                    why, notes=["Shares are computed per request from the server's own prompt/generation timings and the client wall time, then summarized; "
                                "they describe this workload's prompt and generation lengths, not the model in general."])


def rule_residual(ev, label, arm):
    sp = phase_split(arm["requests"]) if arm else None
    if not sp:
        return None
    rs, rms = _med(sp["residual_share"]), _med(sp["residual_ms"])
    if rs is None or rs < OVERHEAD_SHARE_WORTH_NOTING:
        return None
    s = arm.get("settings") or {}
    return _finding("per_request_overhead", "%.0f ms per request (%.0f%%) is spent outside prompt processing and generation" % (rms, 100 * rs),
                    "medium" if rs < 0.15 else "high", "likely", arm["evidence_class"],
                    {"arm": label, "residual_ms_median": round(rms), "residual_ms_p10_p90": [sp["residual_ms"]["p10"], sp["residual_ms"]["p90"]], "residual_share_median": round(rs, 3),
                     "cache_ram_flag": s.get("cache_ram"), "ctx_checkpoints_flag": s.get("ctx_checkpoints")},
                    "This time is paid on every request regardless of prompt or output length: HTTP and JSON handling, tokenization and template rendering, sampler setup, "
                    "and in llama-server the prompt-cache state save and context checkpoints, which copy sequence state between device and host.",
                    opportunity={"bound_share_of_request": round(rs, 3), "basis": "the whole residual; a candidate rarely removes all of it"},
                    experiment={"kind": "server_overhead_flags", "variable": "prompt-cache/checkpoint flags",
                                "arms": [{"name": "default", "args_override": {}}, {"name": "cache-ram-0", "args_override": {"--cache-ram": 0}},
                                         {"name": "ctx-checkpoints-0", "args_override": {"--ctx-checkpoints": 0}}],
                                "metric": "wall_ms median and the residual, prefill unchanged, outputs equivalent",
                                "note": "Flags are llama-server's; a build that lacks one fails to start, which the experiment reports as a launch failure."},
                    risks=["Disabling the prompt cache or checkpoints removes prefix reuse across requests; if the real traffic repeats long prefixes, the saved overhead can cost more prompt processing than it saves. Measure with the real request pattern before adopting.",
                           "Resuming from a checkpoint is not bit-identical to recomputing a prompt; outputs may differ at near-ties either way."])


def rule_microbatch(ev, label, arm):
    s = (arm or {}).get("settings") or {}
    ub = s.get("n_ubatch")
    source = s.get("settings_source") or "experiment args"
    if ub is None:
        ls = (ev.get("launch") or {}).get("launcher_settings") or {}
        ub = _f(ls.get("ubatch"))
        source = "launcher settings"
    if ub is None:
        tbl = (ev.get("launch") or {}).get("session_table") or []
        vals = [r.get("n_ubatch") for r in tbl if r.get("n_ubatch")]
        ub = _f(vals[-1]) if vals else None
        source = "server log (latest session that printed it)"
    if ub is None:
        return _finding("microbatch_unknown", "Microbatch size is not in the evidence", "info", "insufficient", "passive", {},
                        "Prefill throughput depends strongly on the microbatch (prompt tokens processed per GEMM); without the value the prefill rules cannot run.",
                        notes=["Provide an experiment record (the arm's args carry --ubatch-size), launcher settings via launcher_settings_file, or a server log that printed n_ubatch."])
    sp = phase_split(arm["requests"]) if arm else None
    pf_share = _med(sp["prefill_share"]) if sp else None
    pf_tps = _med(percentiles([r.get("prompt_tps") for r in arm["requests"]])) if arm else None
    if ub >= SMALL_UBATCH:
        return _finding("microbatch_not_small", "Microbatch %d is already in the range where prefill is batched GEMM work" % ub, "info", "likely", "passive",
                        {"n_ubatch": ub, "source": source, "prefill_tps_median": pf_tps},
                        "Raising it further buys at most a few percent of prefill and costs compute-buffer VRAM; the prefill rules below look at kernels and overhead instead.",
                        experiment={"kind": "ubatch_sweep", "variable": "--ubatch-size", "arms": [{"name": "ub%d" % int(ub), "args_override": {"--ubatch-size": int(ub)}},
                                                                                                 {"name": "ub%d" % int(min(2 * ub, 2048)), "args_override": {"--ubatch-size": int(min(2 * ub, 2048))}}],
                                    "metric": "prefill t/s median; VRAM per arm; outputs equivalent", "note": "Only worth running if prefill is a large share of the request."},
                        risks=["Compute buffer grows roughly linearly with the microbatch; check free VRAM first."])
    sev = "high" if (pf_share or 0) >= 0.4 else "medium"
    arms = [{"name": "ub%d" % int(ub), "args_override": {"--ubatch-size": int(ub)}}]
    for cand in (min(2 * ub, 512), min(4 * ub, 1024)):
        if cand > ub and all(a["args_override"]["--ubatch-size"] != int(cand) for a in arms):
            arms.append({"name": "ub%d" % int(cand), "args_override": {"--ubatch-size": int(cand)}})
    return _finding("microbatch_limits_prefill", "Microbatch %d is small; prompt processing is likely leaving GEMM efficiency on the table" % ub, sev, "hypothesis", "passive",
                    {"n_ubatch": ub, "source": source, "prefill_tps_median": pf_tps, "prefill_share_of_request": pf_share},
                    "Prompt tokens are processed in microbatch-sized groups; each group streams the full weight set once, so a small microbatch re-reads weights many times per prompt "
                    "and keeps the tensor-core GEMM path under-filled. The effect is largest below 128 and flattens around 256-512 on most GPUs.",
                    opportunity={"bound_share_of_request": pf_share, "basis": "prefill share of the request; the gain inside it is unknown until measured (it has ranged from a few percent to several-fold depending on the starting value)"},
                    experiment={"kind": "ubatch_sweep", "variable": "--ubatch-size", "arms": arms, "metric": "prefill t/s and wall_ms medians; per-arm VRAM; greedy outputs equivalent",
                                "note": "Keep --batch-size >= the largest --ubatch-size; hold CUDA-graph and cache settings constant."},
                    risks=["Larger microbatches need a larger compute buffer (VRAM); on a full card this can spill to system memory and look slower.",
                           "Changing the microbatch changes floating-point accumulation order; outputs can differ at near-ties (a numerical shift, not an error)."])


def _bench_rows(ev, kind):
    rows = []
    for b in ev["bench"]:
        if kind == "pp" and (b.get("n_prompt") or 0) > 0 and (b.get("n_gen") or 0) == 0:
            rows.append(b)
        if kind == "tg" and (b.get("n_gen") or 0) > 0 and (b.get("n_prompt") or 0) == 0:
            rows.append(b)
    return rows


def rule_server_vs_bench(ev, label, arm):
    """The runtime alone (llama-bench) vs the same model served: the gap is server-side work."""
    if not arm or not ev["bench"]:
        return None
    out = []
    pf_tps = _med(percentiles([r.get("prompt_tps") for r in arm["requests"]]))
    dc_tps = _med(percentiles([r.get("predicted_tps") for r in arm["requests"]]))
    pp = _bench_rows(ev, "pp")
    tg = _bench_rows(ev, "tg")
    ub = (arm.get("settings") or {}).get("n_ubatch")
    if pp and pf_tps:
        same = [b for b in pp if ub is None or b.get("n_ubatch") == ub] or pp
        best = max(same, key=lambda b: b.get("avg_ts") or 0)
        frac = pf_tps / best["avg_ts"] if best.get("avg_ts") else None
        if frac is not None and frac < BENCH_GAP_WORTH_NOTING:
            out.append(_finding("prefill_server_gap", "Served prefill reaches %.0f%% of llama-bench prefill at the same settings" % (100 * frac), "medium", "likely", "active",
                                {"server_prefill_tps_median": pf_tps, "bench_pp_tps": best["avg_ts"], "bench_n_prompt": best.get("n_prompt"), "bench_n_ubatch": best.get("n_ubatch"), "fraction": round(frac, 3)},
                                "llama-bench runs the same kernels without HTTP, tokenization, checkpoints or cache saves; the gap is what the server adds per prompt (and prompt-length effects if the bench prompt is shorter).",
                                experiment={"kind": "server_overhead_flags", "variable": "prompt-cache/checkpoint flags",
                                            "arms": [{"name": "default", "args_override": {}}, {"name": "ctx-checkpoints-0", "args_override": {"--ctx-checkpoints": 0}}],
                                            "metric": "prefill t/s median; outputs equivalent"},
                                risks=["See per_request_overhead: checkpoints are what makes partial prompt reuse possible on recurrent/hybrid models."]))
    if tg and dc_tps:
        best = max(tg, key=lambda b: b.get("avg_ts") or 0)
        frac = dc_tps / best["avg_ts"] if best.get("avg_ts") else None
        if frac is not None and frac < BENCH_GAP_WORTH_NOTING:
            out.append(_finding("decode_server_gap", "Served decode reaches %.0f%% of llama-bench decode" % (100 * frac), "medium", "likely", "active",
                                {"server_decode_tps_median": dc_tps, "bench_tg_tps": best["avg_ts"], "bench_n_gen": best.get("n_gen"), "fraction": round(frac, 3)},
                                "Per generated token the server adds sampling (and grammar constraints if the client sends a schema), logit transfer and streaming; llama-bench pays none of it. "
                                "A gap this size is CPU-side per-token work, not GPU kernels.",
                                experiment={"kind": "request_shape", "variable": "client request parameters",
                                            "arms": "measure the same prompts with the application's real request body (schema, sampler settings) and with plain greedy settings on the same server",
                                            "metric": "decode t/s and tokens generated; outputs' validity"},
                                risks=["Removing a grammar or sampler the application relies on changes outputs; the fix is usually in the sampler path, not in the request."]))
    return out or None


def rule_decode_ceiling(ev, label, arm, bandwidth_gbs=None):
    model = ev.get("model")
    bw = bandwidth_gbs or (ev.get("gpu") or {}).get("bandwidth_gbs_known")
    dc_tps = _med(percentiles([r.get("predicted_tps") for r in arm["requests"]])) if arm else None
    if dc_tps is None and ev.get("log_stats"):
        dc_tps = _med((ev["log_stats"] or {}).get("decode_tps"))
    if not model or not dc_tps:
        return _finding("decode_ceiling_unknown", "Decode cannot be placed against a bandwidth ceiling", "info", "insufficient", "passive",
                        {"model_summary": bool(model), "decode_tps": dc_tps, "bandwidth_gbs": bw},
                        "The batch-1 decode ceiling is weight bytes streamed per token divided by memory bandwidth; it needs the GGUF summary (set model_path in hotpath.local.json, or include an observation/experiment record) and a decode measurement.")
    if not bw:
        return _finding("decode_ceiling_unknown", "GPU memory bandwidth unknown; pass --bandwidth-gbs", "info", "insufficient", "passive",
                        {"gpu": (ev.get("gpu") or {}).get("name"), "decode_tps": dc_tps},
                        "Without the peak bandwidth the achieved fraction of the decode ceiling cannot be computed.")
    ceiling = gguf_mod.decode_ceiling_tps(model, bw)
    frac = dc_tps / ceiling if ceiling else None
    env = (arm or {}).get("settings", {}).get("env_ggml") or (ev.get("launch") or {}).get("launcher_env") or {}
    graphs_off = str(env.get("GGML_CUDA_DISABLE_GRAPHS", "")) == "1"
    meas = {"decode_tps_median": dc_tps, "ceiling_tps_at_peak_bandwidth": ceiling, "achieved_fraction": round(frac, 3), "bandwidth_gbs": bw,
            "weight_bytes_per_token_gb": round(model["weight_bytes_streamed_per_decode_token"] / 1e9, 3), "cuda_graphs_disabled_in_env": graphs_off,
            "model_source": ev.get("model_source") or "record"}
    if frac >= DECODE_CEILING_NEAR:
        return _finding("decode_near_ceiling", "Decode is at %.0f%% of the peak-bandwidth ceiling" % (100 * frac), "info", "likely", "passive", meas,
                        "Batch-1 generation is bound by streaming the weights; what remains is kernel efficiency and launch overhead, worth a kernel-level profile only if decode dominates the request.",
                        experiment={"kind": "profile_decode", "variable": "none (profiling run)", "arms": "hotpath profile --pp 0 --tg 64", "metric": "GPU time share by kernel class"})
    sev = "high" if frac < DECODE_CEILING_LOW else "medium"
    exp = {"kind": "cuda_graphs", "variable": "GGML_CUDA_DISABLE_GRAPHS",
           "arms": [{"name": "graphs-on", "env_remove": ["GGML_CUDA_DISABLE_GRAPHS"]}, {"name": "graphs-off", "env_override": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}],
           "metric": "decode t/s median; outputs equivalent", "note": "Whether the build enables graphs on this GPU architecture is decided by the runtime; the experiment shows the effect either way."}
    why = ("Achieved decode is well below what streaming the weights alone would allow. The usual causes, in order of how often they decide it: per-kernel launch overhead "
           "(hundreds of small kernels per token; CUDA graphs remove most of it), mat-vec kernels that do not reach bandwidth for this quantization type, extra passes for "
           "non-attention layers, and CPU-side per-token work in the server.")
    risks = ["CUDA graphs change nothing numerically but can fail to capture on some ops/builds; a launch failure or a fallback warning in the server log is the signal."]
    if graphs_off:
        why = "The launch environment disables CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1). " + why
    return _finding("decode_below_ceiling", "Decode reaches %.0f%% of the peak-bandwidth ceiling (%.1f of %.0f t/s)" % (100 * frac, dc_tps, ceiling), sev,
                    "likely" if graphs_off else "hypothesis", "passive", meas, why,
                    opportunity={"bound": "up to the ceiling fraction: %.0f%% of decode time is above the bandwidth floor" % (100 * (1 - frac)), "basis": "peak theoretical bandwidth; achievable is typically 70-90% of it"},
                    experiment=exp, risks=risks)


def rule_vram(ev, label, arm):
    gpu = ev.get("gpu") or {}
    total = _f(gpu.get("memory.total"))
    if not total:
        return None
    vram = (arm or {}).get("vram") or {}
    after = [g for g in vram.get("gpu_after") or [] if g]
    used = max([_f(g.get("mem_used_mib"), 0) for g in after], default=None)
    if used is None:
        used = _f(((ev.get("launch") or {}).get("gpu_now") or {}).get("mem_used_mib"))
    shared = [x for x in vram.get("process_shared_mib") or [] if x is not None]
    if used is None and not shared:
        return None
    meas = {"gpu_mem_used_mib_max": used, "gpu_mem_total_mib": total, "used_fraction": round(used / total, 3) if used else None,
            "process_shared_mib_max": max(shared) if shared else None, "process_dedicated_mib_max": max([x for x in vram.get("process_dedicated_mib") or [] if x is not None], default=None)}
    pressure = (used and used / total > VRAM_USED_HIGH) or (shared and max(shared) > PROCESS_SHARED_MIB_HIGH)
    if pressure:
        return _finding("vram_pressure", "GPU memory is close to full (%.0f%% used%s)" % (100 * (used or 0) / total, "; server has %.0f MiB in shared/system memory" % max(shared) if shared and max(shared) > PROCESS_SHARED_MIB_HIGH else ""),
                        "high", "likely", arm["evidence_class"] if arm else "passive", meas,
                        "Allocations that spill to system memory are served over PCIe at a fraction of VRAM bandwidth; a configuration can look slower for that reason alone, and larger microbatches or contexts make it worse.",
                        experiment={"kind": "vram_headroom", "variable": "memory footprint", "arms": "smaller --ctx-size or --ubatch-size vs current; or fewer other GPU processes", "metric": "per-process shared MiB (Windows) and decode/prefill t/s"},
                        risks=["Reducing context or microbatch to free memory trades capacity or prefill speed for stability."])
    headroom = total - used if used else None
    return _finding("vram_headroom", "GPU memory headroom %s MiB of %.0f" % (round(headroom) if headroom else "unknown", total), "info", "likely", arm["evidence_class"] if arm else "passive",
                    dict(meas, headroom_mib=round(headroom) if headroom else None),
                    "Headroom bounds which microbatch/context candidates can be tried without spilling; the compute buffer grows with the microbatch.")


def rule_profile(ev):
    out = []
    for p in ev["profiles"]:
        by = p["by_class"]
        if not by:
            continue
        top = list(by.items())[:3]
        mode = "decode" if (p["mode"].get("pp") or 0) == 0 else "prefill"
        kernels = p["kernels"]
        total_inst = sum(k.get("instances", 0) for k in kernels)
        small = sum(k.get("instances", 0) for k in kernels if (k.get("avg_ns") or 0) < 20000)
        meas = {"label": p["label"], "mode": mode, "top_classes_pct": dict(top), "kernel_types": p.get("kernel_count"), "instances_total": total_inst,
                "instances_under_20us_share": round(small / total_inst, 3) if total_inst else None}
        notes = []
        exp = None
        if any(k.startswith("cublas_bf16_fallback") for k, _ in top):
            notes.append("A BF16 cuBLAS fallback GEMM is among the top classes: on GPUs without native BF16 tensor cores the runtime converts and multiplies in a slower path; a build-level fix, not a flag.")
        if mode == "decode" and total_inst and small / total_inst > 0.7:
            notes.append("Most kernel launches are under 20 us: decode is launch-count sensitive; CUDA graphs (if not already on) and kernel fusion are the levers.")
            exp = {"kind": "cuda_graphs", "variable": "GGML_CUDA_DISABLE_GRAPHS", "arms": [{"name": "graphs-on", "env_remove": ["GGML_CUDA_DISABLE_GRAPHS"]}, {"name": "graphs-off", "env_override": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}],
                   "metric": "decode t/s median"}
        out.append(_finding("kernel_time_share_%s" % mode, "%s GPU time: %s" % (mode.capitalize(), ", ".join("%s %.0f%%" % (k, v) for k, v in top)), "info", "confirmed", "active", meas,
                            "Kernel-class shares say where a kernel-level optimization would have to land to matter; a class under 10%% cannot give more than 10%% of that phase.",
                            experiment=exp, notes=notes or None))
    return out or None


VERDICT_METRICS = (("prefill_tps", "higher"), ("decode_tps", "higher"), ("residual_ms", "lower"), ("wall_ms", "lower"))


_REQUEST_FIELD = {"prefill_tps": "prompt_tps", "decode_tps": "predicted_tps", "wall_ms": "wall_ms"}


def _per_prompt(requests, metric):
    out = {}
    for r in requests:
        if metric == "residual_ms":
            w, p, g = _f(r.get("wall_ms")), _f(r.get("prompt_ms")), _f(r.get("predicted_ms"))
            v = None if (w is None or p is None or g is None) else max(0.0, w - p - g)
        else:
            v = _f(r.get(_REQUEST_FIELD.get(metric, metric)))
        if v is not None:
            out.setdefault(r.get("prompt_id"), []).append(v)
    return out


def paired_metric(base_reqs, cand_reqs, metric, better):
    """Compare two arms prompt by prompt: the candidate/baseline ratio of per-prompt medians, judged against the
    within-prompt spread of the baseline (repetitions and rounds of the same prompt). Pooling prompts of different
    lengths would hide a consistent per-request shift behind the length spread; pairing does not."""
    a, b = _per_prompt(base_reqs, metric), _per_prompt(cand_reqs, metric)
    pids = [p for p in a if p in b]
    if not pids:
        return None
    rows = []
    for pid in pids:
        pa, pb = percentiles(a[pid]), percentiles(b[pid])
        if not pa or not pb or not pa["median"]:
            continue
        ratio = pb["median"] / pa["median"] if pa["median"] else None
        noise = (pa["p90"] - pa["p10"]) / pa["median"] if pa["median"] else None
        rows.append({"prompt_id": pid, "baseline_median": pa["median"], "candidate_median": pb["median"], "ratio": round(ratio, 3) if ratio is not None else None,
                     "baseline_relative_spread": round(noise, 3) if noise is not None else None, "n": [pa["n"], pb["n"]]})
    if not rows:
        return None
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    med_ratio = percentiles(ratios)["median"]
    change = med_ratio - 1.0
    improved_dir = (1 if better == "higher" else -1)
    consistent = sum(1 for r in ratios if (r - 1.0) * improved_dir > 0)
    noise = percentiles([r["baseline_relative_spread"] for r in rows if r["baseline_relative_spread"] is not None])
    noise_med = noise["median"] if noise else 0.0
    if abs(change) < 0.02:
        call = "no difference"
    elif consistent >= max(1, (2 * len(ratios) + 2) // 3) and abs(change) > noise_med / 2:
        call = "improvement" if change * improved_dir > 0 else "regression"
    else:
        call = ("improvement" if change * improved_dir > 0 else "regression") + " within noise (weak)"
    return {"ratio": round(med_ratio, 3), "call": call, "prompts": len(ratios), "consistent_direction": consistent,
            "baseline_within_prompt_spread_median": round(noise_med, 3), "per_prompt": rows,
            "baseline_median": percentiles([r["baseline_median"] for r in rows])["median"], "candidate_median": percentiles([r["candidate_median"] for r in rows])["median"]}


def experiment_verdict(rec):
    """What a finished experiment showed: paired per-prompt ratios for prefill, decode, the per-request residual and
    wall time; VRAM delta; greedy-output equivalence; a keep/reject/inconclusive call."""
    from hotpath import compare as compare_mod
    rep = rec.get("comparison") or compare_mod.compare([rec])
    arms = rec["arms"]
    labels = list(arms)
    if len(labels) < 2:
        return None
    base = arms[labels[0]]
    rows = []
    for lab in labels[1:]:
        cand = arms[lab]
        row = {"arm": lab, "vs": labels[0], "metrics": {}, "failures": len(cand.get("failures") or [])}
        for metric, better in VERDICT_METRICS:
            m = paired_metric(base.get("requests") or [], cand.get("requests") or [], metric, better)
            if m:
                row["metrics"][metric] = m
        bd = [x.get("process_counters_after", {}).get("dedicated_mib") for x in base.get("launch_facts") or []]
        cd = [x.get("process_counters_after", {}).get("dedicated_mib") for x in cand.get("launch_facts") or []]
        bd, cd = [x for x in bd if x is not None], [x for x in cd if x is not None]
        if bd and cd:
            row["vram_delta_mib"] = round(max(cd) - max(bd))
        full = "%s/%s" % (rec.get("name"), lab)
        oe = rep.get("output_equivalence") or {}
        ident = [e.get(full, {}).get("identical") for e in oe.values() if full in e]
        row["outputs"] = {"prompts_compared": len(ident), "identical": sum(1 for x in ident if x),
                          "first_divergence_indices": [e[full].get("first_divergence_index") for e in oe.values() if full in e and not e[full].get("identical")]}
        w = row["metrics"].get("wall_ms", {})
        all_identical = row["outputs"]["prompts_compared"] > 0 and row["outputs"]["identical"] == row["outputs"]["prompts_compared"]
        if row["failures"]:
            verdict = "reject: launch failures in this arm"
        elif w.get("call") == "improvement" and all_identical:
            verdict = "keep: end-to-end improvement (%.1f%% wall, consistent on %d/%d prompts) with identical greedy outputs" % (100 * (1 - w["ratio"]), w["consistent_direction"], w["prompts"])
        elif w.get("call") == "improvement":
            verdict = "keep only after explaining the output differences (numerical shift vs behavior change)"
        elif w.get("call") == "regression":
            verdict = "reject: end-to-end regression (%.1f%% wall)" % (100 * (w["ratio"] - 1))
        elif any(m["call"] == "improvement" for m in row["metrics"].values()):
            verdict = "inconclusive: a component improved but wall time did not move beyond the within-prompt noise; more rounds or a longer workload would decide"
        elif any(m["call"].startswith("improvement within noise") for m in row["metrics"].values()):
            verdict = "inconclusive: a small improvement within the within-prompt noise; more rounds would decide"
        else:
            verdict = "no measurable difference"
        row["verdict"] = verdict
        rows.append(row)
    return {"experiment": rec.get("name"), "evidence_class": rep.get("evidence_class"), "comparability_issues": rep.get("comparability_issues"), "arms": rows}


def insufficiencies(ev, arm):
    out = []
    if not ev["bench"]:
        out.append("No llama-bench record: the server's share of prefill/decode cost cannot be separated from the runtime's (hotpath bench).")
    if not ev["profiles"]:
        out.append("No kernel profile: kernel classes cannot be ranked (hotpath profile --pp 0 --tg 64 for decode, --tg 0 for prefill).")
    if not ev["model"]:
        out.append("No GGUF summary: bandwidth ceiling and KV sizing are unavailable (set model_path in hotpath.local.json, or include an observation/experiment record).")
    if arm and len(arm["requests"]) < 6:
        out.append("Fewer than 6 measured requests in the diagnosed arm: shares and medians have little spread information.")
    if not ev["experiments"]:
        out.append("No controlled experiment yet: every configuration finding above is a hypothesis until an interleaved A/B confirms it.")
    out.append("Prefix-cache effectiveness is not measured by hotpath's workloads (they disable the prompt cache on purpose); judge it from the server log's slot-selection lines or an experiment that sends the real request sequence.")
    return out


SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "info": 3}
CONFIDENCE_RANK = {"confirmed": 0, "likely": 1, "hypothesis": 2, "insufficient": 3}


def apply_local_config(ev, cfg):
    """Declared configuration from hotpath.local.json, used only where the records carry nothing: the user's own
    launch arguments (server_args) for a measured server, and the GGUF summary of model_path. Both are labelled as
    declared rather than observed in the findings' `source`."""
    if not cfg:
        return ev
    args = cfg.get("server_args")
    if args:
        for a in ev["arms"].values():
            if a.get("evidence_class") == "active" and a["settings"].get("n_ubatch") is None:
                a["settings"]["n_ubatch"] = _f(_args_value(args, "--ubatch-size"))
                a["settings"]["n_batch"] = _f(_args_value(args, "--batch-size"))
                a["settings"]["settings_source"] = "hotpath.local.json server_args (declared, not observed)"
                a.setdefault("args", list(args))
    if not ev["model"] and cfg.get("model_path"):
        try:
            if Path(cfg["model_path"]).is_file():
                ev["model"] = gguf_mod.summarize(gguf_mod.read_header(cfg["model_path"]))
                ev["model_source"] = "GGUF header of model_path in hotpath.local.json"
        except (OSError, ValueError, struct.error):
            pass
    return ev


def diagnose(records, bandwidth_gbs=None, cfg=None):
    ev = apply_local_config(collect(records), cfg)
    label, arm = _baseline_arm(ev)
    findings = []
    for fn in (rule_phase, rule_residual, rule_microbatch, rule_vram):
        f = fn(ev, label, arm)
        if f:
            findings.append(f)
    f = rule_decode_ceiling(ev, label, arm, bandwidth_gbs)
    if f:
        findings.append(f)
    for fn in (rule_server_vs_bench,):
        fs = fn(ev, label, arm)
        if fs:
            findings.extend(fs)
    fs = rule_profile(ev)
    if fs:
        findings.extend(fs)
    findings.sort(key=lambda f: (SEVERITY_RANK.get(f["severity"], 9), CONFIDENCE_RANK.get(f["confidence"], 9)))
    verdicts = []
    for r in ev["experiments"]:
        v = experiment_verdict(r)
        if not v:
            continue
        # an experiment written by `propose` names the finding it tests; carry that finding's risks as caveats
        prov = ((r.get("spec_redacted") or {}).get("provenance") or {})
        origin = next((f for f in findings if f["id"] == prov.get("finding")), None)
        if origin:
            v["tests_finding"] = origin["id"]
            v["caveats"] = origin.get("risks") or []
        verdicts.append(v)
    # a finding whose experiment has been run in these records is settled by its verdict: do not recommend it again
    tested = {v["tests_finding"]: v for v in verdicts if v.get("tests_finding")}
    for f in findings:
        if f["id"] in tested:
            f["confidence"] = "confirmed"
            f.setdefault("notes", []).append("Tested by experiment %s; see its verdict above." % tested[f["id"]]["experiment"])
    recommended = next((f for f in findings if f["id"] not in tested and f.get("experiment") and isinstance(f["experiment"].get("arms"), list) and f["severity"] in ("high", "medium")), None)
    return {"kind": "diagnosis", "evidence_class": "hypothesis" if not ev["experiments"] else "controlled", "sources": ev["sources"], "diagnosed_arm": label,
            "baseline_args": (arm or {}).get("args"), "findings": findings, "experiment_verdicts": verdicts, "insufficient_evidence": insufficiencies(ev, arm),
            "recommended_next_experiment": {"finding": recommended["id"], **recommended["experiment"]} if recommended else None}


def diagnosis_markdown(d):
    lines = ["# Diagnosis", "", "Sources: %s" % ", ".join(d["sources"]), "Diagnosed arm: %s" % d.get("diagnosed_arm"), ""]
    for v in d.get("experiment_verdicts") or []:
        lines += ["## Experiment verdict: %s  [%s]" % (v["experiment"], v["evidence_class"]), ""]
        if v.get("comparability_issues"):
            lines += ["Not directly comparable: %s" % v["comparability_issues"], ""]
        for row in v["arms"]:
            lines.append("- **%s** vs %s: %s" % (row["arm"], row["vs"], row["verdict"]))
            for m, x in row["metrics"].items():
                lines.append("  - %s: %s (ratio %.3f; %s -> %s; same direction on %d/%d prompts; baseline within-prompt spread %.1f%%)" % (
                    m, x["call"], x["ratio"], x["baseline_median"], x["candidate_median"], x["consistent_direction"], x["prompts"], 100 * x["baseline_within_prompt_spread_median"]))
            lines.append("  - outputs identical %d/%d%s; VRAM delta %s MiB; failures %d" % (row["outputs"]["identical"], row["outputs"]["prompts_compared"],
                         "" if not row["outputs"]["first_divergence_indices"] else " (first divergence at %s)" % row["outputs"]["first_divergence_indices"], row.get("vram_delta_mib"), row["failures"]))
        for c in v.get("caveats") or []:
            lines.append("- caveat (from finding `%s`): %s" % (v.get("tests_finding"), c))
        lines.append("")
    lines += ["## Findings (ranked)", ""]
    for f in d["findings"]:
        lines += ["### %s  [%s, %s, %s]" % (f["title"], f["severity"], f["confidence"], f["evidence_class"]), "", "- id: `%s`" % f["id"],
                  "- measurements: %s" % f["measurements"], "- why it matters: %s" % f["why_it_matters"]]
        if f.get("opportunity"):
            lines.append("- opportunity: %s" % f["opportunity"])
        if f.get("experiment"):
            e = f["experiment"]
            lines.append("- experiment to confirm/reject: %s on `%s`; metric: %s" % (e.get("kind"), e.get("variable"), e.get("metric")))
            if isinstance(e.get("arms"), list):
                lines.append("  - arms: %s" % [a["name"] for a in e["arms"]])
            else:
                lines.append("  - how: %s" % e.get("arms"))
        for r in f.get("risks") or []:
            lines.append("- risk: %s" % r)
        for n in f.get("notes") or []:
            lines.append("- note: %s" % n)
        lines.append("")
    if d.get("recommended_next_experiment"):
        r = d["recommended_next_experiment"]
        lines += ["## Recommended next experiment", "", "`hotpath propose <this diagnosis> --finding %s` writes a spec for: %s on `%s` (arms %s)." % (
            r["finding"], r.get("kind"), r.get("variable"), [a["name"] for a in r["arms"]]), ""]
    lines += ["## What the evidence cannot establish", ""] + ["- " + s for s in d["insufficient_evidence"]]
    return "\n".join(lines)
