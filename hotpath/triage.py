"""hotpath triage: turn an observation into an attribution report with labeled evidence.

Every finding carries: evidence_class, what was observed, what is only a hypothesis, and the
experiment that would settle it. Nothing here claims a speedup.
"""
from hotpath import gguf as gguf_mod


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def triage(obs, bandwidth_gbs=None):
    findings = []
    model = (obs.get("model") or {}).get("summary") or {}
    gpu = obs.get("gpu") or {}
    ls = obs.get("launcher_settings") or {}
    log = obs.get("log") or {}
    stats = log.get("gpu_like_sessions_stats") or {}
    latest = log.get("latest_session_stats") or {}
    props = (obs.get("server") or {}).get("props") or {}
    bw = bandwidth_gbs or gpu.get("bandwidth_gbs_known")

    # 1. request-time attribution (passive, per request)
    share = (stats.get("prefill_share_of_request_pct") or {})
    if share:
        findings.append({
            "id": "request_time_split",
            "evidence_class": "passive",
            "observed": "Across %d logged requests with >= %d prompt tokens, prompt processing took a median %.0f%% of request time (p10 %.0f%%, p90 %.0f%%)." % (
                share["n"], stats["filters"]["min_prompt_tokens"], share["median"], share["p10"], share["p90"]),
            "implication": "Optimizations that only speed up generation cannot move most of this workload's latency.",
        })

    # 2. microbatch
    ub = _f(ls.get("ubatch"))
    pf = stats.get("prefill_tps") or {}
    dc = stats.get("decode_tps") or {}
    if ub is not None and ub < 64:
        obs_txt = "Launcher sets ubatch=%d." % ub
        if pf and dc:
            obs_txt += " Logged prefill median %.1f t/s vs decode median %.1f t/s (ratio %.2f)." % (pf["median"], dc["median"], pf["median"] / dc["median"])
        findings.append({
            "id": "microbatch_throttles_prefill",
            "evidence_class": "passive",
            "observed": obs_txt,
            "mechanism": "With a microbatch of one token, prompt processing re-reads the full weight set per token, so prefill throughput cannot exceed decode throughput.",
            "hypothesis": "A larger microbatch restores batched prefill (MMQ path) and raises prefill throughput several-fold at the cost of a larger compute buffer.",
            "experiment": "Controlled, interleaved arms over ubatch sizes chosen from VRAM headroom, same model/build/prompts/n_predict, graphs setting held constant; record VRAM per arm and output equivalence.",
            "do_not_conclude": "The size of the gain, or that the setting was unnecessary; its history must be checked (see the session table).",
        })

    # 3. decode vs roofline
    if model and bw and dc:
        ceiling = gguf_mod.decode_ceiling_tps(model, bw)
        achieved = dc["median"]
        findings.append({
            "id": "decode_vs_bandwidth_ceiling",
            "evidence_class": "passive",
            "observed": "Decode median %.1f t/s (p10 %.1f, p90 %.1f). Streamed weight bytes per token %.2f GB; peak bandwidth %.0f GB/s gives a batch-1 ceiling of about %.0f t/s before KV reads, launch overhead, and achievable-bandwidth losses." % (
                achieved, dc["p10"], dc["p90"], model["weight_bytes_streamed_per_decode_token"] / 1e9, bw, ceiling),
            "achieved_fraction_of_ceiling": round(achieved / ceiling, 2),
            "hypothesis": "Remaining decode gap is split between quantized mat-vec kernel efficiency, the activation transform, recurrent-layer kernels, and per-kernel launch overhead; only a kernel-level trace ranks them.",
            "experiment": "nsys trace of 32 decode steps on an isolated server (hotpath profile), then test-backend-ops perf at the model's real shapes.",
        })

    # 4. memory budget
    vram = _f(gpu.get("memory.total"))
    if model and vram:
        slots = _f(props.get("total_slots"), 1)
        ctx = _f(props.get("n_ctx_per_slot"), 0)
        kv_mib = (model.get("kv_bytes_per_token_f16") or 0) * ctx * slots / (1 << 20)
        weights_mib = model["total_bytes"] / (1 << 20)
        findings.append({
            "id": "vram_budget",
            "evidence_class": "passive",
            "observed": "Weights %.0f MiB + F16 KV for %d x %d tokens = %.0f MiB, before recurrent state and compute buffers (compute buffer grows with ubatch), on a %.0f MiB card." % (
                weights_mib, slots, ctx, kv_mib, vram),
            "note": "Per-process and adapter counters are reported in the observation but are not summed into a paging verdict: shared allocations (e.g. attributed to dwm.exe) can be double-counted.",
            "experiment": "Per-arm VRAM and the server process's shared-memory counter during the microbatch experiment; NVIDIA 'Prefer No Sysmem Fallback' for the server executable turns silent spills into failures.",
        })

    # 5. CUDA graphs env
    env = obs.get("launcher_env_documented") or {}
    if "GGML_CUDA_DISABLE_GRAPHS" in env:
        findings.append({
            "id": "cuda_graphs_disabled_by_launcher",
            "evidence_class": "passive",
            "observed": "Launcher exports GGML_CUDA_DISABLE_GRAPHS=%s. Whether the runtime honors it and whether the GPU architecture would otherwise allow graphs depends on the exact build; check the pinned source, not a general rule." % env["GGML_CUDA_DISABLE_GRAPHS"],
            "experiment": "Separate experiment from the microbatch one: graphs on vs off at the chosen ubatch, decode-focused.",
        })

    # 6. session history
    tbl = log.get("session_table") or []
    changed = sorted({r["n_ubatch"] for r in tbl if r.get("n_ubatch")})
    if len(changed) > 1:
        findings.append({
            "id": "launch_settings_changed_over_time",
            "evidence_class": "passive",
            "observed": "The log shows sessions with n_ubatch in %s; later sessions did not print n_ubatch (lower verbosity)." % changed,
            "implication": "Earlier settings and any warnings in those sessions are the history behind the current value. See session_table and warnings in the observation.",
        })

    # 7. latest-vs-history drift
    lp = (latest.get("prefill_tps") or {})
    if lp and pf and abs(lp["median"] - pf["median"]) / pf["median"] > 0.15:
        findings.append({
            "id": "latest_session_differs_from_history",
            "evidence_class": "historical",
            "observed": "Latest session prefill median %.1f t/s vs all-GPU-sessions median %.1f t/s." % (lp["median"], pf["median"]),
            "note": "Historical comparison; conditions were not controlled.",
        })

    return {"findings": findings, "bandwidth_gbs_used": bw}


def triage_markdown(rep):
    lines = ["# Triage", ""]
    for f in rep["findings"]:
        lines.append("## %s  [%s]" % (f["id"], f["evidence_class"]))
        lines.append("")
        for k in ("observed", "mechanism", "achieved_fraction_of_ceiling", "implication", "hypothesis", "note", "experiment", "do_not_conclude"):
            if k in f:
                lines.append("- **%s**: %s" % (k, f[k]))
        lines.append("")
    if rep.get("bandwidth_gbs_used"):
        lines.append("Bandwidth used for the roofline: %.0f GB/s (peak theoretical)." % rep["bandwidth_gbs_used"])
    return "\n".join(lines)
