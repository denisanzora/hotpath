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
            log = rec.get("log") or {}
            ev["log_stats"] = log.get("gpu_like_sessions_stats") or ev["log_stats"]
            ev["launch"] = {"launcher_settings": rec.get("launcher_settings") or {}, "launcher_env": rec.get("launcher_env_documented") or {},
                            "session_table": log.get("session_table") or [], "gpu_now": rec.get("gpu_now") or {},
                            "server_process": rec.get("server_process") or {}}
            ev["log"] = {"rows": log.get("requests") or [], "sessions": log.get("session_table") or [], "cache": log.get("cache_reuse") or {},
                         "coverage": log.get("parse_coverage") or {}, "cpu_threshold": _f(log.get("cpu_mode_prefill_tps_below"), 10.0),
                         "available": bool(log.get("sessions")), "source": log.get("source_basename")}
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
    """Per-request shares of prompt processing, generation and the residual.

    With client wall times (measurement/experiment records) the base is the wall and the residual is wall minus
    the server's prompt+generation time. With server-log rows there is no wall time: the base is prompt+generation
    itself, the residual is unknown, and `basis` says so. The two are never mixed in one split."""
    rows = []
    basis = None
    for r in requests:
        w, p, g = _f(r.get("wall_ms")), _f(r.get("prompt_ms")), _f(r.get("predicted_ms"))
        if p is None or g is None:
            continue
        if w and w > 0 and basis != "server":
            basis = "client_wall"
            rows.append({"prefill": p / w, "decode": g / w, "residual": max(0.0, w - p - g) / w, "residual_ms": max(0.0, w - p - g),
                         "prompt_n": r.get("prompt_n"), "predicted_n": r.get("predicted_n"), "wall_ms": w})
        elif not w and p + g > 0 and basis != "client_wall":
            basis = "server"
            rows.append({"prefill": p / (p + g), "decode": g / (p + g), "residual": None, "residual_ms": None,
                         "prompt_n": r.get("prompt_n"), "predicted_n": r.get("predicted_n"), "wall_ms": p + g})
    if len(rows) < MIN_REQUESTS_FOR_SPLIT:
        return None
    out = {"n": len(rows), "basis": basis,
           "prefill_share": percentiles([x["prefill"] for x in rows]), "decode_share": percentiles([x["decode"] for x in rows]),
           "prompt_n": percentiles([x["prompt_n"] for x in rows]), "predicted_n": percentiles([x["predicted_n"] for x in rows]),
           "wall_ms": percentiles([x["wall_ms"] for x in rows])}
    if basis == "client_wall":
        out["residual_share"] = percentiles([x["residual"] for x in rows])
        out["residual_ms"] = percentiles([x["residual_ms"] for x in rows])
    else:
        out["residual_share"] = None
        out["residual_ms"] = None
    return out


def _passive_arm_from_log(ev):
    """When no active record exists, diagnose the newest server-log session that looks GPU-served and has enough
    requests. Rows carry server timings only; the arm says so (`time_basis`)."""
    log = ev.get("log") or {}
    rows = log.get("rows") or []
    if not rows:
        return None, None
    thr = log.get("cpu_threshold") or 10.0
    by_session = {}
    for r in rows:
        by_session.setdefault(r.get("session"), []).append(r)
    for sid in sorted(by_session, reverse=True):
        rs = [r for r in by_session[sid] if r.get("prompt_ms") is not None and r.get("predicted_ms") is not None]
        if len(rs) < MIN_REQUESTS_FOR_SPLIT:
            continue
        pf = percentiles([r.get("prompt_tps") for r in rs if r.get("prompt_tps")])
        if pf and pf["median"] < thr:
            continue  # CPU fallback session: reported by rule_sessions, not diagnosed as the GPU baseline
        reqs = [{"prompt_id": "log-%s-%d" % (sid, i), "prompt_n": r.get("prompt_n"), "prompt_ms": r.get("prompt_ms"), "prompt_tps": r.get("prompt_tps"),
                 "predicted_n": r.get("predicted_n"), "predicted_ms": r.get("predicted_ms"), "predicted_tps": r.get("predicted_tps"),
                 "wall_ms": None, "cache_n": r.get("reused_tokens"), "slot_by": r.get("slot_by")} for i, r in enumerate(rs)]
        sess = next((s for s in log.get("sessions") or [] if s.get("session") == sid), {})
        settings = {"n_ubatch": _f(sess.get("n_ubatch")), "n_batch": _f(sess.get("n_batch")), "env_ggml": (ev.get("launch") or {}).get("launcher_env") or {},
                    "settings_source": "server log, session %s" % sid}
        label = "%s/session-%s" % (log.get("source") or "log", sid)
        return label, {"requests": reqs, "evidence_class": "passive", "settings": settings, "vram": {},
                       "failures": [], "time_basis": "server prompt+generation time from the log; no client wall time", "session": sid}
    return None, None


def _baseline_arm(ev):
    """The arm a diagnosis describes: the first arm of the newest experiment, else the newest measurement, else the
    newest usable server-log session (passive)."""
    if ev["experiments"]:
        rec = ev["experiments"][-1]
        first = next(iter(rec["arms"]))
        return "%s/%s" % (rec["name"], first), ev["arms"]["%s/%s" % (rec["name"], first)]
    if ev["arms"]:
        k = list(ev["arms"])[-1]
        return k, ev["arms"][k]
    label, arm = _passive_arm_from_log(ev)
    if arm:
        ev["arms"][label] = arm
    return label, arm


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
    if sp["basis"] == "client_wall":
        title = "Workload is %s (prefill %.0f%%, decode %.0f%%, other %.0f%% of request wall time)" % (shape, 100 * pf, 100 * dc, 100 * rs)
        note = ("Shares are computed per request from the server's own prompt/generation timings and the client wall time, then summarized; "
                "they describe this workload's prompt and generation lengths, not the model in general.")
    else:
        title = "Workload is %s (prefill %.0f%%, decode %.0f%% of the server's prompt+generation time; overhead outside them is not in a log)" % (shape, 100 * pf, 100 * dc)
        note = ("Computed from the server log's per-request prompt and generation timings (session %s). A server log has no client wall time, so the per-request overhead "
                "outside these two phases (HTTP, cache saves, checkpoints) is unknown here; `hotpath measure` against your own server is the single measurement that adds it." % arm.get("session"))
    band = lambda p: [p["p10"], p["p90"]] if p else None
    meas = {"arm": label, "time_basis": sp["basis"], "requests": sp["n"], "prompt_tokens_median": _med(sp["prompt_n"]), "prompt_tokens_p10_p90": band(sp["prompt_n"]),
            "generated_tokens_median": _med(sp["predicted_n"]), "generated_tokens_p10_p90": band(sp["predicted_n"]),
            "request_ms_median": _med(sp["wall_ms"]), "prefill_share_median": round(pf, 3), "decode_share_median": round(dc, 3),
            "residual_share_median": round(rs, 3) if rs is not None else None,
            "shares_p10_p90": {"prefill": [sp["prefill_share"]["p10"], sp["prefill_share"]["p90"]], "decode": [sp["decode_share"]["p10"], sp["decode_share"]["p90"]]}}
    return _finding("workload_phase_split", title, "info", "confirmed" if sp["n"] >= 6 else "likely", arm["evidence_class"], meas, why, notes=[note])


def cache_pattern_evidence(ev):
    """What the records say about how often real requests reuse a prefix. Decides whether a change that trades
    prefix reuse for per-request overhead can be recommended beyond the isolated workload it was measured on."""
    cache = (ev.get("log") or {}).get("cache") or {}
    if not cache.get("available") or not cache.get("reused_share_of_prompt"):
        return {"available": False, "note": "no slot-selection/release lines in a log; reuse pattern of real traffic unknown"}
    share = cache["reused_share_of_prompt"]
    sel = cache.get("selected_by") or {}
    n_sel = sum(sel.values()) or 1
    lcp = sel.get("LCP similarity", 0) / n_sel
    med = share["median"]
    level = "low" if med < 0.15 and share["p90"] < 0.5 else ("high" if med >= 0.5 else "mixed")
    return {"available": True, "reused_share_median": med, "reused_share_p90": share["p90"], "lcp_fraction": round(lcp, 3), "requests": cache.get("requests_with_reuse_data"),
            "level": level, "source": "server log slot-selection and release lines (real traffic)"}


def rule_residual(ev, label, arm):
    sp = phase_split(arm["requests"]) if arm else None
    if not sp or sp["basis"] != "client_wall":
        return None
    rs, rms = _med(sp["residual_share"]), _med(sp["residual_ms"])
    if rs is None or rs < OVERHEAD_SHARE_WORTH_NOTING:
        return None
    s = arm.get("settings") or {}
    cp = cache_pattern_evidence(ev)
    risks = ["Disabling the prompt cache or checkpoints removes prefix reuse across requests; if the real traffic repeats long prefixes, the saved overhead can cost more prompt processing than it saves."]
    if cp.get("available"):
        risks.append("Cache-pattern evidence from the log: real requests reuse a median %.0f%% of their prompt (p90 %.0f%%; LCP-selected %.0f%%) — reuse is %s, so a flag that removes it is %s." % (
            100 * cp["reused_share_median"], 100 * cp["reused_share_p90"], 100 * cp["lcp_fraction"], cp["level"],
            "defensible for this traffic" if cp["level"] == "low" else "likely to cost more than it saves on this traffic"))
    else:
        risks.append("No cache-pattern evidence: include an `observe --log` record of the real server so the reuse rate of real traffic is known before adopting such a flag.")
    risks.append("Resuming from a checkpoint is not bit-identical to recomputing a prompt; outputs may differ at near-ties either way.")
    return _finding("per_request_overhead", "%.0f ms per request (%.0f%%) is spent outside prompt processing and generation" % (rms, 100 * rs),
                    "medium" if rs < 0.15 else "high", "likely", arm["evidence_class"],
                    {"arm": label, "residual_ms_median": round(rms), "residual_ms_p10_p90": [sp["residual_ms"]["p10"], sp["residual_ms"]["p90"]], "residual_share_median": round(rs, 3),
                     "cache_ram_flag": s.get("cache_ram"), "ctx_checkpoints_flag": s.get("ctx_checkpoints"), "cache_pattern_evidence": cp},
                    "This time is paid on every request regardless of prompt or output length: HTTP and JSON handling, tokenization and template rendering, sampler setup, "
                    "and in llama-server the prompt-cache state save and context checkpoints, which copy sequence state between device and host.",
                    opportunity={"bound_share_of_request": round(rs, 3), "basis": "the whole residual; a candidate rarely removes all of it"},
                    experiment={"kind": "server_overhead_flags", "variable": "prompt-cache/checkpoint flags",
                                "arms": [{"name": "default", "args_override": {}}, {"name": "cache-ram-0", "args_override": {"--cache-ram": 0}},
                                         {"name": "ctx-checkpoints-0", "args_override": {"--ctx-checkpoints": 0}}],
                                "metric": "wall_ms median and the residual, prefill unchanged, outputs equivalent",
                                "note": "Flags are llama-server's; a build that lacks one fails to start, which the experiment reports as a launch failure. "
                                        "The experiment's workload disables prompt caching on purpose, so it measures the overhead only; the reuse side needs the log evidence above."},
                    risks=risks)


def rule_cache_reuse(ev):
    cp = cache_pattern_evidence(ev)
    cache = (ev.get("log") or {}).get("cache") or {}
    if not cp.get("available"):
        if (ev.get("log") or {}).get("available"):
            return _finding("prefix_reuse_unknown", "Prefix reuse of real traffic is not visible in this log", "info", "insufficient", "passive",
                            {"cache_stats": cache}, "The log printed no slot-selection or release lines (verbosity), so how much of each prompt the server reused cannot be read.",
                            notes=["llama-server prints 'selected slot by LRU/LCP similarity' and 'release: ... n_tokens' at its default verbosity; a quieter log level hides them."])
        return None
    sel = cache.get("selected_by") or {}
    meas = {"requests": cp["requests"], "reused_share_of_prompt_median": cp["reused_share_median"], "reused_share_p90": cp["reused_share_p90"],
            "reused_tokens_median": _med(cache.get("reused_tokens")), "selected_by": sel, "f_sim_best_median": _med(cache.get("f_sim_best"))}
    if cp["level"] == "low":
        return _finding("prefix_reuse_low", "Real requests reuse a median %.0f%% of their prompt from the slot (p90 %.0f%%)" % (100 * cp["reused_share_median"], 100 * cp["reused_share_p90"]),
                        "medium", "likely", "passive", meas,
                        "Consecutive requests share only a short prefix (typically the system message), so prompt caching rarely saves prompt processing on this traffic; meanwhile every "
                        "request may pay the server's cache bookkeeping (state saves, checkpoints), which a log cannot show.",
                        opportunity={"bound": "the per-request bookkeeping overhead, measurable only with client wall times", "basis": "low reuse means little is lost by changing cache settings"},
                        experiment={"kind": "server_overhead_flags", "variable": "prompt-cache/checkpoint flags",
                                    "arms": [{"name": "default", "args_override": {}}, {"name": "cache-ram-0", "args_override": {"--cache-ram": 0}}],
                                    "metric": "wall_ms with client timing (hotpath measure/experiment); the log alone cannot show the gain"},
                        risks=["If the application's prompts are reordered so that long static text comes first, reuse would rise and the cache would become valuable; re-check the log after such a change.",
                               "On recurrent/hybrid models partial reuse happens only at checkpoints; reordering prompts without checkpoints at the right place changes little."],
                        notes=["Prompt-order changes in the application (static instructions before per-request content) are usually the larger lever when reuse is this low."])
    return _finding("prefix_reuse_%s" % cp["level"], "Real requests reuse a median %.0f%% of their prompt from the slot (p90 %.0f%%)" % (100 * cp["reused_share_median"], 100 * cp["reused_share_p90"]),
                    "info", "likely", "passive", meas,
                    "Prompt caching is doing useful work for this traffic; changes that trade it for lower per-request overhead would cost prompt processing here.",
                    risks=["Do not disable the prompt cache or checkpoints for this deployment on the strength of an isolated-request experiment."])


def rule_sessions(ev):
    log = ev.get("log") or {}
    table = log.get("sessions") or []
    if not table:
        return None
    thr = log.get("cpu_threshold") or 10.0
    used = [s for s in table if (s.get("requests") or 0) >= MIN_REQUESTS_FOR_SPLIT and s.get("prefill_median_tps")]
    cpu = [s for s in used if s["prefill_median_tps"] < thr]
    gpu = [s for s in used if s["prefill_median_tps"] >= thr]
    out = []
    if cpu:
        n_req = sum(s["requests"] for s in cpu)
        out.append(_finding("cpu_fallback_sessions", "%d of %d sessions with requests ran at CPU-like speed (prefill median < %.0f t/s), %d requests" % (len(cpu), len(used), thr, n_req),
                            "high" if cpu[-1] is used[-1] else "medium", "likely", "passive",
                            {"sessions": [{"session": s["session"], "prefill_median_tps": s["prefill_median_tps"], "decode_median_tps": s["decode_median_tps"], "device": s.get("device"), "requests": s["requests"]} for s in cpu]},
                            "A session whose prompt processing runs at single-digit tokens per second is not using the GPU for the model (no offload, a failed CUDA load, or a deliberate CPU "
                            "launch); every request in it is many times slower than the GPU sessions in the same log." + (" The latest session is one of them." if cpu[-1] is used[-1] else ""),
                            experiment={"kind": "launch_check", "variable": "backend/offload", "arms": "check the launch command and the session's device line and 'offloaded N/M layers' fact; relaunch with full GPU offload and compare the next session",
                                        "metric": "prefill t/s in the next session"},
                            risks=["If the fallback was the launcher's reaction to memory pressure, forcing the GPU can fail to load; check free VRAM first."]))
    if len(gpu) >= 2:
        prev, last = gpu[-2], gpu[-1]
        changes = []
        for k in ("prefill_median_tps", "decode_median_tps"):
            a, b = prev.get(k), last.get(k)
            if a and b and abs(b - a) / a > 0.15:
                changes.append("%s %.1f -> %.1f (%+.0f%%)" % (k, a, b, 100 * (b - a) / a))
        facts = {k: (prev.get(k), last.get(k)) for k in ("n_batch", "n_ubatch", "device", "compute_mib") if prev.get(k) != last.get(k)}
        facts_missing = [k for k in ("n_batch", "n_ubatch", "device") if not last.get(k)]
        if changes:
            out.append(_finding("session_drift", "Latest GPU session differs from the previous one: %s" % "; ".join(changes), "medium", "likely", "historical",
                                {"previous_session": prev["session"], "latest_session": last["session"], "launch_facts_changed": facts, "launch_facts_not_printed_in_latest": facts_missing,
                                 "requests": [prev["requests"], last["requests"]]},
                                "Throughput moved between two launches of the same deployment. If launch facts changed (microbatch, batch, device), that is the first suspect; if nothing "
                                "printed changed, the cause is outside the log (other GPU load, thermals, a different build)." +
                                (" The latest session printed none of %s, so a changed launch setting cannot be excluded from the log alone." % facts_missing if facts_missing else ""),
                                experiment={"kind": "ubatch_sweep" if "n_ubatch" in facts else "launch_check", "variable": "--ubatch-size" if "n_ubatch" in facts else "launch facts",
                                            "arms": [{"name": "ub%s" % v, "args_override": {"--ubatch-size": int(v)}} for v in sorted({int(x) for x in facts.get("n_ubatch", ()) if x})] if "n_ubatch" in facts else "relaunch both configurations under hotpath experiment",
                                            "metric": "prefill and decode t/s medians, controlled"},
                                notes=["Historical comparison: the two sessions ran under conditions hotpath did not control."]))
        best = max(gpu, key=lambda s: s["prefill_median_tps"])
        if best is not last and (best["prefill_median_tps"] - last["prefill_median_tps"]) / best["prefill_median_tps"] > 0.15 and not changes:
            out.append(_finding("regression_vs_best_session", "Latest session prefill %.1f t/s is %.0f%% below the best GPU session in this log (%.1f t/s, session %s)" % (
                last["prefill_median_tps"], 100 * (1 - last["prefill_median_tps"] / best["prefill_median_tps"]), best["prefill_median_tps"], best["session"]),
                "medium", "likely", "historical", {"best_session": best["session"], "latest_session": last["session"]},
                "Something between those launches made prompt processing slower; compare their launch facts in the observation's session table.",
                notes=["Historical comparison; conditions were not controlled."]))
    warn = sum(s.get("warnings") or 0 for s in table)
    err = sum(s.get("errors") or 0 for s in table)
    if err:
        out.append(_finding("log_errors", "%d error lines across sessions (CUDA errors, allocation failures or asserts)" % err, "high", "confirmed", "passive",
                            {"errors": err, "warnings": warn}, "Requests served around an error are not representative, and an allocation failure often precedes a fallback to slower paths.",
                            notes=["See the observation's session table for the sessions that carry them."]))
    return out or None


def rule_coverage(ev):
    log = ev.get("log") or {}
    cov = log.get("coverage") or {}
    if not log.get("available") and not cov:
        return None
    if cov.get("problem"):
        return _finding("log_format_problem", "The server log could not be read as expected: %s" % cov["problem"], "high", "insufficient", "passive", cov,
                        "Every log-based finding depends on the parser recognizing this runtime's lines; with this coverage the absence of findings means nothing.",
                        notes=["Supported: llama-server stderr logs with 'prompt eval time' / 'eval time' / 'total time' per request. Other runtimes and heavily customized builds are not parsed."])
    if cov and cov.get("requests_parsed", 0) < 20:
        return _finding("log_thin", "Only %d requests in the log" % cov.get("requests_parsed", 0), "info", "insufficient", "passive", cov,
                        "Medians over fewer than ~20 requests are fragile; the findings are indicative only.")
    return None


def rule_microbatch(ev, label, arm):
    s = (arm or {}).get("settings") or {}
    ub = s.get("n_ubatch")
    source = s.get("settings_source") or "experiment args"
    if ub is None:
        ls = (ev.get("launch") or {}).get("launcher_settings") or {}
        ub = _f(ls.get("ubatch"))
        source = "launcher settings"
    sp = phase_split(arm["requests"]) if arm else None
    pf_share = _med(sp["prefill_share"]) if sp else None
    pf_tps = _med(percentiles([r.get("prompt_tps") for r in arm["requests"]])) if arm else None
    if ub is None:
        # an older session's value is not this session's value: report it, do not diagnose from it
        tbl = (ev.get("launch") or {}).get("session_table") or []
        older = [(r.get("session"), r.get("n_ubatch")) for r in tbl if r.get("n_ubatch")]
        note = ("Provide an experiment record (the arm's args carry --ubatch-size), launcher settings via launcher_settings_file, or a server log at a verbosity that prints n_ubatch."
                if not older else "The diagnosed session did not print n_ubatch (log verbosity); the last session that did was session %s with n_ubatch=%s, which may not be the current setting." % older[-1])
        return _finding("microbatch_unknown", "Microbatch size of the diagnosed session is not in the evidence", "info", "insufficient", "passive",
                        {"prefill_tps_median": pf_tps, "last_printed": {"session": older[-1][0], "n_ubatch": older[-1][1]} if older else None},
                        "Prefill throughput depends strongly on the microbatch (prompt tokens processed per GEMM); without the value the prefill rules cannot run.", notes=[note])
    if ub < 16 and pf_tps and pf_tps > 150:
        return _finding("microbatch_inconsistent", "Recorded microbatch %d cannot produce the measured prefill of %.0f t/s; the real setting is not in the evidence" % (ub, pf_tps),
                        "info", "insufficient", "passive", {"n_ubatch": ub, "source": source, "prefill_tps_median": pf_tps},
                        "With a microbatch below 16 prompt processing cannot exceed a small multiple of the decode rate; the measurement contradicts the setting, so the setting is stale or belongs to another session.",
                        notes=["Check which session the setting came from; a launcher settings file may describe a different launch than the one in the log."])
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
                                experiment={"kind": "request_shape", "variable": "request body fields (schema / grammar / sampler)",
                                            "arms": [{"name": "as-captured"}, {"name": "no-schema", "request_override": {"remove": ["response_format", "json_schema", "grammar"]}}],
                                            "metric": "decode t/s and generated tokens per request, and output validity; a validity loss is a decision-level difference, not a speedup",
                                            "note": "Needs a `requests` workload built from your captured request bodies (hotpath workload --from-requests). The schema-free arm shows what the constraint costs; it is a diagnosis, not a configuration to adopt if validity drops."},
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
    passive_now = not after and used is not None
    notes = None
    if passive_now:
        pid_found = ((ev.get("launch") or {}).get("server_process") or {}).get("pid_found")
        notes = ["Measured at observation time, not during requests%s; the number is the GPU's state now, not the served configuration's footprint." % (
            " and the observed server was not running" if pid_found is False else "")]
    return _finding("vram_headroom", "GPU memory headroom %s MiB of %.0f%s" % (round(headroom) if headroom else "unknown", total, " (at observation time)" if passive_now else ""),
                    "info", "likely" if not passive_now else "insufficient", arm["evidence_class"] if arm else "passive",
                    dict(meas, headroom_mib=round(headroom) if headroom else None),
                    "Headroom bounds which microbatch/context candidates can be tried without spilling; the compute buffer grows with the microbatch.", notes=notes)


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


COLD_FRACTION_VERIFIED = 0.9   # an arm counts as cold when at least this share of its requests reused no tokens
WARM_FRACTION = 0.5            # and as warm when at least this share reused tokens


def request_cache_state(r):
    """State of one request, always re-derived from the server's own cache_n (a stored label is not trusted): missing,
    null, non-numeric or negative values are "unknown", never cold."""
    from hotpath.measure import cache_state
    return cache_state({"cache_n": r.get("cache_n")})


def _verified(reqs, state):
    """Only the requests whose own state matches the comparison basis; the rest are excluded, not averaged in."""
    keep = [r for r in reqs if request_cache_state(r) == state]
    return keep, len(reqs) - len(keep)


def prompt_reuse_differs(base_reqs, cand_reqs, tolerance=0.10):
    """Warm comparisons: arms must reuse about the same number of cached tokens per prompt, or the difference in reuse,
    not the change under test, drives prefill and wall time. Returns the prompt ids whose median cache_n differs."""
    def med(reqs):
        out = {}
        for r in reqs:
            if request_cache_state(r) == "warm":
                out.setdefault(r.get("prompt_id"), []).append(float(r["cache_n"]))
        return {k: percentiles(v)["median"] for k, v in out.items()}
    a, b = med(base_reqs), med(cand_reqs)
    return sorted(p for p in a if p in b and a[p] and abs(b[p] - a[p]) / a[p] > tolerance)


def arm_cache_evidence(arm):
    """What the server's own cache_n said about an arm's requests, plus how isolation was attempted."""
    reqs = arm.get("requests") or []
    states = [request_cache_state(r) for r in reqs]
    n = len(states)
    cold = states.count("cold")
    warm = states.count("warm")
    info = arm.get("cache") or {}
    if n == 0:
        state = "unknown"
    elif cold / n >= COLD_FRACTION_VERIFIED:
        state = "cold"
    elif warm / n >= WARM_FRACTION:
        state = "warm"
    elif states.count("unknown") == n:
        state = "unknown"
    else:
        state = "mixed"
    return {"mode": info.get("mode"), "isolation": info.get("isolation"), "slot_erase_supported": info.get("slot_erase_supported"),
            "requests": n, "cold": cold, "warm": warm, "unknown": states.count("unknown"), "state": state}


def cache_comparability(base_ev, cand_ev):
    """Whether prefill-side numbers of two arms may be compared, and on what footing."""
    b, c = base_ev["state"], cand_ev["state"]
    if b == "cold" and c == "cold":
        return {"comparable": True, "basis": "cold", "detail": "both arms cold: cache_n = 0 on %d/%d and %d/%d requests (isolation: %s)" % (
            base_ev["cold"], base_ev["requests"], cand_ev["cold"], cand_ev["requests"], base_ev.get("isolation"))}
    if b == "warm" and c == "warm" and base_ev.get("mode") == cand_ev.get("mode") == "warm":
        return {"comparable": True, "basis": "warm", "detail": "both arms warm by design (cache_mode warm, same request sequence): prefill numbers describe reuse behavior, not cold processing"}
    if b == "unknown" and c == "unknown":
        return {"comparable": False, "basis": "unknown", "detail": "the server reported no usable cache_n in either arm; cache state unverified (missing evidence is not read as zero reuse)"}
    if b == "unknown" or c == "unknown":
        return {"comparable": False, "basis": "unknown", "detail": "the server reported no cache_n; cache state unverified"}
    if b == c == "warm" and "cold" in (base_ev.get("mode"), cand_ev.get("mode")):
        return {"comparable": False, "basis": "not cold", "detail": "cold was requested but requests reused cached prompt tokens (cache_n > 0 on %d/%d and %d/%d; isolation: %s): "
                "prefill, residual and wall time describe warm behavior that was not intended" % (base_ev["warm"], base_ev["requests"], cand_ev["warm"], cand_ev["requests"], base_ev.get("isolation"))}
    if b == c == "warm":
        return {"comparable": False, "basis": "unintended warm", "detail": "both arms reused cached prompt tokens without cache_mode warm (mode %s): warm state was not controlled" % base_ev.get("mode")}
    return {"comparable": False, "basis": "mismatch", "detail": "cache states differ or are mixed (baseline %s: cold %d/%d; candidate %s: cold %d/%d): prefill, residual and wall time are contaminated by cache state" % (
        b, base_ev["cold"], base_ev["requests"], c, cand_ev["cold"], cand_ev["requests"])}


def experiment_verdict(rec):
    """What a finished experiment showed: paired per-prompt ratios for prefill, decode, the per-request residual and
    wall time; cache-state comparability; VRAM delta; output equivalence; output validity; a keep/reject/inconclusive call."""
    from hotpath import compare as compare_mod
    rep = rec.get("comparison") or compare_mod.compare([rec])
    arms = rec["arms"]
    labels = list(arms)
    if len(labels) < 2:
        return None
    base = arms[labels[0]]
    base_cache = arm_cache_evidence(base)
    rows = []
    for lab in labels[1:]:
        cand = arms[lab]
        cand_cache = arm_cache_evidence(cand)
        comp = cache_comparability(base_cache, cand_cache)
        row = {"arm": lab, "vs": labels[0], "metrics": {}, "failures": len(cand.get("failures") or []),
               "cache": {"baseline": base_cache, "candidate": cand_cache, "comparability": comp}}
        base_reqs, cand_reqs = base.get("requests") or [], cand.get("requests") or []
        # Cache-sensitive metrics are paired on VERIFIED requests only: a request whose own cache_n does not match the
        # comparison basis is excluded (and counted), and a prompt needs verified requests in BOTH arms to be compared.
        # Arm-level percentages alone cannot hide a contaminated prompt.
        sensitive = ("prefill_tps", "residual_ms", "wall_ms")
        excl = {"baseline": 0, "candidate": 0, "dropped_prompts": []}
        if comp["comparable"]:
            vb, eb = _verified(base_reqs, comp["basis"])
            vc, ec = _verified(cand_reqs, comp["basis"])
            excl["baseline"], excl["candidate"] = eb, ec
            before = set(_per_prompt(base_reqs, "wall_ms")) & set(_per_prompt(cand_reqs, "wall_ms"))
            after = set(_per_prompt(vb, "wall_ms")) & set(_per_prompt(vc, "wall_ms"))
            excl["dropped_prompts"] = sorted(str(p) for p in before - after)
            if comp["basis"] == "warm":
                differs = prompt_reuse_differs(vb, vc)
                if differs:
                    comp = {"comparable": False, "basis": "warm reuse differs",
                            "detail": "both arms warm, but they reused a different number of cached tokens on prompt(s) %s (> 10%% apart): the change in reuse, not the change under test, drives prefill and wall time" % differs[:5]}
                    row["cache"]["comparability"] = comp
            if comp["comparable"] and len(after) < 2:
                comp = {"comparable": False, "basis": "too few verified prompts",
                        "detail": "after excluding requests whose cache state did not match (%d baseline, %d candidate), only %d prompt(s) have verified requests in both arms" % (eb, ec, len(after))}
                row["cache"]["comparability"] = comp
        row["cache"]["excluded"] = excl
        for metric, better in VERDICT_METRICS:
            if metric in sensitive and comp["comparable"]:
                m = paired_metric(vb, vc, metric, better)
            else:
                m = paired_metric(base_reqs, cand_reqs, metric, better)
            if m:
                if not comp["comparable"] and metric in sensitive:
                    m["call"] = "not comparable (cache state %s)" % comp["basis"]
                elif comp["basis"] == "warm" and metric == "prefill_tps":
                    m["call"] += " [warm cache: reuse behavior, not cold prefill]"
                row["metrics"][metric] = m
        bd = [x.get("process_counters_after", {}).get("dedicated_mib") for x in base.get("launch_facts") or []]
        cd = [x.get("process_counters_after", {}).get("dedicated_mib") for x in cand.get("launch_facts") or []]
        bd, cd = [x for x in bd if x is not None], [x for x in cd if x is not None]
        if bd and cd:
            row["vram_delta_mib"] = round(max(cd) - max(bd))
        full = "%s/%s" % (rec.get("name"), lab)
        oe = rep.get("output_equivalence") or {}
        ident = [e.get(full, {}).get("identical") for e in oe.values() if full in e]
        basis = next((e.get("basis") for e in oe.values() if full in e), None)
        row["outputs"] = {"prompts_compared": len(ident), "identical": sum(1 for x in ident if x), "basis": basis,
                          "first_divergence_indices": [e[full].get("first_divergence_index") for e in oe.values() if full in e and not e[full].get("identical")]}
        # validity (application-shaped requests with a schema): judged apart from speed and from output identity, on the
        # strongest level that was actually checked (full schema when the validator is installed, else required keys)
        from hotpath.measure import summarize_requests as _sr
        bv = _sr(base.get("requests") or []).get("validity") or {}
        cv = _sr(cand.get("requests") or []).get("validity") or {}
        if bv or cv:
            # decide on the full-schema check only if EVERY judged request in both arms got it; otherwise on required keys,
            # and say so. Rates, not counts: an arm with fewer requests (a failed round) is not "less valid".
            full = bool(bv and cv and bv.get("schema_checked") == bv.get("judged") and cv.get("schema_checked") == cv.get("judged"))
            key = "schema_valid" if full else "schema_required_present"
            rate = lambda v, k: (v.get(k) or 0) / v["judged"] if v and v.get("judged") else None
            rb, rc = rate(bv, key), rate(cv, key)
            jb, jc = rate(bv, "valid_json"), rate(cv, "valid_json")
            eps = 1e-9
            dropped = bool((rb is not None and rc is not None and rc < rb - eps) or (jb is not None and jc is not None and jc < jb - eps) or (bv and not cv))
            row["validity"] = {"baseline": bv, "candidate": cv, "level": (bv.get("validation_level") or cv.get("validation_level")), "decided_on": key,
                               "downgraded_from_full": (not full) and any((v.get("schema_checked") or 0) > 0 for v in (bv, cv)),
                               "dropped": dropped}
        w = row["metrics"].get("wall_ms", {})
        n_prompts = w.get("prompts") or 0
        n_rounds = int((rec.get("spec_redacted") or {}).get("rounds") or 1)
        small = n_prompts < 3 or n_rounds < 3
        scope = "on this machine and workload (%d prompts x %d rounds%s)" % (n_prompts, n_rounds, "; preliminary" if small else "")
        all_identical = row["outputs"]["prompts_compared"] > 0 and row["outputs"]["identical"] == row["outputs"]["prompts_compared"]
        if row["failures"]:
            verdict = "reject: launch failures in this arm"
        elif row.get("validity", {}).get("dropped"):
            key = row["validity"]["decided_on"]
            verdict = "reject: output validity dropped on %s (%s/%s -> %s/%s): a decision-level difference, whatever the speed" % (
                "full schema validation" if key == "schema_valid" else "the required-key check (not full schema validation)",
                bv.get(key), bv.get("judged"), cv.get(key), cv.get("judged"))
        elif not comp["comparable"]:
            verdict = "inconclusive %s: %s; only decode and validity are comparable here" % (scope, comp["detail"])
        elif w.get("call") == "improvement" and all_identical:
            verdict = "keep %s: %.1f%% wall improvement, same direction on %d/%d prompts, identical outputs (%s)" % (scope, 100 * (1 - w["ratio"]), w["consistent_direction"], w["prompts"], basis)
        elif w.get("call") == "improvement":
            verdict = "faster %s, but outputs differ on %d/%d prompts: explain the difference (numerical shift vs behavior change) before keeping" % (
                scope, row["outputs"]["prompts_compared"] - row["outputs"]["identical"], row["outputs"]["prompts_compared"])
        elif w.get("call") == "regression":
            verdict = "reject %s: %.1f%% wall regression" % (scope, 100 * (w["ratio"] - 1))
        elif any(m["call"] == "improvement" for m in row["metrics"].values()):
            verdict = "inconclusive %s: a component improved but wall time did not move beyond the within-prompt noise; more rounds or a longer workload would decide" % scope
        elif any(m["call"].startswith("improvement within noise") for m in row["metrics"].values()):
            verdict = "inconclusive %s: a small improvement within the within-prompt noise; more rounds would decide" % scope
        else:
            verdict = "no measurable difference %s" % scope
        if comp["comparable"] and comp["basis"] == "warm" and verdict.startswith("keep"):
            verdict += " [warm-cache comparison: prefill reflects reuse, not cold processing]"
        if verdict.startswith("keep") and (excl["baseline"] or excl["candidate"] or excl["dropped_prompts"]):
            verdict += " [cache-sensitive metrics exclude %d baseline / %d candidate unverified requests%s]" % (
                excl["baseline"], excl["candidate"], "; prompts dropped: %s" % excl["dropped_prompts"][:5] if excl["dropped_prompts"] else "")
        if verdict.startswith("keep") and not row.get("validity"):
            verdict += " [no schema in the requests: output validity was not checked]"
        if verdict.startswith("keep") and row.get("validity", {}).get("downgraded_from_full"):
            verdict += " [validity decided on required keys only: not every request got full schema validation]"
        if verdict.startswith("keep"):
            verdict += " [output identity and application-level decision agreement are separate checks; neither implies the other]"
        row["verdict"] = verdict
        if base.get("slot_erase_error") or cand.get("slot_erase_error"):
            row["slot_erase_note"] = "slot erase unsupported on this server (%s); isolation used: %s" % (
                base.get("slot_erase_error") or cand.get("slot_erase_error"), base_cache.get("isolation") or cand_cache.get("isolation"))
        rows.append(row)
    return {"experiment": rec.get("name"), "evidence_class": rep.get("evidence_class"), "comparability_issues": rep.get("comparability_issues"), "arms": rows,
            "scope_note": "A verdict describes this workload on this machine; it is not a property of the runtime in general."}


def insufficiencies(ev, arm):
    out = []
    passive_only = bool(arm) and arm.get("evidence_class") == "passive"
    if passive_only:
        out.append("Only server-log evidence: client wall time, per-request overhead (cache saves, checkpoints, HTTP) and GPU state during requests are not in a log. "
                   "Single best next measurement: `hotpath measure --server <your server> --workload <w>` (a few requests, your own server, cache off) — it adds the wall time and the residual to the same diagnosis.")
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
    if not cache_pattern_evidence(ev).get("available"):
        out.append("Prefix-cache effectiveness of real traffic is unknown: hotpath's workloads disable the prompt cache on purpose; an `observe --log` record of the real server supplies the reuse pattern.")
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
    f = rule_coverage(ev)
    if f:
        findings.append(f)
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
    f = rule_cache_reuse(ev)
    if f:
        findings.append(f)
    fs = rule_sessions(ev)
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
            v["caveats"] = list(origin.get("risks") or [])
        # a flag that trades prefix reuse for per-request overhead: its production applicability depends on real traffic's reuse pattern
        if prov.get("experiment_kind") == "server_overhead_flags" or any("--cache-ram" in str(a.get("args_override")) or "--ctx-checkpoints" in str(a.get("args_override"))
                                                                          for a in (r.get("spec_redacted") or {}).get("arms") or []):
            cp = cache_pattern_evidence(ev)
            if not cp.get("available"):
                gate = "production applicability unknown: no cache-pattern evidence for real traffic (add an `observe --log` record of the real server). The gain was measured on isolated requests with the prompt cache off."
            elif cp["level"] == "low":
                gate = "production applicability supported: real traffic reuses a median %.0f%% of each prompt (p90 %.0f%%), so little prefix reuse is lost." % (100 * cp["reused_share_median"], 100 * cp["reused_share_p90"])
            else:
                gate = "do not apply to production on this evidence: real traffic reuses a median %.0f%% of each prompt; removing the cache would cost more prompt processing than this saves." % (100 * cp["reused_share_median"])
            v["production_gate"] = gate
            for row in v["arms"]:
                if row["verdict"].startswith("keep"):
                    row["verdict"] += " — " + gate
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
            lines.append("  - outputs identical %d/%d by %s%s; VRAM delta %s MiB; failures %d" % (row["outputs"]["identical"], row["outputs"]["prompts_compared"], row["outputs"].get("basis"),
                         "" if not row["outputs"]["first_divergence_indices"] or row["outputs"]["first_divergence_indices"] == [-1] * len(row["outputs"]["first_divergence_indices"]) else " (first divergence at %s)" % row["outputs"]["first_divergence_indices"], row.get("vram_delta_mib"), row["failures"]))
            cc = row.get("cache") or {}
            if cc:
                lines.append("  - cache state: %s" % cc["comparability"]["detail"])
                ex_ = cc.get("excluded") or {}
                if ex_.get("baseline") or ex_.get("candidate") or ex_.get("dropped_prompts"):
                    lines.append("  - excluded from cache-sensitive metrics as unverified: %d baseline / %d candidate requests; prompts dropped: %s" % (
                        ex_.get("baseline", 0), ex_.get("candidate", 0), ex_.get("dropped_prompts") or "none"))
            if row.get("slot_erase_note"):
                lines.append("  - note: %s" % row["slot_erase_note"])
            if row.get("validity"):
                b, c = row["validity"]["baseline"], row["validity"]["candidate"]
                full = "; full schema valid %s/%s -> %s/%s" % (b.get("schema_valid"), b.get("judged"), c.get("schema_valid"), c.get("judged")) if b.get("schema_valid") is not None else "; full schema validation not run (%s)" % row["validity"].get("level")
                lines.append("  - validity: valid JSON %s/%s -> %s/%s; required keys present %s/%s -> %s/%s%s%s" % (
                    b.get("valid_json"), b.get("judged"), c.get("valid_json"), c.get("judged"), b.get("schema_required_present"), b.get("judged"), c.get("schema_required_present"), c.get("judged"),
                    full, " (DROPPED)" if row["validity"]["dropped"] else ""))
                unchecked = sorted(set(b.get("schema_unchecked_keywords") or []) | set(c.get("schema_unchecked_keywords") or []))
                if unchecked:
                    lines.append("  - schema keywords present but not enforced by the validator: %s" % ", ".join(unchecked))
                if row["validity"].get("downgraded_from_full"):
                    lines.append("  - note: not every request got full schema validation; the verdict used the required-key check")
        if v.get("production_gate"):
            lines.append("- production gate: %s" % v["production_gate"])
        for c in v.get("caveats") or []:
            lines.append("- caveat (from finding `%s`): %s" % (v.get("tests_finding"), c))
        lines.append("- scope: %s" % v.get("scope_note"))
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
