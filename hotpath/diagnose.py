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
    flags = {flag} | {{"--ubatch-size": "-ub", "--batch-size": "-b", "--ctx-size": "-c", "--gpu-layers": "-ngl", "--parallel": "-np",
                       "--cache-type-k": "-ctk", "--cache-type-v": "-ctv", "--cache-ram": "-cram"}.get(flag, flag)}
    for i, a in enumerate(args[:-1]):
        if a in flags:
            return args[i + 1]
    return default


# ---- evidence collection -----------------------------------------------------------------------

def launch_settings(launch_rec):
    """Diagnosis settings from an observed command line (observe/measure records), labelled as observed. Values not on
    the command line stay None: a build default or LLAMA_ARG_* variable may set them."""
    from hotpath import launch as launch_mod
    if not launch_rec or not launch_rec.get("available"):
        return {}
    s = launch_rec.get("settings") or {}
    return {"n_ubatch": _f(s.get("ubatch_size")), "n_batch": _f(s.get("batch_size")), "n_ctx": _f(s.get("ctx_size")),
            "n_parallel": _f(s.get("parallel")), "cache_ram": s.get("cache_ram"), "ctx_checkpoints": s.get("ctx_checkpoints"),
            "n_gpu_layers": s.get("n_gpu_layers"), "flash_attn": s.get("flash_attn"), "cache_type_k": s.get("cache_type_k"),
            "cache_type_v": s.get("cache_type_v"), "override_tensor": s.get("override_tensor"), "cpu_moe": s.get("cpu_moe") or s.get("n_cpu_moe"),
            "settings_source": launch_rec.get("source") or "server command line (observed)",
            "argv_redacted": launch_rec.get("argv_redacted"), "describe": launch_mod.describe(s)}


def collect(records):
    """Flatten records into one evidence set. Nothing is interpreted here."""
    ev = {"arms": {}, "gpu": {}, "model": None, "bench": [], "profiles": [], "log_stats": None, "launch": {},
          "classes": set(), "experiments": [], "sources": [], "server_props": None, "observed_launch": None, "metrics": None,
          "process_vram": None}
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
                                 "n_gpu_layers": _args_value(args, "--gpu-layers") or _args_value(args, "--n-gpu-layers"),
                                 "cache_type_k": _args_value(args, "--cache-type-k"), "cache_type_v": _args_value(args, "--cache-type-v"),
                                 "env_ggml": arm.get("env_ggml") or {}},
                    "vram": {"process_dedicated_mib": [x.get("process_counters_after", {}).get("dedicated_mib") for x in lf],
                             "process_shared_mib": [x.get("process_counters_after", {}).get("shared_mib") for x in lf],
                             "gpu_after": [x.get("gpu_after") for x in lf]},
                    "binary": arm.get("binary") or rec.get("binary"), "failures": arm.get("failures") or [], "args": args}
            if not ev["model"] and rec.get("model") and rec["model"].get("summary"):
                ev["model"] = rec["model"]["summary"]
        elif kind == "measurement":
            props = (rec.get("target") or {}).get("props") or {}
            settings = {"n_ctx": _f(props.get("n_ctx_per_slot")), "n_parallel": _f(props.get("total_slots"), 1), "env_ggml": {}}
            observed = launch_settings(rec.get("server_launch"))
            for k, v in observed.items():
                if v is not None and k not in ("n_ctx", "n_parallel"):
                    settings[k] = v
            ev["arms"][rec.get("label") or rec["created"]] = {
                "requests": rec.get("requests") or [], "evidence_class": "active", "settings": settings,
                "vram": {"gpu_after": [s.get("gpu") for s in rec.get("gpu_snapshots") or []],
                         "process_dedicated_mib": [(s.get("process") or {}).get("dedicated_mib") for s in rec.get("gpu_snapshots") or []],
                         "process_shared_mib": [(s.get("process") or {}).get("shared_mib") for s in rec.get("gpu_snapshots") or []]},
                "failures": []}
            if observed:
                ev["observed_launch"] = observed
            ev["server_props"] = ev["server_props"] or props
        elif kind == "observation":
            log = rec.get("log") or {}
            ev["log_stats"] = log.get("gpu_like_sessions_stats") or ev["log_stats"]
            sp = rec.get("server_process") or {}
            ev["launch"] = {"launcher_settings": rec.get("launcher_settings") or {}, "launcher_env": rec.get("launcher_env_documented") or {},
                            "session_table": log.get("session_table") or [], "gpu_now": rec.get("gpu_now") or {},
                            "server_process": sp}
            if launch_settings(sp.get("launch")):
                ev["observed_launch"] = launch_settings(sp.get("launch"))
            if (rec.get("metrics") or {}).get("available"):
                ev["metrics"] = rec["metrics"]
            if (sp.get("gpu_counters") or {}).get("dedicated_mib") is not None:
                ev["process_vram"] = sp["gpu_counters"]
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
        # the newest session is the running server when an observation also read its command line: fill what the log did not print
        obs = ev.get("observed_launch") or {}
        if obs and sid == max(by_session):
            filled = [k for k in ("n_ubatch", "n_batch") if settings.get(k) is None and obs.get(k) is not None]
            for k, v in obs.items():
                if settings.get(k) is None and v is not None:
                    settings[k] = v
            if filled:
                settings["settings_source"] = "server log, session %s; %s from %s" % (sid, "/".join(filled), obs.get("settings_source"))
        label = "%s/session-%s" % (log.get("source") or "log", sid)
        return label, {"requests": reqs, "evidence_class": "passive", "settings": settings, "vram": {},
                       "failures": [], "time_basis": "server prompt+generation time from the log; no client wall time", "session": sid}
    return None, None


def metrics_summary(ev):
    """Real-traffic aggregates from llama-server's /metrics counters (since the server started): prefill and decode
    throughput, the time split between them, and the share of prompt tokens served from the prompt cache. None when
    /metrics was not read or nothing was served yet."""
    m = ev.get("metrics") or {}
    t = m.get("since_server_start") or {}
    ps, gs = _f(t.get("prompt_seconds_total"), 0.0), _f(t.get("tokens_predicted_seconds_total"), 0.0)
    pt, gt = _f(t.get("prompt_tokens_total"), 0.0), _f(t.get("tokens_predicted_total"), 0.0)
    if ps + gs <= 0:
        return None
    cached = _f(t.get("prompt_tokens_cached_total"))
    return {"prompt_tokens": int(pt), "generated_tokens": int(gt), "prompt_seconds": round(ps, 2), "generation_seconds": round(gs, 2),
            "prefill_tps": round(pt / ps, 1) if ps else None, "decode_tps": round(gt / gs, 1) if gs else None,
            "prefill_share_of_server_time": round(ps / (ps + gs), 3), "decode_share_of_server_time": round(gs / (ps + gs), 3),
            "cached_prompt_tokens": int(cached) if cached is not None else None,
            "cached_share_of_prompt_tokens": round(cached / (cached + pt), 3) if cached is not None and cached + pt > 0 else None,
            "n_busy_slots_per_decode": _f(t.get("n_busy_slots_per_decode")), "requests_deferred": _f(t.get("requests_deferred")),
            "source": m.get("source") or "llama-server /metrics", "window_s": m.get("window_s")}


def _passive_arm_from_metrics(ev):
    """No log and no measurement: a passive arm whose rates come from /metrics and settings from the command line."""
    ms = metrics_summary(ev)
    obs = ev.get("observed_launch") or {}
    if not ms and not obs:
        return None, None
    settings = dict(obs) if obs else {}
    settings.setdefault("env_ggml", (ev.get("launch") or {}).get("launcher_env") or {})
    return "running-server", {"requests": [], "evidence_class": "passive", "settings": settings, "vram": {}, "failures": [], "metrics": ms,
                              "time_basis": "aggregate /metrics counters (no per-request detail)" if ms else "command line only (no traffic data)"}


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
    if not arm:
        label, arm = _passive_arm_from_metrics(ev)
    if arm:
        ev["arms"][label] = arm
    return label, arm


def _finding(fid, title, severity, confidence, evidence_class, measurements, why, experiment=None, opportunity=None, risks=None, notes=None,
             known=None, hypothesis=None):
    f = {"id": fid, "title": title, "severity": severity, "confidence": confidence, "evidence_class": evidence_class,
         "measurements": measurements, "why_it_matters": why}
    if known:
        f["known"] = known
    if hypothesis:
        f["hypothesis"] = hypothesis
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
    if not sp and arm and arm.get("metrics"):
        m = arm["metrics"]
        pf, dc = m["prefill_share_of_server_time"], m["decode_share_of_server_time"]
        shape = "prefill-bound" if pf >= 0.6 else ("decode-bound" if dc >= 0.6 else "mixed")
        return _finding("workload_phase_split", "Real traffic is %s (prefill %.0f%%, decode %.0f%% of the server's prompt+generation time since it started)" % (shape, 100 * pf, 100 * dc),
                        "info", "likely", "passive", dict(m, arm=label, time_basis="server metrics totals"),
                        "Which side of a request a change acts on bounds what it can gain; these shares come from the server's own counters for all traffic since it started.",
                        known=["%d prompt tokens processed in %.1f s and %d tokens generated in %.1f s since the server started (/metrics)" % (
                            m["prompt_tokens"], m["prompt_seconds"], m["generated_tokens"], m["generation_seconds"])],
                        notes=["Aggregate counters have no per-request spread and no client wall time; a server log (--log-file) or `hotpath measure` adds both."])
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
        ms = metrics_summary(ev)
        if ms and ms.get("cached_share_of_prompt_tokens") is not None and ms["prompt_tokens"] + (ms["cached_prompt_tokens"] or 0) > 0:
            share = ms["cached_share_of_prompt_tokens"]
            level = "low" if share < 0.15 else ("high" if share >= 0.5 else "mixed")
            # an aggregate has no distribution: the same value stands for median and p90, and the basis says so
            tokens = ms["prompt_tokens"] + (ms["cached_prompt_tokens"] or 0)
            return {"available": True, "reused_share_median": share, "reused_share_p90": share, "lcp_fraction": None, "requests": None,
                    "level": level, "aggregate": True, "tokens": tokens, "thin": tokens < MIN_TOKENS_FOR_REUSE,
                    "source": "llama-server /metrics: share of all prompt tokens served from the prompt cache since the server started (aggregate, real traffic)"}
        return {"available": False, "note": "no slot-selection/release lines in a log and no /metrics prompt-cache counter; reuse pattern of real traffic unknown"}
    share = cache["reused_share_of_prompt"]
    sel = cache.get("selected_by") or {}
    n_sel = sum(sel.values()) or 1
    lcp = sel.get("LCP similarity", 0) / n_sel
    med = share["median"]
    level = "low" if med < 0.15 and share["p90"] < 0.5 else ("high" if med >= 0.5 else "mixed")
    n = cache.get("requests_with_reuse_data") or 0
    return {"available": True, "reused_share_median": med, "reused_share_p90": share["p90"], "lcp_fraction": round(lcp, 3), "requests": n,
            "level": level, "thin": n < MIN_REQUESTS_FOR_REUSE, "source": "server log slot-selection and release lines (real traffic)"}


MIN_REQUESTS_FOR_REUSE = 20     # fewer logged requests say nothing about the traffic's reuse pattern
MIN_TOKENS_FOR_REUSE = 20000    # the /metrics aggregate has no request count; this many prompt tokens is a few dozen typical requests


def reuse_text(cp):
    """How much real traffic reuses, worded for its basis (per-request log rows, or an aggregate /metrics counter)."""
    if cp.get("aggregate"):
        return "%.0f%% of all %d prompt tokens since the server started came from the prompt cache (/metrics aggregate)" % (100 * cp["reused_share_median"], cp.get("tokens") or 0)
    return "real requests reuse a median %.0f%% of their prompt (p90 %.0f%%; LCP-selected %.0f%%)" % (
        100 * cp["reused_share_median"], 100 * cp["reused_share_p90"], 100 * (cp.get("lcp_fraction") or 0))


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
    if cp.get("available") and not cp.get("thin"):
        risks.append("Cache-pattern evidence: %s; reuse is %s, so a flag that removes it is %s." % (
            reuse_text(cp), cp["level"], "defensible for this traffic" if cp["level"] == "low" else "likely to cost more than it saves on this traffic"))
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
                    risks=risks,
                    known=["client wall time minus the server's prompt and generation time is %.0f ms per request (median of %d)" % (rms, sp["n"])],
                    hypothesis="Prompt-cache state saves and context checkpoints are a large part of this time; the flag arms show how much each one costs.")


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
            "reused_tokens_median": _med(cache.get("reused_tokens")), "selected_by": sel, "f_sim_best_median": _med(cache.get("f_sim_best")),
            "basis": cp["source"]}
    if cp.get("thin"):
        return _finding("prefix_reuse_thin", "Too little real traffic to judge prefix reuse (%s)" % (
            "%d prompt tokens in /metrics" % cp["tokens"] if cp.get("aggregate") else "%d logged requests" % cp["requests"]),
            "info", "insufficient", "passive", meas,
            "Whether prompt caching pays off depends on how real requests repeat; a handful of requests cannot show it.",
            notes=["Observe again after the server has served typical traffic (a few dozen requests)."])
    title = "Real traffic: %s" % reuse_text(cp) if cp.get("aggregate") else \
        "Real requests reuse a median %.0f%% of their prompt from the slot (p90 %.0f%%)" % (100 * cp["reused_share_median"], 100 * cp["reused_share_p90"])
    if cp["level"] == "low":
        return _finding("prefix_reuse_low", title,
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
    return _finding("prefix_reuse_%s" % cp["level"], title,
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
    if (arm or {}).get("metrics") and pf_tps is None:
        pf_tps, pf_share = arm["metrics"].get("prefill_tps"), arm["metrics"].get("prefill_share_of_server_time")
    obs = ev.get("observed_launch") or {}
    full_args = (arm or {}).get("evidence_class") == "controlled" and (arm or {}).get("args")
    if ub is None and (full_args or (obs and obs.get("n_ubatch") is None and (arm or {}).get("evidence_class") != "controlled")):
        # the command line is known and does not set it: the build default applies (512 in current llama.cpp)
        where = "the arm's command line (launched by hotpath)" if full_args else obs.get("settings_source")
        return _finding("microbatch_default", "Microbatch not set on the command line: the build's default applies", "info", "likely", "passive",
                        {"source": where, "prefill_tps": pf_tps},
                        "llama-server's default microbatch (512 in current llama.cpp) is already in the range where prompt processing is batched GEMM work.",
                        known=["%s has no -ub/--ubatch-size" % where],
                        notes=["A LLAMA_ARG_UBATCH environment variable would override the default and is not visible in a command line."])
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
                    known=["microbatch %d (%s)" % (ub, source)] + (["prefill %.0f t/s, %.0f%% of the request" % (pf_tps, 100 * pf_share)] if pf_tps and pf_share else []),
                    hypothesis="A larger microbatch makes prompt processing faster on this GPU; how much is unknown until measured.",
                    risks=["Larger microbatches need a larger compute buffer (VRAM); on a full card this can spill to system memory and look slower.",
                           "Changing the microbatch changes floating-point accumulation order; outputs can differ at near-ties (a numerical shift, not an error)."])


KV_TYPE_FACTOR = {"f16": 1.0, "bf16": 1.0, "f32": 2.0, "q8_0": 34 / 64, "q5_1": 24 / 64, "q5_0": 22 / 64, "q4_1": 20 / 64, "q4_0": 18 / 64, "iq4_nl": 18 / 64}
PROCESS_VRAM_PARTIAL = 0.85     # server's dedicated VRAM below this fraction of the model's weights: not all weights can be on the GPU


def offload_evidence(ev, arm):
    """How many of the model's layers are on the GPU, from the strongest source available: the server log's own
    'offloaded N/M layers' line, an explicit -ngl on the command line, or (weakest) the server's dedicated VRAM
    against the model's weight bytes. llama.cpp counts the output layer, so a 64-block model reports N/65."""
    model = ev.get("model") or {}
    total = model["n_layer"] + 1 if model.get("n_layer") else None
    s = (arm or {}).get("settings") or {}
    tbl = [r for r in (ev.get("launch") or {}).get("session_table") or [] if r.get("offloaded")]
    sid = (arm or {}).get("session")
    row = next((r for r in tbl if r.get("session") == sid), None) if sid else (tbl[-1] if tbl and (arm or {}).get("evidence_class") != "controlled" else None)
    if row:
        try:
            n, m = (int(x) for x in str(row["offloaded"]).split("/"))
            return {"on_gpu": n, "total": m, "source": "server log: 'offloaded %s layers to GPU' (session %s)" % (row["offloaded"], row.get("session")),
                    "kind": "log", "session": row}
        except ValueError:
            pass
    from hotpath.launch import int_or_none
    ngl = int_or_none(s.get("n_gpu_layers"))
    if ngl is not None and ngl >= 0 and total:
        obs = ev.get("observed_launch") or {}
        where = obs.get("settings_source") if obs.get("n_gpu_layers") == s.get("n_gpu_layers") and obs.get("settings_source") else (s.get("settings_source") or "launch arguments")
        return {"on_gpu": min(ngl, total), "total": total, "explicit_ngl": ngl,
                "source": "-ngl %d on the %s; the model has %d blocks + the output layer" % (ngl, where, total - 1), "kind": "args"}
    ded = [x for x in ((arm or {}).get("vram") or {}).get("process_dedicated_mib") or [] if x is not None]
    pv = max(ded) if ded else (ev.get("process_vram") or {}).get("dedicated_mib")
    if pv and model.get("total_bytes"):
        w = model["total_bytes"] / (1 << 20)
        return {"on_gpu": None, "total": total, "process_dedicated_mib": pv, "weights_mib": round(w), "partial_by_vram": pv < PROCESS_VRAM_PARTIAL * w,
                "source": "server process dedicated VRAM %.0f MiB vs %.0f MiB of model weights (Windows GPU counters)" % (pv, w), "kind": "vram"}
    return None


def memory_estimate(ev, arm, o):
    """Would all layers fit? Weights + KV cache at the configured context and KV types + the compute buffer, against
    the memory free when the server started (log) or total VRAM minus what other processes use. An estimate with its
    assumptions listed, used only to choose experiment arms; the experiment's per-arm VRAM is the evidence."""
    model = ev.get("model") or {}
    gpu = ev.get("gpu") or {}
    s = (arm or {}).get("settings") or {}
    row = o.get("session") or {}
    total_mib = _f(gpu.get("memory.total"))
    if not model.get("total_bytes") or not total_mib or not o.get("total"):
        return None
    w = model["total_bytes"] / (1 << 20)
    assumptions = []
    n_ctx = _f(row.get("n_ctx")) or s.get("n_ctx") or _f(((ev.get("server_props") or {}).get("n_ctx_per_slot")))
    kv_mib = _f(row.get("kv_mib"))
    if kv_mib is None and n_ctx and model.get("kv_bytes_per_token_f16"):
        fk = KV_TYPE_FACTOR.get(str(s.get("cache_type_k") or "f16").lower(), 1.0)
        fv = KV_TYPE_FACTOR.get(str(s.get("cache_type_v") or "f16").lower(), 1.0)
        kv_mib = model["kv_bytes_per_token_f16"] * n_ctx * (fk + fv) / 2 / (1 << 20)
        assumptions.append("KV cache from the GGUF header at n_ctx %d (%s/%s)" % (n_ctx, s.get("cache_type_k") or "f16", s.get("cache_type_v") or "f16"))
    elif kv_mib is None:
        kv_mib = 0.0
        assumptions.append("KV cache size unknown (no context size in the evidence); counted as 0")
    compute = _f(row.get("compute_mib"))
    if compute is None:
        compute = 512.0
        assumptions.append("compute buffer 512 MiB (an allowance; the log did not print it)")
    free = _f(row.get("free_at_start_mib"))
    if free is None:
        now = (ev.get("launch") or {}).get("gpu_now") or {}
        used, ded = _f(now.get("mem_used_mib")), (ev.get("process_vram") or {}).get("dedicated_mib")
        others = (used - ded) if (used is not None and ded is not None) else None
        free = total_mib - (others if others is not None and others >= 0 else 600.0)
        assumptions.append("free VRAM = total minus %s" % ("what other processes used at observation time" if others is not None else "a 600 MiB allowance for the desktop and other programs"))
    need_all = w + kv_mib + compute
    per_layer = w / o["total"]
    fit_layers = int(max(0.0, (free * 0.95 - kv_mib - compute)) // per_layer) if per_layer else None
    return {"weights_mib": round(w), "kv_mib": round(kv_mib), "compute_mib": round(compute), "need_all_layers_mib": round(need_all),
            "free_mib": round(free), "fits_all": need_all < free * 0.95, "layers_that_fit": min(fit_layers, o["total"]) if fit_layers is not None else None,
            "assumptions": assumptions}


def rule_offload(ev, label, arm):
    o = offload_evidence(ev, arm)
    if not o:
        return None
    if o["kind"] == "vram":
        if not o["partial_by_vram"]:
            return None
    elif o["on_gpu"] >= o["total"]:
        return None
    s = (arm or {}).get("settings") or {}
    deliberate = bool(s.get("override_tensor") or s.get("cpu_moe"))
    sp = phase_split(arm["requests"]) if arm else None
    est = memory_estimate(ev, arm, o) if o.get("total") else None
    n, m = o.get("on_gpu"), o.get("total")
    model = ev.get("model") or {}
    w = (model.get("total_bytes") or 0) / (1 << 20)
    if o["kind"] == "vram":
        title = "The server holds %.0f MiB of VRAM for %.0f MiB of model weights: part of the model is not on the GPU" % (o["process_dedicated_mib"], w)
        known = [o["source"]]
        confidence = "hypothesis"
    else:
        cpu_share = (m - n) / m
        title = "%d of %d layers run on the GPU; %d run on the CPU (about %.0f%% of the weights, %.0f MiB)" % (n, m, m - n, 100 * cpu_share, cpu_share * w)
        known = [o["source"]]
        confidence = "likely"
    if (o.get("session") or {}).get("projected_mib"):
        r = o["session"]
        known.append("at launch the runtime projected %s MiB of device memory against %s MiB free (its automatic fit then chose the layer count)" % (r["projected_mib"], r["free_at_start_mib"]))
    if deliberate:
        known.append("the command line keeps tensors on the CPU on purpose (%s)" % ("--override-tensor" if s.get("override_tensor") else "--cpu-moe/--n-cpu-moe"))
    arms, how = None, None
    if est and o["kind"] != "vram":
        current = {"name": "ngl%d" % o["explicit_ngl"], "args_override": {"--gpu-layers": o["explicit_ngl"]}} if o.get("explicit_ngl") is not None else {"name": "current", "args_override": {}}
        if est["fits_all"]:
            arms = [current, {"name": "ngl%d" % m, "args_override": {"--gpu-layers": m}}]
        elif est["layers_that_fit"] and est["layers_that_fit"] > n + 1:
            arms = [current, {"name": "ngl%d" % est["layers_that_fit"], "args_override": {"--gpu-layers": est["layers_that_fit"]}}]
        else:
            how = ("by the estimate no more layers fit (need %d MiB for all layers, about %d MiB free): free VRAM first (close other GPU programs), shrink the KV cache "
                   "(smaller --ctx-size, or -ctk q8_0 -ctv q8_0 with flash attention), or use a smaller quantization; then compare against the current launch" % (est["need_all_layers_mib"], est["free_mib"]))
    elif o["kind"] == "vram":
        how = "read the server's log for its 'offloaded N/M layers' line (start it with --log-file), or compare an explicit --gpu-layers %s against the current launch" % (m or "N")
    experiment = {"kind": "gpu_layers", "variable": "--gpu-layers", "arms": arms or how,
                  "metric": "decode and prefill t/s and wall_ms medians; per-arm dedicated and shared (system-memory) VRAM; outputs equivalent",
                  "note": "One variable: the number of GPU layers; context, KV types and microbatch stay as launched. The memory estimate behind the candidate is in the measurements."}
    return _finding("gpu_offload_partial", title, "high", confidence, (arm or {}).get("evidence_class") or "passive",
                    {"offload": {k: v for k, v in o.items() if k != "session"}, "memory_estimate": est, "deliberate_cpu_tensors": deliberate},
                    "Layers left on the CPU read their weights from system memory, typically tens of GB/s on a desktop against hundreds of GB/s of VRAM, and the GPU waits for them "
                    "on every generated token; prompt processing also leaves those layers' matrix work to the CPU. A partly offloaded model is usually limited by its CPU part, not by GPU kernels.",
                    experiment=experiment,
                    opportunity={"bound_share_of_request": round(1 - (_med(sp["residual_share"]) or 0), 3) if sp and sp.get("residual_share") else None,
                                 "basis": "prompt processing and generation both include the CPU-resident layers; the gain is unknown until measured"},
                    risks=["More GPU layers need more VRAM: a launch can fail with an allocation error, or on Windows silently spill to shared system memory (watch per-arm shared MiB); "
                           "either is reported by the experiment.",
                           "If the runtime's automatic fit chose this split, forcing more layers removes the memory margin it kept for the context and compute buffers."] +
                          (["The CPU placement looks deliberate (expert tensors kept on the CPU); tune --n-cpu-moe rather than forcing everything onto the GPU."] if deliberate else []),
                    known=known,
                    hypothesis="The CPU-resident layers set the pace of each token; with them in VRAM, decode and prefill rise and the request gets shorter.",
                    notes=["The decode-ceiling comparison below assumes all weights in VRAM; with a partial offload it is not the first thing to fix."])


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
    if dc_tps is None and (arm or {}).get("metrics"):
        dc_tps = arm["metrics"].get("decode_tps")
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
    if graphs_off:
        # the current configuration (graphs off) is the baseline arm; the candidate turns them on
        exp = {"kind": "cuda_graphs", "variable": "GGML_CUDA_DISABLE_GRAPHS",
               "arms": [{"name": "graphs-off", "env_override": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}, {"name": "graphs-on", "env_remove": ["GGML_CUDA_DISABLE_GRAPHS"]}],
               "metric": "decode t/s median; outputs equivalent", "note": "Whether the build enables graphs on this GPU architecture is decided by the runtime; the experiment shows the effect either way."}
    else:
        # graphs are not disabled, so toggling them could only confirm the current state: what decides the next lever is where decode time goes
        exp = {"kind": "profile_decode", "variable": "none (profiling run)",
               "arms": "hotpath profile --label decode --pp 0 --tg 64 (ACTIVE, Nsight Systems): GPU time by kernel class and the share of tiny kernels; with CUDA graphs already on, "
                       "the remaining levers are kernel efficiency for this quantization and the number of kernels per token",
               "metric": "GPU time share by kernel class; kernel launches under 20 us"}
    why = ("Achieved decode is well below what streaming the weights alone would allow. The usual causes, in order of how often they decide it: per-kernel launch overhead "
           "(hundreds of small kernels per token; CUDA graphs remove most of it), mat-vec kernels that do not reach bandwidth for this quantization type, extra passes for "
           "non-attention layers, and CPU-side per-token work in the server.")
    risks = ["CUDA graphs change nothing numerically but can fail to capture on some ops/builds; a launch failure or a fallback warning in the server log is the signal."]
    if graphs_off:
        why = "The launch environment disables CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1). " + why
    return _finding("decode_below_ceiling", "Decode reaches %.0f%% of the peak-bandwidth ceiling (%.1f of %.0f t/s)" % (100 * frac, dc_tps, ceiling), sev,
                    "likely" if graphs_off else "hypothesis", "passive", meas, why,
                    opportunity={"bound": "up to the ceiling fraction: %.0f%% of decode time is above the bandwidth floor" % (100 * (1 - frac)), "basis": "peak theoretical bandwidth; achievable is typically 70-90% of it"},
                    experiment=exp, risks=risks,
                    known=["decode %.1f t/s; %.3f GB of weights per token at %.0f GB/s peak gives a %.0f t/s ceiling" % (dc_tps, model["weight_bytes_streamed_per_decode_token"] / 1e9, bw, ceiling)] +
                          (["the launch environment sets GGML_CUDA_DISABLE_GRAPHS=1"] if graphs_off else []),
                    hypothesis="Per-kernel launch overhead is the largest part of the gap and CUDA graphs remove it" if graphs_off else
                               "Some of the gap is kernel efficiency for this quantization or the number of kernels per token; only a decode profile tells which")


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
            env = (ev.get("launch") or {}).get("launcher_env") or {}
            if str(env.get("GGML_CUDA_DISABLE_GRAPHS", "")) == "1":
                notes.append("Most kernel launches are under 20 us and the launch environment disables CUDA graphs: graphs are the first lever.")
                exp = {"kind": "cuda_graphs", "variable": "GGML_CUDA_DISABLE_GRAPHS",
                       "arms": [{"name": "graphs-off", "env_override": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}, {"name": "graphs-on", "env_remove": ["GGML_CUDA_DISABLE_GRAPHS"]}],
                       "metric": "decode t/s median"}
            else:
                notes.append("Most kernel launches are under 20 us: decode is launch-count sensitive; with CUDA graphs on, fewer kernels per token (fusion, in the runtime) is the lever.")
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
            base_ev["cold"], base_ev["requests"], cand_ev["cold"], cand_ev["requests"], base_ev.get("isolation") or "not recorded; state taken from cache_n")}
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
        # records written before the basis field existed compared token ids, the only method then
        basis = next((e.get("basis") for e in oe.values() if full in e and e.get("basis")), None) or "token ids"
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
        what = "server-log" if arm.get("requests") else ("/metrics" if arm.get("metrics") else "command-line")
        out.append("Only %s evidence: client wall time, per-request overhead (cache saves, checkpoints, HTTP) and GPU state during requests are not in it. "
                   "Single best next measurement: `hotpath measure --server <your server> --workload <w> --i-own-this-server` (a few requests on your own server, cache off) - "
                   "it adds the wall time and the residual to the same diagnosis." % what)
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
    cpe = cache_pattern_evidence(ev)
    if not cpe.get("available") or cpe.get("thin"):
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


def cache_conditions(records):
    """The prompt-cache state under which each measured request set was taken, from the server's own cache_n per request.
    Reported for every measurement and experiment arm so a reader (or an agent) never has to infer it: a record written
    before cache_n was kept, or by a server that does not report it, says `unknown`, never cold."""
    out = []
    for r in records:
        if r.get("kind") == "measurement" and r.get("requests"):
            out.append(dict(arm_cache_evidence({"requests": r["requests"], "cache": r.get("cache")}), source="measurement %s" % (r.get("label") or r.get("name") or "")))
        elif r.get("kind") == "experiment":
            for lab, a in (r.get("arms") or {}).items():
                if a.get("requests"):
                    out.append(dict(arm_cache_evidence(a), source="experiment %s / %s" % (r.get("name"), lab)))
    return out


# which part of a request a finding's candidate change acts on; the measured share of that part bounds what it can gain
AFFECTS = {"gpu_offload_partial": ("prefill", "decode"), "cpu_fallback_sessions": ("all",), "vram_pressure": ("all",),
           "microbatch_limits_prefill": ("prefill",), "microbatch_not_small": ("prefill",), "prefill_server_gap": ("prefill",),
           "decode_below_ceiling": ("decode",), "decode_near_ceiling": ("decode",), "decode_server_gap": ("decode",),
           "per_request_overhead": ("residual",), "kernel_time_share_decode": ("decode",), "kernel_time_share_prefill": ("prefill",)}
CONFIDENCE_WEIGHT = {"confirmed": 1.0, "likely": 0.8, "hypothesis": 0.5, "insufficient": 0.2}
RANK_BASIS = ("Problems are ranked by the share of request time the candidate change can act on (an upper bound from the measured "
              "prefill/decode/overhead shares, not a predicted gain) weighted by confidence (confirmed 1, likely 0.8, hypothesis 0.5); "
              "problems whose share is not measurable follow, then context findings.")


def phase_shares(arm):
    sp = phase_split(arm["requests"]) if arm else None
    if sp:
        return {"prefill": _med(sp["prefill_share"]), "decode": _med(sp["decode_share"]), "residual": _med(sp["residual_share"]),
                "basis": "per-request medians, client wall time" if sp["basis"] == "client_wall" else "per-request medians of server prompt+generation time (log)"}
    m = (arm or {}).get("metrics")
    if m:
        return {"prefill": m["prefill_share_of_server_time"], "decode": m["decode_share_of_server_time"], "residual": None,
                "basis": "server /metrics totals since start"}
    return None


def impact(f, shares):
    aff = AFFECTS.get(f["id"])
    if not aff:
        return None
    if aff == ("all",):
        return {"affects": "the whole request", "bound_share_of_request": 1.0, "basis": "every request in the affected sessions/configuration"}
    vals = [shares.get(p) for p in aff] if shares else [None]
    if any(v is None for v in vals):
        return {"affects": " + ".join(aff), "bound_share_of_request": None,
                "basis": "the share of %s in a request is not in the evidence%s" % ("/".join(aff), " (a log has no client wall time)" if shares else "")}
    return {"affects": " + ".join(aff), "bound_share_of_request": round(min(1.0, sum(vals)), 3), "basis": "share of %s (%s)" % (" + ".join(aff), shares["basis"])}


def rank_key(f):
    imp = f.get("impact") or {}
    b = imp.get("bound_share_of_request")
    problem = f["severity"] in ("high", "medium") and not f.get("superseded_by")
    if problem and b is not None:
        return (0, -b * CONFIDENCE_WEIGHT.get(f["confidence"], 0.2), CONFIDENCE_RANK.get(f["confidence"], 9))
    if problem:
        return (1, SEVERITY_RANK.get(f["severity"], 9), CONFIDENCE_RANK.get(f["confidence"], 9))
    return (2, -(b or 0) * CONFIDENCE_WEIGHT.get(f["confidence"], 0.2), SEVERITY_RANK.get(f["severity"], 9))


def run_rules(ev, label, arm, bandwidth_gbs=None):
    """Every rule on one arm; findings annotated with impact and ranked."""
    findings = []
    f = rule_coverage(ev)
    if f:
        findings.append(f)
    for fn in (rule_offload, rule_phase, rule_residual, rule_microbatch, rule_vram):
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
    ids = {f["id"] for f in findings}
    shares = phase_shares(arm)
    for f in findings:
        if f["id"] == "decode_below_ceiling" and "gpu_offload_partial" in ids:
            f["superseded_by"] = "gpu_offload_partial"
            f.setdefault("notes", []).append("Part of the model runs on the CPU (finding gpu_offload_partial): fix that first; this ceiling assumes all weights in VRAM.")
        imp = impact(f, shares)
        if imp:
            f["impact"] = imp
    findings.sort(key=rank_key)
    return findings, shares


def _winner(rec, verdict):
    """The candidate arm a verdict found faster: a `keep`, else a `faster ... outputs differ` (which assumes the user
    accepts the difference). None when no candidate improved."""
    rows = [r for r in verdict["arms"] if r["verdict"].startswith(("keep", "faster"))]
    if not rows:
        return None
    rows.sort(key=lambda r: (not r["verdict"].startswith("keep"), (r["metrics"].get("wall_ms") or {}).get("ratio", 1.0)))
    return rows[0]


def diagnose(records, bandwidth_gbs=None, cfg=None):
    conditions = cache_conditions(records)
    ev = apply_local_config(collect(records), cfg)
    label, arm = _baseline_arm(ev)
    base_findings, shares = run_rules(ev, label, arm, bandwidth_gbs)
    verdicts = []
    for r in ev["experiments"]:
        v = experiment_verdict(r)
        if not v:
            continue
        # an experiment written by `propose` names the finding it tests; carry that finding's risks as caveats
        prov = ((r.get("spec_redacted") or {}).get("provenance") or {})
        if prov.get("finding"):
            v["tests_finding"] = prov["finding"]
        origin = next((f for f in base_findings if f["id"] == prov.get("finding")), None)
        if origin:
            v["caveats"] = list(origin.get("risks") or [])
        # a flag that trades prefix reuse for per-request overhead: its production applicability depends on real traffic's reuse pattern
        if prov.get("experiment_kind") == "server_overhead_flags" or any("--cache-ram" in str(a.get("args_override")) or "--ctx-checkpoints" in str(a.get("args_override"))
                                                                          for a in (r.get("spec_redacted") or {}).get("arms") or []):
            cp = cache_pattern_evidence(ev)
            if not cp.get("available") or cp.get("thin"):
                gate = "production applicability unknown: no (or too little) cache-pattern evidence for real traffic (add an `observe --log` record of the real server). The gain was measured on isolated requests with the prompt cache off."
            elif cp["level"] == "low":
                gate = "production applicability supported: %s, so little prefix reuse is lost." % reuse_text(cp)
            else:
                gate = "do not apply to production on this evidence: %s; removing the cache would cost more prompt processing than this saves." % reuse_text(cp)
            v["production_gate"] = gate
            for row in v["arms"]:
                if row["verdict"].startswith("keep"):
                    row["verdict"] += " - " + gate
        verdicts.append(v)
    # what remains: after an experiment whose candidate was faster, the remaining findings describe that candidate, so the
    # next proposal builds on the improvement instead of re-diagnosing the configuration it replaced
    findings, diagnosed_why, baseline_from = base_findings, None, None
    newest = ev["experiments"][-1] if ev["experiments"] else None
    v_new = next((v for v in verdicts if newest is not None and v["experiment"] == newest.get("name")), None)
    win = _winner(newest, v_new) if v_new else None
    if win and "%s/%s" % (newest["name"], win["arm"]) in ev["arms"]:
        label = "%s/%s" % (newest["name"], win["arm"])
        arm = ev["arms"][label]
        findings, shares = run_rules(ev, label, arm, bandwidth_gbs)
        diagnosed_why = ("the candidate the verdict found faster (%s); the replaced configuration %s is described by the verdict%s" % (
            win["arm"], win["vs"], "" if win["verdict"].startswith("keep") else
            ". This assumes you accept the output differences the verdict reports; otherwise the replaced configuration still stands"))
        baseline_from = "experiment arm %s (faster than %s in the verdict)" % (label, win["vs"])
    # a finding whose experiment has been run in these records is settled by its verdict: do not recommend it again
    tested = {v["tests_finding"]: v for v in verdicts if v.get("tests_finding")}
    for f in findings:
        if f["id"] in tested:
            v = tested[f["id"]]
            f["confidence"] = "confirmed"
            f.setdefault("notes", []).append("Tested by experiment %s; see its verdict above." % v["experiment"])
            f["measured"] = {"status": "measured", "experiment": v["experiment"], "verdicts": ["%s: %s" % (r["arm"], r["verdict"]) for r in v["arms"]]}
        elif isinstance((f.get("experiment") or {}).get("arms"), list):
            f["measured"] = {"status": "not measured yet", "how": "hotpath propose <this diagnosis> --finding %s, then hotpath experiment <spec>" % f["id"]}
        elif f.get("experiment"):
            f["measured"] = {"status": "not measured yet", "how": f["experiment"].get("arms")}
    recommended = next((f for f in findings if f["id"] not in tested and not f.get("superseded_by") and f.get("experiment")
                        and isinstance(f["experiment"].get("arms"), list) and f["severity"] in ("high", "medium")), None)
    obs = ev.get("observed_launch") or {}
    reqs = (arm or {}).get("requests") or []
    rates = {"prefill_tps": _med(percentiles([r.get("prompt_tps") for r in reqs])), "decode_tps": _med(percentiles([r.get("predicted_tps") for r in reqs])),
             "basis": "per-request medians of %s" % label}
    if not (rates["prefill_tps"] and rates["decode_tps"]) and (arm or {}).get("metrics"):
        rates = {"prefill_tps": arm["metrics"].get("prefill_tps"), "decode_tps": arm["metrics"].get("decode_tps"), "basis": "/metrics totals"}
    return {"kind": "diagnosis", "rates": rates if rates["prefill_tps"] and rates["decode_tps"] else None, "evidence_class": "hypothesis" if not ev["experiments"] else "controlled", "sources": ev["sources"], "diagnosed_arm": label,
            "diagnosed_arm_reason": diagnosed_why, "cache_conditions": conditions, "phase_shares": shares, "rank_basis": RANK_BASIS,
            "observed_launch": {"args_redacted": obs.get("argv_redacted"), "source": obs.get("settings_source"), "describe": obs.get("describe")} if obs else None,
            "baseline_args": (arm or {}).get("args"), "baseline_from_verdict": baseline_from,
            "findings": findings, "experiment_verdicts": verdicts, "insufficient_evidence": insufficiencies(ev, arm),
            "recommended_next_experiment": {"finding": recommended["id"], **recommended["experiment"]} if recommended else None}


def diagnosis_brief(d, path=None, discovery=None, top=5):
    """The short form `inspect` prints: the top problems with what is known and what is not, and the exact next step."""
    L = ["# Diagnosis of %s" % d.get("diagnosed_arm")]
    if d.get("diagnosed_arm_reason"):
        L.append("(%s)" % d["diagnosed_arm_reason"])
    if d.get("phase_shares"):
        p = d["phase_shares"]
        L.append("Time split: prefill %s, decode %s, other %s (%s)" % tuple(
            ["%.0f%%" % (100 * p[k]) if p.get(k) is not None else "unknown" for k in ("prefill", "decode", "residual")] + [p["basis"]]))
    if d.get("observed_launch"):
        L.append("Running configuration: %s" % d["observed_launch"].get("describe"))
    shown = [f for f in d["findings"] if f["severity"] in ("high", "medium") and not f.get("superseded_by")][:top] or d["findings"][:min(top, 3)]
    L.append("")
    for i, f in enumerate(shown, 1):
        imp = f.get("impact") or {}
        bound = (", at most %.0f%% of request time" % (100 * imp["bound_share_of_request"])) if imp.get("bound_share_of_request") is not None else ""
        L.append("%d. %s  [%s, %s%s]" % (i, f["title"], f["severity"], f["confidence"], bound))
        for k in (f.get("known") or [])[:2]:
            L.append("   known: %s" % k)
        if f.get("hypothesis"):
            L.append("   not yet measured: %s" % f["hypothesis"])
        e = f.get("experiment") or {}
        if isinstance(e.get("arms"), list):
            L.append("   verify: %s, arms %s" % (e.get("kind"), ", ".join(a["name"] for a in e["arms"])))
        elif e.get("arms"):
            L.append("   verify: %s" % e["arms"])
    if not shown:
        L.append("No problems found in this evidence.")
    gaps = d.get("insufficient_evidence") or []
    if gaps:
        L += ["", "Not established by this evidence:"] + ["- %s" % g for g in gaps[:3]]
    rec = d.get("recommended_next_experiment")
    L += ["", "Next:"]
    top = shown[0] if shown else None
    if top and isinstance((top.get("experiment") or {}).get("arms"), str) and (not rec or rec["finding"] != top["id"]):
        L.append("- for the top finding (%s): %s" % (top["id"], top["experiment"]["arms"]))
    if rec:
        out = "experiments/%s.json" % rec.get("kind", "next")
        n = 2
        while Path(out).exists():                 # propose never overwrites: suggest a name that is free
            out = "experiments/%s-%d.json" % (rec.get("kind", "next"), n)
            n += 1
        L.append("- hotpath propose %s --finding %s --out %s   (writes a one-variable spec and shows its plan; launches nothing)" % (path or "<diagnosis.json>", rec["finding"], out))
        L.append("- hotpath experiment %s   (ACTIVE: launches its own llama-server per arm on port 8099, interleaved; your server is not touched)" % out)
        if discovery:
            from hotpath import discover as disc
            note = disc.experiment_vram_note(discovery)
            if note:
                L.append("  VRAM: %s." % note)
    else:
        L.append("- no experiment to propose from this evidence; see what is not established above")
    if path:
        L.append("Full report: %s" % (str(path)[:-5] + ".md" if str(path).endswith(".json") else path))
    return "\n".join(L)


def diagnosis_markdown(d):
    lines = ["# Diagnosis", "", "Sources: %s" % ", ".join(d["sources"]), "Diagnosed arm: %s%s" % (
        d.get("diagnosed_arm"), " - %s; the findings below are what remains on it" % d["diagnosed_arm_reason"] if d.get("diagnosed_arm_reason") else ""), ""]
    if d.get("cache_conditions"):
        lines += ["## Cache conditions of the measured requests", ""]
        for c in d["cache_conditions"]:
            if c["state"] == "unknown":
                what = "unknown: the records carry no usable cache_n (older record, or a server that does not report it); not assumed cold"
            else:
                what = "%s (cache_n = 0 on %d/%d requests, > 0 on %d)" % (c["state"], c["cold"], c["requests"], c["warm"])
            lines.append("- %s: %s%s" % (c["source"], what, "; isolation: %s" % c["isolation"] if c.get("isolation") else ""))
        lines.append("")
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
    lines += ["## Findings (ranked)", "", d.get("rank_basis") or "", ""]
    if d.get("observed_launch"):
        lines += ["Running configuration (%s): %s" % (d["observed_launch"].get("source"), d["observed_launch"].get("describe")), ""]
    for i, f in enumerate(d["findings"], 1):
        lines += ["### %d. %s  [%s, %s, %s%s]" % (i, f["title"], f["severity"], f["confidence"], f["evidence_class"],
                                                   ", superseded by %s" % f["superseded_by"] if f.get("superseded_by") else ""), "", "- id: `%s`" % f["id"]]
        imp = f.get("impact") or {}
        if imp:
            lines.append("- acts on: %s; at most %s of request time (%s)" % (
                imp["affects"], "%.0f%%" % (100 * imp["bound_share_of_request"]) if imp.get("bound_share_of_request") is not None else "an unmeasured share", imp["basis"]))
        for k in f.get("known") or []:
            lines.append("- known: %s" % k)
        if f.get("hypothesis"):
            lines.append("- hypothesis (not measured): %s" % f["hypothesis"])
        lines += ["- measurements: %s" % f["measurements"], "- why it may matter: %s" % f["why_it_matters"]]
        if f.get("opportunity"):
            lines.append("- opportunity: %s" % f["opportunity"])
        if f.get("experiment"):
            e = f["experiment"]
            lines.append("- verify with: %s on `%s`; metric: %s" % (e.get("kind"), e.get("variable"), e.get("metric")))
            if isinstance(e.get("arms"), list):
                lines.append("  - arms: %s" % [a["name"] for a in e["arms"]])
            else:
                lines.append("  - how: %s" % e.get("arms"))
        for r in f.get("risks") or []:
            lines.append("- could regress: %s" % r)
        if f.get("measured"):
            m = f["measured"]
            lines.append("- measured: %s" % ("yes, experiment %s: %s" % (m["experiment"], " | ".join(m["verdicts"])) if m["status"] == "measured" else "no (%s)" % m["how"]))
        for n in f.get("notes") or []:
            lines.append("- note: %s" % n)
        lines.append("")
    if d.get("recommended_next_experiment"):
        r = d["recommended_next_experiment"]
        lines += ["## Recommended next experiment", "", "`hotpath propose <this diagnosis> --finding %s` writes a spec for: %s on `%s` (arms %s)." % (
            r["finding"], r.get("kind"), r.get("variable"), [a["name"] for a in r["arms"]]), ""]
    lines += ["## What the evidence cannot establish", ""] + ["- " + s for s in d["insufficient_evidence"]]
    return "\n".join(lines)
