"""hotpath measure: active measurement against a server you control.

Submits real inference. Refuses to target a server unless it is listed as owned in the local
config or --i-own-this-server is passed, because a measurement request consumes GPU time and
changes the prompt cache of whatever is running there.

Per request it records the server's own timings (prompt_n, prompt_ms, predicted_n, predicted_ms,
cache_n), the wall time, and the generated token ids, so prefill, generation, and end-to-end
latency are kept apart and outputs can be compared across arms.
"""
import hashlib
import json
import time

from hotpath import gpu as gpu_mod
from hotpath import server as srv
from hotpath.records import new_record, percentiles
from hotpath.workload import workload_identity


def apply_request_override(body, override):
    """Arm-level change to an application request body: `set` top-level fields, `remove` top-level fields."""
    b = json.loads(json.dumps(body))
    for k in (override or {}).get("remove") or []:
        b.pop(k, None)
    for k, v in ((override or {}).get("set") or {}).items():
        b[k] = v
    return b


def _schema_of(body):
    """The JSON schema a request asks the server to enforce, or None."""
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        return ((rf.get("json_schema") or {}).get("schema")) or {}
    if body.get("json_schema"):
        return body["json_schema"] or {}
    return None


def _schema_required(body):
    s = _schema_of(body)
    return None if s is None else (s.get("required") or [])


def schema_validator():
    """The optional full validator (jsonschema, Draft 2020-12). None when not installed: `pip install "hotpath[schema]"`."""
    try:
        from jsonschema import Draft202012Validator
        return Draft202012Validator
    except ImportError:
        return None


VALIDATION_LEVEL_FULL = "full (jsonschema Draft 2020-12)"
VALIDATION_LEVEL_SUBSET = "required top-level keys only (install hotpath[schema] for full validation)"


def _validity(content, schema):
    """How an output relates to the schema the application asked for.

    valid_json: parses as JSON. schema_required_present: the schema's required top-level keys are present (always
    checked; the subset a plain install can do). schema_valid: the output satisfies the whole schema (nested types,
    enums, arrays, bounds, additionalProperties) when the optional validator is installed; None when it is not.
    validation_level names what was actually checked. Error entries carry the failing keyword and JSON path only,
    never output text. None everywhere when the request asked for no schema."""
    if schema is None:
        return {"valid_json": None, "schema_required_present": None, "schema_valid": None, "validation_level": None}
    required = schema.get("required") or []
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return {"valid_json": False, "schema_required_present": False, "schema_valid": False if schema_validator() else None,
                "validation_level": VALIDATION_LEVEL_FULL if schema_validator() else VALIDATION_LEVEL_SUBSET}
    present = isinstance(obj, dict) and all(k in obj for k in required)
    V = schema_validator()
    if V is None:
        return {"valid_json": True, "schema_required_present": present, "schema_valid": None, "validation_level": VALIDATION_LEVEL_SUBSET}
    try:
        errors = sorted(V(schema).iter_errors(obj), key=lambda e: str(e.json_path))
    except Exception as e:  # noqa: BLE001 - a schema the validator cannot process is reported, not fatal
        return {"valid_json": True, "schema_required_present": present, "schema_valid": None, "validation_level": "validator error: %s" % type(e).__name__}
    return {"valid_json": True, "schema_required_present": present, "schema_valid": not errors, "validation_level": VALIDATION_LEVEL_FULL,
            "schema_errors": [{"keyword": e.validator, "path": e.json_path} for e in errors[:5]]}


def cache_state(timings):
    """What the server itself says about this request's prompt cache: cold when it reused no tokens."""
    cn = timings.get("cache_n")
    if cn is None:
        return "unknown"
    return "cold" if int(cn) == 0 else "warm"


def _content(resp):
    if "choices" in resp:
        ch = (resp.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or ch.get("text") or "", ch.get("finish_reason")
    return resp.get("content", ""), resp.get("stop_type")


def probe_slot_erase(base, slot_id=0):
    """Does this server honor slot erase? llama-server needs --slot-save-path for slot actions; without it the call
    returns HTTP 501. Probing before any request costs nothing and decides how cold state can be obtained."""
    r = srv.erase_slot(base, slot_id)
    if isinstance(r, dict) and r.get("error"):
        return False, r["error"]
    return True, None


CACHE_MODES = ("cold", "warm", "as-is")


def run_requests_workload(base, workload, repetitions=3, warmup=1, request_override=None, store_outputs=False, slot_id=0, pid=None, progress=None):
    """Application-shaped requests under an explicit cache mode.

    cold  (default): every pass over the items starts from an empty slot. Isolation is slot erase when the server
                     supports it; otherwise `cache_prompt: false` is set on every request (same in every arm). Each
                     request's state is then read back from the server's own `cache_n`, not assumed.
    warm           : no isolation; items follow each other as in the application and repetitions reuse the slot.
                     Reuse is intentional and both arms see the same sequence.
    as-is          : no intervention and no fallback; states are recorded from `cache_n` and a verdict treats
                     prefill as unverified.
    Within a pass the items are sent in order, so item-to-item reuse is the application's in every mode."""
    mode = workload.get("cache_mode", "cold")
    if mode not in CACHE_MODES:
        raise ValueError("cache_mode must be one of %s" % (CACHE_MODES,))
    out = {"requests": [], "warmup_requests": [], "gpu_snapshots": [], "slot_erase_error": None,
           "cache": {"mode": mode, "isolation": "none", "slot_erase_supported": None}}
    # Cold isolation is cache_prompt=false on every request, in every arm. Slot erase alone is not enough: llama-server's
    # host prompt cache (--cache-ram) can restore an erased prompt on the next request (observed: slot erase honored,
    # yet every request reused its prompt). Erase is still issued when the server honors it, and the state of every
    # request is verified from the server's cache_n either way.
    force_cold_requests = mode == "cold"
    if mode == "cold":
        ok, err = probe_slot_erase(base, slot_id)
        out["cache"]["slot_erase_supported"] = ok
        out["cache"]["isolation"] = "cache_prompt=false on every request" + (" + slot erase before each pass" if ok else " (slot erase unsupported: %s)" % err)
        if not ok:
            out["slot_erase_error"] = err
    seq = [("warmup", r) for r in range(warmup)] + [("rep", r) for r in range(repetitions)]
    for phase, rep in seq:
        if mode == "cold" and out["cache"]["slot_erase_supported"]:
            srv.erase_slot(base, slot_id)
        for it in workload["items"]:
            body = apply_request_override(it["body"], request_override)
            if force_cold_requests:
                body["cache_prompt"] = False
            before = gpu_mod.gpu_dynamic()
            resp = srv.request(base, it["endpoint"], body)
            after = gpu_mod.gpu_dynamic()
            t = resp.get("timings", {})
            content, stop = _content(resp)
            rec = {"phase": phase, "rep": rep, "prompt_id": it["id"], "prompt_sha256": it["sha256"], "endpoint": it["endpoint"],
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "cache_state": cache_state(t),
                   "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"), "predicted_tps": t.get("predicted_per_second"),
                   "wall_ms": resp.get("_wall_ms"), "stop_type": stop, "tokens": None,
                   "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16], "output_chars": len(content),
                   "output_text": content if store_outputs else None, "gpu_before": before, "gpu_after": after}
            # validity is judged against the application's own request (the workload item), not the arm's edited body:
            # an arm that drops the schema is still measured on whether its output satisfies what the application requires
            rec.update(_validity(content, _schema_of(it["body"])))
            rec["validity_schema_from"] = "workload item" if _schema_of(it["body"]) is not None else None
            (out["warmup_requests"] if phase == "warmup" else out["requests"]).append(rec)
            if progress:
                progress("%s rep %d %s: prompt_n=%s cache_n=%s (%s) pp=%s t/s tg=%s t/s (%s tok) wall=%s ms valid_json=%s schema_valid=%s" % (
                    phase, rep, it["id"], rec["prompt_n"], rec["cache_n"], rec["cache_state"], rec["prompt_tps"], rec["predicted_tps"], rec["predicted_n"], rec["wall_ms"],
                    rec["valid_json"], rec.get("schema_valid")))
        snap = {"phase": phase, "rep": rep, "gpu": gpu_mod.gpu_dynamic()}
        if pid:
            snap["process"] = gpu_mod.windows_process_gpu_counters(pid)
        out["gpu_snapshots"].append(snap)
    return out


def run_workload(base, workload, repetitions=3, warmup=1, erase_between=True, store_prompts=False, store_outputs=True,
                 slot_id=0, pid=None, progress=None, request_override=None):
    """Returns a list of per-request records plus per-repetition GPU snapshots."""
    if workload.get("kind") == "requests":
        return run_requests_workload(base, workload, repetitions=repetitions, warmup=warmup, request_override=request_override,
                                     store_outputs=bool(workload.get("store_outputs", False)), slot_id=slot_id, pid=pid, progress=progress)
    # erase_between=True means "if the server supports it"; cold state comes from cache_prompt=false regardless
    # hotpath's own workloads are cold by construction: cache_prompt=false on every request (the server then processes
    # the whole prompt), plus a slot erase before each request when the server supports it. The state is still read
    # back per request from the server's cache_n.
    out = {"requests": [], "warmup_requests": [], "gpu_snapshots": [], "slot_erase_error": None,
           "cache": {"mode": "cold", "isolation": "cache_prompt=false on every request", "slot_erase_supported": None}}
    if erase_between:
        ok, err = probe_slot_erase(base, slot_id)
        out["cache"]["slot_erase_supported"] = ok
        if ok:
            out["cache"]["isolation"] += " + slot erase before each request"
        else:
            out["slot_erase_error"] = err
            erase_between = False
    n_predict = workload["n_predict"]
    temp = workload.get("temperature", 0.0)
    ignore_eos = bool(workload.get("ignore_eos", True))
    seq = []
    for r in range(warmup):
        seq.append(("warmup", r))
    for r in range(repetitions):
        seq.append(("rep", r))
    for phase, rep in seq:
        for p in workload["prompts"]:
            if erase_between:
                srv.erase_slot(base, slot_id)
            before = gpu_mod.gpu_dynamic()
            resp = srv.completion(base, p["text"], n_predict, temperature=temp, cache_prompt=False, ignore_eos=ignore_eos)
            after = gpu_mod.gpu_dynamic()
            t = resp.get("timings", {})
            content = resp.get("content", "")
            rec = {
                "phase": phase, "rep": rep, "prompt_id": p["id"], "prompt_sha256": p["sha256"],
                "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "cache_state": cache_state(t),
                "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
                "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"), "predicted_tps": t.get("predicted_per_second"),
                "wall_ms": resp.get("_wall_ms"),
                "stop_type": resp.get("stop_type"),
                "tokens": resp.get("tokens") if store_outputs else None,
                "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                "output_text": content if store_outputs else None,
                "gpu_before": before, "gpu_after": after,
            }
            if store_prompts:
                rec["prompt_text"] = p["text"]
            (out["warmup_requests"] if phase == "warmup" else out["requests"]).append(rec)
            if progress:
                progress("%s rep %d %s: prompt_n=%s pp=%s t/s tg=%s t/s wall=%s ms" % (
                    phase, rep, p["id"], rec["prompt_n"], rec["prompt_tps"], rec["predicted_tps"], rec["wall_ms"]))
        snap = {"phase": phase, "rep": rep, "gpu": gpu_mod.gpu_dynamic()}
        if pid:
            snap["process"] = gpu_mod.windows_process_gpu_counters(pid)
        out["gpu_snapshots"].append(snap)
    return out


def summarize_requests(reqs):
    s = {
        "n": len(reqs),
        "prompt_n": percentiles([r["prompt_n"] for r in reqs]),
        "cache_n_max": max([r["cache_n"] or 0 for r in reqs]) if reqs else None,
        "cache_n": percentiles([r.get("cache_n") for r in reqs]),
        "prefill_tps": percentiles([r["prompt_tps"] for r in reqs]),
        "decode_tps": percentiles([r["predicted_tps"] for r in reqs]),
        "prompt_ms": percentiles([r["prompt_ms"] for r in reqs]),
        "predicted_ms": percentiles([r["predicted_ms"] for r in reqs]),
        "wall_ms": percentiles([r["wall_ms"] for r in reqs]),
        "predicted_n": percentiles([r["predicted_n"] for r in reqs]),
        "stop_types": sorted({str(r["stop_type"]) for r in reqs}),
    }
    judged = [r for r in reqs if r.get("valid_json") is not None]
    if judged:
        full = [r for r in judged if r.get("schema_valid") is not None]
        s["validity"] = {"judged": len(judged), "valid_json": sum(1 for r in judged if r["valid_json"]),
                         "schema_required_present": sum(1 for r in judged if r.get("schema_required_present")),
                         "schema_valid": sum(1 for r in full if r["schema_valid"]) if full else None,
                         "validation_level": judged[0].get("validation_level")}
    states = [r.get("cache_state") for r in reqs if r.get("cache_state")]
    if states:
        s["cache_states"] = {k: states.count(k) for k in ("cold", "warm", "unknown") if states.count(k)}
    return s


def measure(cfg, base, workload, repetitions, warmup, owned, background_seconds=10.0, store_prompts=False, label=None):
    owned_list = cfg.get("owned_servers") or []
    if base not in owned_list and not owned:
        raise SystemExit("refusing to submit inference to %s: not in owned_servers and --i-own-this-server not given" % base)
    snap = srv.passive_snapshot(base)
    if snap.get("health", {}).get("status") != "ok":
        raise SystemExit("server not healthy: %s" % snap.get("health"))
    port = int(base.rsplit(":", 1)[1].split("/")[0])
    pid = gpu_mod.pid_listening_on(port)
    rec = new_record("measurement", "active", label=label, target=snap,
                     workload=workload_identity(workload),
                     gpu=gpu_mod.gpu_static(),
                     background_before=gpu_mod.sample_background(seconds=background_seconds, interval=1.0),
                     settings={"repetitions": repetitions, "warmup": warmup, "cache_prompt": False, "erase_slot_between_requests": True,
                               "ignore_eos": bool(workload.get("ignore_eos", True)), "greedy": workload.get("temperature", 0.0) == 0.0})
    t0 = time.time()
    res = run_workload(base, workload, repetitions=repetitions, warmup=warmup, pid=pid, store_prompts=store_prompts,
                       progress=lambda s: print("  " + s))
    rec["duration_s"] = round(time.time() - t0, 1)
    rec["requests"] = res["requests"]
    rec["warmup_requests"] = res["warmup_requests"]
    rec["gpu_snapshots"] = res["gpu_snapshots"]
    rec["cache"] = res.get("cache")
    if res.get("slot_erase_error"):
        rec["slot_erase_error"] = res["slot_erase_error"]
    rec["summary"] = summarize_requests(res["requests"])
    rec["background_after"] = gpu_mod.sample_background(seconds=min(background_seconds, 5.0), interval=1.0)
    return rec
