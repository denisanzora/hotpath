"""hotpath measure: active measurement against a server you control.

Submits real inference. Refuses to target a server unless it is listed as owned in the local
config or --i-own-this-server is passed, because a measurement request consumes GPU time and
changes the prompt cache of whatever is running there.

Per request it records the server's own timings (prompt_n, prompt_ms, predicted_n, predicted_ms,
cache_n), the wall time, and the generated token ids, so prefill, generation, and end-to-end
latency are kept apart and outputs can be compared across arms.
"""
import hashlib
import hmac
import json
import os
import time

from hotpath import gpu as gpu_mod
from hotpath import server as srv
from hotpath.records import new_record, percentiles
from hotpath.workload import is_sampled, request_class, workload_identity


def apply_request_override(body, override):
    """Arm-level change to an application request body: `set` top-level fields, `remove` top-level fields."""
    b = json.loads(json.dumps(body))
    for k in (override or {}).get("remove") or []:
        b.pop(k, None)
    for k, v in ((override or {}).get("set") or {}).items():
        b[k] = v
    return b


def _schema_of(body):
    """The JSON schema (a dict) a request asks the server to enforce, or None."""
    rf = body.get("response_format")
    rf = rf if isinstance(rf, dict) else {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema")
        s = js.get("schema") if isinstance(js, dict) else None
        return s if isinstance(s, dict) else {}
    s = body.get("json_schema")
    if isinstance(s, dict) and s:
        return s
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


SCHEMA_VALIDATION_MODES = ("auto", "full", "required-keys")
INSTALL_HINT = 'install the optional validator with: pip install "hotpath[schema]"   (or: pip install jsonschema)'


def require_schema_validation(mode):
    """Honor an explicit schema_validation request before any server is launched or request sent.
    "full" without the validator is an error with an install hint, never a silent downgrade."""
    if mode not in SCHEMA_VALIDATION_MODES:
        raise SystemExit("error: schema_validation must be one of %s, got %r" % (", ".join(SCHEMA_VALIDATION_MODES), mode))
    if mode == "full" and schema_validator() is None:
        raise SystemExit("error: schema_validation is \"full\" but the jsonschema package is not installed; " + INSTALL_HINT +
                         ". Use \"auto\" to fall back to required-key checks (records then say so), or \"required-keys\" to choose that explicitly.")


VALIDATION_LEVEL_FULL = "full (jsonschema Draft 2020-12)"
VALIDATION_LEVEL_SUBSET = "required top-level keys only (install hotpath[schema] for full validation)"
VALIDATION_LEVEL_CHOSEN_SUBSET = "required top-level keys only (schema_validation: required-keys)"

# Keywords a plain Draft 2020-12 validation does not assert (format is an annotation by default; content* are annotations;
# a $ref that is not a local fragment would need network or file resolution). A schema using them is validated for
# everything else and the record lists what was not enforced.
_UNASSERTED = ("format", "contentEncoding", "contentMediaType", "contentSchema")


def _walk(schema):
    if isinstance(schema, dict):
        yield schema
        for v in schema.values():
            yield from _walk(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _walk(v)


def _unchecked_keywords(schema):
    found = set()
    for node in _walk(schema):
        for k in _UNASSERTED:
            if k in node and not (k == "format" and isinstance(node.get("format"), dict)):
                found.add(k)
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            found.add("$ref (non-local)")
    return sorted(found)


def _declared_keys(schema):
    keys = set()
    for node in _walk(schema):
        props = node.get("properties")
        if isinstance(props, dict):
            keys.update(str(k) for k in props)
        for k in ("required", "dependentRequired"):
            v = node.get(k)
            if isinstance(v, list):
                keys.update(x for x in v if isinstance(x, str))
    return keys


def _safe_path(error, declared):
    """JSON path of a schema error with every object key the schema did not declare replaced by `*`: keys of the model's
    output are output text and must not enter a record. Array indexes and declared property names are kept."""
    parts = ["$"]
    for seg in error.absolute_path:
        parts.append("[%d]" % seg if isinstance(seg, int) else (".%s" % seg if str(seg) in declared else ".*"))
    return "".join(parts)[:160]


def _validity(content, schema, mode="auto"):
    """How an output relates to the schema the application asked for, at separate levels.

    valid_json: parses as JSON. schema_required_present: the schema's required top-level keys are present (always
    checked). schema_valid: the output satisfies the whole schema (nested types, enums, arrays, bounds,
    additionalProperties) by jsonschema Draft 2020-12; None when that check was not run (validator absent, or
    mode "required-keys") or could not run (the validator rejected the schema). validation_level names what was
    actually checked; schema_unchecked_keywords lists schema keywords present but not enforced. Error entries carry the
    failing keyword and a path in which undeclared object keys are masked: never output text. None everywhere when
    the request asked for no schema."""
    if schema is None:
        return {"valid_json": None, "schema_required_present": None, "schema_valid": None, "validation_level": None}
    required = schema.get("required") if isinstance(schema, dict) else None
    required = required if isinstance(required, list) else []
    V = schema_validator() if mode != "required-keys" else None
    subset_level = VALIDATION_LEVEL_CHOSEN_SUBSET if mode == "required-keys" else VALIDATION_LEVEL_SUBSET
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return {"valid_json": False, "schema_required_present": False, "schema_valid": False if V else None,
                "validation_level": VALIDATION_LEVEL_FULL if V else subset_level}
    present = isinstance(obj, dict) and all(k in obj for k in required)
    if V is None:
        return {"valid_json": True, "schema_required_present": present, "schema_valid": None, "validation_level": subset_level}
    try:
        declared = _declared_keys(schema)
        errors = sorted(V(schema).iter_errors(obj), key=lambda e: _safe_path(e, declared))
        out = {"valid_json": True, "schema_required_present": present, "schema_valid": not errors, "validation_level": VALIDATION_LEVEL_FULL,
               "schema_errors": [{"keyword": e.validator, "path": _safe_path(e, declared)} for e in errors[:5]]}
        unchecked = _unchecked_keywords(schema)
        if unchecked:
            out["schema_unchecked_keywords"] = unchecked
        return out
    except Exception as e:  # noqa: BLE001 - a schema the validator cannot process is reported, not fatal
        return {"valid_json": True, "schema_required_present": present, "schema_valid": None, "validation_level": "validator could not run: %s" % type(e).__name__}


def field_kind(prop):
    """What kind of value a schema property holds, for comparing outputs field by field: `categorical` (enum, const,
    boolean: the kind of field an application branches on), `number`, `text` (free strings), `structure` (objects,
    arrays) or `other`."""
    if not isinstance(prop, dict):
        return "other"
    if "enum" in prop or "const" in prop:
        return "categorical"
    t = prop.get("type")
    t = [x for x in t if x != "null"][0] if isinstance(t, list) and [x for x in t if x != "null"] else t
    return {"boolean": "categorical", "integer": "number", "number": "number", "string": "text", "object": "structure",
            "array": "structure"}.get(t, "other")


def output_fields(content, schema, key=None):
    """Per property the schema declares at the top level: a short keyed hash of the output's value (or "missing"), and the
    property's kind. Lets two arms' outputs be compared field by field (did the same decision come out?) without storing
    output text; keys the schema does not declare are output content and are not recorded. The key is a per-run secret
    kept in memory only, so a short value (an enum, a boolean, a score) cannot be recovered by hashing candidates."""
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        return None, None
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    fields, kinds = {}, {}
    for k, prop in schema["properties"].items():
        k = str(k)
        kinds[k] = field_kind(prop)
        if k not in obj:
            fields[k] = "missing"
            continue
        raw = json.dumps(obj[k], sort_keys=True, ensure_ascii=False).encode("utf-8")
        fields[k] = hmac.new(key, raw, hashlib.sha256).hexdigest()[:12] if key else hashlib.sha256(raw).hexdigest()[:12]
    return fields, kinds


def _field_key(workload):
    """The per-run key of output field hashes: set by the experiment runner for all arms of one run (in memory only)."""
    if not workload.get("_field_key"):
        workload["_field_key"] = os.urandom(16)
    return workload["_field_key"]


def cache_state(timings):
    """What the server itself says about this request's prompt cache: cold when it reused no tokens, warm when it
    reused some. Anything else (field missing, null, non-numeric, negative, NaN) is "unknown": missing or malformed
    evidence is never read as zero reuse."""
    cn = timings.get("cache_n") if isinstance(timings, dict) else None
    if isinstance(cn, bool) or not isinstance(cn, (int, float)):
        return "unknown"
    if cn != cn or cn < 0 or (isinstance(cn, float) and cn != int(cn)):
        return "unknown"
    return "cold" if cn == 0 else "warm"


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

    cold  (default): `cache_prompt: false` is set on every request in every arm (overriding the application's own
                     value, which is what makes the state cold), and the slot is erased before each pass when the
                     server honors slot erase. Erase alone is not trusted: the host prompt cache can restore an
                     erased prompt. Each request's state is read back from the server's own `cache_n`.
    warm           : no isolation; items follow each other as in the application and repetitions reuse the slot.
                     Reuse is intentional and both arms see the same sequence.
    as-is          : no intervention; states are recorded from `cache_n` and a verdict compares prefill-side numbers
                     only if both arms verifiably ended up in the same state.
    Items are sent in order. In cold mode nothing is reused, not even between consecutive items; in warm and as-is
    mode item-to-item reuse is whatever the server does."""
    mode = workload.get("cache_mode", "cold")
    if mode not in CACHE_MODES:
        raise ValueError("cache_mode must be one of %s" % (CACHE_MODES,))
    schema_mode = workload.get("schema_validation", "auto")
    require_schema_validation(schema_mode)
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
        # in cold mode a warmup pass only has to warm the runtime (kernels, graphs, allocations): the first item does that, and
        # nothing it leaves behind is reused; in warm and as-is mode a warmup pass establishes the cache state, so it is whole
        items = workload["items"][:1] if (phase == "warmup" and mode == "cold") else workload["items"]
        for it in items:
            body = apply_request_override(it["body"], request_override)
            if force_cold_requests:
                body["cache_prompt"] = False
            before = gpu_mod.gpu_dynamic()
            try:
                resp = srv.request(base, it["endpoint"], body)
            except srv.RequestRejected as e:
                # the server answered with an error (a request larger than the context, a schema it cannot compile): this
                # configuration cannot serve this request. Recorded as a failed request of the arm; the error's type and code
                # are kept, its message is not (it can quote the request)
                rec = {"phase": phase, "rep": rep, "prompt_id": it["id"], "prompt_sha256": it["sha256"], "endpoint": it["endpoint"],
                       "class": request_class(it), "sampled": is_sampled(body), "error": {"http": e.code, "type": e.error_type}}
                (out["warmup_requests"] if phase == "warmup" else out["requests"]).append(rec)
                if progress:
                    progress("%s rep %d %s: REJECTED by the server (HTTP %s, %s)" % (phase, rep, it["id"], e.code, e.error_type))
                continue
            after = gpu_mod.gpu_dynamic()
            t = resp.get("timings", {})
            content, stop = _content(resp)
            rec = {"phase": phase, "rep": rep, "prompt_id": it["id"], "prompt_sha256": it["sha256"], "endpoint": it["endpoint"],
                   "class": request_class(it), "sampled": is_sampled(body),
                   "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "cache_state": cache_state(t),
                   "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"), "predicted_tps": t.get("predicted_per_second"),
                   "wall_ms": resp.get("_wall_ms"), "stop_type": stop, "tokens": None,
                   "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16], "output_chars": len(content),
                   "output_text": content if store_outputs else None, "gpu_before": before, "gpu_after": after}
            # validity is judged against the application's own request (the workload item), not the arm's edited body:
            # an arm that drops the schema is still measured on whether its output satisfies what the application requires
            rec.update(_validity(content, _schema_of(it["body"]), schema_mode))
            rec["validity_schema_from"] = "workload item" if _schema_of(it["body"]) is not None else None
            rec["fields"], rec["field_kinds"] = output_fields(content, _schema_of(it["body"]), _field_key(workload))
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


def run_workload(base, workload, repetitions=3, warmup=1, erase_between=True, store_prompts=False, store_outputs=False,
                 slot_id=0, pid=None, progress=None, request_override=None):
    """Returns a list of per-request records plus per-repetition GPU snapshots.

    Generated text is NOT stored unless store_outputs=True (spec key "store_outputs"): records keep token ids (needed for
    the equivalence check), an output hash and lengths."""
    if workload.get("store_outputs"):
        store_outputs = True
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
                "tokens": resp.get("tokens"),
                "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16], "output_chars": len(content),
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
    g = lambda k: [r.get(k) for r in reqs]
    cache_vals = [r.get("cache_n") for r in reqs if cache_state({"cache_n": r.get("cache_n")}) != "unknown"]
    s = {
        "n": len(reqs),
        "prompt_n": percentiles(g("prompt_n")),
        "cache_n_max": max(cache_vals) if cache_vals else None,
        "cache_n": percentiles(cache_vals),
        "prefill_tps": percentiles(g("prompt_tps")),
        "decode_tps": percentiles(g("predicted_tps")),
        "prompt_ms": percentiles(g("prompt_ms")),
        "predicted_ms": percentiles(g("predicted_ms")),
        "wall_ms": percentiles(g("wall_ms")),
        "predicted_n": percentiles(g("predicted_n")),
        "stop_types": sorted({str(r.get("stop_type")) for r in reqs}),
    }
    judged = [r for r in reqs if r.get("valid_json") is not None]
    if judged:
        full = [r for r in judged if r.get("schema_valid") is not None]
        levels = sorted({str(r.get("validation_level")) for r in judged})
        unchecked = sorted({k for r in judged for k in (r.get("schema_unchecked_keywords") or [])})
        s["validity"] = {"judged": len(judged), "valid_json": sum(1 for r in judged if r["valid_json"]),
                         "schema_required_present": sum(1 for r in judged if r.get("schema_required_present")),
                         "schema_checked": len(full),   # requests that got the full-schema check
                         "schema_valid": sum(1 for r in full if r["schema_valid"]) if full else None,
                         "validation_level": levels[0] if len(levels) == 1 else "mixed: " + "; ".join(levels),
                         "schema_unchecked_keywords": unchecked}
    states = [cache_state({"cache_n": r.get("cache_n")}) if not r.get("cache_state") else r["cache_state"] for r in reqs]
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
    from hotpath import discover as disc
    rec = new_record("measurement", "active", label=label, target=snap,
                     server_launch=disc.launch_for_record(disc.server_launch(pid)) if pid else {"available": False, "note": "no process found on port %d" % port},
                     workload=workload_identity(workload),
                     gpu=gpu_mod.gpu_static(),
                     background_before=gpu_mod.sample_background(seconds=background_seconds, interval=1.0),
                     settings={"repetitions": repetitions, "warmup": warmup,
                               "cache_mode": workload.get("cache_mode", "cold"),
                               "cache_prompt": "as sent by the workload items" if workload.get("kind") == "requests" and workload.get("cache_mode") in ("warm", "as-is") else False,
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
