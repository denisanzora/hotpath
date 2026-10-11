"""llama-server HTTP client.

Passive endpoints (no inference, no state change): /health, /props, /v1/models, /slots (GET).
Active endpoints (consume GPU time and change cache state): /completion, /tokenize, /slots?action=erase.
The passive/active split is enforced by the caller; this module labels each function.
"""
import json
import time
import urllib.error
import urllib.request


class ServerError(RuntimeError):
    pass


def _get(url, timeout=10):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url, body, timeout=600):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
        return json.loads(raw) if raw else {}


# ---- passive -------------------------------------------------------------------------------

def health(base):
    try:
        return _get(base + "/health", timeout=3)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"status": "unreachable", "error": str(e)}


def props(base):
    return _get(base + "/props")


def models(base):
    return _get(base + "/v1/models")


def slots(base):
    try:
        return _get(base + "/slots")
    except urllib.error.HTTPError as e:
        return {"error": "slots endpoint unavailable (HTTP %d); start the server with --slots" % e.code}


def passive_snapshot(base):
    """Everything a running llama-server tells us without being asked to infer."""
    snap = {"base_url": base, "health": health(base)}
    if snap["health"].get("status") == "unreachable":
        return snap
    try:
        p = props(base)
        gen = p.get("default_generation_settings", {})
        snap["props"] = {
            "build_info": p.get("build_info"),
            "model_basename": _basename(p.get("model_path")),
            "n_ctx_per_slot": gen.get("n_ctx"),
            "total_slots": p.get("total_slots"),
            "modalities": p.get("modalities"),
            "default_params": {k: gen.get("params", {}).get(k) for k in
                               ("temperature", "top_k", "top_p", "min_p", "seed", "samplers", "n_predict")},
        }
    except Exception as e:  # noqa: BLE001 - we want the record to carry the failure, not die
        snap["props_error"] = str(e)
    try:
        m = models(base)
        meta = (m.get("data") or [{}])[0].get("meta", {})
        snap["model_meta"] = {k: meta.get(k) for k in ("n_params", "size", "n_ctx_train", "n_embd", "n_vocab", "ftype")}
    except Exception as e:  # noqa: BLE001
        snap["models_error"] = str(e)
    s = slots(base)
    if isinstance(s, list):
        snap["slots"] = [{k: sl.get(k) for k in ("id", "is_processing", "n_ctx", "n_prompt_tokens", "n_prompt_tokens_cache")} for sl in s]
    else:
        snap["slots"] = s
    return snap


def _basename(path):
    if not path:
        return None
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def wait_healthy(base, timeout_s=300, interval=1.0, proc=None):
    t_end = time.time() + timeout_s
    last = None
    while time.time() < t_end:
        if proc is not None and proc.poll() is not None:
            raise ServerError("server process exited with code %s before becoming healthy; see its log. "
                              "On Windows a missing CUDA runtime DLL gives no output: add the CUDA bin directory to path_prepend." % proc.returncode)
        h = health(base)
        last = h
        if h.get("status") == "ok":
            return True
        time.sleep(interval)
    raise ServerError("server at %s not healthy after %ss (last: %s)" % (base, timeout_s, last))


# ---- active --------------------------------------------------------------------------------

def tokenize(base, text):
    r = _post(base + "/tokenize", {"content": text, "add_special": False}, timeout=60)
    return r.get("tokens", [])


def erase_slot(base, slot_id=0):
    """Drop the slot's prompt cache so the next request starts from an empty KV."""
    try:
        return _post(base + "/slots/%d?action=erase" % slot_id, {}, timeout=30)
    except urllib.error.HTTPError as e:
        return {"error": "erase failed HTTP %d" % e.code}


def request(base, endpoint, body, timeout=900):
    """One non-streaming request with an application-shaped body (`/v1/chat/completions` or `/completion`).
    Returns the server JSON plus wall-clock time; the caller decides what to keep from it."""
    body = dict(body)
    body["stream"] = False
    t0 = time.perf_counter()
    r = _post(base + endpoint, body, timeout=timeout)
    r["_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return r


def completion(base, prompt, n_predict, temperature=0.0, seed=42, cache_prompt=False, ignore_eos=True, timeout=900, extra=None):
    """One non-streaming completion. Returns the server JSON plus wall-clock time.

    ignore_eos=True keeps the generated length fixed at n_predict so every arm does the same
    amount of work; the stop reason is recorded either way. Greedy decoding is requested with
    temperature 0 and top_k 1 (llama.cpp treats temperature <= 0 as greedy).
    """
    body = {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "temperature": float(temperature),
        "seed": int(seed),
        "cache_prompt": bool(cache_prompt),
        "ignore_eos": bool(ignore_eos),
        "return_tokens": True,
        "stream": False,
    }
    if temperature == 0.0:
        body["top_k"] = 1
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    r = _post(base + "/completion", body, timeout=timeout)
    r["_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return r
