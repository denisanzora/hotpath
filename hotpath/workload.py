"""Workloads: a deterministic prompt set with a target shape, stored by hash, never by text in records.

Prompts are synthetic by default (deterministic from a seed and target word counts) so they can be
published. A workload file can instead list real prompts; records then carry only their hashes
unless --store-prompts is given.
"""
import hashlib
import json
import random
from pathlib import Path

_WORDS = ("the model answers questions about training schedules recovery nutrition equipment form breathing mobility "
          "sleep hydration consistency progress plateaus warmups cooldowns intervals tempo volume intensity rest "
          "technique posture grip stance balance rhythm focus habit routine goal week month session coach notes").split()


FORMATS = {
    # Raw text: fine for base models; chat models often emit end-of-sequence immediately.
    "raw": "{instruction}{body}\n\nSummary:",
    # ChatML as used by Qwen-family chat models, with an empty think block so no reasoning tokens are generated.
    "chatml": "<|im_start|>user\n{instruction}{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
}


def synthetic_prompt(seed, n_words, instruction="Summarize the following notes in two sentences.\n\n", fmt="chatml"):
    rng = random.Random(seed)
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    body = []
    for i, w in enumerate(words):
        body.append(w)
        if i % 17 == 16:
            body[-1] += "."
    return FORMATS[fmt].format(instruction=instruction, body=" ".join(body))


def build_workload(name, n_prompts, word_counts, seed=1, n_predict=64, instruction=None, fmt="chatml"):
    prompts = []
    for i in range(n_prompts):
        wc = word_counts[i % len(word_counts)]
        kw = {"instruction": instruction} if instruction else {}
        text = synthetic_prompt(seed * 1000 + i, wc, fmt=fmt, **kw)
        prompts.append({"id": "p%02d" % i, "sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "words": wc, "text": text})
    return {"name": name, "seed": seed, "n_predict": n_predict, "temperature": 0.0, "ignore_eos": True, "format": fmt, "prompts": prompts}


def _body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


REQUEST_ENDPOINTS = ("/v1/chat/completions", "/chat/completions", "/completion", "/completions", "/v1/completions")


def request_class(item):
    """The request class of an application request: the label the capture gave it (`class` or `stage`), else one derived
    from its shape: endpoint, the JSON schema's name (or grammar / free text) and a short hash of the system message, so
    requests the application builds the same way fall into one class without storing any of their text."""
    if item.get("class"):
        return str(item["class"])
    body = item.get("body") or {}
    ep = "chat" if "messages" in body else "completion"
    rf = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    js = rf.get("json_schema") if isinstance(rf.get("json_schema"), dict) else {}
    shape = js.get("name") or ("schema" if (rf.get("type") in ("json_schema", "json_object") or body.get("json_schema")) else
                               ("grammar" if body.get("grammar") else "text"))
    sysmsg = next((m.get("content") for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "system"), None)
    tag = hashlib.sha256(json.dumps(sysmsg, ensure_ascii=False).encode("utf-8")).hexdigest()[:6] if sysmsg else "nosys"
    return "%s/%s/%s" % (ep, str(shape)[:40], tag)


def request_size(item):
    """Characters of prompt content plus the requested generation limit: an ordering key for picking small to large
    requests before any of them was measured."""
    body = item.get("body") or {}
    if "messages" in body:
        chars = sum(len(m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")))
                    for m in body["messages"] if isinstance(m, dict))
    else:
        chars = len(str(body.get("prompt") or ""))
    return chars + 4 * int(body.get("max_tokens") or body.get("n_predict") or 0)


def is_sampled(body):
    """Whether a request samples (temperature above 0 without top_k 1): its output is not expected to repeat across
    configurations, so token identity is not a meaningful check for it. llama-server's default temperature is 0.8."""
    try:
        t = float(body.get("temperature", 0.8))
    except (TypeError, ValueError):
        return True
    return t > 0 and body.get("top_k") != 1


def load_workload(path):
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    if w.get("subset_of"):
        # a subset names items of another requests workload by id and holds no request bodies itself, so the user's prompts
        # stay in the one file they keep in workloads/ (never in a results or session directory)
        with open(w["subset_of"], "r", encoding="utf-8") as f:
            if json.load(f).get("subset_of"):
                raise ValueError("subset_of must name a workload of requests, not another subset (%s)" % w["subset_of"])
        parent = load_workload(w["subset_of"])
        if parent.get("kind") != "requests":
            raise ValueError("subset_of must name a requests workload, %s is not one" % w["subset_of"])
        by_id = {it["id"]: it for it in parent["items"]}
        missing = [i for i in w.get("item_ids") or [] if i not in by_id]
        if missing:
            raise ValueError("subset names items %s that %s does not have" % (missing[:5], w["subset_of"]))
        out = {k: v for k, v in parent.items() if k != "items"}
        out.update({k: v for k, v in w.items() if k != "item_ids"})
        out["kind"] = "requests"
        out["items"] = [by_id[i] for i in w["item_ids"]]
        return out
    if w.get("kind") == "requests":
        for it in w["items"]:
            it.setdefault("sha256", _body_hash(it["body"]))
            it.setdefault("endpoint", "/v1/chat/completions" if "messages" in it["body"] else "/completion")
            if it["endpoint"] not in REQUEST_ENDPOINTS:
                # a workload sends inference requests, nothing else (a slot save or a props change is not a request to measure)
                raise ValueError("item %s: endpoint %r is not one of %s" % (it.get("id"), it["endpoint"], ", ".join(REQUEST_ENDPOINTS)))
            it["class"] = request_class(it)
        return w
    for p in w["prompts"]:
        p.setdefault("sha256", hashlib.sha256(p["text"].encode()).hexdigest()[:16])
    return w


def subset(parent_path, item_ids, name, basis):
    """A requests workload made of some items of `parent_path` (by id, in this order), holding no bodies."""
    return {"name": name, "kind": "requests", "subset_of": str(parent_path), "item_ids": list(item_ids), "basis": basis}


def save_workload(w, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=1)


def workload_from_requests(path, name="requests", limit=None):
    """A `requests` workload from your own captured request bodies: a JSON array of bodies, or JSONL with one body
    per line (optionally wrapped as {"endpoint": ..., "body": ...}). The bodies are what your application sends to
    llama-server (`/v1/chat/completions` or `/completion` JSON), kept verbatim including sampler, schema/grammar,
    cache and generation options, so an experiment measures the real request shape.

    The workload file holds the bodies, prompts included: keep it in your local `workloads/` (excluded by hotpath's
    .gitignore) and do not publish it. Records never store bodies or generated text for this kind unless the experiment
    spec sets "store_outputs": true; they store hashes, token counts, timings and validation results. Streaming is
    forced off because timings come from the final response. Prompt-cache handling is chosen per experiment with the
    spec's "cache_mode" (cold, warm, as-is), not here."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        raw = json.loads(text)
    else:
        raw = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    items = []
    for i, r in enumerate(raw):
        label = None
        if isinstance(r, dict) and "body" in r and isinstance(r["body"], dict):
            endpoint, body = r.get("endpoint"), r["body"]
            label = r.get("class") or r.get("stage")          # a capture may label each request with the application step it serves
        else:
            endpoint, body = None, r
        if not isinstance(body, dict) or not ("messages" in body or "prompt" in body):
            raise ValueError("item %d is not a llama-server request body (needs 'messages' or 'prompt')" % i)
        body = dict(body)
        body["stream"] = False
        it = {"id": ("r%02d" if len(raw) <= 100 else "r%04d") % i, "endpoint": endpoint or ("/v1/chat/completions" if "messages" in body else "/completion"),
              "body": body, "sha256": _body_hash(body)}
        if label:
            it["class"] = str(label)[:60]
            it["class_labelled"] = True
        it["class"] = request_class(it)
        items.append(it)
        if limit and len(items) >= limit:
            break
    if not items:
        raise ValueError("no request bodies found in %s" % path)
    return {"name": name, "kind": "requests", "source_basename": Path(path).name, "items": items,
            "n_predict": None, "temperature": None,
            "note": "Real request bodies as sent by the application, prompts included: keep this file local. Arms may override fields with "
                    "request_override. Cache handling is the experiment spec's cache_mode (cold by default: cache_prompt=false on every request). "
                    "Output validity is judged against each item's own schema, whatever an arm removes. "
                    "Records keep hashes, counts, timings and validation results only, unless the spec sets store_outputs."}


def workload_identity(w):
    if w.get("kind") == "requests":
        hashes = [it["sha256"] for it in w["items"]]
        classes = {}
        for it in w["items"]:
            c = request_class(it)
            classes[c] = classes.get(c, 0) + 1
        return {"name": w["name"], "kind": "requests", "n_prompts": len(hashes), "n_predict": None, "temperature": None, "ignore_eos": False,
                "prompt_hashes": hashes, "set_hash": hashlib.sha256("".join(hashes).encode()).hexdigest()[:16],
                "endpoints": sorted({it["endpoint"] for it in w["items"]}), "classes": classes,
                "sampled_requests": sum(1 for it in w["items"] if is_sampled(it["body"])),
                "subset_of": Path(w["subset_of"]).name if w.get("subset_of") else None,
                "fields_present": sorted({k for it in w["items"] for k in it["body"] if k not in ("messages", "prompt")})}
    return {"name": w["name"], "n_prompts": len(w["prompts"]), "n_predict": w["n_predict"], "temperature": w.get("temperature", 0.0),
            "ignore_eos": bool(w.get("ignore_eos", True)),
            "prompt_hashes": [p["sha256"] for p in w["prompts"]],
            "set_hash": hashlib.sha256("".join(p["sha256"] for p in w["prompts"]).encode()).hexdigest()[:16]}
