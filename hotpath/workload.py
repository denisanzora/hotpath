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


def load_workload(path):
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    if w.get("kind") == "requests":
        for it in w["items"]:
            it.setdefault("sha256", _body_hash(it["body"]))
            it.setdefault("endpoint", "/v1/chat/completions" if "messages" in it["body"] else "/completion")
        return w
    for p in w["prompts"]:
        p.setdefault("sha256", hashlib.sha256(p["text"].encode()).hexdigest()[:16])
    return w


def save_workload(w, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=1)


def workload_from_requests(path, name="requests", limit=None):
    """A `requests` workload from your own captured request bodies: a JSON array of bodies, or JSONL with one body
    per line (optionally wrapped as {"endpoint": ..., "body": ...}). The bodies are what your application sends to
    llama-server (`/v1/chat/completions` or `/completion` JSON), kept verbatim including sampler, schema/grammar,
    cache and generation options, so an experiment measures the real request shape.

    The workload file holds the bodies (it lives in your local `workloads/`, which hotpath's .gitignore excludes).
    Records never store bodies or generated text for this kind unless `--store-outputs` is given; they store hashes,
    token counts and timings. Streaming is forced off because timings come from the final response."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        raw = json.loads(text)
    else:
        raw = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    items = []
    for i, r in enumerate(raw):
        if isinstance(r, dict) and "body" in r and isinstance(r["body"], dict):
            endpoint, body = r.get("endpoint"), r["body"]
        else:
            endpoint, body = None, r
        if not isinstance(body, dict) or not ("messages" in body or "prompt" in body):
            raise ValueError("item %d is not a llama-server request body (needs 'messages' or 'prompt')" % i)
        body = dict(body)
        body["stream"] = False
        items.append({"id": "r%02d" % i, "endpoint": endpoint or ("/v1/chat/completions" if "messages" in body else "/completion"), "body": body, "sha256": _body_hash(body)})
        if limit and len(items) >= limit:
            break
    if not items:
        raise ValueError("no request bodies found in %s" % path)
    return {"name": name, "kind": "requests", "source_basename": Path(path).name, "erase_between": "repetition", "items": items,
            "n_predict": None, "temperature": None,
            "note": "Real request bodies as sent by the application; arms may override fields with request_override. "
                    "erase_between: 'repetition' (slot cleared before each pass; items warm each other as in the application), 'item', or false. "
                    "Output validity is judged against each item's own schema, whatever an arm removes. "
                    "Records keep hashes, counts and timings only, unless outputs are explicitly stored."}


def workload_identity(w):
    if w.get("kind") == "requests":
        hashes = [it["sha256"] for it in w["items"]]
        return {"name": w["name"], "kind": "requests", "n_prompts": len(hashes), "n_predict": None, "temperature": None, "ignore_eos": False,
                "prompt_hashes": hashes, "set_hash": hashlib.sha256("".join(hashes).encode()).hexdigest()[:16],
                "endpoints": sorted({it["endpoint"] for it in w["items"]}),
                "fields_present": sorted({k for it in w["items"] for k in it["body"] if k not in ("messages", "prompt")})}
    return {"name": w["name"], "n_prompts": len(w["prompts"]), "n_predict": w["n_predict"], "temperature": w.get("temperature", 0.0),
            "ignore_eos": bool(w.get("ignore_eos", True)),
            "prompt_hashes": [p["sha256"] for p in w["prompts"]],
            "set_hash": hashlib.sha256("".join(p["sha256"] for p in w["prompts"]).encode()).hexdigest()[:16]}
