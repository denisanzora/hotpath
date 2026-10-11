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


def load_workload(path):
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    for p in w["prompts"]:
        p.setdefault("sha256", hashlib.sha256(p["text"].encode()).hexdigest()[:16])
    return w


def save_workload(w, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=1)


def workload_identity(w):
    return {"name": w["name"], "n_prompts": len(w["prompts"]), "n_predict": w["n_predict"], "temperature": w.get("temperature", 0.0),
            "ignore_eos": bool(w.get("ignore_eos", True)),
            "prompt_hashes": [p["sha256"] for p in w["prompts"]],
            "set_hash": hashlib.sha256("".join(p["sha256"] for p in w["prompts"]).encode()).hexdigest()[:16]}
