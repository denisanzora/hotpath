"""Parse llama-server stderr logs into sessions and per-request timings.

Passive: reads a log file. Nothing is inferred beyond what the lines say. Each session is one
model load; a session carries the launch facts the log printed (which depends on verbosity)
and every request's prompt-processing and generation timing.
"""
import re
from collections import Counter

from hotpath.records import percentiles

RE_LOAD = re.compile(r"load_model: loading model '(.+?)'")
RE_PP = re.compile(r"prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_TG = re.compile(r"\|\s+eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?([\d.]+) tokens per second")
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens")
RE_TASK = re.compile(r"task (\d+)")
RE_LCP = re.compile(r"selected slot by (LRU|LCP similarity)(?:, f_sim_best = ([\d.]+))?.*?(?:f_keep = ([\d.]+))?$")
RE_RELEASE = re.compile(r"release: .*n_tokens = (\d+)")
RE_KV = {
    "n_batch": re.compile(r"llama_context: n_batch\s*=\s*(\d+)"),
    "n_ubatch": re.compile(r"llama_context: n_ubatch\s*=\s*(\d+)"),
    "n_ctx": re.compile(r"llama_context: n_ctx\s*=\s*(\d+)"),
    "flash_attn": re.compile(r"llama_context: flash_attn\s*=\s*(\S+)"),
    "n_threads": re.compile(r"threadpool init, n_threads = (\d+)"),
    "n_slots": re.compile(r"n_slots = (\d+)"),
    "n_ctx_slot": re.compile(r"n_ctx_slot = (\d+)"),
    "kv_unified": re.compile(r"kv_unified = '(\w+)'"),
    "offloaded_layers": re.compile(r"offloaded (\d+/\d+) layers to GPU"),
    "model_buffer_gpu_mib": re.compile(r"(?:CUDA0|Vulkan0|Metal|ROCm0) model buffer size\s*=\s*([\d.]+) MiB"),
    "model_buffer_cpu_mib": re.compile(r"CPU_Mapped model buffer size\s*=\s*([\d.]+) MiB"),
    "kv_buffer_gpu_mib": re.compile(r"(?:CUDA0|Vulkan0|Metal|ROCm0) KV buffer size\s*=\s*([\d.]+) MiB"),
    "rs_buffer_gpu_mib": re.compile(r"(?:CUDA0|Vulkan0|Metal|ROCm0) RS buffer size\s*=\s*([\d.]+) MiB"),
    "compute_buffer_gpu_mib": re.compile(r"(?:CUDA0|Vulkan0|Metal|ROCm0) compute buffer size\s*=\s*([\d.]+) MiB"),
    "device_line": re.compile(r"using device (\S+) \((.+?)\)"),
    "fit_warning": re.compile(r"(cannot meet free memory target.*)$"),
    "projected_vs_free": re.compile(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory"),
    "cuda_graphs_disabled_arch": re.compile(r"(disabling CUDA graphs due to GPU architecture)"),
    "error": re.compile(r"(CUDA error|cudaError|illegal memory|out of memory|GGML_ASSERT|failed to allocate)"),
}


def parse_log(path, model_filter=None):
    sessions = []
    cur = None
    pending = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            m = RE_LOAD.search(line)
            if m:
                cur = {"index": len(sessions) + 1, "first_line": lineno, "last_line": lineno,
                       "model_basename": m.group(1).replace("\\", "/").rsplit("/", 1)[-1],
                       "launch": {}, "warnings": [], "errors": [], "requests": []}
                sessions.append(cur)
                pending = {}
                continue
            if cur is None:
                continue
            cur["last_line"] = lineno
            for key, rx in RE_KV.items():
                mm = rx.search(line)
                if not mm:
                    continue
                if key == "fit_warning":
                    cur["warnings"].append(mm.group(1).strip())
                elif key == "error":
                    cur["errors"].append(line.strip()[:200])
                elif key == "cuda_graphs_disabled_arch":
                    cur["launch"]["cuda_graphs_disabled_by_arch"] = True
                elif key == "device_line":
                    cur["launch"]["device"] = mm.group(1)
                    cur["launch"]["device_name"] = mm.group(2)
                elif key == "projected_vs_free":
                    cur["launch"]["projected_mib"] = int(mm.group(1))
                    cur["launch"]["free_at_start_mib"] = int(mm.group(2))
                else:
                    cur["launch"].setdefault(key, mm.group(1))
            task = RE_TASK.search(line)
            tid = task.group(1) if task else None
            ml = RE_LCP.search(line)
            if ml:
                cur["_last_slot_select"] = {"by": ml.group(1), "f_sim_best": float(ml.group(2)) if ml.group(2) else None,
                                           "f_keep": float(ml.group(3)) if ml.group(3) else None}
                continue
            mr = RE_RELEASE.search(line)
            if mr and cur["requests"]:
                cur["requests"][-1].setdefault("n_tokens_in_context", int(mr.group(1)))
                continue
            mp = RE_PP.search(line)
            if mp:
                pending.setdefault(tid, {})["prompt"] = {"ms": float(mp.group(1)), "n": int(mp.group(2)), "tps": float(mp.group(3)), "line": lineno}
                continue
            mt = RE_TG.search(line)
            if mt:
                pending.setdefault(tid, {})["gen"] = {"ms": float(mt.group(1)), "n": int(mt.group(2)), "tps": float(mt.group(3))}
                continue
            mo = RE_TOTAL.search(line)
            if mo and tid in pending:
                rec = pending.pop(tid)
                rec["total"] = {"ms": float(mo.group(1)), "n": int(mo.group(2))}
                rec["task"] = tid
                rec["slot_select"] = cur.pop("_last_slot_select", None)
                cur["requests"].append(rec)
    if model_filter:
        sessions = [s for s in sessions if model_filter.lower() in s["model_basename"].lower()]
    return sessions


def request_stats(requests, min_prompt=100, min_gen=20, min_pp_tps=None):
    """Summaries over a request list. Filters make small or CPU-mode requests explicit, never silent."""
    pp = [r["prompt"]["tps"] for r in requests if "prompt" in r and r["prompt"]["n"] >= min_prompt
          and (min_pp_tps is None or r["prompt"]["tps"] >= min_pp_tps)]
    tg = [r["gen"]["tps"] for r in requests if "gen" in r and r["gen"]["n"] >= min_gen
          and (min_pp_tps is None or ("prompt" in r and r["prompt"]["tps"] >= min_pp_tps))]
    shares = []
    pn = []
    gn = []
    for r in requests:
        if "prompt" in r and "gen" in r and r["prompt"]["n"] >= min_prompt and r["gen"]["n"] >= 1:
            tot = r["prompt"]["ms"] + r["gen"]["ms"]
            if tot > 0:
                shares.append(100.0 * r["prompt"]["ms"] / tot)
                pn.append(r["prompt"]["n"])
                gn.append(r["gen"]["n"])
    return {
        "filters": {"min_prompt_tokens": min_prompt, "min_gen_tokens": min_gen, "min_prompt_tps": min_pp_tps},
        "requests_total": len(requests),
        "prefill_tps": percentiles(pp),
        "decode_tps": percentiles(tg),
        "prompt_tokens": percentiles(pn),
        "gen_tokens": percentiles(gn),
        "prefill_share_of_request_pct": percentiles(shares),
        "note": "prefill share is computed per request from that request's own prompt and generation times; "
                "it is not a ratio of medians",
    }


def session_table(sessions):
    rows = []
    for s in sessions:
        st = request_stats(s["requests"])
        rows.append({
            "session": s["index"],
            "lines": "%d-%d" % (s["first_line"], s["last_line"]),
            "device": s["launch"].get("device"),
            "n_batch": s["launch"].get("n_batch"),
            "n_ubatch": s["launch"].get("n_ubatch"),
            "compute_mib": s["launch"].get("compute_buffer_gpu_mib"),
            "model_mib": s["launch"].get("model_buffer_gpu_mib"),
            "requests": len(s["requests"]),
            "prefill_median_tps": (st["prefill_tps"] or {}).get("median"),
            "decode_median_tps": (st["decode_tps"] or {}).get("median"),
            "warnings": len(s["warnings"]),
            "errors": len(s["errors"]),
        })
    return rows


def launch_facts_coverage(sessions):
    """Which launch facts the log actually printed, so missing values are reported as missing."""
    c = Counter()
    for s in sessions:
        for k in s["launch"]:
            c[k] += 1
    return dict(c)
