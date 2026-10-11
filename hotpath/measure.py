"""hotpath measure: active measurement against a server you control.

Submits real inference. Refuses to target a server unless it is listed as owned in the local
config or --i-own-this-server is passed, because a measurement request consumes GPU time and
changes the prompt cache of whatever is running there.

Per request it records the server's own timings (prompt_n, prompt_ms, predicted_n, predicted_ms,
cache_n), the wall time, and the generated token ids, so prefill, generation, and end-to-end
latency are kept apart and outputs can be compared across arms.
"""
import hashlib
import time

from hotpath import gpu as gpu_mod
from hotpath import server as srv
from hotpath.records import new_record, percentiles
from hotpath.workload import workload_identity


def run_workload(base, workload, repetitions=3, warmup=1, erase_between=True, store_prompts=False, store_outputs=True,
                 slot_id=0, pid=None, progress=None):
    """Returns a list of per-request records plus per-repetition GPU snapshots."""
    out = {"requests": [], "warmup_requests": [], "gpu_snapshots": []}
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
                "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
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
    return {
        "n": len(reqs),
        "prompt_n": percentiles([r["prompt_n"] for r in reqs]),
        "cache_n_max": max([r["cache_n"] or 0 for r in reqs]) if reqs else None,
        "prefill_tps": percentiles([r["prompt_tps"] for r in reqs]),
        "decode_tps": percentiles([r["predicted_tps"] for r in reqs]),
        "prompt_ms": percentiles([r["prompt_ms"] for r in reqs]),
        "predicted_ms": percentiles([r["predicted_ms"] for r in reqs]),
        "wall_ms": percentiles([r["wall_ms"] for r in reqs]),
        "predicted_n": percentiles([r["predicted_n"] for r in reqs]),
        "stop_types": sorted({str(r["stop_type"]) for r in reqs}),
    }


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
    rec["summary"] = summarize_requests(res["requests"])
    rec["background_after"] = gpu_mod.sample_background(seconds=min(background_seconds, 5.0), interval=1.0)
    return rec
