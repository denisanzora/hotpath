"""hotpath observe: a passive baseline of a running deployment.

Collects, without submitting any inference request:
  - server facts from /health, /props, /v1/models, /slots (GET)
  - launch settings from the deployment's launcher configuration (if configured)
  - process-level GPU counters for the server PID (Windows) and whole-GPU state
  - the model's GGUF summary (header only)
  - the runtime binary's --version line and identity
  - request timing history from the server's own log, split by session
"""
import subprocess
from pathlib import Path

from hotpath import gguf as gguf_mod
from hotpath.adapters import env_with_path
from hotpath import gpu as gpu_mod
from hotpath import logparse
from hotpath import server as srv
from hotpath.config import read_launcher_settings
from hotpath.records import env_snapshot, file_identity, new_record


def binary_identity(path, path_prepend=None):
    if not path or not Path(path).is_file():
        return {"available": False}
    ident = file_identity(path)
    try:
        p = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, env=env_with_path(path_prepend))
        ver = (p.stdout + p.stderr).strip().splitlines()
        if p.returncode == 0:
            ident["version_lines"] = ver[:3]
        else:
            ident["version_error"] = "--version exited with code %s" % p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        ident["version_error"] = str(e)
    return ident


def observe(cfg, log_path=None, model_path=None, binary_path=None, base_url=None, sample_seconds=5.0):
    base = base_url or cfg.get("server_url")
    rec = new_record("observation", "passive", target={"base_url": base})
    rec["server"] = srv.passive_snapshot(base) if base else {"skipped": "no server_url"}
    rec["launcher_settings"] = read_launcher_settings(cfg)
    rec["launcher_env_documented"] = cfg.get("launcher_env") or {}
    rec["gpu"] = gpu_mod.gpu_static()
    rec["gpu_now"] = gpu_mod.gpu_dynamic()
    rec["gpu_background_sample"] = gpu_mod.sample_background(seconds=sample_seconds, interval=1.0) if sample_seconds else {}
    port = None
    if base:
        try:
            port = int(base.rsplit(":", 1)[1].split("/")[0])
        except ValueError:
            port = None
    pid = gpu_mod.pid_listening_on(port) if port else None
    rec["server_process"] = {"pid_found": pid is not None}
    if pid:
        rec["server_process"]["gpu_counters"] = gpu_mod.windows_process_gpu_counters(pid)
    rec["adapter_memory"] = gpu_mod.windows_adapter_memory()
    rec["adapter_memory"]["note"] = "adapter-wide counters as reported; per-process values are not summed into a paging verdict"
    bp = binary_path or cfg.get("runtime_binary")
    rec["runtime_binary"] = binary_identity(bp, cfg.get("path_prepend"))
    mp = model_path or cfg.get("model_path")
    if mp and Path(mp).is_file():
        hdr = gguf_mod.read_header(mp)
        rec["model"] = {"identity": file_identity(mp), "summary": gguf_mod.summarize(hdr)}
    else:
        rec["model"] = {"available": False}
    lp = log_path or cfg.get("server_log")
    if lp and Path(lp).is_file():
        sessions = logparse.parse_log(lp, model_filter=cfg.get("model_filter"))
        all_requests = [r for s in sessions for r in s["requests"]]
        rec["log"] = {
            "source_basename": Path(lp).name,
            "sessions": len(sessions),
            "launch_facts_printed": logparse.launch_facts_coverage(sessions),
            "session_table": logparse.session_table(sessions),
            "all_sessions_stats": logparse.request_stats(all_requests),
            "gpu_like_sessions_stats": logparse.request_stats(all_requests, min_pp_tps=cfg.get("log_cpu_mode_prefill_tps_below", 10.0)),
            "latest_session_stats": logparse.request_stats(sessions[-1]["requests"]) if sessions else None,
            "note": "sessions whose prefill is below log_cpu_mode_prefill_tps_below t/s are treated as CPU-mode "
                    "for the gpu_like statistics; the log itself does not say which backend served a request",
        }
    else:
        rec["log"] = {"available": False}
    rec["env_now"] = env_snapshot()
    return rec


def observation_markdown(rec):
    lines = ["# Observation (passive)", "", "Created: %s" % rec["created"], ""]
    s = rec.get("server", {})
    p = s.get("props", {})
    lines += ["## Server", "",
              "- health: %s" % s.get("health", {}).get("status"),
              "- build: %s" % p.get("build_info"),
              "- model: %s" % p.get("model_basename"),
              "- ctx/slot: %s, slots: %s" % (p.get("n_ctx_per_slot"), p.get("total_slots")),
              ""]
    ls = rec.get("launcher_settings", {})
    if ls.get("available"):
        lines += ["## Launcher settings", ""] + ["- %s: %s" % (k, v) for k, v in ls.items() if not k.startswith("_") and k != "available"] + [""]
    g = rec.get("gpu", {})
    lines += ["## GPU", "", "- %s, driver %s (%s), %s MiB, cc %s" % (g.get("name"), g.get("driver_version"), g.get("driver_model"), g.get("memory.total"), g.get("compute_cap")),
              "- now: %s" % rec.get("gpu_now"),
              "- background sample: %s" % rec.get("gpu_background_sample"),
              "- server process counters: %s" % rec.get("server_process", {}).get("gpu_counters"),
              ""]
    lg = rec.get("log", {})
    if lg.get("sessions"):
        lines += ["## Log history (%d sessions)" % lg["sessions"], "",
                  "| session | lines | device | n_batch | n_ubatch | compute MiB | requests | prefill med t/s | decode med t/s | warn | err |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in lg["session_table"]:
            if r["requests"] == 0 and not r["n_ubatch"]:
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["session"], r["lines"], r["device"] or "", r["n_batch"] or "", r["n_ubatch"] or "", r["compute_mib"] or "",
                r["requests"], r["prefill_median_tps"] or "", r["decode_median_tps"] or "", r["warnings"], r["errors"]))
        st = lg.get("gpu_like_sessions_stats", {})
        lines += ["", "GPU-like requests (prefill >= %s t/s): prefill %s, decode %s, prefill share of request %s" % (
            st.get("filters", {}).get("min_prompt_tps"), st.get("prefill_tps"), st.get("decode_tps"), st.get("prefill_share_of_request_pct")), ""]
    m = rec.get("model", {}).get("summary")
    if m:
        lines += ["## Model", "", "- arch %s, %s layers (%s full-attention), vocab %s" % (m["architecture"], m["n_layer"], m["n_full_attention_layers"], m["n_vocab"]),
                  "- weights %.3f GB, streamed per decode token %.3f GB, KV %s bytes/token (F16)" % (
                      m["total_bytes"] / 1e9, m["weight_bytes_streamed_per_decode_token"] / 1e9, m["kv_bytes_per_token_f16"]),
                  "- types: %s" % {k: v["count"] for k, v in m["tensor_types"].items()}, ""]
    return "\n".join(lines)
