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
import time
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
        from hotpath import safepath
        p = safepath.run([path, "--version"], capture_output=True, text=True, timeout=30, env=env_with_path(path_prepend), stdin=subprocess.DEVNULL)
        ver = (p.stdout + p.stderr).strip().splitlines()
        if p.returncode == 0:
            ident["version_lines"] = ver[:3]
        else:
            ident["version_error"] = "--version exited with code %s" % p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        ident["version_error"] = str(e)
    return ident


BUILD_LIBRARIES = ("ggml-cuda.dll", "ggml.dll", "ggml-base.dll", "llama.dll", "libggml-cuda.so", "libggml.so", "libggml-base.so", "libllama.so")


def build_identity(exe, identity=None):
    """The identity of a llama-server build: its executable and the ggml/llama libraries next to it (two builds can share
    an identical llama-server executable and differ only in ggml-cuda), each by size and a hash of its head."""
    ident = identity or file_identity
    if not exe or not Path(exe).is_file():
        return None
    out = {"exe": ident(str(exe)), "libraries": {}}
    for name in BUILD_LIBRARIES:
        p = Path(exe).parent / name
        if p.is_file():
            out["libraries"][name] = ident(str(p))
    return out


def _only_running_server():
    """Zero-config target: the single llama-server on this machine that answers on loopback. Several -> none chosen."""
    from hotpath import discover as disc
    procs = disc.server_processes()
    if not procs:
        return None, "no running llama-server found; pass --server URL (and --log LOG) or set server_url in hotpath.local.json"
    ports = disc.listening_ports()
    live = []
    for p in procs:
        if p.get("launched_by_hotpath"):
            continue
        row = disc.describe_server_process(p, ports)
        if row.get("base_url") and ((row.get("http") or {}).get("health") or {}).get("status") == "ok":
            live.append(row)
    if len(live) == 1:
        return live[0]["base_url"], "the only running llama-server (pid %s)" % live[0]["pid"]
    if not live:
        return None, "llama-server is running but does not answer on loopback yet (still loading?)"
    return None, "several llama-servers answer (%s); choose one with --server" % ", ".join(r["base_url"] for r in live)


def metrics_snapshot(base):
    from hotpath import discover as disc
    h = {}
    try:
        h = disc.parse_metrics(disc._get(base + "/metrics", text=True))
    except Exception:  # noqa: BLE001 - /metrics off (HTTP 501) or unreachable: recorded as unavailable
        return None
    return {"t": time.time(), "values": h}


def observe(cfg, log_path=None, model_path=None, binary_path=None, base_url=None, sample_seconds=5.0):
    base = base_url or cfg.get("server_url")
    selected_by = "--server" if base_url else ("server_url in hotpath.local.json" if base else None)
    if not base:
        base, selected_by = _only_running_server()
    rec = new_record("observation", "passive", target={"base_url": base, "selected_by": selected_by})
    rec["server"] = srv.passive_snapshot(base) if base else {"skipped": selected_by}
    rec["launcher_settings"] = read_launcher_settings(cfg)
    rec["launcher_env_documented"] = cfg.get("launcher_env") or {}
    rec["gpu"] = gpu_mod.gpu_static()
    rec["gpu_now"] = gpu_mod.gpu_dynamic()
    reachable = base and (rec["server"].get("health") or {}).get("status") != "unreachable"
    m0 = metrics_snapshot(base) if reachable else None
    rec["gpu_background_sample"] = gpu_mod.sample_background(seconds=sample_seconds, interval=1.0) if sample_seconds else {}
    m1 = metrics_snapshot(base) if m0 else None
    if m0 and m1:
        rec["metrics"] = {"available": True, "since_server_start": m1["values"], "window_s": round(m1["t"] - m0["t"], 1),
                          "window_delta": {k: round(m1["values"][k] - m0["values"].get(k, 0.0), 3) for k in m1["values"] if k.endswith("_total")},
                          "source": "llama-server /metrics (aggregate counters since the server started; no per-request detail)"}
    else:
        rec["metrics"] = {"available": False, "note": "/metrics not enabled (start llama-server with --metrics) or server not reachable"}
    port = None
    if base:
        try:
            port = int(base.rsplit(":", 1)[1].split("/")[0])
        except ValueError:
            port = None
    pid = gpu_mod.pid_listening_on(port) if port else None
    rec["server_process"] = {"pid_found": pid is not None}
    lnch = None
    if pid:
        from hotpath import discover as disc
        rec["server_process"]["gpu_counters"] = gpu_mod.windows_process_gpu_counters(pid)
        lnch = disc.server_launch(pid)
        rec["server_process"]["launch"] = disc.launch_for_record(lnch)
    ls = (lnch or {}).get("settings") or {} if (lnch or {}).get("available") else {}
    rec["adapter_memory"] = gpu_mod.windows_adapter_memory()
    rec["adapter_memory"]["note"] = "adapter-wide counters as reported; per-process values are not summed into a paging verdict"
    bp = binary_path or cfg.get("runtime_binary") or (lnch or {}).get("exe")
    rec["runtime_binary"] = binary_identity(bp, cfg.get("path_prepend"))
    if lnch and lnch.get("exe") and bp and Path(bp).resolve() != Path(lnch["exe"]).resolve():
        rec["runtime_binary"]["note"] = "the running server's executable is %s, not this binary" % Path(lnch["exe"]).name
    if lnch and lnch.get("exe"):
        # the build the observed server actually runs (from its process), compared with the configured one before an experiment
        rec["server_build"] = build_identity(lnch["exe"])
    props_model = ((rec["server"].get("props") or {}).get("model_basename"))
    candidates = [(model_path, "--model"), (ls.get("model"), "the running server's --model"), (cfg.get("model_path"), "model_path in hotpath.local.json")]
    mp, msrc = next(((m, s) for m, s in candidates if m and Path(m).is_file()), (None, None))
    if mp:
        hdr = gguf_mod.read_header(mp)
        rec["model"] = {"identity": file_identity(mp), "summary": gguf_mod.summarize(hdr), "source": msrc}
        if props_model and Path(mp).name != props_model:
            rec["model"]["note"] = "the server reports model %s; this summary is of %s" % (props_model, Path(mp).name)
    else:
        rec["model"] = {"available": False}
    lp, lsrc = next(((p, s) for p, s in ((log_path, "--log"), (ls.get("log_file"), "the running server's --log-file"), (cfg.get("server_log"), "server_log in hotpath.local.json"))
                     if p and Path(p).is_file()), (None, None))
    if lp and Path(lp).is_file():
        sessions = logparse.parse_log(lp, model_filter=cfg.get("model_filter"))
        all_requests = [r for s in sessions for r in s["requests"]]
        rows = logparse.compact_requests(sessions, limit=int(cfg.get("log_requests_kept", 4000)))
        rec["log"] = {
            "source_basename": Path(lp).name,
            "selected_by": lsrc,
            "sessions": len(sessions),
            "parse_coverage": logparse.parse_coverage(lp, sessions),
            "launch_facts_printed": logparse.launch_facts_coverage(sessions),
            "session_table": logparse.session_table(sessions),
            "all_sessions_stats": logparse.request_stats(all_requests),
            "gpu_like_sessions_stats": logparse.request_stats(all_requests, min_pp_tps=cfg.get("log_cpu_mode_prefill_tps_below", 10.0)),
            "latest_session_stats": logparse.request_stats(sessions[-1]["requests"]) if sessions else None,
            "cache_reuse": logparse.cache_reuse_stats(rows),
            "requests": rows,
            "cpu_mode_prefill_tps_below": cfg.get("log_cpu_mode_prefill_tps_below", 10.0),
            "note": "sessions whose prefill is below log_cpu_mode_prefill_tps_below t/s are treated as CPU-mode "
                    "for the gpu_like statistics; the log itself does not say which backend served a request. "
                    "`requests` holds the newest per-request rows (server timings only; a server log has no client wall time).",
        }
        # no model known yet: the newest session's own load_model line names the file the server loaded
        newest = next((s.get("model_path") for s in reversed(sessions) if s.get("requests") and s.get("model_path")), None)
        if not rec["model"].get("summary") and newest and Path(newest).is_file():
            rec["model"] = {"identity": file_identity(newest), "summary": gguf_mod.summarize(gguf_mod.read_header(newest)),
                            "source": "the log's newest load_model line (session with requests)"}
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
              "- selected by: %s" % (rec.get("target") or {}).get("selected_by")]
    ln = (rec.get("server_process") or {}).get("launch") or {}
    if ln.get("available"):
        from hotpath import launch as launch_mod
        lines.append("- launch (%s): %s" % (ln.get("source"), launch_mod.describe(ln.get("settings") or {})))
    mt = rec.get("metrics") or {}
    if mt.get("available"):
        t = mt["since_server_start"]
        lines.append("- /metrics since start: %s prompt tokens processed (+%s from cache) in %.1f s, %s generated in %.1f s" % (
            int(t.get("prompt_tokens_total", 0)), int(t.get("prompt_tokens_cached_total", 0)), t.get("prompt_seconds_total", 0.0),
            int(t.get("tokens_predicted_total", 0)), t.get("tokens_predicted_seconds_total", 0.0)))
    lines.append("")
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
