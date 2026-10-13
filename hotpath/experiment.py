"""hotpath experiment: controlled arms launched by hotpath itself.

A spec names a base launch (binary, model, common args, env) and arms that override args, env,
or the binary. hotpath launches each arm's server on its own port, waits for health, runs the
workload, records, and stops it. Arms are interleaved across rounds so drift and thermals hit
every arm. The original deployment is never modified: this launches separate processes.

Server launches = rounds x arms. The spec's dry run prints that number.
"""
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from hotpath import gpu as gpu_mod
from hotpath import launch
from hotpath import locks
from hotpath import server as srv
from hotpath.measure import run_workload, summarize_requests
from hotpath.observe import binary_identity
from hotpath.records import env_snapshot, file_identity, new_record, redact_args, redact_env, redact_path
from hotpath.workload import load_workload, workload_identity


def load_spec(path, local_cfg, raw=None):
    """A spec with its ${key} references resolved from the local config. `raw`: the already parsed (and, for agent-run
    launches, already checked) content of `path`, so exactly what was checked is what runs."""
    if raw is None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    spec = copy.deepcopy(raw)
    # values like "${model_path}" resolve from the local config so specs stay path-free
    def resolve(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            key = v[2:-1]
            if key not in local_cfg:
                raise SystemExit("spec references ${%s} but hotpath.local.json has no such key" % key)
            return local_cfg[key]
        if isinstance(v, list):
            return [resolve(x) for x in v]
        if isinstance(v, dict):
            return {k: resolve(x) for k, x in v.items()}
        return v
    return resolve(spec)


def slot_save_path_for(spec, arm, out_dir):
    """Optional --slot-save-path for hotpath-owned launches ("slot_save_path": "auto" or a directory), so the server
    honors slot erase (llama-server returns HTTP 501 for slot actions without it). Off by default: cold isolation
    does not depend on it (see measure.run_requests_workload), and the launched command line stays the user's own."""
    cfg = spec.get("slot_save_path", False)
    if not cfg:
        return None
    if isinstance(cfg, str) and cfg != "auto":
        return cfg
    return str(Path(out_dir) / "slot-cache" / spec["name"])


def arm_launch(spec, arm, out_dir=None):
    binary = arm.get("binary") or spec["binary"]
    args = [str(a) for a in spec.get("common_args", [])]
    # alias-aware: an override of --gpu-layers replaces a -ngl in common_args, so exactly one value reaches the server
    for k, v in (arm.get("args_override") or {}).items():
        args = launch.set_arg(args, k, v)
    for k in arm.get("args_remove") or []:
        args = launch.remove_arg(args, k)
    ssp = slot_save_path_for(spec, arm, out_dir or "results/local")
    if ssp and "--slot-save-path" not in args:
        args += ["--slot-save-path", ssp]
    env = dict(os.environ)
    # directories to put first on PATH (e.g. the CUDA runtime DLLs a native build needs); values resolve via the local config
    if spec.get("path_prepend"):
        env["PATH"] = os.pathsep.join(spec["path_prepend"]) + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    env.update({k: str(v) for k, v in (arm.get("env_override") or {}).items()})
    for k in arm.get("env_remove") or []:
        env.pop(k, None)
    port = int(spec.get("port", 8099))
    model = arm.get("model") or spec["model"]
    cmd = [binary, "--model", model, "--host", "127.0.0.1", "--port", str(port)] + args
    return cmd, env, port


def plan(spec):
    rounds = int(spec.get("rounds", 3))
    arms = spec["arms"]
    n_items, wl_np, warm_items = None, None, None
    try:
        w = load_workload(spec["workload"])
        n_items = len(w.get("items") or w.get("prompts") or [])
        wl_np = w.get("n_predict")
        # a cold pass of application requests warms up with its first request only (measure.run_requests_workload)
        cold = w.get("kind") == "requests" and (spec.get("cache_mode") or w.get("cache_mode") or "cold") == "cold"
        warm_items = 1 if cold else n_items
    except (OSError, ValueError, KeyError, TypeError):
        pass
    per_launch = (int(spec.get("warmup", 1)) * warm_items + int(spec.get("repetitions", 2)) * n_items) if n_items else None
    lines = ["Experiment: %s" % spec["name"], "Rounds: %d, arms: %d -> server launches: %d%s" % (
                 rounds, len(arms), rounds * len(arms), ", requests: %d (%d per launch, warmups included)" % (per_launch * rounds * len(arms), per_launch) if per_launch else ""),
             "Binary: %s" % redact_path(spec["binary"]), "Model: %s" % redact_path(spec["model"]),
             "Common args: %s" % " ".join(redact_args(spec.get("common_args", []))), "Env: %s" % redact_env(spec.get("env") or {}),
             "Workload: %s, n_predict %s, repetitions %s, warmup %s" % (redact_path(spec["workload"]), spec.get("n_predict") or ("%s (from the workload)" % wl_np if wl_np else "per request body"),
                                                                       spec.get("repetitions", 2), spec.get("warmup", 1)),
             "Preconditions: %s" % (spec.get("preconditions") or {}), ""]
    est = (spec.get("provenance") or {}).get("estimated_duration")
    if est:
        lines.insert(-1, "Estimated duration: about %.0f min (rough: %s)" % (max(1.0, est["minutes"]), est["basis"]))
    lines.insert(-1, "Cache mode: %s; slot-save-path for hotpath-owned servers: %s; schema validation: %s" % (
        spec.get("cache_mode", "cold (workload default)"), spec.get("slot_save_path") or "off", spec.get("schema_validation", "auto")))
    for a in arms:
        cmd, env, port = arm_launch(spec, a)
        shown = [redact_path(cmd[0]), "--model", redact_path(cmd[2])] + redact_args(cmd[3:])
        if a.get("binary"):
            shown[0] = "%s(%s)" % (redact_path(cmd[0]), redact_path(str(Path(cmd[0]).parent.parent.parent)))
        ov = {k: env.get(k) for k in (spec.get("env") or {})}
        ov.update(a.get("env_override") or {})
        ov = redact_env(ov)
        req = "   request_override=%s" % a["request_override"] if a.get("request_override") else ""
        lines.append("arm %-12s %s   env=%s%s" % (a["name"], " ".join(shown), ov, req))
    return "\n".join(lines)


def check_preconditions(pre):
    if not pre:
        return {"checked": False}
    s = gpu_mod.sample_background(seconds=float(pre.get("sample_seconds", 10)), interval=1.0)
    d = gpu_mod.gpu_dynamic()
    free = (d.get("mem_total_mib") or 0) - (d.get("mem_used_mib") or 0)
    ok = True
    reasons = []
    if "min_free_vram_mib" in pre and free < pre["min_free_vram_mib"]:
        ok = False
        reasons.append("free VRAM %.0f MiB < %s" % (free, pre["min_free_vram_mib"]))
    if "max_gpu_util_pct" in pre and (s.get("util_pct_max") or 0) > pre["max_gpu_util_pct"]:
        ok = False
        reasons.append("GPU util max %.0f%% > %s" % (s["util_pct_max"], pre["max_gpu_util_pct"]))
    for port in pre.get("ports_must_be_free", []):
        if gpu_mod.pid_listening_on(int(port)):
            ok = False
            reasons.append("port %s is in use" % port)
    return {"checked": True, "ok": ok, "reasons": reasons, "free_vram_mib": free, "background": s}


def holders_hint(pc):
    """When preconditions fail on GPU memory, load or a port: name the llama-server processes that are running, and say
    that hotpath leaves them alone. hotpath only ever stops the servers it launched itself."""
    from hotpath import discover as disc
    try:
        procs = disc.server_processes()
        ports = disc.listening_ports() if procs else {}
    except Exception:  # noqa: BLE001 - the hint is best effort; the precondition result stands on its own
        return ". Pass --wait-s N to wait for the preconditions"
    if not procs:
        gpu = any("VRAM" in r or "util" in r for r in pc.get("reasons") or [])
        return ". No llama-server is running; %s: pass --wait-s N to wait" % (
            "another program is using the GPU (desktop activity counts too)" if gpu else "another program holds the port")
    names = ", ".join("llama-server pid %s%s%s" % (p["pid"], " (port %s)" % ",".join(str(x[1]) for x in ports.get(p["pid"], [])) if ports.get(p["pid"]) else "",
                                                   " launched by another hotpath run" if p.get("launched_by_hotpath") else "")
                      for p in procs)
    return (". Running now: %s. hotpath does not stop, restart or reconfigure servers it did not start: stop it yourself if it may be interrupted, "
            "or pass --wait-s N to wait until the GPU is free" % names)


def _port_free(port):
    """Whether nothing listens on the loopback port: a bind test, which needs no netstat or lsof."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _launch(cmd, env, log_path):
    from hotpath import safepath
    if not Path(cmd[0]).is_absolute() and not Path(cmd[0]).parent.parts:
        cmd = safepath.resolve(cmd)            # a bare runtime_binary name: from PATH, never the working directory
    log = open(log_path, "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, **kw)
    locks.track(proc)
    return proc, log


def _stop(proc, log):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    finally:
        locks.untrack(proc)
        log.close()


def run(spec, out_dir, dry_run=False, wait_for_preconditions_s=0):
    if dry_run:
        print(plan(spec))
        return None
    # Everything that can be decided from the spec alone is validated first, so a spec that can never run fails at once
    # instead of after waiting (possibly minutes) for an idle GPU.
    from hotpath.records import check_name
    try:
        check_name(spec.get("name"), "spec name")
        for a in spec["arms"]:
            check_name(a.get("name"), "arm name")
    except ValueError as e:
        raise SystemExit("error: %s" % e)
    workload = load_workload(spec["workload"])
    if workload.get("kind") != "requests":
        if spec.get("n_predict"):
            workload["n_predict"] = int(spec["n_predict"])
        if "ignore_eos" in spec:
            workload["ignore_eos"] = bool(spec["ignore_eos"])
    else:
        if spec.get("store_outputs"):
            workload["store_outputs"] = True
        if spec.get("cache_mode"):
            workload["cache_mode"] = spec["cache_mode"]
        if spec.get("schema_validation"):
            workload["schema_validation"] = spec["schema_validation"]
    if workload.get("kind") != "requests" and any(a.get("request_override") for a in spec["arms"]):
        raise SystemExit("error: an arm sets request_override, which only applies to `requests` workloads (hotpath workload --from-requests); "
                         "with this workload the arms would run identical requests")
    if workload.get("kind") == "requests":
        from hotpath.measure import CACHE_MODES, require_schema_validation
        if workload.get("cache_mode", "cold") not in CACHE_MODES:
            raise SystemExit("error: cache_mode must be one of %s, got %r" % (", ".join(CACHE_MODES), workload.get("cache_mode")))
        require_schema_validation(workload.get("schema_validation", "auto"))   # fails before any wait or launch
    # one hotpath process does GPU work at a time on this machine (whichever client or working directory started it)
    with locks.gpu_work("experiment", spec["name"], wait_s=wait_for_preconditions_s, log=print):
        return _run_locked(spec, out_dir, workload, wait_for_preconditions_s)


def _run_locked(spec, out_dir, workload, wait_for_preconditions_s):
    pre = spec.get("preconditions") or {}
    port = int(spec.get("port", 8099))
    holder = gpu_mod.pid_listening_on(port) or (None if _port_free(port) else "unknown")
    if holder:
        # whatever the preconditions say: a server already answers on this port, so hotpath's requests (and slot erases) would
        # reach a server it did not start, such as the user's own
        raise SystemExit("refused: port %d is already in use by pid %s; hotpath sends experiment requests only to the server it launches. Choose another "
                         "port (the spec's `port`) or stop that process yourself" % (port, holder))
    t_end = time.time() + wait_for_preconditions_s
    while True:
        pc = check_preconditions(pre)
        if not pc.get("checked") or pc["ok"]:
            break
        if time.time() >= t_end:
            raise SystemExit("preconditions not met: %s%s" % ("; ".join(pc["reasons"]), holders_hint(pc)))
        print("waiting for preconditions: %s" % pc["reasons"])
        time.sleep(60)
    ssp = slot_save_path_for(spec, None, out_dir)
    if ssp:
        Path(ssp).mkdir(parents=True, exist_ok=True)
    rounds = int(spec.get("rounds", 3))
    rec = new_record("experiment", "controlled", name=spec["name"], spec_redacted=json.loads(plan_json(spec)),
                     preconditions_result=pc, workload=workload_identity(workload),
                     gpu=gpu_mod.gpu_static(), binary=binary_identity(spec["binary"], spec.get("path_prepend")), model=file_identity(spec["model"]),
                     env_baseline=env_snapshot(), arms={}, launches=[])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for a in spec["arms"]:
        rec["arms"][a["name"]] = {"requests": [], "warmup_requests": [], "gpu_snapshots": [], "launches": 0, "launch_facts": [], "failures": []}
    rec["cache_mode"] = workload.get("cache_mode", "cold")
    if spec.get("decision_fields"):
        rec["decision_fields"] = spec["decision_fields"]
    private_probes = {}         # application requests: probe tokens stay in memory and only their comparison is recorded
    n_items = len(workload.get("items") or workload.get("prompts") or [])
    total = rounds * len(spec["arms"])
    # a cold pass of application requests warms up with its first request only (measure.run_requests_workload)
    warm_items = 1 if (workload.get("kind") == "requests" and workload.get("cache_mode", "cold") == "cold") else n_items
    print("running %d launches, %d requests" % (total, total * (warm_items * int(spec.get("warmup", 1)) + n_items * int(spec.get("repetitions", 2)))), flush=True)
    for rnd in range(rounds):
        for a in spec["arms"]:
            cmd, env, port = arm_launch(spec, a, out_dir)
            base = "http://127.0.0.1:%d" % port
            log_path = out / ("server-%s-%s-r%d.log" % (spec["name"], a["name"], rnd))
            print("[round %d] launching arm %s (launch %d/%d)" % (rnd, a["name"], len(rec["launches"]) + 1, total), flush=True)
            t0 = time.time()
            proc, log = _launch(cmd, env, log_path)
            arm_rec = rec["arms"][a["name"]]
            arm_rec["launches"] += 1
            if "binary" not in arm_rec:
                # per-arm identity: arms may differ only by binary or model, and version strings can be identical across builds
                arm_rec["binary"] = binary_identity(cmd[0], spec.get("path_prepend"))
                arm_rec["model"] = file_identity(cmd[2])
                arm_rec["env_ggml"] = redact_env({k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_"))})
                arm_rec["args"] = [redact_path(cmd[0]), "--model", redact_path(cmd[2])] + redact_args(cmd[3:])
                sib = Path(cmd[0]).parent / "ggml-cuda.dll"
                if sib.is_file():
                    arm_rec["ggml_cuda_dll"] = file_identity(str(sib))
            try:
                srv.wait_healthy(base, timeout_s=int(spec.get("health_timeout_s", 600)), proc=proc)
                owner = gpu_mod.pid_listening_on(port)
                if owner is not None and owner != proc.pid:
                    raise srv.ServerError("the server answering on port %d is pid %s, not the llama-server hotpath launched (pid %s); no requests sent" % (
                        port, owner, proc.pid))
                load_s = round(time.time() - t0, 1)
                snap = srv.passive_snapshot(base)
                pid = proc.pid
                res = run_workload(base, workload, repetitions=int(spec.get("repetitions", 2)), warmup=int(spec.get("warmup", 1)),
                                   pid=pid, progress=lambda s: print("    " + s), request_override=a.get("request_override"))
                for r in res["requests"]:
                    r["round"] = rnd
                for r in res["warmup_requests"]:
                    r["round"] = rnd
                arm_rec["requests"] += res["requests"]
                arm_rec["warmup_requests"] += res["warmup_requests"]
                if res.get("slot_erase_error"):
                    arm_rec["slot_erase_error"] = res["slot_erase_error"]
                if res.get("cache"):
                    arm_rec["cache"] = res["cache"]
                arm_rec["gpu_snapshots"] += [dict(s, round=rnd) for s in res["gpu_snapshots"]]
                if rnd == 0 and int(spec.get("probe_top_k", 10)) > 0:
                    if workload.get("kind") == "requests":
                        private_probes[a["name"]], arm_rec["probes"] = probe_requests(base, workload, int(spec.get("probe_top_k", 10)), res["requests"],
                                                                                      a.get("request_override"))
                    else:
                        arm_rec["probes"] = probe_outputs(base, workload, int(spec.get("probe_top_k", 10)), res["requests"])
                arm_rec["launch_facts"].append({"round": rnd, "load_s": load_s, "props": snap.get("props"),
                                                "process_counters_after": gpu_mod.windows_process_gpu_counters(pid),
                                                "gpu_after": gpu_mod.gpu_dynamic()})
            except Exception as e:  # noqa: BLE001 - record the failure, continue other arms
                arm_rec["failures"].append({"round": rnd, "error": str(e)[:500]})
                print("    arm %s round %d FAILED: %s" % (a["name"], rnd, str(e)[:200]))
            finally:
                _stop(proc, log)
                rec["launches"].append({"round": rnd, "arm": a["name"], "log": log_path.name})
                time.sleep(float(spec.get("cooldown_s", 5)))
    if private_probes:
        compare_private_probes(rec, spec, private_probes)
    for name, arm_rec in rec["arms"].items():
        arm_rec["summary"] = summarize_requests(arm_rec["requests"])
    rec["server_launches_total"] = len(rec["launches"])
    return rec


def compare_private_probes(rec, spec, private_probes):
    """Each arm's probes against the first arm's, computed before anything is written: the record keeps where outputs
    diverge, the log-probability margins there and how far the divergence carried, never the tokens of the user's outputs."""
    from hotpath import fidelity
    first = spec["arms"][0]["name"]
    ref = private_probes.get(first) or {}
    for name, probes in private_probes.items():
        if name == first:
            continue
        rec["arms"][name]["probe_comparison"] = {pid: fidelity.compare_probes(ref[pid], pr) for pid, pr in probes.items() if pid in ref}
    rec["probe_comparison_basis"] = ("application requests: each arm's probe outputs were compared with arm %s's in memory when the run ended; "
                                     "token ids and top-k candidates are not stored" % first)


def probe_requests(base, workload, k, timed_requests, request_override=None):
    """Untimed: every application request once more as the arm sends it, with top-k log-probabilities (chat: `logprobs` +
    `top_logprobs`, completion: `n_probs`) and the prompt cache off. Returns (private probes with tokens, for comparison in
    memory only; public per-request facts for the record: probe length, whether it reproduced the timed output). Requests
    that sample (temperature above 0) are not probed: identity is not expected for them."""
    from hotpath import fidelity
    from hotpath.measure import _content, apply_request_override, output_digest
    from hotpath.workload import is_sampled
    private, public = {}, {}
    for it in workload["items"]:
        body = apply_request_override(it["body"], request_override)
        if is_sampled(body):
            public[it["id"]] = {"sampled": True}
            continue
        body["cache_prompt"] = False
        chat = it["endpoint"].startswith("/v1/chat")
        if chat:
            body.update({"logprobs": True, "top_logprobs": max(1, min(int(k), 20))})
        else:
            body["n_probs"] = int(k)
        try:
            resp = srv.request(base, it["endpoint"], body)
        except Exception as e:  # noqa: BLE001 - a missing probe makes the comparison weaker, not the run invalid
            public[it["id"]] = {"error": type(e).__name__}
            continue
        pr = fidelity.probe_from_chat_response(resp) if chat else fidelity.probe_from_response(resp)
        content, _stop = _content(resp)
        h = output_digest(content, workload)
        timed = [r.get("output_sha256") for r in timed_requests if r.get("prompt_id") == it["id"] and r.get("output_sha256")]
        public[it["id"]] = {"k": pr["k"], "n_tokens": len(pr["tokens"]), "matches_timed": (h == timed[0]) if timed else None}
        private[it["id"]] = pr
    return private, public


def probe_outputs(base, workload, k, timed_requests):
    """Untimed: every prompt once more with the same greedy settings and n_probs=k, for the fidelity comparison between
    arms (asking for probabilities slows generation, so timed requests never do). Records whether the probe produced the
    same tokens as this arm's timed requests."""
    from hotpath import fidelity
    out = {}
    for p in workload["prompts"]:
        try:
            resp = srv.completion(base, p["text"], workload["n_predict"], temperature=workload.get("temperature", 0.0), cache_prompt=False,
                                  ignore_eos=bool(workload.get("ignore_eos", True)), extra={"n_probs": int(k)})
        except Exception as e:  # noqa: BLE001 - a missing probe makes the comparison weaker, not the run invalid
            out[p["id"]] = {"error": str(e)[:200]}
            continue
        pr = fidelity.probe_from_response(resp)
        timed = [r.get("tokens") for r in timed_requests if r.get("prompt_id") == p["id"] and r.get("tokens")]
        pr["matches_timed"] = (pr["tokens"] == timed[0]) if timed else None
        out[p["id"]] = pr
    return out


def plan_json(spec):
    s = copy.deepcopy(spec)
    s["binary"] = redact_path(s["binary"])
    s["model"] = redact_path(s["model"])
    s["workload"] = redact_path(s["workload"])
    if s.get("path_prepend"):
        s["path_prepend"] = [redact_path(x) for x in s["path_prepend"]]
    s["common_args"] = redact_args(s.get("common_args", []))
    for a in s["arms"]:
        for k in ("args_override",):
            if a.get(k):
                a[k] = {kk: (redact_args([vv])[0] if isinstance(vv, str) else vv) for kk, vv in a[k].items()}
    for a in s["arms"]:
        if "binary" in a:
            a["binary"] = redact_path(a["binary"])
    return json.dumps(s)
