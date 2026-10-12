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
from hotpath import server as srv
from hotpath.measure import run_workload, summarize_requests
from hotpath.observe import binary_identity
from hotpath.records import env_snapshot, file_identity, new_record, redact_args, redact_path
from hotpath.workload import load_workload, workload_identity


def load_spec(path, local_cfg):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
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
    n_items, wl_np = None, None
    try:
        w = load_workload(spec["workload"])
        n_items = len(w.get("items") or w.get("prompts") or [])
        wl_np = w.get("n_predict")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    per_launch = (int(spec.get("warmup", 1)) + int(spec.get("repetitions", 2))) * n_items if n_items else None
    lines = ["Experiment: %s" % spec["name"], "Rounds: %d, arms: %d -> server launches: %d%s" % (
                 rounds, len(arms), rounds * len(arms), ", requests: %d (%d per launch, warmups included)" % (per_launch * rounds * len(arms), per_launch) if per_launch else ""),
             "Binary: %s" % redact_path(spec["binary"]), "Model: %s" % redact_path(spec["model"]),
             "Common args: %s" % " ".join(redact_args(spec.get("common_args", []))), "Env: %s" % (spec.get("env") or {}),
             "Workload: %s, n_predict %s, repetitions %s, warmup %s" % (redact_path(spec["workload"]), spec.get("n_predict") or ("%s (from the workload)" % wl_np if wl_np else "per request body"),
                                                                       spec.get("repetitions", 2), spec.get("warmup", 1)),
             "Preconditions: %s" % (spec.get("preconditions") or {}), ""]
    lines.insert(-1, "Cache mode: %s; slot-save-path for hotpath-owned servers: %s; schema validation: %s" % (
        spec.get("cache_mode", "cold (workload default)"), spec.get("slot_save_path") or "off", spec.get("schema_validation", "auto")))
    for a in arms:
        cmd, env, port = arm_launch(spec, a)
        shown = [redact_path(cmd[0]), "--model", redact_path(cmd[2])] + redact_args(cmd[3:])
        if a.get("binary"):
            shown[0] = "%s(%s)" % (redact_path(cmd[0]), redact_path(str(Path(cmd[0]).parent.parent.parent)))
        ov = {k: env.get(k) for k in (spec.get("env") or {})}
        ov.update(a.get("env_override") or {})
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
        return ""
    if not procs:
        return ""
    names = ", ".join("llama-server pid %s%s" % (p["pid"], " (port %s)" % ",".join(str(x[1]) for x in ports.get(p["pid"], [])) if ports.get(p["pid"]) else "")
                      for p in procs)
    return (". Running now: %s. hotpath does not stop, restart or reconfigure servers it did not start: stop it yourself if it may be interrupted, "
            "or pass --wait-s N to wait until the GPU is free" % names)


def _launch(cmd, env, log_path):
    log = open(log_path, "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, **kw)
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
        log.close()


def run(spec, out_dir, dry_run=False, wait_for_preconditions_s=0):
    if dry_run:
        print(plan(spec))
        return None
    # Everything that can be decided from the spec alone is validated first, so a spec that can never run fails at once
    # instead of after waiting (possibly minutes) for an idle GPU.
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
    pre = spec.get("preconditions") or {}
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
    for rnd in range(rounds):
        for a in spec["arms"]:
            cmd, env, port = arm_launch(spec, a, out_dir)
            base = "http://127.0.0.1:%d" % port
            log_path = out / ("server-%s-%s-r%d.log" % (spec["name"], a["name"], rnd))
            print("[round %d] launching arm %s" % (rnd, a["name"]))
            t0 = time.time()
            proc, log = _launch(cmd, env, log_path)
            arm_rec = rec["arms"][a["name"]]
            arm_rec["launches"] += 1
            if "binary" not in arm_rec:
                # per-arm identity: arms may differ only by binary or model, and version strings can be identical across builds
                arm_rec["binary"] = binary_identity(cmd[0], spec.get("path_prepend"))
                arm_rec["model"] = file_identity(cmd[2])
                arm_rec["env_ggml"] = {k: v for k, v in env.items() if k.startswith(("GGML_", "LLAMA_"))}
                arm_rec["args"] = [redact_path(cmd[0]), "--model", redact_path(cmd[2])] + redact_args(cmd[3:])
                sib = Path(cmd[0]).parent / "ggml-cuda.dll"
                if sib.is_file():
                    arm_rec["ggml_cuda_dll"] = file_identity(str(sib))
            try:
                srv.wait_healthy(base, timeout_s=int(spec.get("health_timeout_s", 600)), proc=proc)
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
    for name, arm_rec in rec["arms"].items():
        arm_rec["summary"] = summarize_requests(arm_rec["requests"])
    rec["server_launches_total"] = len(rec["launches"])
    return rec


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
