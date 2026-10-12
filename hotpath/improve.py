"""hotpath improve: the diagnose -> propose -> experiment -> verdict loop, run for the user within a budget.

Each step tests the top-ranked finding that has an executable experiment, one variable against the configuration
adopted so far, on hotpath's own server (never the user's), with the same workload throughout. A candidate is adopted
only on a `keep` verdict: faster, consistent across prompts, identical outputs, verified cache state, no launch
failure, and, for flags that trade prompt-cache reuse for overhead, real-traffic evidence that little reuse is lost.
Anything else is reported and left for the user to decide.

Speed-ups from separate steps are not multiplied together: when two or more changes were adopted, a final paired run
compares the original launch with the improved one, and that run is the only basis for the cumulative claim. The
result is a recommended launch line (the user's own, with the adopted changes); hotpath applies nothing to the user's
server.
"""
import json
import time
from pathlib import Path

from hotpath import diagnose as dg
from hotpath import launch
from hotpath import propose as prop
from hotpath.records import new_record, write_record

# experiment kinds the loop can run unattended: one variable, arms given as argument/environment overrides
EXECUTABLE_KINDS = frozenset({"gpu_layers", "ubatch_sweep", "cuda_graphs", "server_overhead_flags", "request_shape"})


def _canon(flag, *known):
    """One key per argument in the cumulative changes: the spelling already in use when there is one (`-ngl` and
    `--gpu-layers` are the same argument), else the spelling the experiment used."""
    sp = launch.spellings(flag)
    for keys in known:
        for k in keys:
            if k in sp:
                return k
    return flag


def materialize(common_args, env, arm):
    """The full argument list and environment an arm launches with (runner semantics, without binary/model/port)."""
    args = [str(a) for a in common_args]
    for k, v in (arm.get("args_override") or {}).items():
        args = launch.set_arg(args, k, v)
    for k in arm.get("args_remove") or []:
        args = launch.remove_arg(args, k)
    e = dict(env or {})
    e.update({k: str(v) for k, v in (arm.get("env_override") or {}).items()})
    for k in arm.get("env_remove") or []:
        e.pop(k, None)
    return args, e


def _same_config(a_args, a_env, b_args, b_env):
    return launch.parse(a_args)["settings"] == launch.parse(b_args)["settings"] and \
        sorted(launch.parse(a_args)["other_flags"]) == sorted(launch.parse(b_args)["other_flags"]) and dict(a_env) == dict(b_env)


def executable(finding, requests_workload):
    """Why a finding cannot be run by the loop, or None when it can."""
    if finding.get("superseded_by"):
        return "superseded by %s" % finding["superseded_by"]
    if finding.get("severity") not in ("high", "medium"):
        return "context, not a problem"
    exp = finding.get("experiment") or {}
    if not isinstance(exp.get("arms"), list):
        return "its next step is a procedure, not an experiment (%s)" % (exp.get("kind") or "none")
    if exp.get("kind") not in EXECUTABLE_KINDS:
        return "experiment kind %s is not run unattended" % exp.get("kind")
    if exp.get("kind") == "request_shape" and not requests_workload:
        return "needs a workload of your captured requests (hotpath workload OUT --from-requests FILE)"
    if (finding.get("measured") or {}).get("status") == "measured":
        return "already measured"
    if exp.get("kind") == "server_overhead_flags":
        # these flags trade prompt-cache reuse for per-request overhead; a keep is adopted only when real traffic shows little
        # reuse, so without that evidence the run could not change the recommendation
        cp = (finding.get("measurements") or {}).get("cache_pattern_evidence")
        if cp is not None and (not cp.get("available") or cp.get("thin")):
            return ("trades prompt-cache reuse for overhead, and no real-traffic evidence shows how much reuse would be lost, so a result could not be "
                    "adopted: start from your server with its log (`--log-file`) or /metrics after real traffic")
        if cp is not None and cp.get("level") not in (None, "low"):
            return "real traffic reuses enough of its prompts that removing the cache would cost more than it saves (%s)" % cp.get("level")
    return None


def next_finding(diag, attempted, requests_workload):
    skipped = []
    for f in diag.get("findings") or []:
        if f["id"] in attempted:
            continue
        why = executable(f, requests_workload)
        if why is None:
            return f, skipped
        if f.get("severity") in ("high", "medium"):
            skipped.append({"finding": f["id"], "title": f.get("title"), "why_not_run": why})
    return None, skipped


def adoptable(row, verdict, arm=None):
    """(adopt?, reason). Only `keep` is adopted; a production gate that does not support applying the flag blocks it, and
    so does a change that limits what the server accepts (a smaller context), which is the user's call."""
    if not row["verdict"].startswith("keep"):
        return False, "not adopted: the verdict is not keep (%s)" % row["verdict"].split(":")[0]
    if (arm or {}).get("capacity_change"):
        return False, "not adopted: faster with identical outputs, but it changes what the server accepts (%s); your decision" % arm["capacity_change"]
    gate = verdict.get("production_gate")
    if gate and not gate.startswith("production applicability supported"):
        return False, "not adopted: faster on isolated requests, but %s" % gate
    return True, "adopted: keep verdict"


def _rank_rows(verdict):
    """Candidate rows, keep verdicts first, fastest first."""
    rows = list(verdict["arms"])
    rows.sort(key=lambda r: (not r["verdict"].startswith("keep"), (r["metrics"].get("wall_ms") or {}).get("ratio", 1.0)))
    return rows


def _metric_line(row):
    out = {}
    for m in ("wall_ms", "prefill_tps", "decode_tps"):
        x = row["metrics"].get(m) or {}
        if x.get("baseline_median") is not None:
            out[m] = {"baseline": x.get("baseline_median"), "candidate": x.get("candidate_median"), "ratio": x.get("ratio"), "call": x.get("call")}
    if row.get("vram_delta_mib") is not None:
        out["vram_delta_mib"] = row["vram_delta_mib"]
    out["outputs"] = row.get("outputs")
    return out


def _measured_load_s(records):
    """Median model load time of the newest experiment record hotpath launched, for duration estimates."""
    for rec in reversed(records):
        loads = [x.get("load_s") for a in (rec.get("arms") or {}).values() for x in a.get("launch_facts") or [] if x.get("load_s")]
        if loads:
            loads.sort()
            return loads[len(loads) // 2]
    return None


def _is_requests_workload(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("kind") == "requests"
    except (OSError, ValueError, AttributeError):
        return False


class Budget:
    def __init__(self, max_experiments, max_minutes):
        self.max_experiments, self.max_minutes = int(max_experiments), float(max_minutes)
        self.experiments, self.minutes = 0, 0.0

    def time_left(self):
        return self.max_minutes - self.minutes


def improve(cfg, start_records, workload, out_dir, runner, max_experiments=3, max_minutes=60, rounds=3, repetitions=2, warmup=1, port=8099,
            n_predict=None, dry_run=False, log=print, initial_args=None, bandwidth_gbs=None, session=None):
    """Run the loop. `runner(spec, spec_path)` runs one experiment spec and returns its record (raising SystemExit when
    it cannot start, for example on unmet preconditions). Returns the improvement report (also written to out_dir)."""
    session = session or "improve-%s" % time.strftime("%Y%m%d-%H%M%S")
    sdir = Path(out_dir) / session
    sdir.mkdir(parents=True, exist_ok=True)
    records = list(start_records)
    requests_wl = _is_requests_workload(workload)
    baseline = None
    if not records:
        # nothing running and no log: measure the starting configuration on hotpath's own server first
        args, source = (list(initial_args), "command line") if initial_args else baseline_args_without_server(cfg)
        bspec = baseline_spec(cfg, args, workload, port, repetitions, warmup, n_predict, "%s-baseline" % session)
        bpath = sdir / "baseline.json"
        bpath.write_text(json.dumps(bspec, indent=2) + "\n", encoding="utf-8")
        baseline = {"spec": str(bpath), "args": args, "source": source}
        if dry_run:
            return _early_report(session, out_dir, args, source, workload, max_experiments, max_minutes, "dry run: nothing launched",
                                 next_step={"finding": "baseline", "title": "no running server or log: measure your configuration first, on hotpath's own server",
                                            "spec": str(bpath), "estimated_minutes": None, "arms": ["baseline"], "variable": "none",
                                            "min_free_vram_mib": bspec["preconditions"].get("min_free_vram_mib")})
        log("baseline: no running server or log; measuring %s on hotpath's own server" % " ".join(args))
        t0 = time.time()
        try:
            brec = runner(bspec, str(bpath))
        except SystemExit as e:
            return _early_report(session, out_dir, args, source, workload, max_experiments, max_minutes, "baseline not run: %s" % e)
        baseline["minutes"] = round((time.time() - t0) / 60.0, 1)
        baseline["record"] = write_record(brec, sdir, "experiment-%s" % bspec["name"]).name
        if any(a.get("failures") for a in brec["arms"].values()):
            baseline["failed"] = [f["error"] for a in brec["arms"].values() for f in a.get("failures") or []][:3]
        records = [brec]
        initial_args = args
    diag = dg.diagnose(records, bandwidth_gbs=bandwidth_gbs, cfg=cfg)
    args0, source = prop.baseline_args(diag, cfg, initial_args)
    if baseline:
        source = "%s, measured as the baseline on hotpath's own server" % baseline["source"]
    args0, removed = prop.strip_runner_owned(prop._strip_launch_prefix(args0))
    observed = (diag.get("observed_launch") or {}).get("args_redacted")
    original = {"args": list(args0), "env": {}}
    best = {"args": list(args0), "env": {}, "label": None}
    cum = {"args": {}, "args_remove": set(), "env": {}, "env_remove": set()}
    steps, attempted, adopted_steps = [], set(), []
    budget = Budget(max_experiments, max_minutes)
    if baseline:
        budget.minutes += baseline.get("minutes") or 0.0
        best["label"] = "%s/baseline" % records[0]["name"]
    report = {"session": session, "starting_point": {"args": args0, "source": source, "removed_server_args": sorted(set(removed)),
                                                     "diagnosed_arm": diag.get("diagnosed_arm"), "sources": diag.get("sources")},
              "workload": Path(workload).name, "budget": {"max_experiments": budget.max_experiments, "max_minutes": budget.max_minutes},
              "steps": steps, "skipped_findings": [], "confirmation": None, "next_step": None, "baseline": baseline, "decisions": []}
    stopped = None
    if baseline and baseline.get("failed"):
        stopped = "the baseline launch failed: %s (see the server log in %s)" % ("; ".join(baseline["failed"]), sdir.name)
    while stopped is None:
        if budget.experiments >= budget.max_experiments:
            stopped = "experiment budget reached"
            break
        f, skipped = next_finding(diag, attempted, requests_wl)
        for s in skipped:
            if s["finding"] not in {x["finding"] for x in report["skipped_findings"]}:
                report["skipped_findings"].append(s)
        if f is None:
            stopped = "no further executable finding"
            break
        n = len(steps) + 1
        name = "%s-s%d-%s" % (session, n, f["experiment"]["kind"])
        try:
            spec = step_spec(f, cfg, best, workload, port, rounds, repetitions, warmup, n_predict, name)
        except ValueError as e:
            attempted.add(f["id"])
            report["skipped_findings"].append({"finding": f["id"], "title": f.get("title"), "why_not_run": str(e)})
            continue
        spec["provenance"].update({"improve_session": session, "step": n})
        est = prop.estimate_minutes(spec, diag.get("rates"), load_s=_measured_load_s(records))
        if est:
            spec["provenance"]["estimated_duration"] = est
        need = est["minutes"] if est else None
        if need is not None and need > budget.time_left():
            stopped = "time budget: the next step (%s) needs about %.0f min and %.0f min are left" % (f["id"], need, max(0.0, budget.time_left()))
            break
        spec_path = sdir / ("step-%d-%s.json" % (n, f["experiment"]["kind"]))
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        if dry_run:
            report["next_step"] = {"finding": f["id"], "title": f.get("title"), "spec": str(spec_path), "estimated_minutes": need,
                                   "arms": [a["name"] for a in spec["arms"]], "variable": f["experiment"].get("variable"),
                                   "min_free_vram_mib": spec["preconditions"].get("min_free_vram_mib")}
            stopped = "dry run: nothing launched"
            break
        log("step %d: %s -> %s (%s)" % (n, f["id"], ", ".join(a["name"] for a in spec["arms"]), "about %.0f min" % need if need else "duration unknown"))
        t0 = time.time()
        try:
            rec = runner(spec, str(spec_path))
        except SystemExit as e:
            steps.append({"step": n, "finding": f["id"], "title": f.get("title"), "experiment_kind": f["experiment"]["kind"], "experiment": name,
                          "adopted": False, "verdict": "not run", "why": str(e)})
            stopped = str(e)
            break
        budget.experiments += 1
        budget.minutes += max((time.time() - t0) / 60.0, 0.0)
        attempted.add(f["id"])
        rpath = write_record(rec, sdir, "experiment-%s" % name)
        records.append(rec)
        diag = dg.diagnose(records, bandwidth_gbs=bandwidth_gbs, cfg=cfg)
        verdict = next((v for v in diag.get("experiment_verdicts") or [] if v["experiment"] == rec["name"]), None)
        step = {"step": n, "finding": f["id"], "title": f.get("title"), "experiment_kind": f["experiment"]["kind"], "experiment": name,
                "record": rpath.name, "arms": [a["name"] for a in spec["arms"]], "estimated_minutes": need}
        rows = _rank_rows(verdict) if verdict else []
        arm_of = {a["name"]: a for a in spec["arms"]}
        row, ok, why = None, False, None
        for r in rows:
            r_ok, r_why = adoptable(r, verdict, arm_of.get(r["arm"]))
            if r_ok:
                row, ok, why = r, True, r_why
                break
        if not row and rows:
            row = rows[0]
            ok, why = adoptable(row, verdict, arm_of.get(row["arm"]))
        # faster candidates that only the user can accept: a capacity change, or outputs that differ
        for r in rows:
            if r is row and ok:
                continue
            a = arm_of.get(r["arm"]) or {}
            if (r["verdict"].startswith("keep") and a.get("capacity_change")) or r["verdict"].startswith("faster"):
                report["decisions"].append({"step": n, "arm": r["arm"], "change": _change_text(a), "verdict": r["verdict"],
                                            "needs": a.get("capacity_change") or "the outputs differ: check whether the difference matters for your application",
                                            "measured": _metric_line(r)})
        if not row:
            step.update({"adopted": False, "verdict": "no verdict (the record has fewer than two arms with requests)", "why": "nothing to compare"})
        else:
            step.update({"adopted": ok, "candidate": row["arm"], "verdict": row["verdict"], "why": why, "measured": _metric_line(row),
                         "caveats": verdict.get("caveats") or []})
            if ok:
                arm = arm_of[row["arm"]]
                best["args"], best["env"] = materialize(spec["common_args"], spec.get("env"), arm)
                best["label"] = "%s/%s" % (rec["name"], row["arm"])
                for k, v in (arm.get("args_override") or {}).items():
                    k = _canon(k, cum["args"], cum["args_remove"])
                    cum["args"][k] = v
                    cum["args_remove"].discard(k)
                for k in arm.get("args_remove") or []:
                    k = _canon(k, cum["args"], cum["args_remove"])
                    cum["args"].pop(k, None)
                    cum["args_remove"].add(k)
                for k, v in (arm.get("env_override") or {}).items():
                    cum["env"][k] = str(v)
                    cum["env_remove"].discard(k)
                for k in arm.get("env_remove") or []:
                    cum["env"].pop(k, None)
                    cum["env_remove"].add(k)
                step["change"] = {"args_override": arm.get("args_override") or {}, "args_remove": arm.get("args_remove") or [],
                                  "env_override": arm.get("env_override") or {}, "env_remove": arm.get("env_remove") or []}
                adopted_steps.append(step)
        steps.append(step)
        log("  %s: %s" % ("adopted" if step["adopted"] else "not adopted", step["verdict"]))
        if best["label"] is None or not step["adopted"]:
            # the configuration adopted so far was re-measured as this experiment's first arm: describe that, freshly
            if not best["label"] or best["label"].split("/")[0] != rec["name"]:
                best["label"] = "%s/%s" % (rec["name"], spec["arms"][0]["name"])
        diag = dg.diagnose(records, bandwidth_gbs=bandwidth_gbs, cfg=cfg, focus=best["label"])
    report["stopped_because"] = stopped
    report["experiments_run"] = budget.experiments
    report["result"] = _result(cfg, original, cum, adopted_steps, records, workload, port, rounds, repetitions, warmup, n_predict, budget, sdir, session,
                               runner, dry_run, bandwidth_gbs, log, report)
    report["minutes_used"] = round(budget.minutes, 1)
    report["recommended"] = recommended_launch(observed or args0, cum, bool(observed))
    report["remaining"] = [{"id": f["id"], "title": f.get("title"), "confidence": f.get("confidence"), "next": (f.get("measured") or {}).get("how")}
                           for f in diag.get("findings") or [] if f.get("severity") in ("high", "medium") and f["id"] not in attempted][:5]
    rec = new_record("improvement", "controlled" if budget.experiments else "hypothesis", **report)
    path = write_record(rec, out_dir, "improvement")
    Path(str(path)[:-5] + ".md").write_text(improvement_markdown(rec), encoding="utf-8")
    rec["path"] = str(path)
    return rec


def _early_report(session, out_dir, args, source, workload, max_experiments, max_minutes, stopped, next_step=None):
    """The report when the loop ends before any diagnosis: a dry run of the baseline, or a baseline that could not start."""
    rec = new_record("improvement", "hypothesis", session=session, starting_point={"args": args, "source": source}, workload=Path(workload).name,
                     budget={"max_experiments": max_experiments, "max_minutes": max_minutes}, steps=[], stopped_because=stopped, next_step=next_step,
                     result={"verdict": "dry run: nothing measured" if next_step else "nothing measured", "measured_by": None},
                     recommended=recommended_launch(args, {"args": {}, "args_remove": set(), "env": {}, "env_remove": set()}, False),
                     skipped_findings=[], remaining=[], experiments_run=0, minutes_used=0.0)
    path = write_record(rec, out_dir, "improvement")
    Path(str(path)[:-5] + ".md").write_text(improvement_markdown(rec), encoding="utf-8")
    rec["path"] = str(path)
    return rec


def baseline_args_without_server(cfg):
    """Arguments for a starting configuration when no server runs: server_args from the config, else llama-server's
    usual flags with every layer on the GPU when the model's weights fit in VRAM (from the GGUF header), else the
    number of layers that fit by the same estimate."""
    if cfg.get("server_args"):
        return list(cfg["server_args"]), "hotpath.local.json server_args"
    args = list(prop.DEFAULT_COMMON_ARGS)
    fit = layers_that_fit(cfg)
    if fit is None:
        return args, "hotpath defaults (all layers on the GPU; the model or the GPU's memory is not known)"
    n, total, basis = fit
    if n >= total:
        return args, "hotpath defaults (%s)" % basis
    return launch.set_arg(args, "--gpu-layers", n), "hotpath defaults with %d of %d layers on the GPU (%s)" % (n, total, basis)


def layers_that_fit(cfg, ctx=4096):
    """(layers on the GPU, total layers, basis) from the GGUF header and nvidia-smi: weights + an f16 KV cache at `ctx`
    + a 512 MiB compute allowance against 95% of the free VRAM. An estimate for a starting point, never a verdict."""
    from hotpath import gguf
    from hotpath import gpu as gpu_mod
    mp = cfg.get("model_path")
    if not mp or not Path(mp).is_file():
        return None
    try:
        s = gguf.summarize(gguf.read_header(mp))
        d = gpu_mod.gpu_dynamic()
    except Exception:  # noqa: BLE001 - an estimate only
        return None
    total_mib, used = d.get("mem_total_mib"), d.get("mem_used_mib")
    if not total_mib or not s.get("n_layer") or not s.get("total_bytes"):
        return None
    free = total_mib - (used or 0)
    w = s["total_bytes"] / (1 << 20)
    kv = (s.get("kv_bytes_per_token_f16") or 0) * ctx / (1 << 20)
    total_layers = s["n_layer"] + 1
    per_layer = w / total_layers
    n = int(max(0.0, free * 0.95 - kv - 512) // per_layer) if per_layer else total_layers
    return min(n, total_layers), total_layers, "estimate: %.0f MiB weights + %.0f MiB KV cache at -c %d + 512 MiB compute against %.0f MiB free" % (w, kv, ctx, free)


def baseline_spec(cfg, args, workload, port, repetitions, warmup, n_predict, name):
    spec = {"name": name, "description": "Baseline by hotpath improve: your starting configuration on hotpath's own server, measured once so the loop "
                                         "has evidence to diagnose. Not a comparison.",
            "binary": "${runtime_binary}", "model": "${model_path}", "port": int(port), "common_args": list(args), "env": {},
            "arms": [{"name": "baseline"}], "workload": workload, "rounds": 1, "repetitions": int(repetitions), "warmup": int(warmup),
            "cooldown_s": 5, "health_timeout_s": 600,
            "preconditions": {"sample_seconds": 5, "max_gpu_util_pct": 10, "min_free_vram_mib": 2000, "ports_must_be_free": [int(port)]},
            "provenance": {"experiment_kind": "baseline"}}
    if cfg.get("path_prepend"):
        spec["path_prepend"] = "${path_prepend}"
    if n_predict:
        spec["n_predict"] = int(n_predict)
    need = prop.vram_needed_mib(cfg, args)
    if need is not None:
        spec["preconditions"]["min_free_vram_mib"] = need
    return spec


def step_spec(finding, cfg, best, workload, port, rounds, repetitions, warmup, n_predict, name):
    """A one-variable spec whose first arm is the configuration adopted so far (arguments and environment)."""
    spec = prop.build_spec(finding, cfg, best["args"], workload, port=port, rounds=rounds, repetitions=repetitions, warmup=warmup,
                           n_predict=n_predict, name=name)
    spec["env"] = dict(best["env"])
    first_args, first_env = materialize(spec["common_args"], spec["env"], spec["arms"][0])
    if not _same_config(first_args, first_env, best["args"], best["env"]):
        cur = "current" if all(a["name"] != "current" for a in spec["arms"]) else "adopted-so-far"
        spec["arms"].insert(0, {"name": cur})
    need = [prop.vram_needed_mib(cfg, materialize(spec["common_args"], spec["env"], a)[0]) for a in spec["arms"]]
    if need and all(x is not None for x in need):
        spec["preconditions"]["min_free_vram_mib"] = max(need)
        spec["provenance"]["min_free_vram_basis"] = "weights on the GPU + 10%, KV cache at the arm's context and KV types, 512 MiB compute, for the largest arm (estimate from the GGUF header)"
    return spec


def _result(cfg, original, cum, adopted_steps, records, workload, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner,
            dry_run, bandwidth_gbs, log, report):
    if dry_run:
        return {"verdict": "dry run: nothing measured", "measured_by": None}
    if not budget.experiments:
        return {"verdict": "nothing measured: %s" % (report.get("stopped_because") or "no experiment ran"), "measured_by": None}
    if not adopted_steps:
        return {"verdict": "no change adopted: the configuration you started from stands", "measured_by": None}
    if len(adopted_steps) == 1:
        s = adopted_steps[0]
        return {"verdict": s["verdict"], "measured_by": s["experiment"], "measured": s.get("measured"),
                "basis": "one adopted change: its own paired run already compared your configuration with the change"}
    # two or more adopted changes: one paired run of the original launch against the improved one decides the cumulative claim
    improved = {"name": "improved"}
    if cum["args"]:
        improved["args_override"] = dict(cum["args"])
    if cum["args_remove"]:
        improved["args_remove"] = sorted(cum["args_remove"])
    if cum["env"]:
        improved["env_override"] = dict(cum["env"])
    if cum["env_remove"]:
        improved["env_remove"] = sorted(cum["env_remove"])
    name = "%s-confirm" % session
    spec = {"name": name, "description": "Confirmation by hotpath improve: your original launch against the launch with every adopted change (%s). "
                                         "Several variables differ by design; the one-variable evidence for each change is in its own step." % (
                                             ", ".join(s["finding"] for s in adopted_steps)),
            "binary": "${runtime_binary}", "model": "${model_path}", "port": int(port), "common_args": list(original["args"]), "env": dict(original["env"]),
            "arms": [{"name": "original"}, improved], "workload": workload, "rounds": int(rounds), "repetitions": int(repetitions), "warmup": int(warmup),
            "cooldown_s": 5, "health_timeout_s": 600,
            "preconditions": {"sample_seconds": 5, "max_gpu_util_pct": 10, "min_free_vram_mib": 2000, "ports_must_be_free": [int(port)]},
            "provenance": {"experiment_kind": "confirmation", "improve_session": session, "adopted_steps": [s["experiment"] for s in adopted_steps]}}
    if cfg.get("path_prepend"):
        spec["path_prepend"] = "${path_prepend}"
    if n_predict:
        spec["n_predict"] = int(n_predict)
    need = [prop.vram_needed_mib(cfg, materialize(spec["common_args"], spec["env"], a)[0]) for a in spec["arms"]]
    if need and all(x is not None for x in need):
        spec["preconditions"]["min_free_vram_mib"] = max(need)
    # the original launch's measured rates (the baseline arm of the first step) bound the run: every arm is assumed that slow
    first = (adopted_steps[0].get("measured") or {}) if adopted_steps else {}
    rates = {"prefill_tps": (first.get("prefill_tps") or {}).get("baseline"), "decode_tps": (first.get("decode_tps") or {}).get("baseline"),
             "basis": "the original launch in step %s" % adopted_steps[0]["step"]}
    est = prop.estimate_minutes(spec, rates, load_s=_measured_load_s(records))
    if est:
        spec["provenance"]["estimated_duration"] = est
        if est["minutes"] > budget.time_left():
            return {"verdict": "not confirmed: the joint original-vs-improved run needs about %.0f min and %.0f min of the budget are left; each change has only its own step's verdict" % (
                est["minutes"], max(0.0, budget.time_left())), "measured_by": None, "confirmation_spec": None}
    path = sdir / "confirm.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    log("confirming: original vs improved (%s)" % ", ".join(s["finding"] for s in adopted_steps))
    t0 = time.time()
    try:
        rec = runner(spec, str(path))
    except SystemExit as e:
        return {"verdict": "not confirmed: %s" % e, "measured_by": None, "confirmation_spec": str(path)}
    finally:
        budget.minutes += max((time.time() - t0) / 60.0, 0.0)
    rpath = write_record(rec, sdir, "experiment-%s" % name)
    v = dg.experiment_verdict(rec)
    row = (v or {}).get("arms", [None])[0] if v else None
    report["confirmation"] = {"experiment": name, "record": rpath.name, "verdict": row["verdict"] if row else "no verdict", "measured": _metric_line(row) if row else None}
    if not row:
        return {"verdict": "not confirmed: the confirmation run produced no comparison", "measured_by": name}
    return {"verdict": row["verdict"], "measured_by": name, "measured": _metric_line(row),
            "basis": "one paired run of your original launch against the launch with every adopted change"}


def recommended_launch(user_args, cum, observed):
    """The user's own launch line with the adopted changes, plus the environment changes. Flags the user set that the
    loop did not change (port, host, log file, model) stay as they were."""
    args = [str(a) for a in user_args]
    changes = []
    for flag, value in cum["args"].items():
        name = launch._SPELLING.get(flag)
        before = launch.get(args, name) if name else None
        args = launch.set_arg(args, flag, value)
        changes.append({"flag": flag, "from": before if before is not None else "not set (build default)", "to": str(value)})
    for flag in sorted(cum["args_remove"]):
        name = launch._SPELLING.get(flag)
        before = launch.get(args, name) if name else None
        args = launch.remove_arg(args, flag)
        changes.append({"flag": flag, "from": before, "to": "removed"})
    env = [{"variable": k, "to": v} for k, v in sorted(cum["env"].items())] + [{"variable": k, "to": "unset"} for k in sorted(cum["env_remove"])]
    return {"args": args, "changes": changes, "env": env,
            "basis": "your server's observed command line (paths shown as file names)" if observed else "the arguments the loop started from",
            "note": "hotpath did not change your server. To apply, add these changes to the command (or launcher settings) that starts it, "
                    "restart it yourself, and run `hotpath inspect` afterwards to confirm the new launch."}


def improvement_markdown(rec):
    L = ["# hotpath improve: %s" % rec["session"], ""]
    sp = rec["starting_point"]
    L += ["Started from: %s (%s)." % (" ".join(sp["args"]) or "(no arguments)", sp["source"]),
          "Workload: %s; budget %d experiments / %.0f min; used %d experiments%s / %.1f min in all." % (
              rec["workload"], rec["budget"]["max_experiments"], rec["budget"]["max_minutes"], rec.get("experiments_run", 0),
              " + baseline" if rec.get("baseline") else "", rec.get("minutes_used", 0.0)) + (" The joint confirmation run is not counted as an experiment." if rec.get("confirmation") else ""),
          "Stopped: %s." % rec.get("stopped_because"), ""]
    res = rec.get("result") or {}
    L += ["## Result", "", res.get("verdict") or "", ""]
    if res.get("basis"):
        L += ["Basis: %s." % res["basis"], ""]
    m = res.get("measured") or {}
    for k, lab in (("wall_ms", "wall time per request"), ("prefill_tps", "prompt processing"), ("decode_tps", "generation")):
        if m.get(k):
            x = m[k]
            L.append("- %s: %s -> %s (ratio %s, %s)" % (lab, _fmt(x["baseline"], k), _fmt(x["candidate"], k), x.get("ratio"), x.get("call")))
    if m.get("vram_delta_mib") is not None:
        L.append("- VRAM: %+d MiB" % m["vram_delta_mib"])
    if m.get("outputs"):
        o = m["outputs"]
        L.append("- outputs identical on %s/%s prompts (%s)" % (o.get("identical"), o.get("prompts_compared"), o.get("basis")))
    L += ["", "## Steps", ""]
    if not rec["steps"]:
        L.append("- none run")
    for s in rec["steps"]:
        L.append("%d. **%s** (%s): %s" % (s["step"], s["finding"], s["experiment_kind"], s.get("title") or ""))
        L.append("   - verdict: %s" % s["verdict"])
        L.append("   - %s" % s["why"])
        if s.get("change"):
            L.append("   - change: %s" % _change_text(s["change"]))
        for c in s.get("caveats") or []:
            L.append("   - caveat: %s" % c)
    if rec.get("next_step"):
        ns = rec["next_step"]
        L += ["", "## Next step (dry run)", "", "- %s: %s" % (ns["finding"], ns.get("title")),
              "- arms %s; variable %s; free VRAM needed %s MiB; about %s min" % (", ".join(ns["arms"]), ns.get("variable"), ns.get("min_free_vram_mib"),
                                                                                "%.0f" % ns["estimated_minutes"] if ns.get("estimated_minutes") else "?"),
              "- spec: %s" % Path(ns["spec"]).name]
    if rec.get("decisions"):
        L += ["", "## Needs your decision", "", "Measured faster, but not adopted because only you can accept the trade-off:", ""]
        for d in rec["decisions"]:
            w = (d.get("measured") or {}).get("wall_ms") or {}
            o = (d.get("measured") or {}).get("outputs") or {}
            same = ("outputs identical on %s/%s prompts" % (o.get("identical"), o.get("prompts_compared"))) if o.get("prompts_compared") else "outputs not compared"
            L.append("- step %d, %s (%s): wall %s -> %s, %s; %s" % (d["step"], d["arm"], d["change"], _fmt(w.get("baseline"), "wall_ms"),
                                                                      _fmt(w.get("candidate"), "wall_ms"), same, d["needs"]))
    r = rec["recommended"]
    L += ["", "## Recommended launch", ""]
    if r["changes"] or r["env"]:
        for c in r["changes"]:
            L.append("- %s: %s -> %s" % (c["flag"], c["from"], c["to"]))
        for e in r["env"]:
            L.append("- environment %s: %s" % (e["variable"], e["to"]))
        L += ["", "```", " ".join(r["args"]), "```", "", "(%s)" % r["basis"]]
    else:
        L.append("- no change")
    L += ["", r["note"]]
    if rec.get("skipped_findings"):
        L += ["", "## Findings not run", ""] + ["- %s: %s" % (s["finding"], s["why_not_run"]) for s in rec["skipped_findings"]]
    if rec.get("remaining"):
        L += ["", "## Still open", ""] + ["- %s (%s): %s" % (f["id"], f.get("confidence"), f.get("title")) for f in rec["remaining"]]
    L += ["", "A verdict describes this workload on this machine. Output identity is checked on greedy tokens; it is not a check of what your application "
              "decides from an output."]
    return "\n".join(L) + "\n"


def _fmt(v, k):
    if v is None:
        return "?"
    return ("%.0f ms" % v) if k == "wall_ms" else ("%.1f t/s" % v)


def _change_text(c):
    parts = ["%s %s" % (k, v) for k, v in c.get("args_override", {}).items()] + ["remove %s" % k for k in c.get("args_remove", [])]
    parts += ["%s=%s" % (k, v) for k, v in c.get("env_override", {}).items()] + ["unset %s" % k for k in c.get("env_remove", [])]
    return ", ".join(parts) or "none"
