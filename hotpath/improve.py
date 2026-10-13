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
from hotpath import locks
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


SPILL_MIB = 256              # more system-memory (shared) use than the baseline: the candidate does not really fit in VRAM
HEADROOM_MIB = 300           # free VRAM left while the candidate ran, below which a busier desktop could push it over
PROMPT_REGRESSION = 0.03     # per-prompt wall-time regression allowed beyond twice the prompt's own repetition spread


OUTPUT_POLICIES = ("identical", "schema_valid", "decisions_equal")


def _policy_check(row, cfg):
    """An operator's output policy (hotpath.local.json `output_policy`), checked on this row; the default "identical"
    accepts no difference. "schema_valid": differing outputs are accepted when every judged request of the candidate passed
    full JSON Schema validation and, if the operator configured `decision_fields`, every one of them came out equal.
    "decisions_equal": the same, and `decision_fields` must be configured (an operator's statement of what the application
    decides on); without it the policy never passes. Neither policy says the outputs are equivalent, only that they met
    the operator's stated criteria."""
    policy = (cfg or {}).get("output_policy")
    if policy not in ("schema_valid", "decisions_equal"):
        return policy, None
    v = row.get("validity") or {}
    c = v.get("candidate") or {}
    ok = bool(v and not v.get("dropped") and v.get("decided_on") == "schema_valid" and c.get("judged") and c.get("schema_valid") == c.get("judged"))
    basis = ["full JSON Schema validation %s/%s in the candidate" % (c.get("schema_valid"), c.get("judged")) if v else "no schema in these requests"]
    fa = row.get("fields") or {}
    configured = bool((cfg or {}).get("decision_fields"))
    if configured:
        dec_ok = bool(fa.get("decisions_judged")) and fa.get("decisions_equal") == fa.get("decisions_judged")
        basis.append("decision fields %s equal on %s/%s requests" % (", ".join(fa.get("decision_fields") or []) or "(none found in the outputs)",
                                                                     fa.get("decisions_equal"), fa.get("decisions_judged")))
        ok = ok and dec_ok
    elif policy == "decisions_equal":
        ok = False
        basis.append("decision_fields is not configured, so no decision could be checked")
    return policy, {"passed": ok, "basis": "; ".join(basis)}


def resource_check(rec_arms, base_name, cand_name):
    """(problem, needs_decision) from the per-launch counters: more system-memory spill than the baseline is a rejection,
    too little free VRAM while running is the user's call."""
    def facts(name):
        return (rec_arms or {}).get(name, {}).get("launch_facts") or []
    sb = [((x.get("process_counters_after") or {}).get("shared_mib")) for x in facts(base_name)]
    sc = [((x.get("process_counters_after") or {}).get("shared_mib")) for x in facts(cand_name)]
    sb, sc = [x for x in sb if x is not None], [x for x in sc if x is not None]
    if sb and sc and max(sc) - max(sb) > SPILL_MIB:
        return "it spills %.0f MiB more into system memory than the baseline (shared GPU memory): it does not fit in VRAM" % (max(sc) - max(sb)), False
    free = [(g.get("mem_total_mib") or 0) - (g.get("mem_used_mib") or 0) for g in (x.get("gpu_after") or {} for x in facts(cand_name)) if g.get("mem_total_mib")]
    if free and min(free) < HEADROOM_MIB:
        return "only %.0f MiB of VRAM stayed free while it ran (no headroom: other programs could push it into system memory)" % min(free), True
    return None, False


def prompt_regression(row):
    """The prompts on which the candidate's wall time is worse than the baseline's beyond that prompt's own noise."""
    out = []
    for pp in ((row.get("metrics") or {}).get("wall_ms") or {}).get("per_prompt") or []:
        tol = max(PROMPT_REGRESSION, 2 * (pp.get("baseline_relative_spread") or 0.0))
        if pp.get("ratio") and pp["ratio"] > 1.0 + tol:
            out.append("%s (%+.0f%%)" % (pp["prompt_id"], 100 * (pp["ratio"] - 1)))
    return out


def class_regression(row):
    """Request classes whose median wall time got worse beyond the class's own repetition noise: an average gain must not
    be bought with a slower application step."""
    out = []
    for c, m in sorted((row.get("classes") or {}).items()):
        w = m.get("wall_ms") or {}
        if w.get("ratio") is None:
            continue
        tol = max(PROMPT_REGRESSION, 2 * (w.get("noise") or 0.0))
        if w["ratio"] > 1.0 + tol:
            out.append("%s (%+.0f%% over %d requests)" % (c, 100 * (w["ratio"] - 1), m.get("requests") or 0))
    return out


def adoptable(row, verdict, arm=None, rec_arms=None, base_name=None, cfg=None):
    """(adopt?, reason, needs_decision). Adopted only when faster and consistent, with outputs identical (or accepted by an
    operator's checked output policy), no prompt regressing beyond its noise, no spill into system memory, VRAM headroom
    left, no change to what the server accepts (unless an operator's context policy set it) and, for cache flags, a
    production gate that supports it."""
    faster_differ = row["verdict"].startswith("faster")
    policy, check = _policy_check(row, cfg)
    if not row["verdict"].startswith("keep") and not (faster_differ and check and check.get("passed")):
        return False, "not adopted: the verdict is not keep (%s)" % row["verdict"].split(":")[0], faster_differ
    creg = class_regression(row)
    if creg:
        return False, "not adopted: faster overall but the request class %s got slower beyond its own noise" % "; ".join(creg), False
    reg = prompt_regression(row)
    if reg:
        return False, "not adopted: faster overall but it regresses on %s beyond those prompts' own noise" % ", ".join(reg), False
    problem, decide = resource_check(rec_arms, base_name, row.get("arm"))
    if problem:
        return False, "not adopted: %s" % problem, decide
    if (arm or {}).get("request_override"):
        return False, ("not adopted: faster, but it changes what your application sends (%s), which is a change to your application, not to "
                       "the server launch; your decision" % _request_change_text(arm["request_override"])), True
    if (arm or {}).get("capacity_change"):
        return False, "not adopted: faster with identical outputs, but it changes what the server accepts (%s); your decision" % arm["capacity_change"], True
    gate = verdict.get("production_gate")
    if gate and not gate.startswith("production applicability supported"):
        return False, "not adopted: faster on isolated requests, but %s" % gate, False
    if faster_differ:
        return True, "adopted under the output policy %r: %s" % (policy, check["basis"]), False
    return True, "adopted: keep verdict", False


def _request_change_text(ov):
    parts = ["set %s" % k for k in (ov.get("set") or {})] + ["remove %s" % k for k in ov.get("remove") or []]
    return ", ".join(parts) or "none"


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


STEP_POSITIONS = {1: [0.5], 2: [0.5, 1.0], 3: [0.0, 0.5, 1.0]}


def _spread(items, n, avoid=(), positions=None):
    """n items from a size-sorted list at evenly spaced positions (the largest always included), preferring items not in
    `avoid` near each position, so a validation set is mostly requests the steps did not see."""
    if n >= len(items):
        return list(items)
    positions = positions or ([0.5] if n == 1 else [i / (n - 1) for i in range(n)])
    picks = []
    for q in positions:
        idx = int(round(q * (len(items) - 1)))
        order = sorted(range(len(items)), key=lambda j: (abs(j - idx), j))
        if q >= 1.0:
            pick = items[-1] if items[-1] not in picks else None              # the largest request, seen or not
        else:
            pick = next((items[j] for j in order if items[j]["id"] not in avoid and items[j] not in picks), None) or \
                next((items[j] for j in order if items[j] not in picks), None)
        if pick is not None:
            picks.append(pick)
    return picks


def requests_plan(workload, sdir, cfg, cache_mode=None, step_per_class=2, validation_per_class=4, max_steps=8, max_validation=16):
    """For a workload of the user's own requests: the subset the steps run (per request class the median and the largest
    request) and a broader validation subset (per class, from the smallest to the largest, mostly requests the steps did
    not see), both kept in the session as lists of item ids (the request bodies stay in the workload file). Kept for the
    whole session, so a resumed run measures the same requests."""
    from hotpath import workload as wl
    sdir = Path(sdir)
    sp, vp = sdir / "steps-requests.json", sdir / "validation-requests.json"
    w = wl.load_workload(workload)
    if not (sp.is_file() and vp.is_file()):
        order = {it["id"]: i for i, it in enumerate(w["items"])}
        groups = {}
        for it in w["items"]:
            groups.setdefault(wl.request_class(it), []).append(it)
        if len(groups) > max_steps and any(not it.get("class_labelled") for it in w["items"]):
            # derived classes this fine-grained (a system message that varies per request, say) are not request classes:
            # regroup by endpoint and schema only
            coarse = {}
            for it in w["items"]:
                coarse.setdefault("/".join(wl.request_class(it).split("/")[:2]), []).append(it)
            groups = coarse
        if len(groups) > max_steps:
            # still more classes than the step cap: the most frequent classes, the rest reported as not covered
            keep = sorted(groups, key=lambda c: (-len(groups[c]), c))[:max_steps]
            groups = {c: groups[c] for c in keep}
        for c in groups:
            groups[c].sort(key=lambda it: (wl.request_size(it), order[it["id"]]))
        n_cls = len(groups)
        per_step = max(1, min(int(step_per_class), max_steps // n_cls or 1))
        per_val = max(1, min(int(validation_per_class), max_validation // n_cls or 1))
        steps, val = [], []
        for c, items in groups.items():
            st = _spread(items, per_step, positions=STEP_POSITIONS.get(per_step))
            steps += st
            val += _spread(items, per_val, avoid={x["id"] for x in st})
        steps.sort(key=lambda it: order[it["id"]])          # the application's own order (it matters for warm caches)
        val.sort(key=lambda it: order[it["id"]])
        sp.write_text(json.dumps(wl.subset(workload, [x["id"] for x in steps], "%s-steps" % w["name"],
                                           "per request class: the median and the largest request (%d per class)" % per_step), indent=1) + "\n", encoding="utf-8")
        vp.write_text(json.dumps(wl.subset(workload, [x["id"] for x in val], "%s-validation" % w["name"],
                                           "per request class: %d requests from the smallest to the largest, preferring ones the steps did not run" % per_val),
                                 indent=1) + "\n", encoding="utf-8")
    steps_w, val_w = wl.load_workload(str(sp)), wl.load_workload(str(vp))
    seen = {it["id"] for it in steps_w["items"]}
    ident = wl.workload_identity(w)

    def per_class(ws):
        out = {}
        for it in ws["items"]:
            out[wl.request_class(it)] = out.get(wl.request_class(it), 0) + 1
        return out
    mode = cache_mode or w.get("cache_mode") or "cold"
    summary = {"source": Path(workload).name, "requests": len(w["items"]), "classes": ident["classes"], "sampled_requests": ident["sampled_requests"],
               "steps": {"requests": len(steps_w["items"]), "per_class": per_class(steps_w), "basis": json.loads(sp.read_text(encoding="utf-8")).get("basis")},
               "validation": {"requests": len(val_w["items"]), "per_class": per_class(val_w), "held_out": sum(1 for it in val_w["items"] if it["id"] not in seen),
                              "basis": json.loads(vp.read_text(encoding="utf-8")).get("basis")},
               "cache_mode": mode, "decision_fields": cfg.get("decision_fields") or None,
               "note": ("cache_mode cold: every request with cache_prompt false, so each change is measured on full prompt processing; your application's "
                        "prefix reuse is not part of these numbers (cache_mode warm replays the requests in order with reuse)" if mode == "cold" else
                        "cache_mode %s: the requests are replayed in your application's order with the prompt cache as they send it" % mode)}
    return {"steps": str(sp), "validation": str(vp), "cache_mode": mode, "decision_fields": cfg.get("decision_fields") or None,
            "validation_basis": "%d of your own requests (%s; %d not run in the steps)" % (len(val_w["items"]), summary["validation"]["basis"], summary["validation"]["held_out"]),
            "summary": summary}


def _apply_rq(spec, rq):
    """Request-workload settings every spec of the session carries: the cache mode, the operator's decision fields, and no
    n_predict (each request has its own generation limit)."""
    if not rq:
        return spec
    spec["cache_mode"] = rq["cache_mode"]
    if rq.get("decision_fields"):
        spec["decision_fields"] = rq["decision_fields"]
    spec.pop("n_predict", None)
    return spec


def _measured_requests(records):
    """From the newest experiment of the session: each application request's measured wall time (first arm) and the median
    generated length per request class, so later estimates use what the requests really cost."""
    for rec in reversed(records):
        if rec.get("kind") != "experiment" or not rec.get("arms"):
            continue
        first = next(iter(rec["arms"].values()))
        reqs = [r for r in first.get("requests") or [] if r.get("wall_ms") and not r.get("error")]
        if not reqs:
            continue
        wall, gen = {}, {}
        for r in reqs:
            wall.setdefault(r.get("prompt_id"), []).append(r["wall_ms"] / 1000.0)
            if r.get("class") and r.get("predicted_n"):
                gen.setdefault(r["class"], []).append(r["predicted_n"])
        med = lambda v: sorted(v)[len(v) // 2]
        return {"wall_s": {k: med(v) for k, v in wall.items()}, "class_generated": {k: med(v) for k, v in gen.items()}}
    return {}


def _gen_hint(diag):
    shape = (diag or {}).get("traffic_shape") or {}
    return shape.get("generated_tokens_p90") or shape.get("generated_tokens_median")


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


def build_problem(seen, cfg, identity):
    """The observed server's build (executable and ggml/llama libraries, from its process) against the configured
    runtime_binary's. None when they match or either is unknown."""
    from hotpath.observe import build_identity
    if not seen or not cfg.get("runtime_binary"):
        return None
    mine = build_identity(cfg["runtime_binary"], identity) or {}
    key = lambda x: ((x or {}).get("size_bytes"), (x or {}).get("sha256_head64m"))
    if not mine or not (mine.get("exe") or {}).get("size_bytes") or not (seen.get("exe") or {}).get("size_bytes"):
        return None
    differs = []
    if key(seen.get("exe")) != key(mine.get("exe")):
        differs.append("llama-server executable (%s vs %s)" % ((seen.get("exe") or {}).get("basename"), (mine.get("exe") or {}).get("basename")))
    for name in sorted(set(seen.get("libraries") or {}) | set(mine.get("libraries") or {})):
        if key((seen.get("libraries") or {}).get(name)) != key((mine.get("libraries") or {}).get(name)):
            differs.append(name)
    if differs:
        return ("the observed server runs another llama-server build than runtime_binary in hotpath.local.json (different %s): experiments would "
                "measure another setup than the one diagnosed. Point runtime_binary at the server's executable, or start from a new observation" % ", ".join(differs))
    return None


def identity_problem(records, cfg, identity):
    """Why evidence about the user's server cannot be carried into experiments with the configured model and binary, or None.
    Compared on size and a hash of the first 64 MiB; a record without identity is not judged. The build is the observed
    server's own executable and libraries (its process), not the configured binary's identity."""
    for rec in records:
        if rec.get("kind") != "observation":
            continue
        problem = build_problem(rec.get("server_build"), cfg, identity)
        if problem:
            return problem
        for key, cfg_key, what in (("model", "model_path", "model"),):
            seen = rec.get(key) or {}
            seen = seen.get("identity") if key == "model" else seen
            if not seen or not seen.get("size_bytes") or not cfg.get(cfg_key):
                continue
            mine = identity(cfg[cfg_key]) or {}
            if not mine.get("size_bytes"):
                continue
            if (seen.get("size_bytes"), seen.get("sha256_head64m")) != (mine.get("size_bytes"), mine.get("sha256_head64m")):
                return ("the observed server used a different %s (%s, %s bytes) than %s in hotpath.local.json (%s, %s bytes): experiments would measure "
                        "another setup than the one diagnosed. Point %s at the server's %s, or start from a new observation" % (
                            what, seen.get("basename"), seen.get("size_bytes"), cfg_key, mine.get("basename"), mine.get("size_bytes"), cfg_key, what))
    return None


def spec_fingerprint(spec, cfg=None):
    """What makes two runs of a step the same experiment: launch arguments, environment, arms, sizes, cache mode, decision
    fields, the workload's content (its set hash) and the files the ${key} references resolve to (size and modification
    time of the configured binary and model), not the spec's file name. A record is reused on resume only for this exact spec."""
    import hashlib
    import os as _os
    from hotpath.workload import load_workload, workload_identity

    def files(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}") and cfg:
            v = cfg.get(v[2:-1])
        try:
            st = _os.stat(str(v))
            return [Path(str(v)).name, st.st_size, int(st.st_mtime)]
        except (OSError, TypeError, ValueError):
            return [str(v)]
    try:
        w = workload_identity(load_workload(spec["workload"]))
        wl = [w.get("set_hash"), w.get("n_predict")]
    except (OSError, ValueError, KeyError, TypeError):
        wl = [Path(str(spec.get("workload"))).name, None]
    keep = ("name", "args_override", "args_remove", "env_override", "env_remove", "request_override", "binary", "model", "control")
    body = {"binary": spec.get("binary"), "model": spec.get("model"), "common_args": [str(a) for a in spec.get("common_args") or []],
            "env": spec.get("env") or {}, "arms": [{k: a.get(k) for k in keep if a.get(k) is not None} for a in spec.get("arms") or []],
            "sizes": [spec.get(k) for k in ("rounds", "repetitions", "warmup", "n_predict", "probe_top_k", "ignore_eos")],
            "cache_mode": spec.get("cache_mode"), "schema_validation": spec.get("schema_validation"), "workload": wl,
            "decision_fields": spec.get("decision_fields"), "files": [files(spec.get("binary")), files(spec.get("model"))] +
            [[a.get("name"), files(a["binary"]) if a.get("binary") else None, files(a["model"]) if a.get("model") else None] for a in spec.get("arms") or []
             if a.get("binary") or a.get("model")]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _matching_record(sdir, spec, cfg=None):
    """(path, None) for the newest record in the session made from this exact spec, else (None, why the recorded ones do
    not count, or None when there are none)."""
    done = sorted(Path(sdir).glob("experiment-%s-2*.json" % spec["name"]))
    if not done:
        return None, None
    fp = spec_fingerprint(spec, cfg)
    for p in reversed(done):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("spec_fingerprint") == fp:
            return p, None
    return None, "it was recorded for a different spec (another starting point, workload or hotpath version), so it is not evidence for this one"


def _reusing(runner, sdir, log, cfg=None):
    """A runner that returns the record already in the session directory for this exact spec (resume), else runs it and
    stamps the record with the spec's fingerprint."""
    def run(spec, spec_path):
        found, why = _matching_record(sdir, spec, cfg)
        if found:
            log("  reusing %s (completed earlier in this session)" % found.name)
            rec = json.loads(found.read_text(encoding="utf-8"))
            rec["_reused_from"] = found.name
            return rec
        if why:
            log("  not reusing the recorded %s: %s" % (spec["name"], why))
        rec = runner(spec, spec_path)
        rec["spec_fingerprint"] = spec_fingerprint(spec, cfg)
        return rec
    run.cfg = cfg
    return run


def _never_launch(spec, spec_path):
    """The runner of a dry run: recorded experiments are reused by _reusing; anything else would launch, so it refuses."""
    raise _DryRunStop(spec["name"])


class _DryRunStop(Exception):
    pass


def _recorded(sdir, spec, cfg=None):
    return _matching_record(sdir, spec, cfg)[0] is not None


def _check_saved_state(saved, sdir):
    """In a process the MCP server started, a resumed session's saved state is an input an agent could have edited: its
    starting arguments and workload must pass the same checks as an agent's own `args` and `workload`."""
    from hotpath import agentgate
    if not agentgate.active():
        return
    try:
        if saved.get("initial_args"):
            agentgate.check_args_string(saved["initial_args"], "%s/session.json initial_args" % Path(sdir).name)
        if saved.get("workload"):
            agentgate.inside_roots(saved["workload"], "%s/session.json workload" % Path(sdir).name)
            bad = agentgate.workload_problems(saved["workload"])
            if bad:
                raise ValueError("; ".join(bad))
    except ValueError as e:
        raise SystemExit("refused to resume %s: %s" % (Path(sdir).name, e))


def _spec_dir(sdir, dry_run):
    """Where a run writes its spec files: the session directory, or for a dry run a subdirectory of its own, so a plan made
    while the session runs never rewrites a file the running process uses."""
    d = Path(sdir) / "dry-run" if dry_run else Path(sdir)
    d.mkdir(parents=True, exist_ok=True)
    return d


# flags of the user's launch that experiments leave out without changing what is computed: the network binding, files the
# server writes, the model given another way (the configured model file, whose identity is checked), logging and endpoints
PLUMBING_FLAGS = frozenset({"--host", "--port", "--log-file", "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file", "--path", "--media-path",
                            "--webui-config-file", "--slot-save-path", "--prompt-cache", "--prompt-cache-all", "--prompt-cache-ro", "-m", "--model",
                            "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "--hf-token", "-hft", "-mu", "--model-url", "-dr", "--docker-repo", "-a", "--alias",
                            "--metrics", "--slots", "--no-slots", "--no-webui", "--webui", "--offline", "--props", "--api-prefix", "--threads-http",
                            "-to", "--timeout", "-v", "--verbose", "--log-verbose", "--log-disable", "--log-colors", "--log-prefix", "--log-timestamps",
                            "-lv", "--log-verbosity"})


def left_out_of_measurement(observed, measured):
    """Flags of the user's own launch line that the experiments did not run with, split into plumbing and the rest. The
    rest (a draft model, a vision projector, adapters, template or grammar files, ...) changes what the server computes
    or how much memory it needs, so a result measured without them does not describe a launch with them."""
    if not observed:
        return [], []
    have = {f for f, _ in launch.split_args(measured) if f}
    have_names = {launch._SPELLING.get(f, f) for f in have}
    gone = []
    for f, _v in launch.split_args(observed):
        if f and f not in have and launch._SPELLING.get(f, f) not in have_names and f not in gone:
            gone.append(f)
    return [f for f in gone if f in PLUMBING_FLAGS], [f for f in gone if f not in PLUMBING_FLAGS]


def _write_exp(rec, sdir, name):
    if rec.get("_reused_from"):
        return Path(sdir) / rec["_reused_from"]
    return write_record(rec, sdir, "experiment-%s" % name)


def improve(cfg, start_records, workload, out_dir, runner, max_experiments=3, max_minutes=60, rounds=3, repetitions=2, warmup=1, port=8099,
            n_predict=None, dry_run=False, log=print, initial_args=None, bandwidth_gbs=None, session=None, resume=False, identity=None,
            cache_mode=None, step_requests=2, validation_requests=4):
    """Run the loop. `runner(spec, spec_path)` runs one experiment spec and returns its record (raising SystemExit when
    it cannot start, for example on unmet preconditions). Returns the improvement report (also written to out_dir).
    With `resume`, the session's saved starting point is used and experiments already recorded in it are not run again."""
    from hotpath.records import file_identity
    session = session or "improve-%s" % time.strftime("%Y%m%d-%H%M%S")
    if not resume:
        # a new run never shares a session directory (two runs in one second, or a reused name): its records would be reused
        base_name, k = session, 2
        while (Path(out_dir) / session).exists():
            session, k = "%s-%d" % (base_name, k), k + 1
    sdir = Path(out_dir) / session
    sdir.mkdir(parents=True, exist_ok=True)
    state = sdir / "session.json"
    if resume:
        if not state.is_file():
            raise SystemExit("error: no resumable session in %s (session.json missing)" % sdir)
        saved = json.loads(state.read_text(encoding="utf-8"))
        _check_saved_state(saved, sdir)
        start_records, workload, initial_args = saved["start_records"], saved["workload"], saved.get("initial_args")
        cache_mode = saved.get("cache_mode", cache_mode)
        log("resuming %s: %d starting record(s), workload %s" % (session, len(start_records), Path(workload).name))
    elif not dry_run:
        # starting arguments are kept without runner-owned flags (ports, files, keys): those never reach a launch anyway
        saved_args = prop.strip_runner_owned(initial_args)[0] if initial_args else initial_args
        state.write_text(json.dumps({"start_records": list(start_records), "workload": workload, "initial_args": saved_args, "cache_mode": cache_mode}) + "\n",
                         encoding="utf-8")
    if dry_run:
        runner = _never_launch          # a dry run reads what a session recorded and launches nothing, also when resuming
    runner = _reusing(runner, sdir, log, cfg)
    lock = None if dry_run else locks.take_session(sdir)
    try:
        return _improve(cfg, start_records, workload, out_dir, runner, max_experiments, max_minutes, rounds, repetitions, warmup, port, n_predict,
                        dry_run, log, initial_args, bandwidth_gbs, session, sdir, identity, file_identity,
                        {"cache_mode": cache_mode, "step": step_requests, "validation": validation_requests})
    finally:
        if lock is not None:
            locks.drop_session(lock)


def _improve(cfg, start_records, workload, out_dir, runner, max_experiments, max_minutes, rounds, repetitions, warmup, port, n_predict, dry_run, log,
             initial_args, bandwidth_gbs, session, sdir, identity, file_identity, rq_opts=None):
    problem = identity_problem(start_records, cfg, identity or (lambda path: file_identity(path) if Path(path).is_file() else None))
    if problem:
        return _early_report(session, out_dir, initial_args or [], "the records you started from", workload, max_experiments, max_minutes,
                             "refused before launching anything: %s" % problem)
    records = list(start_records)
    requests_wl = _is_requests_workload(workload)
    rq = None
    if requests_wl:
        try:
            rq = requests_plan(workload, sdir, cfg, (rq_opts or {}).get("cache_mode"), (rq_opts or {}).get("step") or 2, (rq_opts or {}).get("validation") or 4)
        except (OSError, ValueError, KeyError) as e:
            return _early_report(session, out_dir, initial_args or [], "the records you started from", workload, max_experiments, max_minutes,
                                 "refused before launching anything: the requests workload cannot be used: %s" % e)
        log("your requests: %d in %d class(es) from %s; steps run %d, the validation %d (%d not seen by the steps); cache mode %s" % (
            rq["summary"]["requests"], len(rq["summary"]["classes"]), rq["summary"]["source"], rq["summary"]["steps"]["requests"],
            rq["summary"]["validation"]["requests"], rq["summary"]["validation"]["held_out"], rq["cache_mode"]))
    step_wl = rq["steps"] if rq else workload
    baseline = None
    if not records:
        # nothing running and no log: measure the starting configuration on hotpath's own server first
        args, source = (list(initial_args), "command line") if initial_args else baseline_args_without_server(cfg)
        args, removed_b = prop.strip_runner_owned(args)
        if removed_b:
            source += "; left out (the runner sets its own binding, model and files): %s" % ", ".join(sorted(set(removed_b)))
        bspec = _apply_rq(baseline_spec(cfg, args, step_wl, port, repetitions, warmup, n_predict, "%s-baseline" % session), rq)
        bpath = _spec_dir(sdir, dry_run) / "baseline.json"
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
        except _DryRunStop:
            return _early_report(session, out_dir, args, source, workload, max_experiments, max_minutes, "dry run: nothing launched")
        except SystemExit as e:
            return _early_report(session, out_dir, args, source, workload, max_experiments, max_minutes, "baseline not run: %s" % e)
        baseline["minutes"] = round((time.time() - t0) / 60.0, 1)
        baseline["record"] = _write_exp(brec, sdir, bspec["name"]).name
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
    shape = diag.get("traffic_shape")
    best = {"args": list(args0), "env": {}, "label": None}
    cum = {"args": {}, "args_remove": set(), "env": {}, "env_remove": set()}
    steps, attempted, adopted_steps = [], set(), []
    budget = Budget(max_experiments, max_minutes)
    if baseline:
        budget.minutes += baseline.get("minutes") or 0.0
        best["label"] = "%s/baseline" % records[0]["name"]
    top = next((f for f in diag.get("findings") or [] if f.get("severity") in ("high", "medium") and not f.get("superseded_by")), None)
    report_bottleneck = {"id": top["id"], "title": top.get("title"), "severity": top.get("severity"), "confidence": top.get("confidence"),
                         "known": (top.get("known") or [])[:2], "bound": (top.get("impact") or {}).get("bound_share_of_request")} if top else None
    report = {"bottleneck": report_bottleneck, "session": session, "starting_point": {"args": args0, "source": source, "removed_server_args": sorted(set(removed)),
                                                     "diagnosed_arm": diag.get("diagnosed_arm"), "sources": diag.get("sources")},
              "workload": Path(workload).name, "budget": {"max_experiments": budget.max_experiments, "max_minutes": budget.max_minutes},
              "steps": steps, "skipped_findings": [], "confirmation": None, "next_step": None, "baseline": baseline, "decisions": [],
              "requests": rq["summary"] if rq else None}
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
            spec = _apply_rq(step_spec(f, cfg, best, step_wl, port, rounds, repetitions, warmup, n_predict, name), rq)
        except ValueError as e:
            attempted.add(f["id"])
            report["skipped_findings"].append({"finding": f["id"], "title": f.get("title"), "why_not_run": str(e)})
            continue
        spec["provenance"].update({"improve_session": session, "step": n})
        est = prop.estimate_minutes(spec, diag.get("rates"), load_s=_measured_load_s(records), measured=_measured_requests(records), gen_hint=_gen_hint(diag))
        if est:
            spec["provenance"]["estimated_duration"] = est
        need = est["minutes"] if est else None
        reusable = _recorded(sdir, spec, cfg)          # a step recorded earlier in the session costs nothing on resume
        over = need is not None and need > budget.time_left() and not reusable
        if over and not dry_run:
            stopped = "time budget: the next step (%s) needs about %.0f min and %.0f min are left" % (f["id"], need, max(0.0, budget.time_left()))
            break
        spec_path = _spec_dir(sdir, dry_run) / ("step-%d-%s.json" % (n, f["experiment"]["kind"]))
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        if dry_run and not reusable:
            report["next_step"] = {"finding": f["id"], "title": f.get("title"), "spec": str(spec_path), "estimated_minutes": need,
                                   "arms": [a["name"] for a in spec["arms"]], "variable": f["experiment"].get("variable"), "port": port,
                                   "min_free_vram_mib": spec["preconditions"].get("min_free_vram_mib"),
                                   "changes": {a["name"]: _change_text(a) for a in spec["arms"]},
                                   "exceeds_budget": {"needs_minutes": need, "left_minutes": round(max(0.0, budget.time_left()), 1)} if over else None}
            vw = rq["validation"] if rq else validation_workload(_spec_dir(sdir, True), shape, _ctx_tokens(best["args"]), n_predict)[0]
            vest = prop.estimate_minutes({"workload": vw, "arms": [{}, {}], "rounds": rounds, "repetitions": repetitions, "warmup": warmup, "cooldown_s": 5,
                                          "cache_mode": rq["cache_mode"] if rq else None},
                                         diag.get("rates"), load_s=_measured_load_s(records), measured=_measured_requests(records), gen_hint=_gen_hint(diag))
            report["validation_estimate_minutes"] = vest["minutes"] if vest else None
            report["validation_size"] = rq["validation_basis"] if rq else "6 synthetic prompts from the smallest to the largest real request size"
            report["plan_queue"] = [{"finding": x["id"], "title": x.get("title")} for x in diag.get("findings") or []
                                    if x["id"] != f["id"] and executable(x, requests_wl) is None]
            report["not_run_unattended"] = [{"finding": x["id"], "title": x.get("title"), "why": executable(x, requests_wl)}
                                            for x in diag.get("findings") or [] if x.get("severity") in ("high", "medium") and executable(x, requests_wl)]
            stopped = "dry run: nothing launched" + (" (the first step needs more than the time budget; see the plan)" if over else "")
            break
        log("step %d of at most %d: %s -> %s (%s; %.0f of %.0f budget minutes used)" % (
            n, budget.max_experiments, f["id"], ", ".join(a["name"] for a in spec["arms"]), "about %.0f min" % need if need else "duration unknown",
            budget.minutes, budget.max_minutes))
        t0 = time.time()
        try:
            rec = runner(spec, str(spec_path))
        except _DryRunStop:
            stopped = "dry run: nothing launched"
            break
        except SystemExit as e:
            steps.append({"step": n, "finding": f["id"], "title": f.get("title"), "experiment_kind": f["experiment"]["kind"], "experiment": name,
                          "adopted": False, "verdict": "not run", "why": str(e)})
            stopped = str(e)
            break
        budget.experiments += 1
        budget.minutes += max((time.time() - t0) / 60.0, 0.0)
        attempted.add(f["id"])
        rpath = _write_exp(rec, sdir, name)
        records.append(rec)
        diag = dg.diagnose(records, bandwidth_gbs=bandwidth_gbs, cfg=cfg)
        verdict = next((v for v in diag.get("experiment_verdicts") or [] if v["experiment"] == rec["name"]), None)
        step = {"step": n, "finding": f["id"], "title": f.get("title"), "experiment_kind": f["experiment"]["kind"], "experiment": name,
                "record": rpath.name, "arms": [a["name"] for a in spec["arms"]], "estimated_minutes": need}
        rows = _rank_rows(verdict) if verdict else []
        arm_of = {a["name"]: a for a in spec["arms"]}
        base_name = spec["arms"][0]["name"]
        judged = {}
        for r in rows:
            judged[r["arm"]] = adoptable(r, verdict, arm_of.get(r["arm"]), rec.get("arms"), base_name, cfg)
        row, ok, why = None, False, None
        for r in rows:
            if judged[r["arm"]][0] and not (arm_of.get(r["arm"]) or {}).get("control"):
                row, ok, why = r, True, judged[r["arm"]][1]
                break
        if not row and rows:
            row = rows[0]
            ok, why = judged[row["arm"]][0], judged[row["arm"]][1]
        # faster candidates that only the user can accept: a capacity change, outputs that differ, no VRAM headroom
        for r in rows:
            if (r is row and ok) or (arm_of.get(r["arm"]) or {}).get("control"):
                continue
            a = arm_of.get(r["arm"]) or {}
            if judged[r["arm"]][2]:
                report["decisions"].append({"step": n, "arm": r["arm"], "change": _change_text(a), "verdict": r["verdict"],
                                            "change_spec": {k: a.get(k) or ({} if k.endswith("override") else []) for k in ("args_override", "args_remove", "env_override", "env_remove")},
                                            "needs": judged[r["arm"]][1].replace("not adopted: ", ""), "fidelity": (r.get("fidelity") or {}).get("class"),
                                            "measured": _metric_line(r), "decomposition": decomposition(rows, arm_of, r["arm"])})
        if not row:
            step.update({"adopted": False, "verdict": "no verdict (the record has fewer than two arms with requests)", "why": "nothing to compare"})
        else:
            step.update({"adopted": ok, "candidate": row["arm"], "verdict": row["verdict"], "why": why, "measured": _metric_line(row),
                         "fidelity": (row.get("fidelity") or {}).get("class"), "caveats": verdict.get("caveats") or [],
                         "other_arms": [{"arm": r["arm"], "verdict": r["verdict"].split(":")[0],
                                         "why": ("control arm (%s): wall ratio %s, outputs %s" % (arm_of[r["arm"]]["control"], ((r.get("metrics") or {}).get("wall_ms") or {}).get("ratio"),
                                                                                                  (r.get("fidelity") or {}).get("class")))
                                         if (arm_of.get(r["arm"]) or {}).get("control") else judged[r["arm"]][1]} for r in rows if r is not row]})
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
                               runner, dry_run, bandwidth_gbs, log, report, shape, rq)
    report["minutes_used"] = round(budget.minutes, 1)
    validated = report["result"].get("validated")
    if validated and report.get("validated_steps") and len(report["validated_steps"]) < len(adopted_steps):
        ch = _cum_change([st for st in adopted_steps if st["experiment"] in report["validated_steps"]])
        cum = {"args": dict(ch["args_override"]), "args_remove": set(ch["args_remove"]), "env": dict(ch["env_override"]), "env_remove": set(ch["env_remove"])}
    plumbing, not_measured = left_out_of_measurement(observed, args0)
    report["starting_point"]["left_out"] = {"plumbing": plumbing, "not_measured": not_measured}
    if validated and not_measured:
        # the validated run did not include these: applying its changes to the full launch line is not what was measured
        report["result"]["validated"] = validated = False
        report["result"]["verdict"] = ("not recommended automatically: the validated run measured your launch without %s (hotpath cannot run them in "
                                       "its experiments), and those change what the server computes or how much memory it needs. Result without "
                                       "them: %s" % (", ".join(not_measured), report["result"]["verdict"]))
        report["decisions"].append({"step": "validation", "arm": "validated without %s" % ", ".join(not_measured), "change": _change_text(_cum_change(adopted_steps)),
                                    "verdict": report["result"]["verdict"], "needs": "the measurement left out %s; check the change with them loaded" % ", ".join(not_measured),
                                    "fidelity": None, "measured": report["result"].get("measured"), "decomposition": None})
    report["recommended"] = recommended_launch(observed or args0, cum if validated else {"args": {}, "args_remove": set(), "env": {}, "env_remove": set()},
                                               bool(observed), validated=validated, adopted=bool(adopted_steps))
    if not dry_run and budget.experiments:
        validate_decisions(cfg, original, report, records, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner, log, shape, rq)
    report["remaining"] = [{"id": f["id"], "title": f.get("title"), "confidence": f.get("confidence"), "next": (f.get("measured") or {}).get("how")}
                           for f in diag.get("findings") or [] if f.get("severity") in ("high", "medium") and f["id"] not in attempted][:5]
    report["outcomes"] = outcomes(report)
    report["next_measurement"] = next_measurement(report)
    rec = new_record("improvement", "controlled" if budget.experiments else "hypothesis", **report)
    path = write_record(rec, out_dir, "improvement")
    Path(str(path)[:-5] + ".md").write_text(improvement_markdown(rec), encoding="utf-8")
    rec["path"] = str(path)
    if not dry_run:
        (sdir / "finished").write_text(Path(path).name + "\n", encoding="utf-8")
    return rec


OUTCOME_ORDER = ("recommended", "needs_review", "rejected", "inconclusive")
OUTCOME_LABEL = {"recommended": "Recommended", "needs_review": "Needs review", "rejected": "Rejected", "inconclusive": "Inconclusive"}


def outcomes(report):
    """Every candidate the run measured, in exactly one of four outcomes:
    recommended  - faster, and the required correctness criteria held in a validation run of exactly this set;
    needs_review - faster, but with output differences, a capacity limit, a change to what the application sends, too little
                   VRAM headroom, or launch flags the measurement could not include: only the user can accept it;
    rejected     - slower, a regression on some prompt or request class, failed launches or rejected requests, a validity drop,
                   memory spill, or a production gate that rules it out;
    inconclusive - no difference beyond the noise, or not enough trustworthy evidence (including not validated for lack of budget)."""
    res = report.get("result") or {}
    out = {k: [] for k in OUTCOME_ORDER}
    validated_steps = set(report.get("validated_steps") or [])
    decided = set()
    for d in report.get("decisions") or []:
        key = (d.get("step"), d.get("arm"))
        if key in decided:
            continue
        decided.add(key)
        out["needs_review"].append({"what": "%s%s" % ("step %s " % d["step"] if isinstance(d.get("step"), int) else "", d.get("arm")), "change": d.get("change"),
                                    "why": d.get("needs"), "fidelity": d.get("fidelity"), "wall": ((d.get("measured") or {}).get("wall_ms") or {}).get("ratio"),
                                    "validated_on_broad_workload": bool(d.get("validation"))})
    for st in report.get("steps") or []:
        what = "step %d %s" % (st["step"], st.get("candidate") or st["finding"])
        item = {"what": what, "change": _change_text(st["change"]) if st.get("change") else None, "why": st.get("why"), "fidelity": st.get("fidelity"),
                "wall": (((st.get("measured") or {}).get("wall_ms") or {}).get("ratio"))}
        if st.get("adopted"):
            if res.get("validated") and (not validated_steps or st["experiment"] in validated_steps):
                out["recommended"].append(dict(item, why="validated: %s" % (res.get("verdict") or "").split(" [")[0]))
            elif (st.get("step"), st.get("candidate")) in decided or any(d.get("step") == "validation" for d in report.get("decisions") or []):
                continue                     # its fate is the validation decision listed under needs review
            elif "budget" in (res.get("verdict") or "") or res.get("not_validated"):
                out["inconclusive"].append(dict(item, why="kept by its step, not validated: %s" % (res.get("verdict") or "")))
            else:
                v = (res.get("verdict") or "").lower()
                bucket = "rejected" if any(w in v for w in ("regress", "reject", "slower")) else "needs_review" if "outputs differ" in v or "near-tie" in v or "differ" in v                     else "inconclusive"
                out[bucket].append(dict(item, why="kept by its step, but the validation did not pass: %s" % (res.get("verdict") or "")))
            continue
        if (st.get("step"), st.get("candidate")) in decided:
            continue
        verdict = (st.get("verdict") or "").lower()
        why = (st.get("why") or "").lower()
        if verdict.startswith("reject") or "regress" in why or "spills" in why or "would cost more" in why:
            out["rejected"].append(item)
        else:
            out["inconclusive"].append(item)
    return out


def next_measurement(report):
    """What to measure next when the evidence did not decide, instead of a conclusion it cannot support."""
    out = []
    res = report.get("result") or {}
    oc = report.get("outcomes") or {}
    if res.get("not_validated") or "budget" in (res.get("verdict") or ""):
        out.append("validate what the steps kept: `hotpath improve --resume %s --max-minutes N` (completed steps are reused)" % report.get("session"))
    if any("within the within-prompt noise" in (x.get("why") or "") or "more rounds" in (x.get("why") or "") for x in oc.get("inconclusive") or []) or \
            any("within noise" in (st.get("verdict") or "") or "more rounds" in (st.get("verdict") or "") for st in report.get("steps") or []):
        out.append("a small effect within the noise: repeat with more rounds (`--rounds 5`) before deciding")
    if not report.get("requests"):
        out.append("the steps used synthetic prompts: run again with your application's own requests (`--requests captured.jsonl`) so the decision rests on "
                   "your request shapes, schemas and output lengths")
    if any((x.get("fidelity") or "").startswith(("numerically close", "unverified")) for x in oc.get("needs_review") or []):
        out.append("faster options change outputs: review the differing requests with your application's own check, or state what it decides on "
                   "(`decision_fields` in hotpath.local.json) so hotpath can compare exactly those fields")
    rq = report.get("requests") or {}
    if rq and rq.get("cache_mode") == "cold" and any(st.get("experiment_kind") == "server_overhead_flags" for st in report.get("steps") or []):
        out.append("prompt-cache flags were measured without reuse (cache mode cold); `--cache-mode warm` replays your requests in order with reuse")
    return out


def unfinished_sessions(out_dir):
    """Sessions that started experiments but wrote no report (interrupted): (name, completed experiment records)."""
    out = []
    for d in sorted(Path(out_dir).glob("improve-*")):
        if d.is_dir() and (d / "session.json").is_file() and not (d / "finished").exists():
            out.append((d.name, len(list(d.glob("experiment-*.json")))))
    return out


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
    # an arm that launches the same configuration as an earlier one measures nothing new
    kept = []
    for a in spec["arms"]:
        args, env = materialize(spec["common_args"], spec["env"], a)
        if any(_same_config(args, env, *materialize(spec["common_args"], spec["env"], k)) for k in kept):
            continue
        kept.append(a)
    spec["arms"] = kept
    if len(kept) < 2:
        raise ValueError("every candidate arm repeats the configuration already adopted")
    need = [prop.vram_needed_mib(cfg, materialize(spec["common_args"], spec["env"], a)[0]) for a in spec["arms"]]
    if need and all(x is not None for x in need):
        spec["preconditions"]["min_free_vram_mib"] = max(need)
        spec["provenance"]["min_free_vram_basis"] = "weights on the GPU + 10%, KV cache at the arm's context and KV types, 512 MiB compute, for the largest arm (estimate from the GGUF header)"
    return spec


def validation_workload(sdir, shape, ctx_tokens, n_predict=None):
    """A broader workload than the steps used, for the final original-vs-improved run: six prompts from the smallest to the
    largest real request size (p10, median, p90 and the largest, plus two in between), generating the longest observed
    output (bounded), so a gain on the step workload must also hold on long requests and long generations. Without real
    traffic: six default sizes. Every prompt + generation fits in `ctx_tokens` (the smallest context of the two arms)."""
    from hotpath import workload as wl
    if shape:
        toks = sorted({shape["prompt_tokens_p10"], (shape["prompt_tokens_p10"] + shape["prompt_tokens_median"]) // 2, shape["prompt_tokens_median"],
                       (shape["prompt_tokens_median"] + shape["prompt_tokens_p90"]) // 2, shape["prompt_tokens_p90"],
                       shape.get("prompt_tokens_max") or shape["prompt_tokens_p90"]})
        gen = int(n_predict or max(32, min(256, shape.get("generated_tokens_max") or shape["generated_tokens_median"])))
        basis = "sizes from %s; generation = the longest observed output (bounded to 32-256)" % shape["basis"]
    else:
        toks = [250, 520, 1050, 1500, 2300, 3450]
        gen = int(n_predict or 128)
        basis = "default sizes (no real traffic in the evidence)"
    cap = int(ctx_tokens * 0.9) - gen if ctx_tokens else None
    if cap:
        toks = [t for t in toks if t <= cap] or [max(64, cap)]
    while len(toks) < 6:
        toks.append(toks[len(toks) % len(toks)])
    words = [max(20, int(round((t - 20) / 1.15))) for t in toks[:6]]
    path = Path(sdir) / "validation-workload.json"
    w = wl.build_workload("hotpath-validation", len(words), words, seed=7, n_predict=gen, fmt="chatml")
    w["basis"] = basis
    path.write_text(json.dumps(w, indent=1) + "\n", encoding="utf-8")
    return str(path), basis


def _ctx_tokens(args):
    c = launch.int_or_none(launch.get(args, "ctx_size"))
    np_ = launch.int_or_none(launch.get(args, "parallel")) or 1
    return (c // np_) if c else 4096


def decomposition(rows, arm_of, name):
    """For a smaller-context arm measured with its context-only control: how much of the wall-time change the context alone
    explains and how much the extra GPU layers do (ratios against the same baseline in the same experiment)."""
    a = arm_of.get(name) or {}
    ctrl = next((x for x in arm_of.values() if x.get("control") and (x.get("args_override") or {}).get("--ctx-size") == (a.get("args_override") or {}).get("--ctx-size")
                 and x is not a), None)
    if not ctrl or "--ctx-size" not in (a.get("args_override") or {}):
        return None
    ratio = {r["arm"]: ((r.get("metrics") or {}).get("wall_ms") or {}).get("ratio") for r in rows}
    if ratio.get(ctrl["name"]) is None or ratio.get(name) is None:
        return None
    return {"context_only_wall_ratio": ratio[ctrl["name"]], "with_more_layers_wall_ratio": ratio[name],
            "reading": "the context change alone moves wall time by %+.0f%%; with the extra GPU layers it allows, %+.0f%%" % (
                100 * (ratio[ctrl["name"]] - 1), 100 * (ratio[name] - 1))}


def _result(cfg, original, cum, adopted_steps, records, workload, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner,
            dry_run, bandwidth_gbs, log, report, shape=None, rq=None):
    if dry_run:
        return {"verdict": "dry run: nothing measured", "measured_by": None, "validated": None}
    if not budget.experiments:
        return {"verdict": "nothing measured: %s" % (report.get("stopped_because") or "no experiment ran"), "measured_by": None, "validated": None}
    if not adopted_steps:
        return {"verdict": "no change adopted: the configuration you started from stands", "measured_by": None, "validated": None}
    # every adopted change is validated in one paired run of the original launch against the improved one, on a broader
    # workload than the steps used; only such a run supports a recommendation. When the full set fails and it has more than
    # one change, each change is validated alone (largest step gain first) while the budget allows, and the first that passes
    # is recommended instead; every failed set that was faster is offered as a decision
    # the original launch's own speed bounds every validation run (step 1's first arm is the original launch)
    s1 = ((report.get("steps") or [{}])[0].get("measured") or {})
    original = dict(original, rates={"prefill_tps": (s1.get("prefill_tps") or {}).get("baseline"), "decode_tps": (s1.get("decode_tps") or {}).get("baseline"),
                                     "basis": "the original launch in step 1"})
    candidates = [("all adopted changes", adopted_steps)]
    if len(adopted_steps) > 1:
        gain = lambda st: ((st.get("measured") or {}).get("wall_ms") or {}).get("ratio") or 1.0
        candidates += [("only %s" % st["finding"], [st]) for st in sorted(adopted_steps, key=gain)]
    tried = []
    for k, (label, steps_) in enumerate(candidates):
        out = _validate(cfg, original, steps_, records, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner, log, shape,
                        "%s-confirm" % session if k == 0 else "%s-confirm-%d" % (session, k + 1), rq)
        out["label"] = label
        tried.append(out)
        if out.get("validated") or out.get("stop"):
            break
        if out.get("row") and (out["row"].get("verdict") or "").startswith(("faster", "keep")):
            report["decisions"].append({"step": "validation", "arm": label, "change": _change_text(_cum_change(steps_)), "verdict": out["row"]["verdict"],
                                        "needs": out["why"].replace("not adopted: ", ""), "fidelity": (out["row"].get("fidelity") or {}).get("class"),
                                        "measured": _metric_line(out["row"]), "decomposition": None})
    report["validations"] = [{k: v for k, v in t.items() if k not in ("row", "steps")} for t in tried]
    last = tried[-1]
    report["confirmation"] = (last if last.get("validated") else tried[0]).get("confirmation")
    if last.get("validated"):
        kept = last["steps"]
        report["validated_steps"] = [st["experiment"] for st in kept]
        if len(kept) < len(adopted_steps):
            last["verdict"] += " (validated without %s, which did not pass alone or together)" % ", ".join(st["finding"] for st in adopted_steps if st not in kept)
        return {"verdict": last["verdict"], "measured_by": last["measured_by"], "measured": last.get("measured"), "validated": True,
                "validated_changes": [st["finding"] for st in kept], "basis": last["basis"]}
    first = tried[0]
    untried = [label for label, _ in candidates[len(tried):]]
    note = ""
    if untried:
        note = (". Not validated, so not supported by any evidence: %s (the time budget ran out); `hotpath improve --resume %s --max-minutes N` validates them "
                "without repeating the steps" % ("; ".join(untried), session))
    return {"verdict": "not validated: %s; your original launch stays the recommendation%s%s" % (
                first["why"].replace("not adopted: ", ""), ". Validation run: %s" % first["row"]["verdict"] if first.get("row") else "", note),
            "measured_by": first.get("measured_by"), "measured": first.get("measured"), "validated": False, "not_validated": untried}


def validate_decisions(cfg, original, report, records, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner, log, shape, rq=None):
    """A step-level decision (a smaller context) was measured on the step workload only; validate the fastest such option
    against the original launch on the same broad workload the adopted changes faced, so the user decides on evidence of
    the same standard. Within the time budget; the result is attached to the decision, nothing is adopted."""
    cands = [d for d in report.get("decisions") or [] if isinstance(d.get("step"), int) and d.get("change_spec") and "context" in (d.get("needs") or "")]
    if not cands:
        return
    cands.sort(key=lambda d: (((d.get("measured") or {}).get("wall_ms") or {}).get("ratio") or 1.0))
    d = cands[0]
    s1 = ((report.get("steps") or [{}])[0].get("measured") or {})
    orig = dict(original, rates={"prefill_tps": (s1.get("prefill_tps") or {}).get("baseline"), "decode_tps": (s1.get("decode_tps") or {}).get("baseline"),
                                 "basis": "the original launch in step 1"})
    pseudo = {"finding": "option %s" % d["arm"], "step": d["step"], "experiment": d["arm"], "change": d["change_spec"], "measured": d.get("measured")}
    out = _validate(cfg, orig, [pseudo], records, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner, log, shape,
                    "%s-option-%s" % (session, d["arm"]), rq)
    row = out.get("row")
    fid = (row.get("fidelity") or {}) if row else {}
    d["validation"] = {"run": out.get("measured_by"), "verdict": out.get("verdict"), "measured": out.get("measured"), "fidelity": fid.get("class"),
                       "fidelity_why": fid.get("why"),
                       "summary": ("validated on the broad workload: %s" % out["verdict"].split(" [")[0]) if row else out.get("verdict")}


def _cum_change(steps_):
    c = {"args_override": {}, "args_remove": [], "env_override": {}, "env_remove": []}
    for st in steps_:
        ch = st.get("change") or {}
        c["args_override"].update(ch.get("args_override") or {})
        c["args_remove"] += [x for x in ch.get("args_remove") or [] if x not in c["args_remove"]]
        c["env_override"].update(ch.get("env_override") or {})
        c["env_remove"] += [x for x in ch.get("env_remove") or [] if x not in c["env_remove"]]
    return c


def _validate(cfg, original, steps_, records, port, rounds, repetitions, warmup, n_predict, budget, sdir, session, runner, log, shape, name, rq=None):
    """One validation run: the original launch against the original plus `steps_`' changes, on the broader workload."""
    ch = _cum_change(steps_)
    improved = {"name": "improved"}
    for k in ("args_override", "args_remove", "env_override", "env_remove"):
        if ch[k]:
            improved[k] = ch[k]
    imp_args = materialize(original["args"], original["env"], improved)[0]
    if rq:
        vw, vbasis = rq["validation"], rq["validation_basis"]
    else:
        vw, vbasis = validation_workload(sdir, shape, min(_ctx_tokens(original["args"]), _ctx_tokens(imp_args)), n_predict)
    spec = {"name": name, "description": "Validation by hotpath improve: your original launch against the launch with %s, on a broader workload than the "
                                         "steps used. Several variables may differ by design; the one-variable evidence for each change is in its own step." % (
                                             ", ".join(st["finding"] for st in steps_)),
            "binary": "${runtime_binary}", "model": "${model_path}", "port": int(port), "common_args": list(original["args"]), "env": dict(original["env"]),
            "arms": [{"name": "original"}, improved], "workload": vw, "rounds": int(rounds), "repetitions": int(repetitions), "warmup": int(warmup),
            "cooldown_s": 5, "health_timeout_s": 600,
            "preconditions": {"sample_seconds": 5, "max_gpu_util_pct": 10, "min_free_vram_mib": 2000, "ports_must_be_free": [int(port)]},
            "provenance": {"experiment_kind": "confirmation", "improve_session": session, "adopted_steps": [st["experiment"] for st in steps_],
                           "workload_basis": vbasis}}
    if cfg.get("path_prepend"):
        spec["path_prepend"] = "${path_prepend}"
    _apply_rq(spec, rq)
    need = [prop.vram_needed_mib(cfg, materialize(spec["common_args"], spec["env"], a)[0]) for a in spec["arms"]]
    if need and all(x is not None for x in need):
        spec["preconditions"]["min_free_vram_mib"] = max(need)
    rates = original.get("rates")
    est = prop.estimate_minutes(spec, rates, load_s=_measured_load_s(records), measured=_measured_requests(records))
    base = {"steps": steps_, "measured_by": None, "validated": False}
    if est:
        spec["provenance"]["estimated_duration"] = est
        if est["minutes"] > budget.time_left():
            return dict(base, verdict="not validated: the validation run needs about %.0f min and %.0f min of the budget are left (raise --max-minutes, or "
                                      "--resume this session with a larger budget)" % (est["minutes"], max(0.0, budget.time_left())),
                        why="not adopted: the budget ran out before validation", stop=True)
    path = sdir / ("%s.json" % name.replace(session + "-", ""))
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    log("validating: original vs %s on %s%s" % (", ".join(st["finding"] for st in steps_), vbasis.split(" (")[0] if rq else "6 prompts",
                                                ", about %.0f min" % est["minutes"] if est else ""))
    t0 = time.time()
    try:
        rec = runner(spec, str(path))
    except SystemExit as e:
        return dict(base, verdict="not validated: %s" % e, why="not adopted: %s" % e, stop=True)
    finally:
        budget.minutes += max((time.time() - t0) / 60.0, 0.0)
    rpath = _write_exp(rec, sdir, name)
    v = dg.experiment_verdict(rec)
    row = (v or {}).get("arms", [None])[0] if v else None
    if not row:
        return dict(base, measured_by=name, verdict="not validated: the validation run produced no comparison", why="not adopted: no comparison")
    ok, why, _ = adoptable(row, v, improved, rec.get("arms"), "original", cfg)
    if ok and "preliminary" in (row.get("verdict") or ""):
        ok, why = False, "not adopted: insufficient evidence: the validation run was preliminary (%s); more rounds or requests are needed" % (
            row["verdict"].split("(", 1)[1].split(")", 1)[0] if "(" in row["verdict"] else "too few prompts or rounds")
    conf = {"experiment": name, "record": rpath.name, "verdict": row["verdict"], "workload_basis": vbasis, "measured": _metric_line(row),
            "fidelity": row.get("fidelity"), "passed": ok, "why": why, "classes": row.get("classes"), "fields": row.get("fields"),
            "request_errors": row.get("request_errors")}
    return dict(base, measured_by=name, row=row, why=why, verdict=row["verdict"], measured=_metric_line(row), validated=ok, confirmation=conf,
                basis="one paired run of your original launch against the launch with %s, on a broader workload (%s)" % (
                    ", ".join(st["finding"] for st in steps_), vbasis))


def recommended_launch(user_args, cum, observed, validated=True, adopted=False):
    """The user's own launch line with the adopted changes, plus the environment changes; and the line to go back to.
    Flags the user set that the loop did not change (port, host, log file, model) stay as they were."""
    revert = [str(a) for a in user_args]
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
    if adopted and not validated:
        note = ("No change is recommended: the adopted changes did not pass the final validation, so your original launch stays (shown below). "
                "hotpath did not change your server.")
    elif changes or env:
        note = ("hotpath did not change your server. To apply, add these changes to the command (or launcher settings) that starts it, restart it "
                "yourself, and run `hotpath inspect` afterwards to confirm the new launch. To revert, start it again with your original line.")
    else:
        note = "No change is recommended; your original launch stays. hotpath did not change your server."
    return {"args": args, "changes": changes, "env": env, "revert_args": revert,
            "basis": "your server's observed command line (paths shown as file names)" if observed else "the arguments the loop started from",
            "note": note}


def _metrics_lines(m):
    L = []
    for k, lab in (("wall_ms", "wall time per request"), ("prefill_tps", "prompt processing"), ("decode_tps", "generation")):
        if (m or {}).get(k):
            x = m[k]
            L.append("- %s: %s -> %s (ratio %s, %s)" % (lab, _fmt(x["baseline"], k), _fmt(x["candidate"], k), x.get("ratio"), x.get("call")))
    if (m or {}).get("vram_delta_mib") is not None:
        L.append("- VRAM: %+d MiB" % m["vram_delta_mib"])
    if (m or {}).get("outputs"):
        o = m["outputs"]
        L.append("- outputs identical on %s/%s prompts (%s)" % (o.get("identical"), o.get("prompts_compared"), o.get("basis")))
    return L


def improvement_markdown(rec):
    """The report a developer reads: what was found, what was tested, what was kept or rejected, what needs their decision,
    the validated result, and exactly how to apply or revert."""
    res = rec.get("result") or {}
    r = rec["recommended"]
    sp = rec["starting_point"]
    L = ["# hotpath improve: %s" % rec["session"], ""]
    # summary first
    if res.get("validated"):
        head = "Validated improvement. " + (res.get("verdict") or "").split(" [")[0]
    elif res.get("validated") is False:
        head = (res.get("verdict") or "").split(" Validation run:")[0]
        if res.get("not_validated"):
            head += " Not validated (no evidence either way): %s." % "; ".join(res["not_validated"])
    else:
        head = res.get("verdict") or ""
    L += ["**%s**" % head, ""]
    if rec.get("outcomes") and any(rec["outcomes"].values()):
        L += ["## Outcome", "", "| | What | Change | Why |", "|---|---|---|---|"]
        for k in OUTCOME_ORDER:
            for x in rec["outcomes"].get(k) or []:
                L.append("| %s | %s | `%s` | %s%s |" % (OUTCOME_LABEL[k], x["what"], x.get("change") or "-", (x.get("why") or "").replace("not adopted: ", "").replace("|", "/"),
                                                       " (outputs: %s)" % x["fidelity"] if x.get("fidelity") else ""))
        L.append("")
    if rec.get("next_measurement"):
        L += ["Next useful measurement:"] + ["- %s" % x for x in rec["next_measurement"]] + [""]
    L += ["## What was found", "",
          "- Starting configuration: `%s` (%s)" % (" ".join(sp["args"]) or "(no arguments)", sp["source"])]
    lo = sp.get("left_out") or {}
    if lo.get("not_measured"):
        L.append("  - not in the measured launch (hotpath cannot run them; results do not cover them): %s" % ", ".join(lo["not_measured"]))
    if lo.get("plumbing"):
        L.append("  - left out as plumbing (hotpath's server sets its own binding, model file and logging): %s" % ", ".join(lo["plumbing"]))
    rqs = rec.get("requests")
    if rqs:
        L.append("- Your requests: %d from %s in %d class(es) (%s); each step runs %d of them, the validation %d (%d not run by the steps); cache mode %s" % (
            rqs["requests"], rqs["source"], len(rqs["classes"]), ", ".join("%s x%d" % kv for kv in sorted(rqs["classes"].items())), rqs["steps"]["requests"],
            rqs["validation"]["requests"], rqs["validation"]["held_out"], rqs["cache_mode"]))
        if rqs.get("sampled_requests"):
            L.append("  - %d of them sample (temperature above 0): their outputs are not compared token by token" % rqs["sampled_requests"])
    b = rec.get("bottleneck")
    if b:
        L.append("- Biggest bottleneck: %s [%s, %s%s]" % (b["title"], b.get("severity"), b.get("confidence"),
                                                            ", at most %.0f%% of request time" % (100 * b["bound"]) if b.get("bound") is not None else ""))
        for k in b.get("known") or []:
            L.append("  - known: %s" % k)
    L += ["- Workload: %s; budget %d experiments / %.0f min; used %d experiment(s)%s, %.1f min in all" % (
              rec["workload"], rec["budget"]["max_experiments"], rec["budget"]["max_minutes"], rec.get("experiments_run", 0),
              " + baseline" if rec.get("baseline") else "", rec.get("minutes_used", 0.0)),
          "- Stopped: %s" % rec.get("stopped_because"), ""]
    if rec.get("next_step"):
        ns = rec["next_step"]
        L += ["## Plan (dry run, nothing launched)", "",
              "1. %s: %s" % (ns["finding"], ns.get("title")),
              "   - arms: %s; variable: %s" % ("; ".join("%s (%s)" % (a, (ns.get("changes") or {}).get(a) or "as configured") for a in ns["arms"]), ns.get("variable")),
              "   - free VRAM needed: %s MiB; about %s min (rough: every arm at the current configuration's speed, so a faster candidate finishes sooner)" % (
                  ns.get("min_free_vram_mib"), "%.0f" % ns["estimated_minutes"] if ns.get("estimated_minutes") else "?")]
        if ns.get("exceeds_budget"):
            eb = ns["exceeds_budget"]
            L.append("   - **this step needs about %.0f min and the budget leaves %.0f**: run with `--max-minutes %d` (MCP: max_minutes %d) or more to include it "
                     "and the validation" % (eb["needs_minutes"], eb["left_minutes"], int(eb["needs_minutes"] + (rec.get("validation_estimate_minutes") or 0) + 10),
                                             int(eb["needs_minutes"] + (rec.get("validation_estimate_minutes") or 0) + 10)))
        L += [
              "2. Then, re-diagnosed on whatever is adopted and within the budget: %s." % (
                  "; ".join("%s (%s)" % (q["finding"], q["title"]) for q in rec.get("plan_queue") or []) or "no other finding is testable unattended now"),
              "3. If anything was adopted: a validation run of your original launch against the improved one on a broader workload: %s%s." % (
                  rec.get("validation_size") or "more prompts than the steps",
                  " (about %.0f min)" % rec["validation_estimate_minutes"] if rec.get("validation_estimate_minutes") else ""),
              ""] + (["Not run unattended (they need a profile or your evidence, not a launch change):"] +
                     ["- %s: %s" % (q["finding"], q["why"]) for q in rec.get("not_run_unattended") or []] + [""] if rec.get("not_run_unattended") else []) + [
              "Safety: experiments run on hotpath's own llama-server (port %s); your server must be stopped by you if it holds the GPU; nothing is "
                  "applied to it." % ns.get("port", 8099), ""]
    if rec["steps"]:
        L += ["## What was tested", ""]
        for st in rec["steps"]:
            L.append("%d. **%s** (%s): %s" % (st["step"], st["finding"], st["experiment_kind"], st.get("title") or ""))
            why = st["why"][len("not adopted: "):] if st["why"].startswith("not adopted: ") else (st["why"][len("adopted: "):] if st["why"].startswith("adopted: ") else st["why"])
            L.append("   - %s: %s" % ("kept" if st["adopted"] else "not kept", why))
            if st.get("fidelity"):
                L.append("   - outputs: %s" % st["fidelity"])
            if st.get("change"):
                L.append("   - change: %s" % _change_text(st["change"]))
            for o in st.get("other_arms") or []:
                L.append("   - %s %s: %s" % ("" if o["why"].startswith("control arm") else "other arm", o["arm"], o["why"].replace("not adopted: ", "")))
        L.append("")
    if rec.get("decisions"):
        L += ["## Needs your decision", "", "Measured faster, but not adopted because only you can accept the trade-off:", ""]
        for d in rec["decisions"]:
            w = (d.get("measured") or {}).get("wall_ms") or {}
            o = (d.get("measured") or {}).get("outputs") or {}
            same = ("outputs identical on %s/%s prompts" % (o.get("identical"), o.get("prompts_compared"))) if o.get("prompts_compared") else "outputs not compared"
            L.append("- %s, %s (`%s`): wall %s -> %s, %s%s" % ("step %d" % d["step"] if isinstance(d["step"], int) else d["step"], d["arm"], d["change"], _fmt(w.get("baseline"), "wall_ms"),
                                                                  _fmt(w.get("candidate"), "wall_ms"), same,
                                                                  " (%s)" % d["fidelity"] if d.get("fidelity") else ""))
            L.append("  - trade-off: %s" % d["needs"])
            if d.get("decomposition"):
                L.append("  - %s" % d["decomposition"]["reading"])
            if d.get("validation"):
                v = d["validation"]
                vw = ((v.get("measured") or {}).get("wall_ms") or {})
                L.append("  - %s%s%s" % (v.get("summary") or "", "; wall %s -> %s" % (_fmt(vw.get("baseline"), "wall_ms"), _fmt(vw.get("candidate"), "wall_ms")) if vw else "",
                                           "; outputs: %s" % v["fidelity"] if v.get("fidelity") else ""))
            elif isinstance(d.get("step"), int):
                L.append("  - measured on the step workload only; not validated on the broad workload (budget)")
        L += ["", "To accept one, add its change to your launch yourself; `hotpath improve` will then start from it.", ""]
    L += ["## Measured result", ""]
    if res.get("basis"):
        L += ["Basis: %s." % res["basis"], ""]
    L += _metrics_lines(res.get("measured")) or ["- nothing measured end to end"]
    conf = rec.get("confirmation") or {}
    if conf.get("fidelity"):
        f = conf["fidelity"]
        L.append("- output fidelity in the validation run: %s (%s; %s)" % (f.get("class"), f.get("why"), f.get("basis")))
    for c, m in sorted((conf.get("classes") or {}).items()):
        w, p_, d_ = m.get("wall_ms") or {}, m.get("prefill_tps") or {}, m.get("decode_tps") or {}
        L.append("  - %s (%d requests): wall ratio %s (worst request %s), prefill %s, decode %s" % (
            c, m.get("requests") or 0, w.get("ratio"), w.get("worst_request_ratio"), p_.get("ratio"), d_.get("ratio")))
    fa = conf.get("fields") or {}
    if fa.get("decisions_judged"):
        L.append("- decision fields (%s): equal on %d of %d requests (%s)" % (", ".join(fa.get("decision_fields") or []), fa["decisions_equal"], fa["decisions_judged"],
                                                                          fa.get("decision_basis")))
    L += ["", "## Apply or revert", ""]
    if r["changes"] or r["env"]:
        for c in r["changes"]:
            L.append("- %s: %s -> %s" % (c["flag"], c["from"], c["to"]))
        for e in r["env"]:
            L.append("- environment %s: %s" % (e["variable"], e["to"]))
        L += ["", "Recommended launch (%s):" % r["basis"], "```", " ".join(r["args"]), "```", "Revert to your original launch:", "```",
              " ".join(r.get("revert_args") or []), "```"]
    else:
        L += ["Your launch stays as it is:", "```", " ".join(r.get("revert_args") or r["args"]), "```"]
    L += ["", r["note"]]
    if rec.get("skipped_findings"):
        L += ["", "## Not tested", ""] + ["- %s: %s" % (x["finding"], x["why_not_run"]) for x in rec["skipped_findings"]]
    if rec.get("remaining"):
        L += ["", "## Still open", ""] + ["- %s (%s): %s" % (f["id"], f.get("confidence"), f.get("title")) for f in rec["remaining"]]
    L += ["", "A verdict describes this workload on this machine. Identical greedy tokens on these prompts are evidence, not proof, that other prompts "
              "behave the same; output identity is not a check of what your application decides from an output."]
    return "\n".join(L) + "\n"


def _fmt(v, k):
    if v is None:
        return "?"
    return ("%.0f ms" % v) if k == "wall_ms" else ("%.1f t/s" % v)


def _change_text(c):
    parts = ["%s %s" % (k, v) for k, v in c.get("args_override", {}).items()] + ["remove %s" % k for k in c.get("args_remove", [])]
    parts += ["%s=%s" % (k, v) for k, v in c.get("env_override", {}).items()] + ["unset %s" % k for k in c.get("env_remove", [])]
    return ", ".join(parts) or "none"
