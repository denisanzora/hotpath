"""hotpath propose: turn a diagnosis finding into an experiment spec for the existing runner.

The spec changes one variable between arms and keeps everything else identical: same binary, model,
common arguments, workload, interleaved rounds, preconditions. Paths are `${key}` references into
hotpath.local.json, so the spec can be shared. The baseline arm is the configuration the diagnosis
described (the first arm of the diagnosed experiment, or the launch arguments you pass / configure),
so the comparison answers "does this change help *my* current setup".
"""
import copy
import json
from pathlib import Path

from hotpath.records import load_record

DEFAULT_COMMON_ARGS = ["--ctx-size", "4096", "--gpu-layers", "99", "--batch-size", "512", "--ubatch-size", "512", "--parallel", "1", "--slots", "--no-webui"]


def _set_arg(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]
    return args


def baseline_args(diagnosis_record, cfg, override=None):
    """Common arguments for the spec: explicit override > cfg['server_args'] > the diagnosed experiment's arm args > defaults."""
    if override:
        return list(override), "command line"
    if cfg.get("server_args"):
        return list(cfg["server_args"]), "hotpath.local.json server_args"
    arm = (diagnosis_record or {}).get("baseline_args")
    if arm:
        return list(arm), "diagnosed experiment arm"
    return list(DEFAULT_COMMON_ARGS), "hotpath defaults (adapt --ctx-size and --gpu-layers to your model)"


def _strip_launch_prefix(args):
    """Experiment records store argv as [binary, --model, model, ...]; the spec wants only what follows."""
    args = list(args)
    if len(args) >= 3 and args[1] == "--model":
        return args[3:]
    return args


def build_spec(finding, cfg, common_args, workload, port=8099, rounds=3, repetitions=2, warmup=1, n_predict=None, name=None):
    exp = finding.get("experiment") or {}
    if not isinstance(exp.get("arms"), list):
        raise ValueError("finding %s has no arm list to propose from (its experiment is a procedure, not a spec): %s" % (finding.get("id"), exp.get("arms")))
    kind = exp.get("kind")
    args = list(common_args)
    arms = []
    for a in exp["arms"]:
        arm = {"name": a["name"]}
        if a.get("args_override"):
            arm["args_override"] = a["args_override"]
            # a larger microbatch needs --batch-size at least as large; keep every other argument identical
            if kind == "ubatch_sweep":
                ub = int(a["args_override"].get("--ubatch-size", 0))
                nb = int(dict(zip(args[::2], args[1::2])).get("--batch-size", 0) or 0)
                if ub and nb and nb < ub:
                    arm["args_override"] = dict(arm["args_override"], **{"--batch-size": ub})
        for k in ("env_override", "env_remove", "args_remove", "request_override"):
            if a.get(k):
                arm[k] = a[k]
        arms.append(arm)
    if kind == "request_shape" and not str(workload).endswith(".json"):
        raise ValueError("a request_shape experiment needs a `requests` workload (hotpath workload --from-requests FILE)")
    spec = {
        "name": name or "%s-%s" % (kind, finding["id"]),
        "description": "Proposed by hotpath diagnose from finding '%s': %s. One variable (%s) differs between arms; everything else is identical. "
                       "Run with --dry-run first. Metric: %s." % (finding["id"], finding.get("title"), exp.get("variable"), exp.get("metric")),
        "binary": "${runtime_binary}", "model": "${model_path}", "port": int(port),
        "common_args": args, "env": {}, "arms": arms, "workload": workload,
        "rounds": int(rounds), "repetitions": int(repetitions), "warmup": int(warmup), "cooldown_s": 5, "health_timeout_s": 600,
        "preconditions": {"sample_seconds": 5, "max_gpu_util_pct": 10, "min_free_vram_mib": 2000, "ports_must_be_free": [int(port)]},
        "provenance": {"finding": finding["id"], "experiment_kind": kind, "note": exp.get("note")},
    }
    if cfg.get("path_prepend"):
        spec["path_prepend"] = "${path_prepend}"
    if n_predict:
        spec["n_predict"] = int(n_predict)
    return spec


def propose(diagnosis_path, finding_id, cfg, workload, out_path, common_args=None, port=8099, rounds=3, repetitions=2, warmup=1, n_predict=None):
    d = load_record(diagnosis_path)
    findings = {f["id"]: f for f in d.get("findings", [])}
    if finding_id is None:
        rec = d.get("recommended_next_experiment")
        if not rec:
            raise ValueError("the diagnosis has no recommended experiment; pass --finding ID")
        finding_id = rec["finding"]
    if finding_id not in findings:
        raise ValueError("no finding %r in the diagnosis; available: %s" % (finding_id, sorted(findings)))
    args, source = baseline_args(d, cfg, common_args)
    spec = build_spec(findings[finding_id], cfg, _strip_launch_prefix(args), workload, port=port, rounds=rounds, repetitions=repetitions, warmup=warmup, n_predict=n_predict)
    spec["provenance"]["common_args_source"] = source
    spec["provenance"]["diagnosis"] = Path(diagnosis_path).name
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec, source


def spec_for_display(spec):
    s = copy.deepcopy(spec)
    return json.dumps(s, indent=2)
