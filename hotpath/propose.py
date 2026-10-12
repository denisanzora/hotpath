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
    from hotpath import launch
    return launch.set_arg(args, flag, value)


_PATH_SETTINGS = ("model", "log_file", "model_draft", "host", "port")


def _speed_settings(args):
    """The settings of an argv that matter for a comparison (paths, host and port excluded)."""
    from hotpath import launch
    s = launch.parse(args)["settings"]
    return {k: str(v) for k, v in s.items() if k not in _PATH_SETTINGS}


def baseline_args(diagnosis_record, cfg, override=None):
    """Common arguments for the spec, so the baseline arm is the configuration that was diagnosed:
    explicit override > the arm the last verdict found faster > the diagnosed server's observed command line (or server_args in the config when it matches it, since
    the config keeps full paths) > server_args > the diagnosed experiment's arm args > defaults."""
    if override:
        return list(override), "command line"
    d = diagnosis_record or {}
    if d.get("baseline_from_verdict") and d.get("baseline_args"):
        # the improve step: the next experiment starts from the configuration the last verdict found faster
        return list(d["baseline_args"]), d["baseline_from_verdict"]
    obs = d.get("observed_launch") or {}
    if obs.get("args_redacted"):
        from hotpath import launch
        if cfg.get("server_args") and _speed_settings(cfg["server_args"]) == _speed_settings(obs["args_redacted"]):
            return list(cfg["server_args"]), "hotpath.local.json server_args (same settings as the observed command line, %s)" % obs.get("source")
        kept, dropped = launch.drop_path_args(obs["args_redacted"])
        src = "the diagnosed server's observed command line (%s)" % obs.get("source")
        if cfg.get("server_args"):
            src += "; server_args in hotpath.local.json differ from it and were not used"
        if [d for d in dropped if d not in RUNNER_OWNED_FLAGS]:
            src += "; arguments with file paths left out (records keep only basenames): %s; put the full arguments in server_args to keep them" % ", ".join(
                d for d in dropped if d not in RUNNER_OWNED_FLAGS)
        return kept, src
    if cfg.get("server_args"):
        return list(cfg["server_args"]), "hotpath.local.json server_args"
    arm = (diagnosis_record or {}).get("baseline_args")
    if arm:
        return list(arm), "diagnosed experiment arm"
    return list(DEFAULT_COMMON_ARGS), "hotpath defaults (adapt --ctx-size and --gpu-layers to your model)"


COMPUTE_ALLOWANCE_MIB = 512


def vram_needed_mib(cfg, args):
    """Free VRAM an experiment arm needs before launch, from the GGUF header: the weights on the GPU (+10%), the KV cache at
    the arm's context size and KV types, and a compute-buffer allowance; None when the model file is unknown (then the
    spec keeps a generic floor). An estimate for the precondition only; the arm's real footprint is measured per arm."""
    from hotpath import gguf
    mp = cfg.get("model_path")
    if not mp or not Path(mp).is_file():
        return None
    try:
        s = gguf.summarize(gguf.read_header(mp))
    except (OSError, ValueError, KeyError):
        return None
    from hotpath.launch import int_or_none, get
    from hotpath.diagnose import KV_TYPE_FACTOR
    ngl = int_or_none(get(args, "n_gpu_layers"))
    frac = 1.0 if ngl is None or not s.get("n_layer") else min(1.0, max(0, ngl) / (s["n_layer"] + 1))
    need = s["total_bytes"] / (1 << 20) * frac * 1.1
    ctx = int_or_none(get(args, "ctx_size")) or 4096
    if s.get("kv_bytes_per_token_f16") and ctx > 0:
        fk = KV_TYPE_FACTOR.get(str(get(args, "cache_type_k") or "f16").lower(), 1.0)
        fv = KV_TYPE_FACTOR.get(str(get(args, "cache_type_v") or "f16").lower(), 1.0)
        need += s["kv_bytes_per_token_f16"] * ctx * (fk + fv) / 2 / (1 << 20) * frac
    if frac > 0:
        need += COMPUTE_ALLOWANCE_MIB
    return int(need)


# llama-server arguments the experiment runner owns or that must never be shared with the user's own server: it adds
# --model, --host 127.0.0.1 and --port itself (a copied --port could take over the user's production port), and an
# inherited log file, slot directory or credential would make an experiment server write to or authenticate like the
# production one. The MCP server refuses the same flags in agent-authored specs.
RUNNER_OWNED_FLAGS = frozenset({
    "--host", "--port", "--model", "-m", "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file", "--path", "--media-path", "--webui-config-file",
    "--log-file", "--slot-save-path", "--prompt-cache", "--prompt-cache-all", "--prompt-cache-ro",
    "--hf-repo", "-hf", "-hfr", "--hf-file", "-hff", "--hf-repo-draft", "-hfd", "--hf-token", "-hft", "--model-url", "-mu", "--docker-repo", "-dr",
    "--model-draft", "-md", "--lora", "--lora-scaled", "--lora-init-without-apply", "--control-vector", "--control-vector-scaled",
    "--mmproj", "-mm", "--mmproj-url", "-mmu",
})


def strip_runner_owned(args):
    """Drop runner-owned flags (and their values) from copied server arguments. Returns (kept_args, removed_flag_names);
    only flag names are reported, since their values are often private paths."""
    from hotpath import launch
    args = [launch.canonical(a) for a in args]
    kept, removed, i = [], [], 0
    while i < len(args):
        flag = args[i].split("=", 1)[0]
        if flag in RUNNER_OWNED_FLAGS:
            removed.append(flag)
            has_value = "=" not in args[i] and i + 1 < len(args) and not args[i + 1].startswith("-")
            i += 2 if has_value else 1
            continue
        kept.append(args[i])
        i += 1
    return kept, removed


def _is_requests_workload(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("kind") == "requests"
    except (OSError, ValueError, AttributeError):
        return False


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
                from hotpath import launch
                ub = int(a["args_override"].get("--ubatch-size", 0))
                nb = launch.int_or_none(launch.get(args, "batch_size")) or 0
                if ub and nb and nb < ub:
                    arm["args_override"] = dict(arm["args_override"], **{"--batch-size": ub})
        for k in ("env_override", "env_remove", "args_remove", "request_override", "capacity_change", "context_policy", "control"):
            if a.get(k):
                arm[k] = a[k]
        arms.append(arm)
    if kind == "request_shape" and not _is_requests_workload(workload):
        raise ValueError("a request_shape experiment needs a `requests` workload built from your captured request bodies "
                         "(hotpath workload OUT --from-requests FILE); %r is not one, and the arms would run identical requests" % str(workload))
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


LAUNCH_S_ASSUMED = 30


def estimate_minutes(spec, rates, load_s=None):
    """Rough duration of a run from the diagnosed configuration's own prefill/decode rates and the workload's size, for
    planning (and for an agent's polling); every arm is assumed to run at the baseline's rates, so a faster candidate
    finishes sooner. Model load time is an assumption. None when rates or the workload are unknown."""
    from hotpath.workload import load_workload
    if not rates or not rates.get("prefill_tps") or not rates.get("decode_tps"):
        return None
    try:
        w = load_workload(spec["workload"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if w.get("kind") == "requests":
        sizes = [(len(json.dumps(it["body"].get("messages") or it["body"].get("prompt") or "")) / 4.0,
                  it["body"].get("max_tokens") or it["body"].get("n_predict") or 256) for it in w["items"]]
    else:
        n_predict = spec.get("n_predict") or w.get("n_predict") or 64
        sizes = [(1.2 * (p.get("words") or len(p["text"].split())) + 15, n_predict) for p in w["prompts"]]
    per_pass = sum(t / rates["prefill_tps"] + n / rates["decode_tps"] + 0.2 for t, n in sizes)
    load = float(load_s) if load_s else LAUNCH_S_ASSUMED
    per_launch = (int(spec.get("warmup", 1)) + int(spec.get("repetitions", 2))) * per_pass + load + float(spec.get("cooldown_s", 5))
    launches = int(spec.get("rounds", 3)) * len(spec["arms"])
    return {"minutes": round(launches * per_launch / 60.0, 1),
            "basis": "baseline rates %.0f/%.1f t/s prefill/decode (%s) for every arm, prompt tokens estimated from the workload, %s" % (
                rates["prefill_tps"], rates["decode_tps"], rates.get("basis"),
                "%.0f s per model load (measured)" % load if load_s else "%d s per model load assumed" % LAUNCH_S_ASSUMED)}


def _arm_args(common, arm):
    args = list(common)
    for k, v in (arm.get("args_override") or {}).items():
        args = _set_arg(args, k, v)
    return args


def check_output_path(out_path, force=False):
    """A spec is written to a new .json file: never over an existing file unless forced, and never over a local config."""
    out = Path(out_path)
    if out.suffix.lower() != ".json":
        raise ValueError("the spec path must end in .json (got %s)" % out.name)
    if out.name.lower().endswith(".local.json"):
        raise ValueError("refusing to write a spec over %s: *.local.json files are local configuration" % out.name)
    if out.exists() and not force:
        raise ValueError("%s already exists; choose another path (or pass --force from the CLI to replace it)" % out.name)
    return out


def propose(diagnosis_path, finding_id, cfg, workload, out_path, common_args=None, port=8099, rounds=3, repetitions=2, warmup=1, n_predict=None, force=False):
    out = check_output_path(out_path, force)
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
    args, removed = strip_runner_owned(_strip_launch_prefix(args))
    spec = build_spec(findings[finding_id], cfg, args, workload, port=port, rounds=rounds, repetitions=repetitions, warmup=warmup, n_predict=n_predict)
    spec["provenance"]["common_args_source"] = source
    # the largest arm decides how much free VRAM a launch needs; a generic floor when the model is unknown
    need = [vram_needed_mib(cfg, _arm_args(args, a)) for a in spec["arms"]]
    if all(n is not None for n in need) and need:
        spec["preconditions"]["min_free_vram_mib"] = max(need)
        spec["provenance"]["min_free_vram_basis"] = "weights on the GPU + 10%, KV cache at the arm's context and KV types, 512 MiB compute, for the largest arm (estimate from the GGUF header)"
    if removed:
        spec["provenance"]["removed_server_args"] = sorted(set(removed))
    spec["provenance"]["diagnosis"] = Path(diagnosis_path).name
    est = estimate_minutes(spec, d.get("rates"))
    if est:
        spec["provenance"]["estimated_duration"] = est
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec, source


def spec_for_display(spec):
    s = copy.deepcopy(spec)
    return json.dumps(s, indent=2)
