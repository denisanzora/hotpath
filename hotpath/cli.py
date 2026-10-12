"""Command line: init, tools, observe, triage, diagnose, propose, gguf, workload, measure, experiment, compare, bench, profile, kernels."""
import argparse
import json
import os
import sys
from pathlib import Path

from hotpath import __version__
from hotpath import adapters
from hotpath import gguf as gguf_mod
from hotpath import compare as compare_mod
from hotpath import experiment as exp_mod
from hotpath import measure as measure_mod
from hotpath import observe as observe_mod
from hotpath import triage as triage_mod
from hotpath import workload as wl_mod
from hotpath.config import DEFAULT_NAME, load_local
from hotpath.records import load_record, scrub_text, write_record


def _build_dir(cfg):
    return cfg.get("build_dir") or cfg.get("fork_build_dir")


def _require(value, what, flag, cfg_key):
    if not value:
        raise SystemExit("error: no %s given. Pass %s or set %s in hotpath.local.json (see `hotpath init`)." % (what, flag, cfg_key))
    if not Path(value).exists():
        raise SystemExit("error: %s does not exist: %s" % (what, value))


def _out(args):
    return args.out or "results/local"


def _write(rec, args, stem, md=None):
    path = write_record(rec, _out(args), stem)
    print("wrote %s" % path)
    if md:
        mdp = Path(str(path)[:-5] + ".md")
        mdp.write_text(scrub_text(md), encoding="utf-8")
        print("wrote %s" % mdp)
    return path


def cmd_observe(args):
    cfg = load_local(args.local)
    rec = observe_mod.observe(cfg, log_path=args.log, model_path=args.model, binary_path=args.binary,
                              base_url=args.server, sample_seconds=args.sample_seconds)
    md = observe_mod.observation_markdown(rec)
    print(md)
    _write(rec, args, "observation", md)
    return 0


def cmd_triage(args):
    obs = load_record(args.observation)
    rep = triage_mod.triage(obs, bandwidth_gbs=args.bandwidth_gbs)
    rec = {"kind": "triage", "evidence_class": "passive", "source_observation": Path(args.observation).name, **rep}
    md = triage_mod.triage_markdown(rep)
    print(md)
    _write(rec, args, "triage", md)
    return 0


def cmd_diagnose(args):
    """Ranked bottleneck findings from any mix of records; verdicts for finished experiments."""
    from hotpath import diagnose as diag_mod
    recs = [load_record(p) for p in args.records]
    cfg = load_local(args.local)
    d = diag_mod.diagnose(recs, bandwidth_gbs=args.bandwidth_gbs, cfg=None if cfg.get("_missing") else cfg)
    d["source_files"] = [Path(p).name for p in args.records]
    md = diag_mod.diagnosis_markdown(d)
    print(md)
    _write(d, args, "diagnosis", md)
    return 0


def cmd_propose(args):
    """Write an experiment spec for a diagnosis finding (one variable between arms), then show its dry-run plan."""
    from hotpath import propose as prop_mod
    cfg = load_local(args.local)
    workload = args.workload or cfg.get("workload")
    if not workload:
        workload = str(Path("workloads") / "hotpath-default.json")
        if not Path(workload).exists():
            w = wl_mod.build_workload("hotpath-default", 3, [450, 900, 1300], seed=1, n_predict=64, fmt="chatml")
            wl_mod.save_workload(w, workload)
            print("no workload given; wrote a default synthetic one: %s (3 ChatML prompts of 450/900/1300 words, n_predict 64)" % workload)
    try:
        spec, source = prop_mod.propose(args.diagnosis, args.finding, cfg, workload, args.out, common_args=args.common_args, port=args.port,
                                        rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup, n_predict=args.n_predict, force=args.force)
    except ValueError as e:
        raise SystemExit("error: %s" % e)
    print("wrote %s (baseline arguments from: %s)" % (args.out, source))
    if spec["provenance"].get("removed_server_args"):
        print("left out of the copied server arguments (the runner supplies its own, so the experiment cannot reuse your server's port, log, "
              "slot directory or model flags): %s" % ", ".join(spec["provenance"]["removed_server_args"]))
    print("")
    if cfg.get("_missing"):
        print("hotpath.local.json not found: the spec references ${runtime_binary} and ${model_path}; create the config before running it.")
        return 0
    try:
        resolved = exp_mod.load_spec(args.out, cfg)
        print(exp_mod.plan(resolved))
    except SystemExit as e:
        print("dry run not possible yet: %s" % e)
    print("")
    print("Next: review the plan, then `hotpath experiment %s` (ACTIVE: launches servers), then `hotpath diagnose results/local/experiment-%s-*.json`." % (args.out, spec["name"]))
    return 0


def cmd_gguf(args):
    hdr = gguf_mod.read_header(args.path)
    s = gguf_mod.summarize(hdr)
    if args.bandwidth_gbs:
        s["decode_ceiling_tps_at_bandwidth"] = {"bandwidth_gbs": args.bandwidth_gbs,
                                               "n_past_0": gguf_mod.decode_ceiling_tps(s, args.bandwidth_gbs, 0),
                                               "n_past_2048": gguf_mod.decode_ceiling_tps(s, args.bandwidth_gbs, 2048)}
    if args.tensors:
        s["tensors"] = [{k: t[k] for k in ("name", "dims", "type", "bytes")} for t in hdr["tensors"]]
    print(json.dumps(s, indent=1))
    return 0


def cmd_workload(args):
    if args.from_requests:
        try:
            w = wl_mod.workload_from_requests(args.from_requests, name=args.name if args.name != "synthetic" else "requests", limit=args.limit)
        except ValueError as e:
            raise SystemExit("error: %s" % e)
        wl_mod.save_workload(w, args.out_file)
        ident = wl_mod.workload_identity(w)
        print("wrote %s: %d application request bodies (%s), fields %s" % (args.out_file, ident["n_prompts"], ", ".join(ident["endpoints"]), ident["fields_present"]))
        print("The workload file holds your request bodies; keep it local (workloads/ is gitignored here). Records store hashes, counts and timings only, "
              "unless the spec sets store_outputs.")
        return 0
    w = wl_mod.build_workload(args.name, args.n_prompts, args.words, seed=args.seed, n_predict=args.n_predict, fmt=args.format)
    wl_mod.save_workload(w, args.out_file)
    print("wrote %s: %d prompts, word counts %s, n_predict %d, format %s" % (args.out_file, len(w["prompts"]), args.words, args.n_predict, args.format))
    return 0


def cmd_measure(args):
    cfg = load_local(args.local)
    w = wl_mod.load_workload(args.workload)
    if args.n_predict:
        w["n_predict"] = args.n_predict
    rec = measure_mod.measure(cfg, args.server, w, args.repetitions, args.warmup, args.i_own_this_server,
                              background_seconds=args.background_seconds, store_prompts=args.store_prompts, label=args.label)
    print(json.dumps(rec["summary"], indent=1))
    _write(rec, args, "measurement-" + (args.label or "run"))
    return 0


def cmd_experiment(args):
    cfg = load_local(args.local)
    spec = exp_mod.load_spec(args.spec, cfg)
    if not args.dry_run:
        _require(spec.get("binary"), "runtime binary", "'binary' in the spec", "runtime_binary")
        _require(spec.get("model"), "model file", "'model' in the spec", "model_path")
        _require(spec.get("workload"), "workload file", "'workload' in the spec", "-")
        for a in spec["arms"]:
            for k in ("binary", "model"):
                if a.get(k):
                    _require(a[k], "arm %s %s" % (a["name"], k), k, "-")
    rec = exp_mod.run(spec, _out(args), dry_run=args.dry_run, wait_for_preconditions_s=args.wait_s)
    if rec is None:
        return 0
    rep = compare_mod.compare([rec])
    rec["comparison"] = rep
    md = compare_mod.compare_markdown(rep)
    print(md)
    _write(rec, args, "experiment-" + spec["name"], md)
    failed = [n for n, a in rec["arms"].items() if a["failures"]]
    if failed:
        print("warning: arms with failed launches: %s (see the server logs next to the results)" % ", ".join(failed), file=sys.stderr)
        return 1
    return 0


def cmd_compare(args):
    recs = [load_record(p) for p in args.records]
    rep = compare_mod.compare(recs)
    md = compare_mod.compare_markdown(rep)
    print(md)
    _write({"kind": "comparison", "sources": [Path(p).name for p in args.records], **rep}, args, "comparison", md)
    return 0


def _roots(args, cfg):
    from hotpath import discover as disc
    roots = []
    for r in list(getattr(args, "root", None) or []) + list(cfg.get("discovery_roots") or []):
        try:
            roots.append(disc.check_root(r))
        except ValueError as e:
            raise SystemExit("error: %s" % e)
    return roots


def _discover(args, cfg, prefer=None):
    from hotpath import discover as disc
    from hotpath.records import new_record
    rec = disc.discover(None if cfg.get("_missing") else cfg, roots=_roots(args, cfg), run_found=not getattr(args, "no_exec", False),
                        run_in_roots=not getattr(args, "no_exec_roots", False), max_depth=getattr(args, "max_depth", None) or disc.MAX_DEPTH, prefer=prefer)
    return rec, new_record("discovery", "passive", **rec)


def cmd_discover(args):
    """Find the local inference setup (running servers, installs, models, GPU, tools, logs) without changing anything."""
    from hotpath import discover as disc
    cfg = load_local(args.local)
    rec, full = _discover(args, cfg)
    md = disc.discovery_markdown(rec)
    print(md)
    _write(full, args, "discovery", md)
    print("")
    if cfg.get("_missing"):
        print("Next: `hotpath init` writes the suggested values to hotpath.local.json%s." % (
            " (choose among several candidates with --binary/--model/--server)" if rec["ambiguous"] else ""))
    if disc.healthy_user_servers(rec) or cfg.get("server_url"):
        print("Next: `hotpath inspect` diagnoses the running server from its command line, log and /metrics (passive).")
    return 0


def cmd_init(args):
    """Write hotpath.local.json from discovery (refuses to overwrite; --update adds only missing keys; --template writes placeholders)."""
    target = Path(args.path or DEFAULT_NAME)
    if args.template:
        if target.exists():
            print("%s already exists; not overwriting" % target)
            return 1
        template = {"server_url": "http://127.0.0.1:8080", "model_path": "/path/to/model.gguf", "runtime_binary": "/path/to/llama-server",
                    "runtime_bin_dir": "/path/to/llama.cpp/build/bin", "build_dir": "/path/to/llama.cpp/build", "owned_servers": ["http://127.0.0.1:8099"]}
        target.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
        print("wrote %s; edit the paths, then run: hotpath tools" % target)
        return 0
    if target.exists() and not args.update:
        print("%s already exists; not overwriting. `hotpath init --update` adds discovered keys it does not have yet (existing values are kept)." % target)
        return 1
    if target.name.lower() != DEFAULT_NAME and not target.name.lower().endswith(".local.json"):
        raise SystemExit("error: the config file name must end in .local.json (it holds local paths and is gitignored by that pattern)")
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    cfg = dict(existing, _path=str(target)) if existing else {"_missing": True}
    prefer = {k: v for k, v in (("runtime_binary", args.binary), ("model_path", args.model), ("server_url", args.server)) if v}
    try:
        rec, full = _discover(args, cfg, prefer)
    except ValueError as e:
        raise SystemExit("error: %s" % e)
    from hotpath import discover as disc
    _write(full, args, "discovery", disc.discovery_markdown(rec))
    sug, src = rec["suggested_config"], rec["suggested_config_sources"]
    if args.root:
        sug["discovery_roots"] = [str(r) for r in _roots(args, {})]
        src["discovery_roots"] = "--root (searched again by later discover/inspect runs)"
    added = {k: v for k, v in sug.items() if k not in existing}
    data = dict(existing) if existing else {"_comment": "Written by `hotpath init` from discovery; every value says where it came from in the init output. "
                                                         "Keep this file out of version control: it holds local paths."}
    data.update(added)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("%s %s" % ("updated" if existing else "wrote", target))
    for k, v in added.items():
        print("  %-16s %s\n  %-16s   (%s)" % (k, " ".join(v) if k == "server_args" else v, "", src.get(k)))
    if existing and not added:
        print("  nothing to add: every discovered key is already set (existing values are never changed)")
    for k, v in rec["ambiguous"].items():
        if k not in existing:
            flag = {"runtime_binary": "--binary", "model_path": "--model", "server_url": "--server"}[k]
            print("  %-16s not set: several candidates; choose with `hotpath init %s%s PATH`:\n%s" % (
                k, "--update " if existing else "", flag, "\n".join("    - %s" % c for c in v)))
    for x in rec["issues"]:
        if x["severity"] == "high":
            print("  problem: %s: %s. Fix: %s" % (x["where"], x["what"], x["fix"]))
    missing = [k for k in ("runtime_binary", "model_path") if k not in data]
    if missing:
        print("  still missing: %s. Pass --binary/--model, or --root DIR to search where they live." % ", ".join(missing))
    print("Next: `hotpath inspect` (passive) diagnoses the running server; `hotpath tools` checks the external tools.")
    return 0


def cmd_inspect(args):
    """Passive, one step: discover the setup, observe the running (or configured) server with its log and /metrics, diagnose."""
    from hotpath import discover as disc
    from hotpath import diagnose as diag_mod
    cfg = load_local(args.local)
    rec, full = _discover(args, cfg)
    _write(full, args, "discovery", disc.discovery_markdown(rec))
    print(disc.discovery_brief(rec))
    live = disc.healthy_user_servers(rec)
    base = args.server or cfg.get("server_url") or (live[0]["base_url"] if len(live) == 1 else None)
    if not base and len(live) > 1:
        raise SystemExit("several llama-servers are running (%s); choose one with --server" % ", ".join(s["base_url"] for s in live))
    if not (base or args.log or cfg.get("server_log")):
        print("\nNo running llama-server and no log to read: start your server (with --log-file and --metrics for the most evidence) "
              "or pass --log LOG, then run `hotpath inspect` again.")
        return 0
    obs = observe_mod.observe(cfg, log_path=args.log, base_url=base, sample_seconds=args.sample_seconds)
    opath = _write(obs, args, "observation", observe_mod.observation_markdown(obs))
    d = diag_mod.diagnose([obs], bandwidth_gbs=args.bandwidth_gbs, cfg=None if cfg.get("_missing") else cfg)
    d["source_files"] = [Path(str(opath)).name]
    dpath = _write(d, args, "diagnosis", diag_mod.diagnosis_markdown(d))
    print("")
    print(diag_mod.diagnosis_brief(d, str(dpath), rec))
    return 0


def cmd_mcp(args):
    """Serve hotpath as MCP tools over stdio (see docs/agents.md)."""
    from hotpath import mcp_server
    mcp_server.main()
    return 0


def cmd_tools(args):
    """Environment check: Python, nvidia-smi, local config, and the external tools hotpath wraps."""
    import platform
    from hotpath import gpu as gpu_mod
    cfg = load_local(args.local)
    print("hotpath %s, Python %s, %s" % (__version__, platform.python_version(), platform.platform()))
    if cfg.get("_missing"):
        print("config:            not found (%s). Run `hotpath init` to create one; commands that need paths will ask for them." % cfg["_path"])
    else:
        print("config:            %s" % cfg["_path"])
    g = gpu_mod.gpu_static()
    if g:
        print("nvidia-smi         found=True  %s, driver %s, %s MiB, compute capability %s" % (
            g.get("name"), g.get("driver_version"), g.get("memory.total"), g.get("compute_cap")))
    else:
        print("nvidia-smi         found=False install the NVIDIA driver; GPU facts and background-load checks need it")
    st = adapters.all_status(bin_dir=cfg.get("runtime_bin_dir"), build_dir=_build_dir(cfg))
    for s in st:
        print("%-18s found=%-5s %s" % (s["tool"], s["found"], s["path"] or s["install_hint"]))
    from hotpath.measure import schema_validator
    print("%-18s found=%-5s %s" % ("jsonschema", bool(schema_validator()),
                                   "full JSON Schema validation of request-shape outputs" if schema_validator() else "optional: pip install \"hotpath[schema]\" (otherwise only required top-level keys are checked)"))
    return 0


def cmd_bench(args):
    """llama-bench wrapper: whole-model pp/tg with repetitions; refuses under background load."""
    cfg = load_local(args.local)
    from hotpath import gpu as gpu_mod
    ad = adapters.LlamaBenchAdapter(cfg.get("runtime_bin_dir"))
    if not ad.locate():
        print(ad.status())
        return 2
    bg = gpu_mod.sample_background(seconds=args.background_seconds, interval=1.0)
    if (bg.get("util_pct_max") or 0) > args.max_background_util:
        print("refusing: background GPU utilization max %.0f%% > %s" % (bg["util_pct_max"], args.max_background_util))
        return 3
    _require(args.model or cfg.get("model_path"), "model file", "--model", "model_path")
    cmd = ad.recipe(args.model or cfg.get("model_path"), ngl=args.ngl, ub=args.ub, pp=args.pp, tg=args.tg, reps=args.reps)
    print(" ".join(cmd))
    res = adapters.run_recipe(cmd, path_prepend=cfg.get("path_prepend"))
    from hotpath.records import new_record, file_identity
    rec = new_record("bench", "active", background_before=bg, gpu=gpu_mod.gpu_static(), model=file_identity(args.model or cfg.get("model_path")),
                     binary=observe_mod.binary_identity(str(ad.locate()), cfg.get("path_prepend")), result=res)
    try:
        rec["parsed"] = json.loads(res["stdout"])
    except ValueError:
        rec["parsed"] = None
    print(res["stdout"][-3000:])
    if res["returncode"] != 0:
        print("llama-bench exited with code %s. %s" % (res["returncode"], res["stderr"][-500:].strip() or
              "No output: on Windows a missing CUDA runtime DLL does this; add the CUDA bin directory to path_prepend in hotpath.local.json."),
              file=sys.stderr)
    _write(rec, args, "bench")
    return 0 if res["returncode"] == 0 else 1


def cmd_profile(args):
    """nsys kernel-share profile of a prefill-only or decode-only llama-bench run (not a throughput number)."""
    from hotpath import profile as prof_mod
    cfg = load_local(args.local)
    bin_dir = args.bin_dir or cfg.get("runtime_bin_dir")
    model = args.model or cfg.get("model_path")
    _require(bin_dir, "runtime bin directory", "--bin-dir", "runtime_bin_dir")
    _require(model, "model file", "--model", "model_path")
    rec = prof_mod.profile(bin_dir, model, args.label, _out(args), pp=args.pp, tg=args.tg, ngl=args.ngl, ub=args.ub,
                           path_prepend=cfg.get("path_prepend"))
    md = prof_mod.profile_markdown(rec)
    print(md)
    _write(rec, args, "profile-" + args.label, md)
    return 0 if not rec.get("error") else 1


def cmd_kernels(args):
    """test-backend-ops on a build dir: 'test' (NMSE vs CPU) or 'perf' for ops matching a filter."""
    from hotpath import kernels as k_mod
    cfg = load_local(args.local)
    build_dir = args.build_dir or _build_dir(cfg)
    _require(build_dir, "llama.cpp build directory", "--build-dir", "build_dir")
    rec = k_mod.run_tbo(build_dir, args.mode, args.op, params_regex=args.params, backend=args.backend,
                       path_prepend=cfg.get("path_prepend"))
    rec["label"] = args.label
    md = k_mod.kernels_markdown(rec)
    print(md)
    _write(rec, args, "kernels-%s-%s" % (args.mode, args.label or "run"), md)
    return 0 if rec["returncode"] == 0 else 1


def build_parser():
    p = argparse.ArgumentParser(prog="hotpath", description="find real GPU inference bottlenecks, run fair experiments, keep the evidence")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--local", help="local deployment config (default hotpath.local.json)")
    p.add_argument("--out", help="results directory (default results/local)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("observe", help="passive baseline: server facts, launcher settings, GPU state, log history (no inference)")
    s.add_argument("--server"); s.add_argument("--log"); s.add_argument("--model"); s.add_argument("--binary")
    s.add_argument("--sample-seconds", type=float, default=5.0)
    s.set_defaults(fn=cmd_observe)

    s = sub.add_parser("triage", help="attribution report from an observation record")
    s.add_argument("observation"); s.add_argument("--bandwidth-gbs", type=float)
    s.set_defaults(fn=cmd_triage)

    s = sub.add_parser("diagnose", help="ranked bottleneck findings (with confidence, measurements, opportunity, confirming experiment) from any records; verdicts for finished experiments")
    s.add_argument("records", nargs="+", help="observation, measurement, experiment, bench and/or profile records")
    s.add_argument("--bandwidth-gbs", type=float, help="GPU peak memory bandwidth if not in the built-in table")
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("propose", help="write an experiment spec (one variable between arms) for a diagnosis finding, then print its dry-run plan")
    s.add_argument("diagnosis"); s.add_argument("--finding", help="finding id (default: the diagnosis's recommended experiment)")
    s.add_argument("--out", required=True, help="where to write the spec JSON")
    s.add_argument("--workload", help="workload file (default: 'workload' in the local config, else a generated synthetic one)")
    s.add_argument("--common-args", nargs="+", help="baseline llama-server arguments (default: server_args in the local config, else the diagnosed experiment's, else hotpath defaults)")
    s.add_argument("--port", type=int, default=8099); s.add_argument("--rounds", type=int, default=3); s.add_argument("--repetitions", type=int, default=2)
    s.add_argument("--warmup", type=int, default=1); s.add_argument("--n-predict", type=int)
    s.add_argument("--force", action="store_true", help="replace an existing spec file (never a *.local.json)")
    s.set_defaults(fn=cmd_propose)

    s = sub.add_parser("gguf", help="model header summary: shapes, types, bytes per decode token")
    s.add_argument("path"); s.add_argument("--tensors", action="store_true"); s.add_argument("--bandwidth-gbs", type=float)
    s.set_defaults(fn=cmd_gguf)

    s = sub.add_parser("workload", help="generate a deterministic synthetic prompt set")
    s.add_argument("out_file"); s.add_argument("--name", default="synthetic"); s.add_argument("--n-prompts", type=int, default=4)
    s.add_argument("--words", type=int, nargs="+", default=[450, 900, 1300, 2000]); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--n-predict", type=int, default=64)
    s.add_argument("--format", choices=sorted(wl_mod.FORMATS), default="chatml", help="prompt wrapper; chat models need chatml")
    s.add_argument("--from-requests", help="build a `requests` workload from your captured llama-server request bodies (JSON array or JSONL) instead of synthetic prompts")
    s.add_argument("--limit", type=int, help="with --from-requests: keep only the first N bodies")
    s.set_defaults(fn=cmd_workload)

    s = sub.add_parser("measure", help="ACTIVE: run a workload against a server you own and record timings")
    s.add_argument("--server", required=True); s.add_argument("--workload", required=True); s.add_argument("--label")
    s.add_argument("--repetitions", type=int, default=3); s.add_argument("--warmup", type=int, default=1); s.add_argument("--n-predict", type=int)
    s.add_argument("--background-seconds", type=float, default=10.0); s.add_argument("--store-prompts", action="store_true")
    s.add_argument("--i-own-this-server", action="store_true")
    s.set_defaults(fn=cmd_measure)

    s = sub.add_parser("experiment", help="ACTIVE: launch arms from a spec, interleaved, and record everything")
    s.add_argument("spec"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--wait-s", type=int, default=0,
                                                                                               help="wait up to N seconds for preconditions (GPU idle, VRAM free)")
    s.set_defaults(fn=cmd_experiment)

    s = sub.add_parser("compare", help="side-by-side of experiment arms or measurement records, with comparability checks")
    s.add_argument("records", nargs="+")
    s.set_defaults(fn=cmd_compare)

    def discovery_args(s):
        s.add_argument("--root", action="append", help="also search this directory (repeatable; bounded depth; drive roots are refused)")
        s.add_argument("--no-exec", action="store_true", help="do not run `llama-server --version` on found binaries")
        s.add_argument("--no-exec-roots", action="store_true", help=argparse.SUPPRESS)

    s = sub.add_parser("discover", help="find the local setup: running llama-servers and their launch settings, installs, GGUF models, GPU, tools, logs (passive)")
    discovery_args(s)
    s.add_argument("--max-depth", type=int, help="directory depth for --root searches (default 4)")
    s.set_defaults(fn=cmd_discover)

    s = sub.add_parser("init", help="write hotpath.local.json from discovery (the running server's binary, model, args, log; CUDA path fixes)")
    s.add_argument("path", nargs="?", help="where to write it (default hotpath.local.json)")
    discovery_args(s)
    s.add_argument("--binary", help="choose this llama-server when several are found"); s.add_argument("--model", help="choose this GGUF model")
    s.add_argument("--server", help="choose this running server (URL) when several answer")
    s.add_argument("--update", action="store_true", help="add discovered keys to an existing file; existing values are never changed")
    s.add_argument("--template", action="store_true", help="write the old placeholder template instead of discovering")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("inspect", help="passive, one step: discover + observe the running server (command line, log, /metrics, GPU) + diagnose")
    discovery_args(s)
    s.add_argument("--server", help="server URL (default: server_url in the config, else the only running llama-server)")
    s.add_argument("--log", help="server log (default: the server's --log-file, else server_log in the config)")
    s.add_argument("--sample-seconds", type=float, default=5.0); s.add_argument("--bandwidth-gbs", type=float)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("mcp", help="run the MCP server on stdio so AI coding agents can call hotpath (active tools are off unless HOTPATH_MCP_ALLOW_ACTIVE=1)")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("tools", help="environment check: GPU, config, and which profiling/benchmark tools are available")
    s.set_defaults(fn=cmd_tools)

    s = sub.add_parser("bench", help="ACTIVE: llama-bench pp/tg with repetitions; refuses under background GPU load")
    s.add_argument("--model"); s.add_argument("--ngl", type=int, default=99); s.add_argument("--ub", type=int, default=512)
    s.add_argument("--pp", type=int, default=512); s.add_argument("--tg", type=int, default=128); s.add_argument("--reps", type=int, default=5)
    s.add_argument("--background-seconds", type=float, default=10.0); s.add_argument("--max-background-util", type=float, default=10.0)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("profile", help="ACTIVE: nsys kernel-share of a prefill-only (--tg 0) or decode-only (--pp 0) llama-bench run")
    s.add_argument("--label", required=True); s.add_argument("--bin-dir"); s.add_argument("--model")
    s.add_argument("--pp", type=int, default=512); s.add_argument("--tg", type=int, default=0)
    s.add_argument("--ub", type=int, default=256); s.add_argument("--ngl", type=int, default=99)
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("kernels", help="ACTIVE: test-backend-ops test/perf on a build directory for ops matching a filter")
    s.add_argument("mode", choices=["test", "perf"]); s.add_argument("--build-dir", help="llama.cpp build tree containing test-backend-ops (default: build_dir in the local config)"); s.add_argument("--label")
    s.add_argument("--op", default="MUL_MAT"); s.add_argument("--params", help="regex on the case parameter string, e.g. type_a=q4_0")
    s.add_argument("--backend", default="CUDA0")
    s.set_defaults(fn=cmd_kernels)
    return p


def _tolerant_console():
    """Reports contain characters (dashes, arrows) that legacy Windows code pages cannot encode; printing one must not
    abort a command after its record was already written. Unencodable characters become '?' on the console only."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _tolerant_console()
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (OSError, ValueError, KeyError) as e:
        if os.environ.get("HOTPATH_DEBUG"):
            raise
        print("error: %s: %s (set HOTPATH_DEBUG=1 for a traceback)" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
