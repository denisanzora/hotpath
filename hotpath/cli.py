"""Command line: init, tools, observe, triage, diagnose, propose, gguf, workload, measure, experiment, compare, bench, profile, kernels."""
import argparse
import json
import shlex
import os
import sys
from pathlib import Path

from hotpath import __version__
from hotpath import adapters
from hotpath import agentgate
from hotpath import locks
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


def choose_workload(workload, cfg, shape):
    """The workload for an experiment: the one given, else `workload` in the config, else synthetic prompts sized like
    the real traffic a diagnosis saw, else three default synthetic prompts. Generated files go to workloads/."""
    workload = workload or cfg.get("workload")
    if not workload and shape:
        # synthetic prompts sized like the real traffic the diagnosis saw (p10 / median / p90 prompt tokens, median output),
        # so the experiment measures the request shape the user actually serves
        sizes = [shape["prompt_tokens_p10"], shape["prompt_tokens_median"], shape["prompt_tokens_p90"]]
        words = [max(20, int(round((t - 20) / 1.15))) for t in sizes]          # ~1.15 tokens per word of the synthetic vocabulary
        n_predict = max(16, min(512, shape["generated_tokens_median"]))
        workload = str(Path("workloads") / ("hotpath-shaped-%s-n%d.json" % ("-".join(str(w) for w in words), n_predict)))
        if not Path(workload).exists():
            wl_mod.save_workload(wl_mod.build_workload("hotpath-shaped", 3, words, seed=1, n_predict=n_predict, fmt="chatml"), workload)
        print("no workload given; using synthetic prompts shaped like the observed traffic: %s (prompts of about %s tokens, %d generated; from %s)" % (
            workload, "/".join(str(t) for t in sizes), n_predict, shape["basis"]))
    if not workload:
        workload = str(Path("workloads") / "hotpath-default.json")
        if not Path(workload).exists():
            w = wl_mod.build_workload("hotpath-default", 3, [450, 900, 1300], seed=1, n_predict=64, fmt="chatml")
            wl_mod.save_workload(w, workload)
            print("no workload given; wrote a default synthetic one: %s (3 ChatML prompts of 450/900/1300 words, n_predict 64)" % workload)
    return workload


def cmd_propose(args):
    """Write an experiment spec for a diagnosis finding (one variable between arms), then show its dry-run plan."""
    from hotpath import propose as prop_mod
    cfg = load_local(args.local)
    shape = (load_record(args.diagnosis).get("traffic_shape") or None) if not (args.workload or cfg.get("workload")) else None
    workload = choose_workload(args.workload, cfg, shape)
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
    raw = None
    if agentgate.active():
        # the MCP server checked the file; check what this process reads, too. Dry runs as well: a plan resolves ${key}
        # references and prints them, so a spec that may not run may not be planned either
        try:
            raw = agentgate.check_spec_file(args.spec)
            if not args.dry_run:
                agentgate.harden(raw)
        except ValueError as e:
            raise SystemExit("refused: %s" % e)
    spec = exp_mod.load_spec(args.spec, cfg, raw=raw)
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
    if not args.server and cfg.get("server_url") and len(live) == 1 and cfg["server_url"].rstrip("/") != live[0]["base_url"]:
        from hotpath import server as srv_mod
        if srv_mod.health(cfg["server_url"].rstrip("/")).get("status") == "unreachable":
            print("\nserver_url in the config (%s) does not answer; inspecting the running server %s instead (`hotpath init --update` does not change "
                  "existing values: edit server_url, or pass --server)." % (cfg["server_url"], live[0]["base_url"]))
            base = live[0]["base_url"]
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


def _experiment_runner(cfg, wait_s):
    """Runs one spec file through the normal experiment runner (preconditions, own port, interleaving) and returns its
    record with the comparison attached; server logs go next to the spec."""
    def run(spec, spec_path):
        # the spec the loop built is the one launched (the file next to it is a copy for the user): nothing re-read from disk
        raw = json.loads(json.dumps(spec))
        if agentgate.active():
            # started by the MCP server: whatever produced this spec (an agent, a saved session, the loop from an observed command
            # line), it launches only what an agent-authored spec may carry, with the idle-GPU and free-port preconditions kept
            try:
                agentgate.check_spec(raw)
            except ValueError as e:
                raise SystemExit("refused: %s" % e)
            agentgate.harden(raw)
        resolved = exp_mod.load_spec(spec_path, cfg, raw=raw)
        for k in ("binary", "model"):
            if not resolved.get(k) or not Path(resolved[k]).exists():
                raise SystemExit("error: %s %s does not exist; set %s in hotpath.local.json (`hotpath init --binary/--model`)" % (
                    k, resolved.get(k), "runtime_binary" if k == "binary" else "model_path"))
        rec = exp_mod.run(resolved, str(Path(spec_path).parent), wait_for_preconditions_s=wait_s)
        rec["comparison"] = compare_mod.compare([rec])
        return rec
    return run


def _last_observation(out, server_url, max_age_h=24):
    """The newest observation in the results directory of the configured server that recorded its launch command line,
    if it is less than a day old; None otherwise."""
    import datetime as _dt
    if not server_url:
        return None
    for p in sorted(Path(out).glob("observation-*.json"), reverse=True):
        try:
            rec = load_record(p)
        except (OSError, ValueError):
            continue
        if ((rec.get("server") or {}).get("base_url") or "").rstrip("/") != server_url.rstrip("/"):
            continue
        if not ((rec.get("server_process") or {}).get("launch") or {}).get("available"):
            continue
        try:
            age = _dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(rec["created"])
        except (KeyError, ValueError, TypeError):
            continue
        return (p, rec) if age.total_seconds() < max_age_h * 3600 else None
    return None


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _session_gpu(args):
    """An active improve run holds the machine-wide GPU lock from its first launch to its report (a dry run takes nothing)."""
    if args.dry_run:
        return _NoLock()
    return locks.gpu_work("improve", args.resume or "new session", wait_s=args.wait_s, log=lambda m: print(m, flush=True))


def split_args_string(text):
    """A launch line given as one string. On Windows, backslashes in paths are kept (POSIX shell rules would drop them)
    and surrounding quotes are removed."""
    if os.name != "nt":
        return shlex.split(text)
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(text, posix=False)]


def requests_workload(captured):
    """The requests workload for a capture file: workloads/<name>.requests.json, built once from the capture (an existing
    one is reused when it was built from the same file contents, so a session keeps measuring the same requests)."""
    import hashlib
    src = Path(captured)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out = Path("workloads") / ("%s.requests.json" % src.stem)
    if out.is_file():
        try:
            if json.loads(out.read_text(encoding="utf-8")).get("source_sha256") == digest:
                return str(out)
        except ValueError:
            pass
        out = Path("workloads") / ("%s-%s.requests.json" % (src.stem, digest[:6]))
        if out.is_file():
            return str(out)
    try:
        w = wl_mod.workload_from_requests(str(src), name=src.stem)
    except ValueError as e:
        raise SystemExit("error: %s" % e)
    w["source_sha256"] = digest
    wl_mod.save_workload(w, str(out))
    ident = wl_mod.workload_identity(w)
    print("wrote %s: %d of your requests in %d class(es) (%s); it holds your prompts, keep it local" % (
        out, ident["n_prompts"], len(ident["classes"]), ", ".join("%s x%d" % kv for kv in sorted(ident["classes"].items()))))
    return str(out)


def cmd_improve(args):
    """ACTIVE (unless --dry-run): diagnose, test the top finding with an experiment, adopt it only on a keep verdict, repeat
    within the budget, confirm the cumulative change, and print the launch line to apply."""
    from hotpath import diagnose as diag_mod
    from hotpath import discover as disc
    from hotpath import improve as imp
    cfg = load_local(args.local)
    if cfg.get("_missing"):
        raise SystemExit("error: hotpath.local.json not found. Run `hotpath init` first: it finds your llama-server and model "
                         "(`hotpath init --root DIR` when they live outside the standard locations).")
    out = _out(args)
    records, start = [], None
    pending = [x for x in imp.unfinished_sessions(out) if x[0] != args.resume]
    if pending:
        print("note: interrupted improve session(s) can be continued instead of starting over: %s (`hotpath improve --resume NAME`; "
              "MCP: hotpath_improve with resume=NAME)" % ", ".join("%s (%d experiment(s) recorded)" % x for x in pending[-3:]))
    if args.resume:
        with _session_gpu(args):
            rep = imp.improve(cfg, [], args.workload or "", out, None if args.dry_run else _experiment_runner(cfg, args.wait_s), max_experiments=args.max_experiments,
                              max_minutes=args.max_minutes, rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup, port=args.port,
                              n_predict=args.n_predict, bandwidth_gbs=args.bandwidth_gbs, session=args.resume, resume=True, dry_run=args.dry_run,
                              log=lambda m: print(m, flush=True))
        print("")
        print(imp.improvement_markdown(rep))
        print("wrote %s" % rep["path"])
        return 0
    if args.from_records:
        records = [load_record(p) for p in args.from_records]
        start = "the records you named: %s" % ", ".join(Path(p).name for p in args.from_records)
    else:
        rec, full = _discover(args, cfg)
        _write(full, args, "discovery", disc.discovery_markdown(rec))
        live = disc.healthy_user_servers(rec)
        base = args.server or cfg.get("server_url") if (args.server or cfg.get("server_url")) else (live[0]["base_url"] if len(live) == 1 else None)
        if base and not args.server:
            from hotpath import server as srv_mod
            if srv_mod.health(base.rstrip("/")).get("status") == "unreachable":
                base = live[0]["base_url"] if len(live) == 1 else None
        if not base and len(live) > 1:
            raise SystemExit("several llama-servers are running (%s); choose one with --server" % ", ".join(s["base_url"] for s in live))
        saved = None if (base or args.log) else _last_observation(out, cfg.get("server_url"))
        if saved:
            # the usual sequence on one GPU: improve refused while the user's server held the VRAM, the user stopped it, and
            # runs improve again; the last observation still has that server's launch command line, which a log alone lacks
            path, obs = saved
            records = [obs]
            start = "the last observation of your server %s (%s, %s; it is not running now)" % (cfg.get("server_url"), path.name, obs.get("created"))
        elif base or args.log or cfg.get("server_log"):
            obs = observe_mod.observe(cfg, log_path=args.log, base_url=base, sample_seconds=5.0)
            _write(obs, args, "observation", observe_mod.observation_markdown(obs))
            records = [obs]
            start = "your running server %s (observed passively)" % base if base else "your server log"
            if live:
                names = ", ".join("pid %s (%s)" % (s.get("pid"), s["base_url"]) for s in live)
                print("note: your llama-server %s is running. Experiments run on hotpath's own server (port %d) and need the GPU memory it holds; "
                      "hotpath never stops, restarts or reconfigures your server. Stop it yourself before the run, or pass --wait-s N." % (names, args.port))
        else:
            start = "no running server or log: a baseline of your configuration on hotpath's own server"
    shape = None
    if args.requests:
        args.workload = requests_workload(args.requests)
    if records and not (args.workload or cfg.get("workload")):
        shape = diag_mod.diagnose(records, cfg=cfg).get("traffic_shape")
    workload = choose_workload(args.workload, cfg, shape)
    print("starting point: %s" % start)
    with _session_gpu(args):
        rep = imp.improve(cfg, records, workload, out, None if args.dry_run else _experiment_runner(cfg, args.wait_s), max_experiments=args.max_experiments,
                          max_minutes=args.max_minutes, rounds=args.rounds, repetitions=args.repetitions, warmup=args.warmup, port=args.port,
                              n_predict=args.n_predict, dry_run=args.dry_run, initial_args=split_args_string(args.args) if args.args else None,
                          bandwidth_gbs=args.bandwidth_gbs, log=lambda m: print(m, flush=True), cache_mode=args.cache_mode,
                          step_requests=args.step_requests, validation_requests=args.validation_requests)
    print("")
    print(imp.improvement_markdown(rep))
    print("wrote %s" % rep["path"])
    if args.dry_run and rep.get("next_step"):
        print("\nNothing was launched. Run without --dry-run to start (ACTIVE: launches hotpath's own llama-server on port %d, one experiment at a time)." % args.port)
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

    s = sub.add_parser("improve", help="ACTIVE (unless --dry-run): test the top findings one experiment at a time on hotpath's own server, "
                                       "adopt only keep verdicts, confirm the result, print the launch line to apply")
    discovery_args(s)
    s.add_argument("--from", dest="from_records", nargs="+", metavar="RECORD", help="start from these records (observation, measurement, diagnosis inputs) "
                   "instead of observing the running server")
    s.add_argument("--server", help="the running server to start from (default: server_url, else the only running llama-server)")
    s.add_argument("--log", help="server log to diagnose from (default: the server's --log-file, else server_log)")
    s.add_argument("--workload", help="workload file (default: 'workload' in the config, else synthetic prompts shaped like your observed traffic)")
    s.add_argument("--requests", metavar="CAPTURED", help="your application's own request bodies (JSON array or JSONL of llama-server request bodies, "
                   "optionally {\"class\": ..., \"body\": ...}): every step and the final validation run these real requests instead of synthetic prompts; "
                   "the workload is written to workloads/ (keep it local: it holds your prompts)")
    s.add_argument("--cache-mode", choices=("cold", "warm", "as-is"), help="prompt-cache handling for request workloads: %s" % (
        "cold (default): every request fully processed, prompt cache off; warm: your requests in your application's order with the prompt cache as "
        "it sends them; as-is: no intervention, for diagnostics only"))
    s.add_argument("--step-requests", type=int, default=2, help="requests per request class in each step (default 2: the median and the largest)")
    s.add_argument("--validation-requests", type=int, default=4, help="requests per request class in the final validation (default 4, smallest to "
                   "largest, preferring ones the steps did not run)")
    s.add_argument("--args", help="starting llama-server arguments as one quoted string, when no server runs, e.g. --args \"-ngl 99 -c 8192\" "
                   "(default: server_args, else defaults fitted to your GPU)")
    s.add_argument("--max-experiments", type=int, default=3, help="experiments to run at most, not counting the baseline and the final confirmation (default 3)")
    s.add_argument("--max-minutes", type=float, default=60, help="stop before a step whose estimated duration would exceed this (default 60)")
    s.add_argument("--dry-run", action="store_true", help="passive: diagnose and write the first spec, launch nothing")
    s.add_argument("--resume", metavar="SESSION", help="continue an interrupted session (its directory name under results/local): experiments it already "
                   "recorded are reused, not run again; the session's starting point and workload are kept")
    s.add_argument("--wait-s", type=int, default=0, help="wait up to this long for each experiment's preconditions (free GPU)")
    s.add_argument("--port", type=int, default=8099); s.add_argument("--rounds", type=int, default=3); s.add_argument("--repetitions", type=int, default=2)
    s.add_argument("--warmup", type=int, default=1); s.add_argument("--n-predict", type=int); s.add_argument("--bandwidth-gbs", type=float)
    s.set_defaults(fn=cmd_improve)

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


_GPU_COMMANDS = {"cmd_bench": "bench", "cmd_profile": "profile", "cmd_kernels": "kernels", "cmd_measure": "measure"}


def main(argv=None):
    _tolerant_console()
    args = build_parser().parse_args(argv)
    # started by the MCP server: stop this process's servers and exit if that server is killed outright
    locks.watch_parent()
    try:
        kind = _GPU_COMMANDS.get(getattr(args.fn, "__name__", ""))
        if kind:
            with locks.gpu_work(kind, getattr(args, "label", None) or getattr(args, "mode", None), wait_s=getattr(args, "wait_s", 0) or 0):
                return args.fn(args)
        return args.fn(args)
    except (OSError, ValueError, KeyError) as e:
        if os.environ.get("HOTPATH_DEBUG"):
            raise
        print("error: %s: %s (set HOTPATH_DEBUG=1 for a traceback)" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
