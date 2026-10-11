"""hotpath MCP server: the CLI exposed as Model Context Protocol tools over local stdio.

There is no second implementation: every tool runs `python -m hotpath <subcommand>` as a child
process, with a timeout, and returns its output plus the files it wrote. Safety model:

- Read-only tools (environment check, GGUF header, passive observe, triage, compare, list results)
  are always available.
- Active tools launch servers or benchmarks and use the GPU (experiment, bench, kernels, profile,
  measure). They are refused unless the server was started with HOTPATH_MCP_ALLOW_ACTIVE=1 AND the
  call passes confirm=true. `measure` can only target servers listed in `owned_servers`; the
  --i-own-this-server override is deliberately not exposed.
- Every child process has a timeout (default 1800 s for active tools, 300 s for read-only ones,
  capped by HOTPATH_MCP_MAX_TIMEOUT_S, default 7200) and is killed together with its children on expiry.
- Files that hotpath reads as inputs it will act on (experiment specs, workloads, records) or writes
  must stay inside the server's working directory or a directory listed in HOTPATH_MCP_EXTRA_ROOTS
  (os.pathsep-separated). Read-only data you point at, namely models (GGUF headers, `--model`) and
  server logs, may live anywhere. Runtime binaries, model paths and build directories come from your
  hotpath.local.json or the spec, never from MCP tool arguments, so an agent cannot redirect
  hotpath to launch an arbitrary program.

Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout. Logs go to stderr only.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hotpath import __version__

PROTOCOL_VERSION = "2024-11-05"
MAX_OUTPUT_CHARS = 20000


def _active_allowed():
    return os.environ.get("HOTPATH_MCP_ALLOW_ACTIVE") == "1"


def _max_timeout():
    return int(os.environ.get("HOTPATH_MCP_MAX_TIMEOUT_S", "7200"))


def _roots():
    roots = [Path.cwd().resolve()]
    for extra in os.environ.get("HOTPATH_MCP_EXTRA_ROOTS", "").split(os.pathsep):
        if extra.strip():
            roots.append(Path(extra).resolve())
    return roots


def _inside_cwd(value, what):
    p = Path(value).resolve()
    roots = _roots()
    if not any(r == p or r in p.parents for r in roots):
        raise ValueError("%s must be inside the server working directory or HOTPATH_MCP_EXTRA_ROOTS (allowed: %s)" % (
            what, ", ".join(str(r) for r in roots)))
    return str(p)


def _s(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_CONFIRM = {"type": "boolean", "description": "Must be true. Also requires the server to run with HOTPATH_MCP_ALLOW_ACTIVE=1."}
_TIMEOUT = {"type": "integer", "description": "Seconds before the child process is killed."}

# name -> (description, schema, active, argv builder)
TOOLS = {}


def tool(name, description, schema, active=False):
    def deco(fn):
        TOOLS[name] = (description, schema, active, fn)
        return fn
    return deco


@tool("hotpath_tools", "Environment check: GPU via nvidia-smi, local config, and which profiling/benchmark tools were found. Read-only.", _s({}))
def _tools(a):
    return ["tools"]


@tool("hotpath_gguf_summary", "Summarize a GGUF model header: tensor types, bytes streamed per decoded token, optional bandwidth-bound decode ceiling. Reads only the header.",
      _s({"path": _STR, "bandwidth_gbs": {"type": "number"}}, ["path"]))
def _gguf(a):
    argv = ["gguf", a["path"]]
    if a.get("bandwidth_gbs"):
        argv += ["--bandwidth-gbs", str(a["bandwidth_gbs"])]
    return argv


@tool("hotpath_observe", "Passive baseline of a running llama-server (health, props, slots, GPU state, log history). Submits no inference. Writes results/local/observation-*.json.",
      _s({"server": _STR, "log": _STR, "sample_seconds": {"type": "number"}}))
def _observe(a):
    argv = ["observe"]
    if a.get("server"):
        argv += ["--server", a["server"]]
    if a.get("log"):
        argv += ["--log", a["log"]]  # read-only; may live outside the working directory
    if a.get("sample_seconds") is not None:
        argv += ["--sample-seconds", str(a["sample_seconds"])]
    return argv


@tool("hotpath_triage", "Attribution report from an observation record: request-time split, decode vs bandwidth ceiling, VRAM budget, labeled hypotheses and the experiments that would test them.",
      _s({"observation": _STR, "bandwidth_gbs": {"type": "number"}}, ["observation"]))
def _triage(a):
    argv = ["triage", _inside_cwd(a["observation"], "observation")]
    if a.get("bandwidth_gbs"):
        argv += ["--bandwidth-gbs", str(a["bandwidth_gbs"])]
    return argv


@tool("hotpath_diagnose", "Ranked bottleneck findings from any mix of hotpath records (observation, measurement, experiment, bench, profile): each with confidence level, the measurements it rests on, why it matters, an opportunity bound where defensible, the controlled experiment that would confirm it, and risks. Finished experiments get a keep/reject/inconclusive verdict. Writes results/local/diagnosis-*.json. Read-only.",
      _s({"records": {"type": "array", "items": _STR, "minItems": 1}, "bandwidth_gbs": {"type": "number"}}, ["records"]))
def _diagnose(a):
    argv = ["diagnose"] + [_inside_cwd(r, "record") for r in a["records"]]
    if a.get("bandwidth_gbs"):
        argv += ["--bandwidth-gbs", str(a["bandwidth_gbs"])]
    return argv


@tool("hotpath_propose", "Write an experiment spec for a diagnosis finding (one variable between arms, baseline = the diagnosed configuration) and print its dry-run plan. Launches nothing; the spec is run with hotpath_experiment_run after the user agrees.",
      _s({"diagnosis": _STR, "finding": _STR, "out": _STR, "workload": _STR, "rounds": _INT, "repetitions": _INT, "n_predict": _INT, "port": _INT}, ["diagnosis", "out"]))
def _propose(a):
    argv = ["propose", _inside_cwd(a["diagnosis"], "diagnosis"), "--out", _inside_cwd(a["out"], "out")]
    if a.get("finding"):
        argv += ["--finding", a["finding"]]
    if a.get("workload"):
        argv += ["--workload", _inside_cwd(a["workload"], "workload")]
    for k in ("rounds", "repetitions", "n_predict", "port"):
        if a.get(k) is not None:
            argv += ["--" + k.replace("_", "-"), str(a[k])]
    return argv


@tool("hotpath_compare", "Compare experiment/measurement/kernel records: medians with spread, ratios, comparability issues, greedy-output equivalence.",
      _s({"records": {"type": "array", "items": _STR, "minItems": 1}}, ["records"]))
def _compare(a):
    return ["compare"] + [_inside_cwd(r, "record") for r in a["records"]]


@tool("hotpath_workload", "Generate a deterministic synthetic prompt workload file.",
      _s({"out_file": _STR, "n_prompts": _INT, "words": {"type": "array", "items": _INT}, "n_predict": _INT, "format": {"type": "string", "enum": ["chatml", "raw"]}}, ["out_file"]))
def _workload(a):
    argv = ["workload", _inside_cwd(a["out_file"], "out_file")]
    if a.get("n_prompts"):
        argv += ["--n-prompts", str(a["n_prompts"])]
    if a.get("words"):
        argv += ["--words"] + [str(w) for w in a["words"]]
    if a.get("n_predict"):
        argv += ["--n-predict", str(a["n_predict"])]
    if a.get("format"):
        argv += ["--format", a["format"]]
    return argv


@tool("hotpath_experiment_plan", "Dry run of an experiment spec: prints the arms, launch count and redacted command lines. Launches nothing.",
      _s({"spec": _STR}, ["spec"]))
def _plan(a):
    return ["experiment", _inside_cwd(a["spec"], "spec"), "--dry-run"]


@tool("hotpath_list_results", "List result records in results/local (name, size, modified time).", _s({}))
def _list(a):
    return None  # handled in-process, no child needed


@tool("hotpath_experiment_run", "ACTIVE: launch the spec's arms as separate llama-server processes on the spec's port, interleaved, and record results. Uses the GPU for minutes to hours. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"spec": _STR, "wait_s": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT}, ["spec", "confirm"]), active=True)
def _run(a):
    argv = ["experiment", _inside_cwd(a["spec"], "spec")]
    if a.get("wait_s"):
        argv += ["--wait-s", str(a["wait_s"])]
    return argv


@tool("hotpath_measure", "ACTIVE: run a workload against a running server that is listed in owned_servers in the local config. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"server": _STR, "workload": _STR, "repetitions": _INT, "warmup": _INT, "label": _STR, "confirm": _CONFIRM, "timeout_s": _TIMEOUT}, ["server", "workload", "confirm"]), active=True)
def _measure(a):
    argv = ["measure", "--server", a["server"], "--workload", _inside_cwd(a["workload"], "workload")]
    for k, f in (("repetitions", "--repetitions"), ("warmup", "--warmup"), ("label", "--label")):
        if a.get(k) is not None:
            argv += [f, str(a[k])]
    return argv


@tool("hotpath_bench", "ACTIVE: llama-bench prefill/decode throughput; refuses under background GPU load. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"model": _STR, "pp": _INT, "tg": _INT, "ub": _INT, "reps": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT}, ["confirm"]), active=True)
def _bench(a):
    argv = ["bench"]
    if a.get("model"):
        argv += ["--model", a["model"]]
    for k in ("pp", "tg", "ub", "reps"):
        if a.get(k) is not None:
            argv += ["--" + k, str(a[k])]
    return argv


@tool("hotpath_kernels", "ACTIVE: test-backend-ops correctness ('test', NMSE vs CPU) or timing ('perf') for ops matching a filter. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"mode": {"type": "string", "enum": ["test", "perf"]}, "op": _STR, "params": _STR, "label": _STR, "confirm": _CONFIRM, "timeout_s": _TIMEOUT}, ["mode", "confirm"]), active=True)
def _kernels(a):
    argv = ["kernels", a["mode"]]
    if a.get("op"):
        argv += ["--op", a["op"]]
    if a.get("params"):
        argv += ["--params", a["params"]]
    if a.get("label"):
        argv += ["--label", a["label"]]
    return argv


@tool("hotpath_profile", "ACTIVE: Nsight Systems kernel-share profile of a decode-only (pp=0) or prefill-only (tg=0) llama-bench run. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"label": _STR, "pp": _INT, "tg": _INT, "ub": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT}, ["label", "confirm"]), active=True)
def _profile(a):
    argv = ["profile", "--label", a["label"]]
    for k in ("pp", "tg", "ub"):
        if a.get(k) is not None:
            argv += ["--" + k, str(a[k])]
    return argv


def _kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def _run_child(argv, timeout_s):
    kw = {}
    if os.name != "nt":
        kw["start_new_session"] = True
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "hotpath"] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=env, **kw)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        timed_out = True
    text = out or ""
    wrote = [ln[len("wrote "):] for ln in text.splitlines() if ln.startswith("wrote ")]
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS // 2] + "\n...[truncated]...\n" + text[-MAX_OUTPUT_CHARS // 2:]
    body = text
    if err and err.strip():
        body += "\n[stderr]\n" + err.strip()[-4000:]
    if timed_out:
        body += "\n[killed: exceeded %d s timeout]" % timeout_s
    body += "\n[exit code %s, %.1f s]" % (proc.returncode, time.time() - t0)
    if wrote:
        body += "\n[files written: %s]" % ", ".join(wrote)
    return body, (proc.returncode != 0 or timed_out)


def call_tool(name, args):
    if name not in TOOLS:
        raise ValueError("unknown tool %s" % name)
    _desc, _schema, active, builder = TOOLS[name]
    args = args or {}
    if active:
        if args.get("confirm") is not True:
            return "refused: %s is an active tool; pass confirm=true after the user has agreed to it." % name, True
        if not _active_allowed():
            return ("refused: active tools are disabled. The operator must start the MCP server with HOTPATH_MCP_ALLOW_ACTIVE=1 "
                    "to allow launching servers and benchmarks."), True
    if name == "hotpath_list_results":
        d = Path("results/local")
        rows = ["%s  %d bytes  %s" % (p.name, p.stat().st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)))
                for p in sorted(d.glob("*.json"))] if d.is_dir() else []
        return "\n".join(rows) or "no records in results/local", False
    argv = builder(args)
    timeout_s = int(args.get("timeout_s") or (1800 if active else 300))
    timeout_s = max(1, min(timeout_s, _max_timeout()))
    return _run_child(argv, timeout_s)


def _reply(msg_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:  # notification
        return
    if method == "initialize":
        _reply(mid, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                     "serverInfo": {"name": "hotpath", "version": __version__},
                     "instructions": "Read-only tools are always on. Active tools (experiments, benchmarks, profiling) need the operator's "
                                     "HOTPATH_MCP_ALLOW_ACTIVE=1 and confirm=true, and should only be called after the user agrees."})
    elif method == "ping":
        _reply(mid, {})
    elif method == "tools/list":
        _reply(mid, {"tools": [{"name": n, "description": d + (" (active)" if act else ""), "inputSchema": s,
                               "annotations": {"readOnlyHint": not act, "destructiveHint": False, "openWorldHint": False}}
                              for n, (d, s, act, _f) in TOOLS.items()]})
    elif method == "tools/call":
        p = msg.get("params") or {}
        try:
            text, is_err = call_tool(p.get("name"), p.get("arguments"))
        except (ValueError, KeyError, OSError) as e:
            text, is_err = "error: %s" % e, True
        _reply(mid, {"content": [{"type": "text", "text": text}], "isError": bool(is_err)})
    else:
        _reply(mid, error={"code": -32601, "message": "method not found: %s" % method})


def main():
    print("hotpath MCP server %s on stdio (active tools %s)" % (__version__, "ENABLED" if _active_allowed() else "disabled"), file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(json.loads(line))
        except json.JSONDecodeError:
            _reply(None, error={"code": -32700, "message": "parse error"})


if __name__ == "__main__":
    main()
