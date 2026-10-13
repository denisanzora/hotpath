"""hotpath MCP server: the CLI exposed as Model Context Protocol tools over local stdio.

There is no second implementation: every tool runs `python -m hotpath <subcommand>` as a child
process, with a timeout, and returns its output plus the files it wrote. Safety model:

- Read-only tools (environment check, GGUF header, passive observe, triage, compare, list results)
  are always available.
- Active tools launch servers or benchmarks and use the GPU (experiment, bench, kernels, profile,
  measure). They are refused unless the server was started with HOTPATH_MCP_ALLOW_ACTIVE=1 AND the
  call passes confirm=true. `measure` can only target servers listed in `owned_servers`; the
  --i-own-this-server override is deliberately not exposed.
- Every child process has a timeout (default 7200 s for active tools, 300 s for read-only ones,
  capped by HOTPATH_MCP_MAX_TIMEOUT_S, default 7200) and is killed together with its children on expiry.
- Active tools run as jobs: the call returns the result if the job finishes within `wait_for_result_s`
  (default 45 s, below the ~60 s some clients allow a tool call), otherwise a job id to poll with
  hotpath_job_status. One active job at a time (they share the GPU). hotpath_job_cancel stops a job's
  process tree; jobs still running when the MCP server exits are stopped too. Only processes hotpath
  started are ever stopped.
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
import threading
import time
import uuid
from pathlib import Path

from hotpath import __version__

PROTOCOL_VERSION = "2024-11-05"
MAX_OUTPUT_CHARS = 20000
DEFAULT_RESULT_WAIT_S = 45
MAX_RESULT_WAIT_S = 55          # default cap per call; an operator whose client allows longer tool calls raises it (HOTPATH_MCP_MAX_WAIT_S)


def _max_wait():
    """Longest a call may wait for a job: 55 s by default (below the ~60 s some clients allow a tool call), or the operator's
    HOTPATH_MCP_MAX_WAIT_S for clients that allow longer calls, which means fewer status polls."""
    try:
        return max(5, min(3600, int(os.environ.get("HOTPATH_MCP_MAX_WAIT_S", MAX_RESULT_WAIT_S))))
    except ValueError:
        return MAX_RESULT_WAIT_S
ENABLE_ACTIVE_HINT = ("To allow them, the user restarts this MCP server with the environment variable HOTPATH_MCP_ALLOW_ACTIVE=1, e.g. "
                      "Claude Code: `claude mcp add hotpath -e HOTPATH_MCP_ALLOW_ACTIVE=1 -- hotpath-mcp` (or \"env\": {\"HOTPATH_MCP_ALLOW_ACTIVE\": \"1\"} "
                      "in the server's entry in .mcp.json); OpenCode: \"environment\": {\"HOTPATH_MCP_ALLOW_ACTIVE\": \"1\"} in the mcp entry of opencode.json; "
                      "then start a new session. An agent cannot enable this itself.")


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


def _no_unc(value, what):
    """A network path (\\\\host\\share, //host/share) would make Windows contact that host, and send the user's credentials, just
    to read it: refused for every path argument, including read-only ones."""
    v = str(value)
    if v.startswith(("\\\\", "//")) or v.lower().startswith(("\\\\?\\unc", "file:")):
        raise ValueError("%s must be a local path, not a network path" % what)
    return value


def _inside_cwd(value, what):
    _no_unc(value, what)
    p = Path(value).resolve()
    roots = _roots()
    if not any(r == p or r in p.parents for r in roots):
        raise ValueError("%s must be inside the server working directory or HOTPATH_MCP_EXTRA_ROOTS (allowed: %s)" % (
            what, ", ".join(str(r) for r in roots)))
    return str(p)


def _loopback(url, what="server"):
    """Read-only tools may only talk to a local server: an agent-supplied URL must not make hotpath issue requests elsewhere."""
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("%s must be a loopback URL (127.0.0.1, localhost or ::1), got host %r" % (what, host))
    return url


def _no_local_config_name(value, what):
    if str(value).lower().replace("\\", "/").rsplit("/", 1)[-1].endswith(".local.json"):
        raise ValueError("%s must not be a *.local.json file: those hold local configuration" % what)
    return value


def _s(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_STR = {"type": "string"}
_INT = {"type": "integer"}
_CONFIRM = {"type": "boolean", "description": "Must be true. Also requires the server to run with HOTPATH_MCP_ALLOW_ACTIVE=1."}
_TIMEOUT = {"type": "integer", "description": "Seconds before the job's process tree is killed (default 7200, capped by HOTPATH_MCP_MAX_TIMEOUT_S)."}
_RESULT_WAIT = {"type": "integer", "description": "Return the result if the job finishes within this many seconds (default 45, max 55 unless the operator raised it); "
                "otherwise return a job id to poll with hotpath_job_status."}

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


_ROOTS_ARG = {"type": "array", "items": _STR, "description": "Extra directories to search for llama-server binaries, GGUF models and launch scripts "
              "(bounded depth; drive roots refused). Binaries found there are inspected but not executed."}


@tool("hotpath_inspect", "START HERE. Passive, one call: discovers the local llama.cpp setup (running llama-servers and their real launch "
      "settings, installs, GGUF models, GPU, missing CUDA libraries), observes the running server (command line, log, /metrics, GPU state; "
      "no inference), and diagnoses it: ranked findings with what is known, what is a hypothesis, the experiment that would verify it, "
      "and the exact next step. Writes discovery/observation/diagnosis records in results/local.",
      _s({"server": _STR, "log": _STR, "roots": _ROOTS_ARG, "sample_seconds": {"type": "number"}, "bandwidth_gbs": {"type": "number"}}))
def _inspect(a):
    argv = ["inspect", "--no-exec-roots"]
    if a.get("server"):
        argv += ["--server", _loopback(a["server"], "server")]
    if a.get("log"):
        argv += ["--log", _no_unc(a["log"], "log")]       # read-only; may live outside the working directory
    for r in a.get("roots") or []:
        argv += ["--root", _no_unc(r, "roots")]
    if a.get("sample_seconds") is not None:
        argv += ["--sample-seconds", str(a["sample_seconds"])]
    if a.get("bandwidth_gbs"):
        argv += ["--bandwidth-gbs", str(a["bandwidth_gbs"])]
    return argv


@tool("hotpath_discover", "Passive: find the local inference setup without changing anything: running llama-servers (launch settings from "
      "their command lines, loopback /health /props /metrics), llama-server installs (backends, sibling tools, unresolved CUDA DLLs and the "
      "directory that fixes them), GGUF models (headers only, VRAM fit), GPUs and driver CUDA version, profilers, logs. Lists where it looked.",
      _s({"roots": _ROOTS_ARG}))
def _discover(a):
    argv = ["discover", "--no-exec-roots"]
    for r in a.get("roots") or []:
        argv += ["--root", _no_unc(r, "roots")]
    return argv


@tool("hotpath_gguf_summary", "Summarize a GGUF model header: tensor types, bytes streamed per decoded token, optional bandwidth-bound decode ceiling. Reads only the header.",
      _s({"path": _STR, "bandwidth_gbs": {"type": "number"}}, ["path"]))
def _gguf(a):
    argv = ["gguf", _no_unc(a["path"], "path")]
    if a.get("bandwidth_gbs"):
        argv += ["--bandwidth-gbs", str(a["bandwidth_gbs"])]
    return argv


@tool("hotpath_observe", "Passive baseline of a running llama-server (health, props, slots, GPU state, log history). Submits no inference. Writes results/local/observation-*.json.",
      _s({"server": _STR, "log": _STR, "sample_seconds": {"type": "number"}}))
def _observe(a):
    argv = ["observe"]
    if a.get("server"):
        argv += ["--server", _loopback(a["server"], "server")]
    if a.get("log"):
        argv += ["--log", _no_unc(a["log"], "log")]  # read-only; may live outside the working directory
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
    # --force is deliberately not exposed: an agent can write a new spec file, never replace an existing file
    argv = ["propose", _inside_cwd(a["diagnosis"], "diagnosis"), "--out", _no_local_config_name(_inside_cwd(a["out"], "out"), "out")]
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


@tool("hotpath_workload", "Generate a deterministic synthetic prompt workload file, or (from_requests) build a `requests` workload from the user's captured llama-server request bodies (file inside the working directory; bodies stay local).",
      _s({"out_file": _STR, "n_prompts": _INT, "words": {"type": "array", "items": _INT}, "n_predict": _INT, "format": {"type": "string", "enum": ["chatml", "raw"]},
          "from_requests": _STR, "limit": _INT}, ["out_file"]))
def _workload(a):
    out = _no_local_config_name(_inside_cwd(a["out_file"], "out_file"), "out_file")
    # a new .json file only: an agent may write a workload, never replace an existing file (the user's captured workload,
    # a spec, a record or anything else in the working directory)
    if not out.lower().endswith(".json"):
        raise ValueError("out_file must be a .json file")
    if Path(out).exists():
        raise ValueError("out_file %s already exists; choose a new file name (hotpath_workload never replaces a file)" % Path(out).name)
    argv = ["workload", out]
    if a.get("from_requests"):
        argv += ["--from-requests", _inside_cwd(a["from_requests"], "from_requests")]
        if a.get("limit"):
            argv += ["--limit", str(a["limit"])]
        return argv
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


# Spec contents an agent may have written. The spec file is the one input that decides WHAT gets launched, so its contents
# are checked here, and again in the child on the content it actually launches (hotpath/agentgate.py): programs and models
# only as ${key} references to the operator's hotpath.local.json, and only allowlisted llama-server tuning arguments and
# runtime variables. `hotpath propose` strips runner-owned flags from the baseline it copies from the user's config.
from hotpath import agentgate  # noqa: E402


def _flag_of(token):
    """`--flag=value` and `--flag` name the same argument."""
    return str(token).split("=", 1)[0]


def check_spec_for_agent(path):
    """Raise ValueError if the spec would let an agent-authored file launch something other than the operator's configured
    runtime with tuning settings (see hotpath/agentgate.py)."""
    agentgate.check_spec_file(path)


@tool("hotpath_experiment_run", "ACTIVE: launch the spec's arms as separate llama-server processes on the spec's port, interleaved, and record results. Uses the GPU for minutes to hours; runs as a job (poll hotpath_job_status). Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1. The spec's contents are checked: binary/model must be ${key} references to the operator's config. wait_s waits for the GPU preconditions (free VRAM, idle GPU) before failing.",
      _s({"spec": _STR, "wait_s": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT, "wait_for_result_s": _RESULT_WAIT}, ["spec", "confirm"]), active=True)
def _run(a):
    spec = _inside_cwd(a["spec"], "spec")
    check_spec_for_agent(spec)
    argv = ["experiment", spec]
    if a.get("wait_s"):
        argv += ["--wait-s", str(a["wait_s"])]
    return argv


_IMPROVE_PROPS = {
    "max_experiments": {"type": "integer", "description": "Experiments to run at most (default 3), not counting the baseline and the final confirmation."},
    "max_minutes": {"type": "number", "description": "Stop before a step whose estimated duration would exceed the time left (default 60)."},
    "workload": {"type": "string", "description": "Workload file inside the working directory (default: synthetic prompts shaped like the observed traffic)."},
    "requests": {"type": "string", "description": "The user's captured application request bodies (JSONL or JSON array, inside the working directory): steps and the "
                 "final validation run these real requests. Prefer this whenever the user has them."},
    "cache_mode": {"type": "string", "enum": ["cold", "warm", "as-is"], "description": "Prompt-cache handling for request workloads (default cold)."},
    "from_records": {"type": "array", "items": _STR, "description": "Start from these records (inside the working directory) instead of observing the running server."},
    "server": {"type": "string", "description": "The user's running server to start from (loopback URL); it is observed, never restarted or loaded."},
    "args": {"type": "string", "description": "Starting llama-server arguments as one string, only when no server runs (default: server_args in the config, else defaults fitted to the GPU)."},
    "resume": {"type": "string", "description": "Continue an interrupted session by its name (improve-YYYYMMDD-HHMMSS, from the report or results/local): completed experiments are reused."},
}


def _improve_argv(a, dry_run):
    argv = ["improve", "--no-exec-roots"]
    if dry_run:
        argv.append("--dry-run")
    if a.get("max_experiments") is not None:
        argv += ["--max-experiments", str(int(a["max_experiments"]))]
    if a.get("max_minutes") is not None:
        argv += ["--max-minutes", str(float(a["max_minutes"]))]
    if a.get("workload"):
        argv += ["--workload", _inside_cwd(a["workload"], "workload")]
    if a.get("requests"):
        argv += ["--requests", _inside_cwd(a["requests"], "requests")]
    if a.get("cache_mode"):
        argv += ["--cache-mode", str(a["cache_mode"])]
    if a.get("from_records"):
        argv += ["--from"] + [_inside_cwd(r, "record") for r in a["from_records"]]
    if a.get("server"):
        argv += ["--server", _loopback(a["server"], "server")]
    if a.get("args"):
        import shlex
        agentgate.check_args_string(shlex.split(a["args"]), "args")
        argv.append("--args=" + a["args"])
    if a.get("wait_s"):
        argv += ["--wait-s", str(int(a["wait_s"]))]
    if a.get("resume"):
        import re
        if not re.fullmatch(r"improve-[0-9]{8}-[0-9]{6}(-[0-9]+)?", str(a["resume"])):
            raise ValueError("resume must be a session name such as improve-20261012-091500 (a directory under results/local)")
        argv += ["--resume", a["resume"]]
    return argv


@tool("hotpath_improve_plan", "Passive dry run of hotpath_improve: diagnoses the starting point (the running server observed passively, or nothing "
      "running) and writes the first experiment spec; reports the finding it would test, the arms, the free VRAM needed and a rough duration. "
      "Starts no llama-server and no GPU work (discovery may run `llama-server --version` on installed binaries). Show this to the user before "
      "asking to run hotpath_improve. With `requests` (the user's captured request bodies) the steps and the validation use those real requests.", _s(dict(_IMPROVE_PROPS)))
def _improve_plan(a):
    return _improve_argv(a, True)


@tool("hotpath_improve", "ACTIVE, the whole loop in one job: tests the top finding with a one-variable experiment on hotpath's own llama-server "
      "(never the user's), adopts a change only on a keep verdict (faster, consistent, identical outputs, verified cache state; cache flags "
      "only with real-traffic evidence), repeats within the budget, validates the adopted changes against the original launch on a broader "
      "workload (the user's own requests with `requests`), and reports each option as recommended, needs review, rejected or inconclusive, "
      "with the launch line to apply only for what was validated. Starts with a baseline run when no server or log exists. Uses the GPU for tens of minutes; runs as "
      "a job (poll hotpath_job_status). Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1. Changes nothing on the user's server.",
      _s(dict(_IMPROVE_PROPS, wait_s=_INT, confirm=_CONFIRM, timeout_s=_TIMEOUT, wait_for_result_s=_RESULT_WAIT), ["confirm"]), active=True)
def _improve(a):
    # the job is killed at its timeout; a budget it cannot finish would lose the final validation, so refuse it up front
    budget_s = float(a.get("max_minutes") if a.get("max_minutes") is not None else 60) * 60
    limit = min(int(a.get("timeout_s") or _max_timeout()), _max_timeout())
    if budget_s * 1.25 > limit:
        raise ValueError("max_minutes %.0f does not fit the job timeout of %d s (the final validation and model loads need room beyond the budget): "
                         "lower max_minutes to %d or less, or the operator raises HOTPATH_MCP_MAX_TIMEOUT_S" % (budget_s / 60, limit, int(limit / 1.25 / 60)))
    return _improve_argv(a, False)


@tool("hotpath_measure", "ACTIVE: run a workload against a running server that is listed in owned_servers in the local config. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"server": _STR, "workload": _STR, "repetitions": _INT, "warmup": _INT, "label": _STR, "confirm": _CONFIRM, "timeout_s": _TIMEOUT,
          "wait_for_result_s": _RESULT_WAIT}, ["server", "workload", "confirm"]), active=True)
def _measure(a):
    argv = ["measure", "--server", _loopback(a["server"], "server"), "--workload", _inside_cwd(a["workload"], "workload")]
    for k, f in (("repetitions", "--repetitions"), ("warmup", "--warmup"), ("label", "--label")):
        if a.get(k) is not None:
            argv += [f, _label(a[k]) if k == "label" else str(a[k])]
    return argv


@tool("hotpath_bench", "ACTIVE: llama-bench prefill/decode throughput; refuses under background GPU load. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"model": _STR, "pp": _INT, "tg": _INT, "ub": _INT, "reps": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT, "wait_for_result_s": _RESULT_WAIT}, ["confirm"]), active=True)
def _bench(a):
    argv = ["bench"]
    if a.get("model"):
        argv += ["--model", _no_unc(a["model"], "model")]
    for k in ("pp", "tg", "ub", "reps"):
        if a.get(k) is not None:
            argv += ["--" + k, str(a[k])]
    return argv


@tool("hotpath_kernels", "ACTIVE: test-backend-ops correctness ('test', NMSE vs CPU) or timing ('perf') for ops matching a filter. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"mode": {"type": "string", "enum": ["test", "perf"]}, "op": _STR, "params": _STR, "label": _STR, "confirm": _CONFIRM, "timeout_s": _TIMEOUT,
          "wait_for_result_s": _RESULT_WAIT}, ["mode", "confirm"]), active=True)
def _kernels(a):
    argv = ["kernels", a["mode"]]
    if a.get("op"):
        argv += ["--op", a["op"]]
    if a.get("params"):
        argv += ["--params", a["params"]]
    if a.get("label"):
        argv += ["--label", _label(a["label"])]
    return argv


@tool("hotpath_profile", "ACTIVE: Nsight Systems kernel-share profile of a decode-only (pp=0) or prefill-only (tg=0) llama-bench run. Requires confirm=true and HOTPATH_MCP_ALLOW_ACTIVE=1.",
      _s({"label": _STR, "pp": _INT, "tg": _INT, "ub": _INT, "confirm": _CONFIRM, "timeout_s": _TIMEOUT, "wait_for_result_s": _RESULT_WAIT}, ["label", "confirm"]), active=True)
def _profile(a):
    argv = ["profile", "--label", _label(a["label"])]
    for k in ("pp", "tg", "ub"):
        if a.get(k) is not None:
            argv += ["--" + k, str(a[k])]
    return argv


def _label(v):
    """Labels become file names (records, profiler captures)."""
    from hotpath.records import check_name
    return check_name(str(v), "label")


def _kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, stdin=subprocess.DEVNULL)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def _child_argv(argv):
    """`python -m hotpath` puts the working directory first on sys.path, so a .py file an agent wrote there (argparse.py,
    json.py) would run inside every child. Children run isolated (-I: no working directory, user site or PYTHON*
    variables on the path) with this hotpath's own location added explicitly."""
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    boot = "import sys; sys.path.insert(0, %r); from hotpath.cli import main; sys.exit(main(sys.argv[1:]))" % pkg_parent
    return [sys.executable, "-I", "-c", boot] + list(argv)


def _child_env():
    from hotpath import locks
    env = dict(os.environ)
    # every launch in the child is checked against the agent allowlist, and the child stops its servers and exits if this
    # server is killed outright (no stdin close, so stop_all_jobs never ran)
    env[agentgate.GATE_ENV] = "1"
    env[locks.PARENT_ENV] = "%d:%s" % (os.getpid(), locks.process_start(os.getpid()) or "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"             # a job's progress lines reach its output file (and hotpath_job_status) as they happen
    # the child must import this same hotpath, also when it runs from a source checkout and the working directory is elsewhere
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([pkg_parent] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _format(text, err, returncode, seconds, timed_out=None, cancelled=False):
    text = text or ""
    wrote = [ln[len("wrote "):] for ln in text.splitlines() if ln.startswith("wrote ")]
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS // 2] + "\n...[truncated]...\n" + text[-MAX_OUTPUT_CHARS // 2:]
    body = text
    if err and err.strip():
        body += "\n[stderr]\n" + err.strip()[-4000:]
    if timed_out:
        body += "\n[killed: exceeded %d s timeout]" % timed_out
    if cancelled:
        body += "\n[cancelled with hotpath_job_cancel]"
    body += "\n[exit code %s, %.1f s]" % (returncode, seconds)
    if wrote:
        body += "\n[files written: %s]" % ", ".join(wrote)
    return body


def _run_child(argv, timeout_s):
    kw = {}
    if os.name != "nt":
        kw["start_new_session"] = True
    t0 = time.time()
    # stdin=DEVNULL: the child must never read (or wait on) this server's JSON-RPC pipe
    proc = subprocess.Popen(_child_argv(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            text=True, encoding="utf-8", errors="replace", env=_child_env(), **kw)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        timed_out = None
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        timed_out = timeout_s
    return _format(out, err, proc.returncode, time.time() - t0, timed_out), (proc.returncode != 0 or bool(timed_out))


# ---- jobs: active tools run in the background so no client tool-call timeout cuts an experiment short --------------

JOBS = {}
_JOBS_LOCK = threading.Lock()
_START_LOCK = threading.Lock()


class Job:
    def __init__(self, tool_name, argv, timeout_s):
        self.id = "job-" + uuid.uuid4().hex[:8]
        self.tool, self.argv, self.timeout_s = tool_name, argv, timeout_s
        d = Path("results/local/jobs")
        d.mkdir(parents=True, exist_ok=True)
        self.out_path, self.err_path = d / (self.id + ".out"), d / (self.id + ".err")
        self.started = time.time()
        self.ended = None
        self.timed_out = None
        self.cancelled = False
        kw = {"start_new_session": True} if os.name != "nt" else {}
        self._out, self._err = open(self.out_path, "w", encoding="utf-8"), open(self.err_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(_child_argv(argv), stdout=self._out, stderr=self._err, stdin=subprocess.DEVNULL,
                                     env=_child_env(), **kw)
        self.done = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        try:
            self.proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            self.timed_out = self.timeout_s
            _kill_tree(self.proc)
            self.proc.wait()
        self.ended = time.time()
        for f in (self._out, self._err):
            f.close()
        self.done.set()

    def stop(self):
        if not self.done.is_set():
            self.cancelled = True
            _kill_tree(self.proc)
            self.done.wait(30)

    def state(self):
        if not self.done.is_set():
            return "running"
        if self.cancelled:
            return "cancelled"
        if self.timed_out:
            return "killed (timeout)"
        return "finished" if self.proc.returncode == 0 else "failed"

    def _read(self, p):
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def report(self, tail_lines=30):
        elapsed = (self.ended or time.time()) - self.started
        head = "%s %s: %s after %.0f s (`hotpath %s`)" % (self.id, self.tool, self.state(), elapsed, " ".join(self.argv))
        if not self.done.is_set():
            lines = self._read(self.out_path).splitlines()
            import re
            totals = next((re.search(r"running (\d+) launches, (\d+) requests", ln) for ln in lines if ln.startswith("running ")), None)
            launches = [re.search(r"\(launch (\d+)/(\d+)\)", ln) for ln in lines if "launching arm" in ln]
            done_requests = sum(1 for ln in lines if ln.strip().startswith(("warmup rep", "rep rep")))
            # improve sessions: which part of the loop runs (a step and its estimate, the validation), then the current experiment
            phase = [ln for ln in lines if ln.startswith(("step ", "validating:", "baseline:", "your requests:", "  adopted", "  not adopted", "resuming "))]
            if phase:
                head += "\nphase: %s" % phase[-1].strip()
                kept = [ln.strip() for ln in phase if ln.startswith(("  adopted", "  not adopted"))]
                if kept:
                    head += "\nsteps finished so far: %s" % "; ".join(k.split(":")[0] for k in kept)
            if totals:
                head += "\nprogress of the current experiment: launch %s of %s, %d of %s requests done" % (
                    launches[-1].group(1) if launches and launches[-1] else "0", totals.group(1), done_requests, totals.group(2))
            return head + "\nlatest output:\n" + "\n".join(lines[-tail_lines:]) + \
                ("\n\nStill running. Poll with hotpath_job_status(job_id=%r, wait_s=%d): each call waits up to %d s and returns early when the job "
                 "ends, so there is no need to poll faster; stop with hotpath_job_cancel." % (self.id, _max_wait(), _max_wait())), False
        body = _format(self._read(self.out_path), self._read(self.err_path), self.proc.returncode, elapsed, self.timed_out, self.cancelled)
        return head + "\n" + body, self.state() != "finished"


def _running_job():
    with _JOBS_LOCK:
        return next((j for j in JOBS.values() if not j.done.is_set()), None)


def start_job(name, argv, timeout_s, wait_s):
    # the check and the start are one step: two calls arriving together must not both start a job
    with _START_LOCK:
        running = _running_job()
        if running:
            return ("refused: %s (%s) is still running and active jobs share the GPU; wait with hotpath_job_status(job_id=%r) or stop it with "
                    "hotpath_job_cancel." % (running.id, running.tool, running.id)), True
        job = Job(name, argv, timeout_s)
        with _JOBS_LOCK:
            JOBS[job.id] = job
    job.done.wait(max(0, min(int(wait_s), _max_wait())))
    return job.report()


def stop_all_jobs():
    """On shutdown: stop every job this server started (and only those)."""
    with _JOBS_LOCK:
        jobs = list(JOBS.values())
    for j in jobs:
        j.stop()


@tool("hotpath_job_status", "State and latest output of an active job (improve, experiment, bench, measure, profile, kernels) started by this server: "
      "the loop's phase and the current experiment's progress; waits up to wait_s seconds (the operator's limit, 55 by default) for it to finish "
      "and returns its full result when it has, so one call per wait is enough. Without job_id: lists jobs, and names any other hotpath process "
      "holding the GPU. Read-only.",
      _s({"job_id": _STR, "wait_s": _INT}))
def _job_status(a):
    return None  # handled in-process


@tool("hotpath_job_cancel", "Stop an active job this server started (its whole process tree, including the llama-servers it launched). "
      "Only hotpath's own jobs can be stopped; nothing else is touched.", _s({"job_id": _STR}, ["job_id"]))
def _job_cancel(a):
    return None  # handled in-process


def _job_call(name, args):
    if name == "hotpath_job_status" and not args.get("job_id"):
        from hotpath import locks
        with _JOBS_LOCK:
            rows = ["%s  %s  %s" % (j.id, j.tool, j.state()) for j in JOBS.values()]
            mine = {j.proc.pid for j in JOBS.values() if not j.done.is_set()}
        holder = locks.gpu_holder()
        if holder and holder.get("pid") not in mine:
            # a job of an earlier MCP server (another session or client), or a CLI run: it holds the GPU and this server cannot poll it
            rows.append("GPU in use by %s; not a job of this server: its records are in that working directory (results/local/), and an "
                        "improve session there can be continued with resume once it ends" % locks.describe(holder))
        return "\n".join(rows) or "no jobs started by this server; the GPU is not in use by hotpath", False
    with _JOBS_LOCK:
        job = JOBS.get(args.get("job_id"))
    if not job:
        return "error: no job %r started by this server (jobs do not survive a server restart; their output stays in results/local/jobs/)" % args.get("job_id"), True
    if name == "hotpath_job_cancel":
        job.stop()
        return job.report()
    job.done.wait(max(0, min(int(args.get("wait_s") if args.get("wait_s") is not None else min(DEFAULT_RESULT_WAIT_S, _max_wait())), _max_wait())))
    return job.report()


_LIMITS = {"max_minutes": (1, 24 * 60), "max_experiments": (0, 20), "wait_s": (0, 86400), "timeout_s": (1, 7 * 86400), "wait_for_result_s": (0, 3600),
           "rounds": (1, 20), "repetitions": (1, 50), "n_predict": (1, 32768), "port": (1024, 65535), "sample_seconds": (0, 600),
           "n_prompts": (1, 1000), "limit": (1, 100000), "pp": (0, 1 << 20), "tg": (0, 1 << 20), "ub": (1, 1 << 16), "reps": (1, 1000),
           "warmup": (0, 50), "bandwidth_gbs": (1, 100000)}


def _type_errors(args, schema):
    """Arguments of the wrong JSON type, or numbers that are not finite or out of range: answered with an error instead of
    failing inside the tool (where a TypeError would leave the client waiting) or slipping past a budget check (NaN)."""
    import math
    out = []
    for k, v in args.items():
        t = ((schema.get("properties") or {}).get(k) or {}).get("type")
        ok = True
        if t == "integer":
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif t == "number":
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        elif t == "string":
            ok = isinstance(v, str)
        elif t == "boolean":
            ok = isinstance(v, bool)
        elif t == "array":
            ok = isinstance(v, list) and all(isinstance(x, (str if ((schema["properties"][k].get("items") or {}).get("type") == "string") else (int, str))) for x in v)
        if not ok:
            out.append("%s must be of type %s" % (k, t))
            continue
        if k in _LIMITS and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = _LIMITS[k]
            if not lo <= v <= hi:
                out.append("%s must be between %s and %s" % (k, lo, hi))
    return out


def call_tool(name, args):
    if name not in TOOLS:
        raise ValueError("unknown tool %s" % name)
    _desc, _schema, active, builder = TOOLS[name]
    args = args or {}
    # agents guess argument names; say exactly what this tool takes instead of failing on a missing key later
    if not isinstance(args, dict):
        return "error: arguments must be an object", True
    unknown = sorted(set(args) - set(_schema.get("properties") or {}))
    missing = [k for k in _schema.get("required") or [] if k not in args and k != "confirm"]    # confirm has its own refusal below
    bad = _type_errors(args, _schema)
    if bad and not unknown and not missing:
        return "error: %s. Arguments of %s: %s." % ("; ".join(bad), name, ", ".join(sorted(_schema.get("properties") or {}))), True
    if unknown or missing:
        return ("error: %s%s%s. Arguments of %s: %s (required: %s)." % (
            "unknown argument(s) %s" % ", ".join(unknown) if unknown else "", "; " if unknown and missing else "",
            "missing required argument(s) %s" % ", ".join(missing) if missing else "", name,
            ", ".join(sorted(_schema.get("properties") or {})) or "none", ", ".join(_schema.get("required") or []) or "none")), True
    if active:
        if args.get("confirm") is not True:
            return "refused: %s is an active tool; pass confirm=true after the user has agreed to it." % name, True
        if not _active_allowed():
            return ("refused: active tools are disabled: this MCP server was started without HOTPATH_MCP_ALLOW_ACTIVE=1, so it does not launch "
                    "servers, benchmarks or profilers. " + ENABLE_ACTIVE_HINT), True
    if name == "hotpath_list_results":
        d = Path("results/local")
        rows = ["%s  %d bytes  %s" % (p.name, p.stat().st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)))
                for p in sorted(d.glob("*.json"))] if d.is_dir() else []
        return "\n".join(rows) or "no records in results/local", False
    if name in ("hotpath_job_status", "hotpath_job_cancel"):
        return _job_call(name, args)
    argv = builder(args)
    timeout_s = int(args.get("timeout_s") or (_max_timeout() if active else 300))
    timeout_s = max(1, min(timeout_s, _max_timeout()))
    if active:
        wait = args.get("wait_for_result_s")
        return start_job(name, argv, timeout_s, DEFAULT_RESULT_WAIT_S if wait is None else wait)
    return _run_child(argv, timeout_s)


# tools that change something beyond hotpath's own records in results/local: not advertised as read-only
_WRITING_TOOLS = frozenset({"hotpath_job_cancel", "hotpath_workload", "hotpath_propose"})

_OUT_LOCK = threading.Lock()
_IN_FLIGHT = []


def _reply(msg_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    with _OUT_LOCK:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def _instructions():
    return ("hotpath diagnoses local llama.cpp-family inference. Start with hotpath_inspect (passive: discovers the setup, observes the running "
            "server, diagnoses it and names the next step). Read-only tools are always on. Active tools (experiments, benchmarks, profiling, "
            "measure) are %s on this server%s; they need confirm=true after the user agrees, run as jobs (poll hotpath_job_status), and never "
            "touch the user's own server: experiments launch their own llama-server on the spec's port. When the user wants their setup "
            "improved, show hotpath_improve_plan (passive) and, after they agree, run hotpath_improve: it runs the experiments, adopts only "
            "keep verdicts and reports the launch line to apply. For single experiments, specs come from hotpath_propose." % (
                "ENABLED" if _active_allowed() else "DISABLED", "" if _active_allowed() else " (the user must restart it with HOTPATH_MCP_ALLOW_ACTIVE=1)"))


def _tool_call(mid, p):
    try:
        text, is_err = call_tool(p.get("name"), p.get("arguments"))
    except (ValueError, KeyError, OSError) as e:
        text, is_err = "error: %s" % e, True
    except Exception as e:  # noqa: BLE001 - every call gets an answer; a client waiting for a reply that never comes hangs
        text, is_err = "error: internal error in %s: %s: %s" % (p.get("name"), type(e).__name__, e), True
    _reply(mid, {"content": [{"type": "text", "text": text}], "isError": bool(is_err)})


def handle(msg, threaded=False):
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:  # notification
        return
    if method == "initialize":
        _reply(mid, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                     "serverInfo": {"name": "hotpath", "version": __version__}, "instructions": _instructions()})
    elif method == "ping":
        _reply(mid, {})
    elif method == "tools/list":
        _reply(mid, {"tools": [{"name": n, "description": d + (" (active)" if act else ""), "inputSchema": s,
                               "annotations": {"readOnlyHint": not act and n not in _WRITING_TOOLS, "destructiveHint": False, "openWorldHint": False}}
                              for n, (d, s, act, _f) in TOOLS.items()]})
    elif method == "tools/call":
        p = msg.get("params") or {}
        if threaded:
            # calls run concurrently (a job_status that waits must not hold up a cancel or a ping); replies are matched by id
            t = threading.Thread(target=_tool_call, args=(mid, p), daemon=True)
            _IN_FLIGHT.append(t)
            t.start()
        else:
            _tool_call(mid, p)
    else:
        _reply(mid, error={"code": -32601, "message": "method not found: %s" % method})


def main():
    print("hotpath MCP server %s on stdio (active tools %s)" % (__version__, "ENABLED" if _active_allowed() else "disabled"), file=sys.stderr)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _reply(None, error={"code": -32700, "message": "parse error"})
                continue
            if not isinstance(msg, dict):
                _reply(None, error={"code": -32600, "message": "invalid request: not a JSON-RPC object"})
                continue
            try:
                handle(msg, threaded=True)
            except Exception as e:  # noqa: BLE001 - one bad message must not end the server
                _reply(msg.get("id"), error={"code": -32603, "message": "internal error: %s" % type(e).__name__})
    finally:
        # input ended: answer the calls already received (a piped client still reads them; each is bounded by its own
        # timeout), then stop the jobs this server started, so no experiment keeps the GPU without anyone to report to
        for t in list(_IN_FLIGHT):
            t.join(timeout=330)
        stop_all_jobs()


if __name__ == "__main__":
    main()
