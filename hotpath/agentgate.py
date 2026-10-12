"""What an agent may make hotpath launch.

The MCP server runs every tool as a child process with HOTPATH_AGENT_GATE=1. In that mode every spec hotpath is about to
launch, whoever wrote it (an agent through hotpath_experiment_run, or hotpath's own improve loop from an observed command
line, a saved session or `args`), is checked here at the one place that launches servers, immediately before the launch
and on the same parsed content that is launched.

The check is an allowlist. llama-server keeps gaining arguments and environment variables that read or write files,
reach the network or start programs: this build alone has --tools/--agent (built-in tools and a CORS proxy),
--mcp-servers-config and LLAMA_ARG_MCP_SERVERS_JSON (programs the server starts), --rpc (remote backends),
--log-prompts-dir (writes prompts to disk) and GGML_BACKEND_PATH / CUDA_INJECTION64_PATH (load a library). A denylist
misses the next one; an allowlist refuses it until someone decides it is a pure tuning setting. Outside agent mode
(the operator's own CLI) nothing here applies: the operator's launch line is theirs.
"""
import json
import os
from pathlib import Path

from hotpath import launch

GATE_ENV = "HOTPATH_AGENT_GATE"

# llama-server settings an agent-run experiment may change or carry: how the configured model runs on the configured
# runtime, nothing that names a file, a host, a program or a download.
_TUNING_SETTINGS = ("n_gpu_layers", "ctx_size", "batch_size", "ubatch_size", "parallel", "flash_attn", "cache_type_k", "cache_type_v",
                    "threads", "threads_batch", "cache_ram", "ctx_checkpoints", "split_mode", "tensor_split", "main_gpu", "override_tensor",
                    "n_cpu_moe", "fit", "alias", "device")
VALUE_FLAGS = frozenset(s for name in _TUNING_SETTINGS for s in launch.FLAGS[name]) | frozenset({
    "--reasoning-format", "--reasoning-budget", "--chat-template", "--cache-reuse", "-dt", "--defrag-thold", "-lm", "--load-mode",
    "-n", "--n-predict", "--keep", "--threads-http", "-to", "--timeout", "-sps", "--slot-prompt-similarity", "--numa", "--prio", "--poll",
    "--prio-batch", "--poll-batch", "-fitc", "--fit-ctx", "-fitt", "--fit-target", "--log-colors", "-lv", "--log-verbosity",
    "-ctkd", "-ctvd", "--cache-type-k-draft", "--cache-type-v-draft",
})
BOOLEAN_FLAGS = frozenset({
    "--metrics", "--slots", "--no-slots", "--no-webui", "-cb", "--cont-batching", "-nocb", "--no-cont-batching", "--jinja", "--no-jinja",
    "--mmap", "--no-mmap", "--mlock", "-cmoe", "--cpu-moe", "--swa-full", "-kvu", "--kv-unified", "-kvo", "--kv-offload", "-nkvo",
    "--no-kv-offload", "--warmup", "--no-warmup", "--check-tensors", "--perf", "--no-perf", "--context-shift", "--no-context-shift",
    "--op-offload", "--no-op-offload", "--offline", "-dio", "--direct-io", "-ndio", "--no-direct-io", "--repack", "-nr", "--no-repack",
    "--no-host", "--cache-idle-slots", "--no-cache-idle-slots", "--cache-prompt", "--no-cache-prompt", "-fa-off", "--log-disable",
    "--log-timestamps", "--no-log-timestamps", "--log-prefix", "--no-log-prefix", "-v", "--verbose", "--log-verbose",
})
ALLOWED_FLAGS = VALUE_FLAGS | BOOLEAN_FLAGS

# environment variables: runtime tuning knobs only. Not GGML_* or CUDA_* wholesale: GGML_BACKEND_PATH and
# CUDA_INJECTION64_PATH load a library, CUDA_CACHE_PATH writes files, and LLAMA_ARG_* mirror every argument above and below.
ENV_PREFIXES = ("GGML_CUDA_", "GGML_SCHED_")
ENV_NAMES = frozenset({
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING", "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE", "CUDA_AUTO_BOOST", "CUDA_SCALE_LAUNCH_QUEUES",
    "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH", "LLAMA_ARG_N_PARALLEL", "LLAMA_ARG_FLASH_ATTN",
    "LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V", "LLAMA_ARG_THREADS", "LLAMA_ARG_THREADS_BATCH", "LLAMA_ARG_CONT_BATCHING",
    "LLAMA_ARG_MMAP", "LLAMA_ARG_MLOCK", "LLAMA_ARG_KV_OFFLOAD", "LLAMA_ARG_NO_KV_OFFLOAD", "LLAMA_ARG_KV_UNIFIED", "LLAMA_ARG_SWA_FULL",
    "LLAMA_ARG_CACHE_REUSE", "LLAMA_ARG_DEFRAG_THOLD", "LLAMA_ARG_CACHE_PROMPT", "LLAMA_ARG_CACHE_RAM", "LLAMA_ARG_CTX_CHECKPOINTS",
    "LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_MAIN_GPU", "LLAMA_ARG_TENSOR_SPLIT", "LLAMA_ARG_JINJA",
    "LLAMA_ARG_THINK", "LLAMA_ARG_THINK_BUDGET", "LLAMA_ARG_CONTEXT_SHIFT", "LLAMA_ARG_ENDPOINT_METRICS", "LLAMA_ARG_ENDPOINT_SLOTS",
    "LLAMA_ARG_OFFLINE", "LLAMA_ARG_THREADS_HTTP", "LLAMA_ARG_REPACK", "LLAMA_ARG_FIT", "LLAMA_ARG_FIT_CTX", "LLAMA_ARG_FIT_TARGET",
    "LLAMA_ARG_PERF", "LLAMA_ARG_NO_HOST", "LLAMA_ARG_N_PREDICT", "LLAMA_ARG_ALIAS", "LLAMA_ARG_OVERRIDE_TENSOR",
})
REFUSAL_TAIL = (". In agent-run experiments hotpath accepts only llama-server tuning settings (GPU layers, context, batch sizes, slots, "
                "flash attention, KV types, cache sizes, threads, ...) and GGML_CUDA_*/selected CUDA_*/LLAMA_ARG_* tuning variables; arguments or "
                "variables that name files, hosts, programs, tools or downloads are refused whatever they are. The operator can run this from the CLI")


def active():
    """True in a process the MCP server started: every launch is checked here."""
    return os.environ.get(GATE_ENV) == "1"


def roots():
    """Directories agent-supplied inputs must stay in: the working directory and HOTPATH_MCP_EXTRA_ROOTS."""
    out = [Path.cwd().resolve()]
    for extra in os.environ.get("HOTPATH_MCP_EXTRA_ROOTS", "").split(os.pathsep):
        if extra.strip():
            out.append(Path(extra).resolve())
    return out


def inside_roots(value, what):
    p = Path(value).resolve()
    rs = roots()
    if not any(r == p or r in p.parents for r in rs):
        raise ValueError("%s must be inside the server working directory or HOTPATH_MCP_EXTRA_ROOTS (allowed: %s)" % (
            what, ", ".join(str(r) for r in rs)))
    return str(p)


def _flag_of(token):
    return str(token).split("=", 1)[0]


def arg_problems(argv, where):
    """Flags in an argument list that agent-run experiments may not carry (values of allowed flags pass)."""
    out = []
    for flag, _value in launch.split_args(argv):
        if flag is None:
            continue
        if _flag_of(flag) not in ALLOWED_FLAGS:
            out.append("%s: server argument %s is not an allowed tuning setting" % (where, _flag_of(flag)))
    return out


def env_problems(env, where):
    out = []
    for k in env or {}:
        k = str(k)
        if not (k in ENV_NAMES or k.startswith(ENV_PREFIXES)):
            out.append("%s: environment variable %s is not an allowed runtime tuning variable" % (where, k))
    return out


def _is_ref(v):
    return isinstance(v, str) and len(v) > 3 and v.startswith("${") and v.endswith("}")


def spec_problems(spec):
    """Everything in an unresolved spec (as written on disk) that an agent-run launch may not carry. Besides each field,
    the full argument list every arm launches with is checked, so values cannot smuggle flags past a key check."""
    if not isinstance(spec, dict) or not isinstance(spec.get("arms"), list):
        return ["spec is not an experiment spec (needs an `arms` list)"]
    problems = []
    try:
        port = int(spec.get("port", 8099))
        if not 1024 <= port <= 65535:
            problems.append("port %s is outside 1024-65535" % port)
    except (TypeError, ValueError):
        problems.append("port %r is not a number" % spec.get("port"))
    from hotpath.records import check_name
    for n, what in [(spec.get("name"), "spec name")] + [(a.get("name") if isinstance(a, dict) else None, "arm name") for a in spec["arms"]]:
        try:
            check_name(n, what)
        except ValueError as e:
            problems.append(str(e))
    for a in spec["arms"]:
        if isinstance(a, dict):
            argv = [str(x) for x in spec.get("common_args") or []]
            for k, v in (a.get("args_override") or {}).items():
                argv = launch.set_arg(argv, k, v)
            for k in a.get("args_remove") or []:
                argv = launch.remove_arg(argv, k)
            for flag, value in launch.split_args(argv):
                if flag is None:
                    problems.append("arm %s: stray value %r in its launch arguments" % (a.get("name"), value))

    def fields(where, d):
        for k in ("binary", "model"):
            if k in d and not _is_ref(d[k]):
                problems.append("%s: `%s` must be a ${key} reference into hotpath.local.json, not a path" % (where, k))
        if "path_prepend" in d and not (_is_ref(d["path_prepend"]) or d["path_prepend"] in ([], None)):
            problems.append("%s: `path_prepend` must be a ${key} reference" % where)
        for envkey in ("env", "env_override"):
            problems.extend(env_problems(d.get(envkey), where))
        for k, v in (d.get("args_override") or {}).items():
            problems.extend(arg_problems([k] if isinstance(v, bool) and v else [k, str(v)], where))
    fields("spec", spec)
    for a in spec["arms"]:
        if not isinstance(a, dict):
            problems.append("an arm is not an object")
            continue
        fields("arm %s" % a.get("name"), a)
    problems.extend(arg_problems(spec.get("common_args") or [], "common_args"))
    ssp = spec.get("slot_save_path", False)
    try:
        if isinstance(ssp, str) and ssp != "auto":
            inside_roots(ssp, "slot_save_path")
        if spec.get("workload"):
            inside_roots(spec["workload"], "workload")
            problems.extend(workload_problems(spec["workload"]))
    except ValueError as e:
        problems.append(str(e))
    return problems


def workload_problems(path):
    """A workload that is a subset of another file must not reach outside the allowed directories through it."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
    except (OSError, ValueError):
        return []       # the runner reports an unreadable workload itself
    if isinstance(w, dict) and w.get("subset_of"):
        try:
            inside_roots(resolve_relative(w["subset_of"], path), "the workload's subset_of")
        except ValueError as e:
            return [str(e)]
    return []


def resolve_relative(target, relative_to_file):
    """`subset_of` is written relative to the working directory; an absolute path is taken as given."""
    p = Path(target)
    return str(p) if p.is_absolute() else str(Path.cwd() / p)


AGENT_PRECONDITIONS = {"sample_seconds": 5, "max_gpu_util_pct": 10}


def harden(spec):
    """An agent-run spec keeps the idle-GPU and free-port preconditions whatever it says (it may make them stricter)."""
    pre = dict(spec.get("preconditions") or {})
    pre["sample_seconds"] = max(float(pre.get("sample_seconds") or 0), AGENT_PRECONDITIONS["sample_seconds"])
    pre["max_gpu_util_pct"] = min(float(pre.get("max_gpu_util_pct", AGENT_PRECONDITIONS["max_gpu_util_pct"])), AGENT_PRECONDITIONS["max_gpu_util_pct"])
    port = int(spec.get("port", 8099))
    pre["ports_must_be_free"] = sorted({int(x) for x in pre.get("ports_must_be_free") or []} | {port})
    spec["preconditions"] = pre
    return spec


def check_spec(spec, what="spec"):
    problems = spec_problems(spec)
    if problems:
        raise ValueError("%s refused for an agent-run experiment: %s%s" % (what, "; ".join(problems[:6]), REFUSAL_TAIL))


def check_spec_file(path):
    """Read a spec file once, check it, and return the parsed content: the caller launches exactly what was checked."""
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    check_spec(spec)
    return spec


def check_args_string(tokens, what="args"):
    problems = arg_problems(tokens, what)
    if problems:
        raise ValueError("; ".join(problems[:6]) + REFUSAL_TAIL)
