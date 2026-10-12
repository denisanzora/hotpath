"""What a llama-server command line says about its configuration.

A running server's command line is the most direct passive evidence of how it was launched: which model, which
port, how many layers on the GPU, microbatch, context, flash attention, KV types, prompt-cache flags, log file.
This module turns an argv list into named settings and keeps track of which values are file paths, so records can
drop them and an experiment never inherits a path it cannot resolve.

Only the command line is read. Settings given through LLAMA_ARG_* environment variables or defaults compiled into
the build are not visible here; `unset` in the result means "not on the command line", never "off".
"""

# canonical name -> spellings (llama.cpp common/arg.cpp; older and newer spellings both accepted)
FLAGS = {
    "model": ("-m", "--model"),
    "hf_repo": ("-hf", "-hfr", "--hf-repo"),
    "host": ("--host",),
    "port": ("--port",),
    "n_gpu_layers": ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    "ctx_size": ("-c", "--ctx-size"),
    "batch_size": ("-b", "--batch-size"),
    "ubatch_size": ("-ub", "--ubatch-size"),
    "parallel": ("-np", "--parallel"),
    "flash_attn": ("-fa", "--flash-attn"),
    "cache_type_k": ("-ctk", "--cache-type-k"),
    "cache_type_v": ("-ctv", "--cache-type-v"),
    "threads": ("-t", "--threads"),
    "threads_batch": ("-tb", "--threads-batch"),
    "cache_ram": ("-cram", "--cache-ram"),
    "ctx_checkpoints": ("-ctxcp", "--ctx-checkpoints"),
    "log_file": ("--log-file",),
    "split_mode": ("-sm", "--split-mode"),
    "tensor_split": ("-ts", "--tensor-split"),
    "main_gpu": ("-mg", "--main-gpu"),
    "override_tensor": ("-ot", "--override-tensor"),
    "n_cpu_moe": ("-ncmoe", "--n-cpu-moe"),
    "fit": ("-fit", "--fit"),
    "model_draft": ("-md", "--model-draft"),
    "alias": ("-a", "--alias"),
    "device": ("-dev", "--device"),
}

# flags that never take a value (a following token is the next flag or a positional)
BOOLEAN_FLAGS = frozenset({
    "--metrics", "--slots", "--no-slots", "--no-mmap", "--mlock", "--no-webui", "--webui", "-cb", "--cont-batching", "-nocb", "--no-cont-batching",
    "-v", "--verbose", "--log-verbose", "--log-disable", "--log-colors", "--log-timestamps", "--log-prefix", "--embedding", "--embeddings",
    "--reranking", "--rerank", "--jinja", "--no-jinja", "-cmoe", "--cpu-moe", "--swa-full", "-kvu", "--kv-unified", "-nkvo", "--no-kv-offload",
    "--no-warmup", "--check-tensors", "--no-perf", "--context-shift", "--no-context-shift", "--props", "--no-op-offload", "--offline",
    "--no-mmproj", "--no-mmproj-offload", "-dio", "--direct-io", "--no-direct-io", "--escape", "--no-escape", "--special", "-sp",
    "--no-prefill-assistant", "-fa-off", "--no-repack", "-nr", "--numa-distribute",
})

# flags whose value is a local file or directory
PATH_FLAGS = frozenset({
    "-m", "--model", "--log-file", "--slot-save-path", "--prompt-cache", "--lora", "--lora-scaled", "--control-vector", "--control-vector-scaled",
    "--mmproj", "-mm", "-md", "--model-draft", "--chat-template-file", "--grammar-file", "--json-schema-file", "-jf", "--ssl-key-file",
    "--ssl-cert-file", "--api-key-file", "--path", "--media-path", "--webui-config-file", "--models-dir", "--models-preset", "-f", "--file",
    "--model-vocoder", "-mv", "--logits-file",
})

_SPELLING = {s: name for name, spellings in FLAGS.items() for s in spellings}


def _is_value(tok):
    """A token that is a value, not a flag: anything not starting with '-', and negative numbers ("-1")."""
    if not tok.startswith("-"):
        return True
    try:
        float(tok)
        return True
    except ValueError:
        return False


def canonical(tok):
    """llama.cpp reads `_` as `-` in long options (`--log_file` is `--log-file`), so they are one argument here too."""
    tok = str(tok)
    if tok.startswith("--") and "_" in tok.split("=", 1)[0]:
        head, sep, rest = tok.partition("=")
        return head.replace("_", "-") + sep + rest
    return tok


SECRET_FLAGS = frozenset({"--api-key", "--hf-token", "-hft"})


def split_args(argv):
    """[(flag, value_or_None), ...] for everything after the program name. `--flag=value` is split; boolean flags
    and flags followed by another flag get None. Positionals are reported with flag None. Long options are canonical
    (`--log_file` -> `--log-file`)."""
    out = []
    i = 0
    toks = [canonical(a) if str(a).startswith("--") else str(a) for a in argv]
    while i < len(toks):
        t = toks[i]
        if t.startswith("-") and not _is_value(t):
            if "=" in t and t.startswith("--"):
                f, v = t.split("=", 1)
                out.append((f, v))
                i += 1
                continue
            if t in BOOLEAN_FLAGS or i + 1 >= len(toks) or not _is_value(toks[i + 1]):
                # `-fa` was a boolean in older builds and takes on/off/auto in newer ones; both parse here
                out.append((t, None))
                i += 1
                continue
            out.append((t, toks[i + 1]))
            i += 2
            continue
        out.append((None, t))
        i += 1
    return out


def parse(argv):
    """Settings named by the command line. `argv` excludes the program. Unknown flags are kept in `other_flags`."""
    settings, other, path_flags = {}, [], []
    for flag, value in split_args(argv):
        if flag is None:
            continue
        if flag in PATH_FLAGS:
            path_flags.append(flag)
        name = _SPELLING.get(flag)
        if name == "flash_attn":
            settings[name] = "on" if value is None else value
        elif name:
            settings[name] = value if value is not None else True
        elif flag == "--metrics":
            settings["metrics"] = True
        elif flag in ("--slots", "--no-slots"):
            settings["slots_endpoint"] = flag == "--slots"
        elif flag in ("-cmoe", "--cpu-moe"):
            settings["cpu_moe"] = True
        elif flag in ("--no-mmap",):
            settings["mmap"] = False
        elif flag in ("--mlock",):
            settings["mlock"] = True
        elif flag in ("-nkvo", "--no-kv-offload"):
            settings["kv_offload"] = False
        elif flag in ("--api-key", "--api-key-file"):
            settings["api_key_set"] = True
        else:
            other.append(flag)
    return {"settings": settings, "path_flags": sorted(set(path_flags)), "other_flags": sorted(set(other))}


def spellings(flag):
    """Every spelling of the same argument (`-ngl`, `--gpu-layers`, `--n-gpu-layers`), or just the flag itself."""
    flag = canonical(flag)
    name = _SPELLING.get(flag)
    return set(FLAGS[name]) if name else {flag}


def remove_arg(argv, flag):
    """argv without any spelling of `flag` (and its value, in either `--flag value` or `--flag=value` form)."""
    names = spellings(flag)
    out = []
    for f, v in split_args(argv):
        if f in names:
            continue
        if f is not None:
            out.append(f)
        if v is not None:
            out.append(v)
    return out


def set_arg(argv, flag, value):
    """argv with `flag value` replacing every spelling of the same argument, so exactly one value reaches the server."""
    return remove_arg(argv, flag) + [flag, str(value)]


def get(argv, name, default=None):
    """One setting by canonical name (see FLAGS), from an argv list."""
    return parse(argv)["settings"].get(name, default)


def int_or_none(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def drop_path_args(argv):
    """argv without flags whose value is a local path (and without positionals). Returns (kept, dropped_flag_names).
    Used when arguments come from a record, where paths were reduced to basenames and would not resolve."""
    kept, dropped = [], []
    for flag, value in split_args(argv):
        if flag is None:
            continue
        if flag in PATH_FLAGS or (value is not None and ("/" in value or "\\" in value) and not value.startswith(("http://", "https://"))):
            dropped.append(flag)
            continue
        kept.append(flag)
        if value is not None:
            kept.append(value)
    return kept, sorted(set(dropped))


def redact(argv):
    """argv with every path-valued flag's value reduced to its basename, for records and console output."""
    out = []
    for flag, value in split_args(argv):
        if flag is None:
            out.append(_base(value))
            continue
        out.append(flag)
        if value is not None:
            out.append("<redacted>" if flag in SECRET_FLAGS else (_base(value) if (flag in PATH_FLAGS or "/" in value or "\\" in value) else value))
    return out


def _base(v):
    return str(v).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def describe(settings):
    """One line for reports: the settings that most often decide speed, with `unset` for the ones not given."""
    keys = (("n_gpu_layers", "-ngl"), ("ctx_size", "-c"), ("batch_size", "-b"), ("ubatch_size", "-ub"), ("parallel", "-np"),
            ("flash_attn", "-fa"), ("cache_type_k", "-ctk"), ("cache_type_v", "-ctv"), ("cache_ram", "--cache-ram"), ("threads", "-t"))
    return ", ".join("%s %s" % (short, settings.get(k, "unset")) for k, short in keys)
