"""Result records: what every hotpath output carries so two results can be judged comparable.

Evidence classes (every record states one):
  passive     - read from existing logs, configuration, and non-inference HTTP endpoints
  active      - produced by hotpath submitting inference requests to a server
  historical  - comparison of records collected under conditions hotpath did not control
  controlled  - arms launched by hotpath under one spec, interleaved, same workload
  hypothesis  - a claim that still needs an experiment
"""
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import socket
from pathlib import Path

from hotpath import __version__

EVIDENCE_CLASSES = ("passive", "active", "historical", "controlled", "hypothesis")


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def host_identity():
    return {
        "hostname_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12],
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def file_identity(path, full_hash=False):
    """Identity for a model or binary without re-reading multi-GB files by default.

    Default: size + sha256 of the first 64 MiB. Pass full_hash=True for the whole file.
    Paths are reduced to basenames so records stay free of private directory layouts.
    """
    p = Path(path)
    if not p.is_file():
        return {"basename": p.name, "exists": False}
    h = hashlib.sha256()
    size = p.stat().st_size
    with open(p, "rb") as f:
        if full_hash:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        else:
            h.update(f.read(64 << 20))
    return {
        "basename": p.name,
        "exists": True,
        "size_bytes": size,
        "sha256_" + ("full" if full_hash else "head64m"): h.hexdigest()[:24],
        "mtime": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
    }


def redact_path(path):
    """Keep only the basename of a path. Used for every path that goes into a record."""
    if path is None:
        return None
    return Path(str(path)).name


_ABS_PATH = re.compile(r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|(?<![\w:/.])/(?=[\w.-]+/[\w.-]+/))[^\s\"'<>|,;]*")


def scrub_text(text):
    """Replace anything that looks like an absolute filesystem path in free text with its basename."""
    def _base(m):
        return re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1]
    return _ABS_PATH.sub(_base, text)


def scrub(obj):
    """Deep-copy a JSON-like structure with every string passed through scrub_text."""
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj


def redact_args(args):
    """Redact command-line values that look like filesystem paths (e.g. --lora, --mmproj, --log-file)."""
    return [redact_path(a) if (isinstance(a, str) and not a.startswith("-") and ("/" in a or "\\" in a)) else a for a in args]


def new_record(kind, evidence_class, **fields):
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError("unknown evidence class %r" % evidence_class)
    rec = {
        "hotpath_version": __version__,
        "kind": kind,
        "evidence_class": evidence_class,
        "created": now_iso(),
        "host": host_identity(),
    }
    rec.update(fields)
    return rec


def write_record(rec, out_dir, stem):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    data = json.dumps(scrub(rec), indent=1, ensure_ascii=False)
    # Created exclusively: two records of the same kind written within one second (a diagnosis of two inputs run back
    # to back by an agent, say) must not overwrite each other. A numeric suffix keeps names unique; the .md companion
    # takes its name from the returned path.
    for n in range(0, 1000):
        path = out / ("%s-%s%s.json" % (stem, ts, "" if n == 0 else "-%d" % (n + 1)))
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(data)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not find a free record name for %s-%s" % (stem, ts))


def load_record(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentiles(values):
    """Median and spread of a sample. Returns None for an empty sample."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    n = len(vals)

    def q(p):
        if n == 1:
            return vals[0]
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)

    return {
        "n": n,
        "min": vals[0],
        "p10": round(q(0.10), 3),
        "median": round(q(0.5), 3),
        "p90": round(q(0.90), 3),
        "max": vals[-1],
    }


def env_snapshot(keys=None):
    """Only GGML_*/LLAMA_*/CUDA_* variables, since those change kernel behavior."""
    out = {}
    for k, v in os.environ.items():
        if k.startswith(("GGML_", "LLAMA_", "CUDA_")) or (keys and k in keys):
            out[k] = v
    return out
