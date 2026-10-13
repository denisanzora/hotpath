"""Local deployment configuration.

Everything machine- or deployment-specific lives in a gitignored JSON file (default
hotpath.local.json next to the working directory) so reusable code never embeds a path,
endpoint, or model identity. See hotpath.local.example.json for the fields.
"""
import json
import os
from pathlib import Path

DEFAULT_NAME = "hotpath.local.json"


SNAPSHOT_ENV = "HOTPATH_LOCAL_SHA256"


def config_digest(path=None):
    """sha256 of the local config file's bytes, or "missing"."""
    import hashlib
    p = Path(path or os.environ.get("HOTPATH_LOCAL") or DEFAULT_NAME)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"


def load_local(path=None):
    p = Path(path or os.environ.get("HOTPATH_LOCAL") or DEFAULT_NAME)
    # in a process the MCP server started, the configuration is the one the operator started that server with: the file
    # names the programs and models hotpath launches, and it sits in a working directory an agent may write to
    import hashlib
    expected = os.environ.get(SNAPSHOT_ENV)
    data = p.read_bytes() if p.is_file() else None
    # the bytes that were checked are the bytes that are parsed (no second read the file could change in between)
    if expected and (hashlib.sha256(data).hexdigest() if data is not None else "missing") != expected:
        raise SystemExit("refused: %s changed after the MCP server started; it decides which programs and models hotpath launches, so a "
                         "change takes effect only when the operator restarts the MCP server" % p.name)
    if data is None:
        return {"_path": str(p), "_missing": True}
    d = json.loads(data.decode("utf-8-sig"))
    d["_path"] = str(p)
    return d


def read_launcher_settings(cfg):
    """Read the launch-relevant settings of the deployment's launcher, if the local config points at them.

    `launcher_settings_file` is a JSON file; `launcher_settings_keys` maps hotpath names to keys in it.
    Values are copied, paths are reduced to basenames.
    """
    f = cfg.get("launcher_settings_file")
    keys = cfg.get("launcher_settings_keys") or {}
    if not f or not Path(f).is_file():
        return {"available": False}
    with open(f, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = {"available": True, "source_basename": Path(f).name}
    for name, key in keys.items():
        v = raw.get(key)
        if isinstance(v, str) and ("\\" in v or "/" in v):
            v = Path(v).name
        out[name] = v
    return out
