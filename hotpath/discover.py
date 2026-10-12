"""hotpath discover: find the local llama.cpp-family inference setup without changing anything.

Bounded, read-only and transparent. What it looks at:

  - running llama-server processes: their command lines (model, port, offload, microbatch, context, log file, ...)
    and the loopback HTTP endpoints they serve (GET /health, /props, /metrics only; no inference)
  - llama-server installations: the running ones, the configured one, PATH, a short list of standard install
    locations, and directories you name (walked to a fixed depth with an entry cap; drive roots are refused)
  - per installation: compute backends present, sibling tools, the libraries its executables import that cannot be
    found (on Windows: a CUDA runtime DLL missing next to the binary and not on PATH makes llama-server exit with no
    output), the directory that would fix it, and `--version` (the only program discovery executes)
  - GGUF models: headers only, in the running servers' model directories, standard model caches and your directories
  - NVIDIA GPUs, the driver's CUDA version, processes on the GPU, profilers, the optional jsonschema validator
  - launch scripts in directories you name that start llama-server (their command lines, as written)

Every search records where it looked and what it found; anything not found is reported as not found, not guessed.
Nothing is written except the discovery record (and hotpath.local.json when `hotpath init` asks for it).
"""
import glob
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from hotpath import gguf as gguf_mod
from hotpath import gpu as gpu_mod
from hotpath import launch

SERVER_NAMES = ("llama-server",)
SIBLING_TOOLS = ("llama-bench", "llama-perplexity", "test-backend-ops", "llama-cli", "llama-fit-params")
BACKEND_PATTERNS = {"cuda": "ggml-cuda", "vulkan": "ggml-vulkan", "hip": "ggml-hip", "sycl": "ggml-sycl", "metal": "ggml-metal",
                    "opencl": "ggml-opencl", "blas": "ggml-blas", "rpc": "ggml-rpc", "cpu": "ggml-cpu"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages", "$Recycle.Bin", "System Volume Information",
             "AppData", "Windows", ".cache-loader", "dist-packages"}
MAX_DEPTH = 4
MAX_ENTRIES = 40000
MAX_MODELS_LISTED = 200
MAX_MODEL_HEADERS = 40
MAX_VERSION_RUNS = 8
EXE = ".exe" if os.name == "nt" else ""

# Windows NTSTATUS exit codes a process gets when the loader cannot start it
_LOADER_STATUS = {0xC0000135: "a DLL it imports was not found (on CUDA builds: usually the CUDA runtime, cudart/cublas)",
                  0xC0000139: "an imported function was not found in a DLL (a DLL of a different version was loaded first on PATH)",
                  0xC000007B: "an imported DLL has the wrong architecture (32/64-bit mismatch)"}

FTYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
          14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
          23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
          36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


class Log:
    """What discovery looked at and what it executed, so the report can say exactly that."""

    def __init__(self):
        self.searched = []
        self.executed = []

    def search(self, what, where, found, note=None):
        row = {"what": what, "where": str(where), "found": found}
        if note:
            row["note"] = note
        self.searched.append(row)


# ---- processes -----------------------------------------------------------------------------------

def _win_argv(cmdline):
    """Split a Windows command line exactly as the C runtime does (CommandLineToArgvW)."""
    import ctypes
    from ctypes import wintypes
    fn = ctypes.windll.shell32.CommandLineToArgvW
    fn.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.POINTER(wintypes.LPWSTR)
    n = ctypes.c_int(0)
    p = fn(cmdline, ctypes.byref(n))
    if not p:
        return []
    try:
        return [p[i] for i in range(n.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(p, ctypes.c_void_p))


def _run(cmd, timeout=20, env=None):
    try:
        # stdin is never inherited: under the MCP server it is the JSON-RPC pipe, and PowerShell waits for it to close
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, errors="replace", stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout or "", p.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", str(e)


def server_processes(names=SERVER_NAMES):
    """Running processes whose executable is one of `names`: pid, executable path, argv. Command lines of processes
    owned by other users may be unreadable; those are listed with `argv: None`."""
    procs = []
    if os.name == "nt":
        where = " OR ".join("Name = '%s.exe'" % n.replace("'", "") for n in names)
        # the parent's command line tells a server hotpath launched for an experiment from the user's own
        ps = ("Get-CimInstance Win32_Process -Filter \"%s\" | ForEach-Object { $pp = Get-CimInstance Win32_Process -Filter (\"ProcessId = \" + $_.ParentProcessId) "
              "-ErrorAction SilentlyContinue; [pscustomobject]@{ProcessId=$_.ProcessId; ExecutablePath=$_.ExecutablePath; CommandLine=$_.CommandLine; "
              "ParentCommandLine=$pp.CommandLine} } | ConvertTo-Json -Compress" % where)
        rc, out, _ = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=30)
        if rc != 0 or not out.strip():
            return procs
        data = json.loads(out)
        for d in data if isinstance(data, list) else [data]:
            argv = _win_argv(d["CommandLine"]) if d.get("CommandLine") else None
            procs.append({"pid": d.get("ProcessId"), "exe": d.get("ExecutablePath"), "argv": argv,
                          "launched_by_hotpath": _is_hotpath_cmdline(d.get("ParentCommandLine"))})
        return procs
    if Path("/proc").is_dir():
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                raw = (p / "cmdline").read_bytes()
            except OSError:
                continue
            argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
            if not argv:
                continue
            try:
                exe = os.readlink(p / "exe")
            except OSError:
                exe = argv[0]
            if Path(exe).name in names or Path(argv[0]).name in names:
                parent = None
                try:
                    ppid = (p / "stat").read_text().rsplit(")", 1)[1].split()[1]
                    parent = (Path("/proc") / ppid / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                except (OSError, IndexError):
                    pass
                procs.append({"pid": int(p.name), "exe": exe, "argv": argv, "launched_by_hotpath": _is_hotpath_cmdline(parent)})
        return procs
    rc, out, _ = _run(["ps", "-axww", "-o", "pid=,args="])     # macOS and other Unix: approximate split (untested)
    for line in (out or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            argv = shlex.split(parts[1])
        except ValueError:
            continue
        if argv and Path(argv[0]).name in names:
            procs.append({"pid": int(parts[0]), "exe": argv[0], "argv": argv})
    return procs


def _is_hotpath_cmdline(cmdline):
    """A parent command line that is hotpath itself (`hotpath experiment ...`, `python -m hotpath ...`)."""
    c = (cmdline or "").lower().replace("\\", "/")
    return bool(re.search(r"(-m hotpath\b|/hotpath(\.exe)?[\"']?\s|^hotpath(\.exe)?\s)", c)) and any(k in c for k in ("experiment", "bench", "profile", "kernels"))


def process_argv(pid):
    """{exe, argv} of one process, or None when it cannot be read (another user's process, or it exited)."""
    if not pid:
        return None
    if os.name == "nt":
        ps = "Get-CimInstance Win32_Process -Filter \"ProcessId = %d\" | Select-Object ExecutablePath,CommandLine | ConvertTo-Json -Compress" % int(pid)
        rc, out, _ = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=30)
        if rc != 0 or not out.strip():
            return None
        d = json.loads(out)
        return {"exe": d.get("ExecutablePath"), "argv": _win_argv(d["CommandLine"]) if d.get("CommandLine") else None}
    p = Path("/proc/%d" % int(pid))
    if p.is_dir():
        try:
            argv = [a.decode("utf-8", "replace") for a in (p / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            return None
        try:
            exe = os.readlink(p / "exe")
        except OSError:
            exe = argv[0] if argv else None
        return {"exe": exe, "argv": argv}
    rc, out, _ = _run(["ps", "-ww", "-o", "args=", "-p", str(int(pid))])
    try:
        argv = shlex.split(out.strip()) if out.strip() else None
    except ValueError:
        argv = None
    return {"exe": argv[0] if argv else None, "argv": argv}


def server_launch(pid):
    """How the server process was launched, from its command line: the observed settings that records and diagnosis use.
    Paths stay as they are here (records reduce them to basenames when written)."""
    pa = process_argv(pid)
    if not pa or not pa.get("argv"):
        return {"available": False, "note": "command line of pid %s not readable" % pid}
    parsed = launch.parse(pa["argv"][1:])
    return {"available": True, "pid": pid, "exe": pa.get("exe"), "argv": pa["argv"][1:], "settings": parsed["settings"],
            "path_flags": parsed["path_flags"], "source": "command line of pid %s (observed)" % pid}


def launch_for_record(sl):
    """A server_launch() result fit for a record: path-valued arguments reduced to basenames per argument (a path
    with spaces cannot leak its directories), and no full executable path."""
    if not sl or not sl.get("available"):
        return sl
    st = dict(sl["settings"])
    for k in ("model", "log_file", "model_draft"):
        if isinstance(st.get(k), str):
            st[k] = launch._base(st[k])
    return {"available": True, "pid": sl.get("pid"), "exe": launch._base(sl.get("exe") or ""), "argv_redacted": launch.redact(sl["argv"]),
            "settings": st, "path_flags": sl.get("path_flags"), "source": sl.get("source")}


def listening_ports():
    """{pid: [(address, port), ...]} for listening TCP sockets. Windows uses netstat; elsewhere the port comes
    from the command line and is confirmed by the HTTP probe instead."""
    if os.name != "nt":
        return {}
    out_map = {}
    for proto in ("tcp", "tcpv6"):
        rc, out, _ = _run(["netstat", "-ano", "-p", proto])
        for line in (out or "").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING" and parts[4].isdigit():
                addr, _, port = parts[1].rpartition(":")
                if port.isdigit():
                    out_map.setdefault(int(parts[4]), []).append((addr.strip("[]"), int(port)))
    return out_map


def _get(url, timeout=3.0, text=False):
    req = urllib.request.Request(url, headers={"Accept": "text/plain" if text else "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
        return raw if text else json.loads(raw)


def parse_metrics(text):
    """Prometheus text from llama-server's /metrics -> {name: value} for the llamacpp:* series."""
    out = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0][len("llamacpp:"):]] = float(parts[1])
                except ValueError:
                    pass
    return out


def probe_http(base):
    """GET /health, /props and /metrics on a loopback base URL. No inference, no state change."""
    res = {"base_url": base}
    try:
        res["health"] = _get(base + "/health", timeout=2.0)
    except urllib.error.HTTPError as e:
        res["health"] = {"status": "http %d" % e.code}
    except (urllib.error.URLError, OSError, ValueError) as e:
        res["health"] = {"status": "unreachable", "error": str(e)[:120]}
        return res
    try:
        p = _get(base + "/props")
        gen = p.get("default_generation_settings") or {}
        res["props"] = {"build_info": p.get("build_info"), "model_path": p.get("model_path"), "n_ctx_per_slot": gen.get("n_ctx"),
                        "total_slots": p.get("total_slots")}
    except urllib.error.HTTPError as e:
        res["props"] = {"error": "HTTP %d%s" % (e.code, " (API key required)" if e.code == 401 else "")}
    except (urllib.error.URLError, OSError, ValueError) as e:
        res["props"] = {"error": str(e)[:120]}
    try:
        res["metrics"] = parse_metrics(_get(base + "/metrics", text=True))
        res["metrics_enabled"] = True
    except urllib.error.HTTPError as e:
        res["metrics_enabled"] = False
        res["metrics_note"] = "HTTP %d: start llama-server with --metrics to expose aggregate prompt/generation counters" % e.code
    except (urllib.error.URLError, OSError, ValueError) as e:
        res["metrics_enabled"] = False
        res["metrics_note"] = str(e)[:120]
    return res


_LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::", "")


def describe_server_process(proc, ports_by_pid):
    """Settings from a process's command line plus the loopback endpoint it serves."""
    argv = proc.get("argv")
    row = {"pid": proc.get("pid"), "exe": proc.get("exe"), "argv": argv, "launched_by_hotpath": bool(proc.get("launched_by_hotpath"))}
    if not argv:
        row["note"] = "command line not readable (process of another user?)"
        return row
    parsed = launch.parse(argv[1:])
    s = parsed["settings"]
    row.update({"settings": s, "path_flags": parsed["path_flags"], "other_flags": parsed["other_flags"]})
    host = str(s.get("host") or "127.0.0.1")
    listening = ports_by_pid.get(proc.get("pid")) or []
    port = launch.int_or_none(s.get("port")) or 8080
    if listening and port not in [p for _, p in listening]:
        port = listening[0][1]          # the port came from LLAMA_ARG_PORT or similar, not the command line
    row["listening"] = listening
    row["host"] = host
    row["port"] = port
    if host not in _LOOPBACK:
        row["http"] = {"skipped": "listens on %s; hotpath only contacts loopback addresses" % host}
        row["exposed"] = True
    else:
        row["exposed"] = host in ("0.0.0.0", "::")
        row["base_url"] = "http://127.0.0.1:%d" % port
        row["http"] = probe_http(row["base_url"])
    return row


# ---- installations -------------------------------------------------------------------------------

def pe_imports(path):
    """DLL names a Windows PE image imports (normal and delay-load tables). Reads headers and import strings only.
    None for a file that is not a PE image."""
    try:
        f = open(path, "rb")
    except OSError:
        return None
    with f:
        mz = f.read(64)
        if len(mz) < 64 or mz[:2] != b"MZ":
            return None
        (e_lfanew,) = struct.unpack_from("<I", mz, 0x3C)
        f.seek(e_lfanew)
        if f.read(4) != b"PE\0\0":
            return None
        coff = f.read(20)
        if len(coff) < 20:
            return None
        nsec, opt_size = struct.unpack_from("<H", coff, 2)[0], struct.unpack_from("<H", coff, 16)[0]
        opt = f.read(opt_size)
        if len(opt) < 2:
            return None
        magic = struct.unpack_from("<H", opt, 0)[0]
        dd = {0x20B: 112, 0x10B: 96}.get(magic)
        if dd is None or len(opt) < dd:
            return None
        ndirs = struct.unpack_from("<I", opt, dd - 4)[0]
        secs = []
        for _ in range(nsec):
            s = f.read(40)
            if len(s) < 40:
                break
            vsize, vaddr, rsize, rptr = struct.unpack_from("<IIII", s, 8)
            secs.append((vaddr, max(vsize, rsize), rptr))

        def off(rva):
            for va, size, ptr in secs:
                if va <= rva < va + size:
                    return rva - va + ptr
            return None

        def cstr(rva):
            o = off(rva)
            if o is None:
                return None
            f.seek(o)
            return f.read(260).split(b"\0", 1)[0].decode("ascii", "replace") or None

        names = []
        for index, width, name_at in ((1, 20, 12), (13, 32, 4)):      # import table, delay-load import table
            if index >= ndirs:
                continue
            rva, size = struct.unpack_from("<II", opt, dd + 8 * index)
            o = off(rva) if rva else None
            for i in range(4096 if o is not None else 0):
                f.seek(o + width * i)
                d = f.read(width)
                if len(d) < width or d == b"\0" * width:
                    break
                n = cstr(struct.unpack_from("<I", d, name_at)[0])
                if n:
                    names.append(n)
        return names


def _system_dirs():
    root = os.environ.get("SystemRoot") or "C:/Windows"
    return [Path(root) / "System32", Path(root)]


def _path_dirs(extra=None):
    return [Path(p) for p in (list(extra or []) + os.environ.get("PATH", "").split(os.pathsep)) if p]


def cuda_toolkit_bin_dirs():
    """Directories a CUDA toolkit install puts its runtime libraries in (CUDA 13 moved them to bin/x64)."""
    dirs = []
    for k, v in os.environ.items():
        if k.upper().startswith("CUDA_PATH") and v:
            dirs += [Path(v) / "bin", Path(v) / "bin" / "x64"]
    if os.name == "nt":
        for d in sorted(glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"), reverse=True):
            dirs += [Path(d) / "bin", Path(d) / "bin" / "x64"]
    else:
        for d in sorted(glob.glob("/usr/local/cuda*"), reverse=True):
            dirs += [Path(d) / "lib64", Path(d) / "lib"]
    seen, out = set(), []
    for d in dirs:
        k = str(d).lower()
        if k not in seen and d.is_dir():
            seen.add(k)
            out.append(d)
    return out


_CUDA_LIB = re.compile(r"^(cudart|cublas|cublaslt|nvrtc|nvjitlink|cufft|curand|cusparse)(64_(\d+))?", re.I)


def windows_dependency_check(bin_dir, path_prepend=None):
    """Which DLLs the llama/ggml executables in bin_dir import that Windows would not find (next to the binary,
    System32 or PATH), and, for CUDA runtime DLLs, a toolkit directory that has them."""
    bin_dir = Path(bin_dir)
    files = [p for p in bin_dir.iterdir() if p.is_file() and p.suffix.lower() in (".exe", ".dll")
             and p.name.lower().startswith(("llama-server", "ggml", "llama.dll", "llama-common", "llama-server-impl"))]
    imported = {}
    for p in files:
        for n in pe_imports(p) or []:
            imported.setdefault(n.lower(), set()).add(p.name)
    local = {p.name.lower() for p in bin_dir.iterdir() if p.is_file()}
    search = _system_dirs() + _path_dirs(path_prepend)
    missing = []
    for name in sorted(imported):
        if name.startswith(("api-ms-win-", "ext-ms-")) or name in local:
            continue
        if any((d / name).is_file() for d in search):
            continue
        missing.append(name)
    cuda_needed = sorted({n for n in imported if _CUDA_LIB.match(n)})
    majors = sorted({int(m.group(3)) for n in cuda_needed for m in [_CUDA_LIB.match(n)] if m and m.group(3)})
    cuda_missing = [n for n in missing if _CUDA_LIB.match(n)]
    fix = None
    if cuda_missing:
        for d in cuda_toolkit_bin_dirs():
            if all((d / n).is_file() for n in cuda_missing):
                fix = str(d)
                break
    return {"imports_checked": sorted(p.name for p in files), "cuda_libraries_imported": cuda_needed, "cuda_major_required": majors[-1] if majors else None,
            "unresolved": missing, "cuda_unresolved": cuda_missing, "path_prepend_fix": fix}


def linux_dependency_check(binary):
    """`ldd` on the binary and its ggml CUDA library: shared libraries reported as not found (untested outside Windows)."""
    if not shutil.which("ldd"):
        return {"note": "ldd not available; shared-library dependencies not checked"}
    targets = [Path(binary)] + [p for p in Path(binary).parent.glob("libggml-cuda*.so*")]
    missing = set()
    for t in targets:
        rc, out, _ = _run(["ldd", str(t)], timeout=15)
        for line in (out or "").splitlines():
            if "not found" in line:
                missing.add(line.split("=>")[0].strip())
    return {"unresolved": sorted(missing), "cuda_unresolved": sorted(n for n in missing if _CUDA_LIB.match(n))}


def _exit_reason(rc):
    if rc is None:
        return None
    code = rc & 0xFFFFFFFF
    return _LOADER_STATUS.get(code)


def run_version(binary, log, path_prepend=None):
    """`binary --version`: the version line and whether the program starts at all. Recorded in the executed list."""
    env = dict(os.environ)
    if path_prepend:
        env["PATH"] = os.pathsep.join(str(x) for x in path_prepend) + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    rc, out, err = _run([str(binary), "--version"], timeout=15, env=env)
    lines = [ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip()]
    log.executed.append({"cmd": "%s --version" % Path(binary).name, "with_path_prepend": bool(path_prepend), "returncode": rc,
                         "seconds": round(time.time() - t0, 1)})
    res = {"returncode": rc, "starts": rc == 0, "lines": lines[:3]}
    m = next((re.search(r"version:\s*(.+)", ln) for ln in lines if "version:" in ln), None)
    if m:
        res["version"] = m.group(1).strip()
    reason = _exit_reason(rc)
    if reason:
        res["failure"] = reason
    elif rc not in (0, None):
        res["failure"] = (lines[-1] if lines else "exited with code %s and printed nothing" % rc)[:200]
    elif rc is None:
        res["failure"] = err[:200]
    return res


def inspect_install(binary, origin, log, run=True, cfg_path_prepend=None):
    """One llama-server installation: backends, sibling tools, unresolved libraries, and whether it starts."""
    b = Path(binary)
    d = b.parent
    names = [p.name.lower() for p in d.iterdir()] if d.is_dir() else []
    backends = sorted({k for k, pat in BACKEND_PATTERNS.items() if any(n.startswith(pat) or n.startswith("lib" + pat) for n in names)})
    siblings = {t: str(d / (t + EXE)) for t in SIBLING_TOOLS if (d / (t + EXE)).is_file()}
    row = {"binary": str(b), "dir": str(d), "origin": origin, "backends": backends or ["not visible as separate libraries (static build?)"],
           "siblings": siblings}
    if os.name == "nt":
        row["dependencies"] = windows_dependency_check(d, cfg_path_prepend)
    elif "cuda" in backends:
        row["dependencies"] = linux_dependency_check(b)
    if run:
        v = run_version(b, log)
        row["version_check"] = v
        fix = (row.get("dependencies") or {}).get("path_prepend_fix")
        if not v["starts"] and fix:
            v2 = run_version(b, log, path_prepend=[fix])
            row["version_check_with_fix"] = v2
            if v2["starts"]:
                row["needs_path_prepend"] = fix
        elif not v["starts"] and cfg_path_prepend:
            v2 = run_version(b, log, path_prepend=cfg_path_prepend)
            row["version_check_with_config_path_prepend"] = v2
            if v2["starts"]:
                row["needs_path_prepend"] = os.pathsep.join(cfg_path_prepend)
    else:
        row["version_check"] = {"skipped": "found by a directory search; not executed"}
    return row


def standard_install_candidates():
    home = Path.home()
    c = []
    if os.name == "nt":
        la = os.environ.get("LOCALAPPDATA")
        if la:
            c.append(Path(la) / "Microsoft" / "WinGet" / "Links" / "llama-server.exe")
            c += [Path(p) / "llama-server.exe" for p in glob.glob(str(Path(la) / "Microsoft" / "WinGet" / "Packages" / "ggml.llamacpp*"))]
        c += [home / "scoop" / "apps" / "llama.cpp" / "current" / "llama-server.exe",
              home / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe", home / "llama.cpp" / "build" / "bin" / "llama-server.exe"]
    else:
        c += [Path("/usr/local/bin/llama-server"), Path("/usr/bin/llama-server"), Path("/opt/homebrew/bin/llama-server"),
              home / ".local" / "bin" / "llama-server", home / "llama.cpp" / "build" / "bin" / "llama-server", home / "llama.cpp" / "llama-server"]
    return c


# ---- bounded directory search --------------------------------------------------------------------

def check_root(root):
    """A directory the user asked to search. Drive and filesystem roots are refused: name a narrower directory."""
    p = Path(root).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass
    if not p.is_dir():
        raise ValueError("not a directory: %s" % root)
    if p.parent == p:
        raise ValueError("refusing to search a whole drive or filesystem root (%s); name the directory that holds your runtime or models" % p)
    return p


def walk(root, max_depth=MAX_DEPTH, max_entries=MAX_ENTRIES, want=None):
    """Files under root matching `want(name)`, to a fixed depth, without following links. Returns (files, entries_seen, truncated)."""
    found, seen = [], 0
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            seen += 1
            if seen > max_entries:
                return found, seen, True
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < max_depth and e.name not in SKIP_DIRS and not (e.name.startswith(".") and depth > 0):
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file() and (want is None or want(e.name)):
                    found.append(Path(e.path))
            except OSError:
                continue
    return found, seen, False


def _is_server_exe(name):
    return name.lower() in {n + EXE for n in SERVER_NAMES}


def _is_gguf(name):
    return name.lower().endswith(".gguf")


_SCRIPT_EXT = (".bat", ".cmd", ".ps1", ".sh", ".yaml", ".yml", ".json", ".toml")


def launch_scripts(files, limit=20):
    """Launch scripts and configs (llama-swap YAML, batch/shell files) that start llama-server: the command lines
    as written. Only small text files are read."""
    out = []
    for p in files:
        try:
            if p.stat().st_size > 256 * 1024:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "llama-server" in line and "--" in line:
                out.append({"file": str(p), "line": line.strip()[:400]})
                break
        if len(out) >= limit:
            break
    return out


# ---- models --------------------------------------------------------------------------------------

def standard_model_dirs():
    home, env = Path.home(), os.environ
    d = []
    if env.get("LLAMA_CACHE"):
        d.append(Path(env["LLAMA_CACHE"]))
    elif os.name == "nt" and env.get("LOCALAPPDATA"):
        d.append(Path(env["LOCALAPPDATA"]) / "llama.cpp")
    elif sys.platform == "darwin":
        d.append(home / "Library" / "Caches" / "llama.cpp")
    else:
        d.append(Path(env.get("XDG_CACHE_HOME") or home / ".cache") / "llama.cpp")
    hub = env.get("HF_HUB_CACHE") or (str(Path(env["HF_HOME"]) / "hub") if env.get("HF_HOME") else str(home / ".cache" / "huggingface" / "hub"))
    d += [Path(hub), home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models", home / "models", home / "Models"]
    return d


_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def model_info(path, gpu_total_mib=None):
    p = Path(path)
    row = {"path": str(p), "size_bytes": p.stat().st_size if p.is_file() else None}
    m = _SPLIT.search(p.name)
    if m:
        parts = sorted(p.parent.glob(_SPLIT.sub("-*-of-%s.gguf" % m.group(2), p.name)))
        row["split_parts"] = len(parts)
        row["size_bytes"] = sum(x.stat().st_size for x in parts)
    try:
        hdr = gguf_mod.read_header(p)
    except (OSError, ValueError, struct.error, KeyError, UnicodeDecodeError) as e:
        row["header_error"] = "%s: %s" % (type(e).__name__, str(e)[:80])
        return row
    meta = hdr["meta"]
    arch = meta.get("general.architecture")
    s = gguf_mod.summarize(hdr) if arch and arch != "clip" else {}
    ft = meta.get("general.file_type")
    role = "projector (mmproj)" if arch == "clip" else ("embedding/reranker" if ("%s.pooling_type" % arch) in meta else "generative")
    row.update({"architecture": arch, "name": meta.get("general.name"), "size_label": meta.get("general.size_label"), "role": role,
                "file_type": FTYPES.get(ft, "ftype %s" % ft if ft is not None else None),
                "n_layer": s.get("n_layer"), "n_ctx_train": s.get("n_ctx_train"), "kv_bytes_per_token_f16": s.get("kv_bytes_per_token_f16"),
                "weight_bytes": s.get("total_bytes") or row["size_bytes"]})
    if gpu_total_mib and row["weight_bytes"]:
        w = row["weight_bytes"] / (1 << 20)
        row["vram_fit"] = ("weights %.0f MiB of %.0f MiB VRAM: fit, before KV cache and compute buffers" % (w, gpu_total_mib) if w < gpu_total_mib * 0.95
                           else "weights %.0f MiB exceed %.0f MiB VRAM: only part of the model can be offloaded to the GPU" % (w, gpu_total_mib))
    return row


# ---- the whole picture ---------------------------------------------------------------------------

def _resolved(p):
    try:
        return str(Path(p).resolve()).lower()
    except OSError:
        return str(p).lower()


def discover(cfg=None, roots=(), run_found=True, run_in_roots=True, max_depth=MAX_DEPTH, read_headers=True, prefer=None):
    """Everything above. `roots`: extra directories to search (bounded). `run_found=False` executes nothing;
    `run_in_roots=False` executes `--version` only for running, configured, PATH and standard-location binaries,
    never for a binary that was found by searching a directory. `prefer`: {"runtime_binary"|"model_path"|"server_url":
    value} chosen by the user where several candidates qualify."""
    cfg = cfg or {}
    prefer = prefer or {}
    roots = list(roots)
    for k in ("runtime_binary", "model_path"):
        if prefer.get(k):
            if not Path(prefer[k]).is_file():
                raise ValueError("%s does not exist: %s" % (k, prefer[k]))
            try:
                roots.append(check_root(Path(prefer[k]).parent))
            except ValueError:
                pass
    log = Log()
    t0 = time.time()
    names = tuple(cfg.get("discovery_process_names") or SERVER_NAMES)

    gpus = gpu_mod.gpus_all()
    gpu_total = max([float(g["memory.total"]) for g in gpus if str(g.get("memory.total", "")).replace(".", "", 1).isdigit()], default=None)
    driver_cuda = gpu_mod.driver_cuda_version()
    apps = gpu_mod.compute_apps()

    procs = server_processes(names)
    log.search("running llama-server processes", "process table (%s)" % ", ".join(names), len(procs))
    ports = listening_ports() if procs else {}
    servers = [describe_server_process(p, ports) for p in procs]

    # endpoints with no visible process (another user's server, a container with a forwarded port)
    probed = {s.get("base_url") for s in servers}
    extra_urls = [u for u in [cfg.get("server_url")] if u and u not in probed]
    endpoints = []
    for u in extra_urls:
        host = (urlparse(u).hostname or "").lower()
        if host in ("127.0.0.1", "localhost", "::1"):
            h = probe_http(u.rstrip("/"))
            if (h.get("health") or {}).get("status") not in ("unreachable",):
                endpoints.append(h)
            log.search("configured server_url", u, int((h.get("health") or {}).get("status") not in ("unreachable",)))

    # installations
    cands = []
    for s in servers:
        if s.get("exe"):
            cands.append((s["exe"], "running process %s" % s.get("pid")))
    if cfg.get("runtime_binary"):
        cands.append((cfg["runtime_binary"], "hotpath.local.json runtime_binary"))
    for n in names:
        w = shutil.which(n)
        if w:
            cands.append((w, "PATH"))
    log.search("llama-server on PATH", "PATH", sum(1 for _, o in cands if o == "PATH"))
    std = [p for p in standard_install_candidates() if p.is_file()]
    log.search("standard install locations", "; ".join(str(p.parent) for p in standard_install_candidates()[:4]) + " ...", len(std))
    cands += [(str(p), "standard location") for p in std]

    root_files, scripts = [], []
    for r in roots:
        files, seen, trunc = walk(r, max_depth=max_depth, want=lambda n: _is_server_exe(n) or _is_gguf(n) or n.lower().endswith(_SCRIPT_EXT))
        exes = [f for f in files if _is_server_exe(f.name)]
        root_files += files
        cands += [(str(f), "found under %s" % r) for f in exes]
        scripts += launch_scripts([f for f in files if f.suffix.lower() in _SCRIPT_EXT and "hotpath" not in f.name.lower()])
        log.search("directory you named", r, len(exes) + sum(1 for f in files if _is_gguf(f.name)),
                   "depth %d, %d entries visited%s" % (max_depth, seen, "; stopped at the entry cap" if trunc else ""))

    installs, seen_bin = [], set()
    runs_left = MAX_VERSION_RUNS
    for path, origin in cands:
        k = _resolved(path)
        if k in seen_bin or not Path(path).is_file():
            continue
        seen_bin.add(k)
        may_run = run_found and runs_left > 0 and (run_in_roots or not origin.startswith("found under"))
        runs_left -= 1 if may_run else 0
        installs.append(inspect_install(Path(path).resolve() if origin == "standard location" else path, origin, log, run=may_run,
                                        cfg_path_prepend=cfg.get("path_prepend")))

    # models
    model_paths = []
    for s in servers:
        mp = (s.get("settings") or {}).get("model")
        props_mp = ((s.get("http") or {}).get("props") or {}).get("model_path")
        for m in (mp, props_mp):
            if m and Path(m).is_file():
                model_paths.append((m, "running process %s" % s.get("pid")))
    if cfg.get("model_path") and Path(cfg["model_path"]).is_file():
        model_paths.append((cfg["model_path"], "hotpath.local.json model_path"))
    dirs = []
    for m, _ in model_paths:
        dirs.append((Path(m).parent, "next to a known model"))
    dirs += [(d, "standard model location") for d in standard_model_dirs()]
    for d, why in dirs:
        if not d.is_dir():
            continue
        files, seen, trunc = walk(d, max_depth=5, max_entries=MAX_ENTRIES // 4, want=_is_gguf)
        log.search("GGUF models (%s)" % why, d, len(files), "stopped at the entry cap" if trunc else None)
        model_paths += [(str(f), why) for f in files]
    model_paths += [(str(f), "found under a directory you named") for f in root_files if _is_gguf(f.name)]
    models, seen_m = [], set()
    for m, origin in model_paths:
        k = _resolved(m)
        mm = _SPLIT.search(Path(m).name)
        if k in seen_m or (mm and mm.group(1) != "00001"):
            continue
        seen_m.add(k)
        if len(models) >= MAX_MODELS_LISTED:
            break
        if read_headers and len(models) < MAX_MODEL_HEADERS:
            row = model_info(m, gpu_total)
        else:
            row = {"path": m, "size_bytes": Path(m).stat().st_size}
        row["origin"] = origin
        models.append(row)

    from hotpath import adapters
    ranked = sorted(installs, key=lambda i: (not i["origin"].startswith("running"), "cuda" not in i["backends"]))
    bench_dir = cfg.get("runtime_bin_dir") or next((i["dir"] for i in ranked if "llama-bench" in i["siblings"]), None)
    tbo_dir = cfg.get("build_dir") or cfg.get("fork_build_dir") or next((i["dir"] for i in ranked if "test-backend-ops" in i["siblings"]), None)
    tools = adapters.all_status(bin_dir=bench_dir, build_dir=tbo_dir)
    from hotpath.measure import schema_validator
    tools.append({"tool": "jsonschema", "found": bool(schema_validator()), "path": None,
                  "install_hint": None if schema_validator() else 'optional: pip install "hotpath[schema]"'})

    logs = []
    for s in servers:
        lf = (s.get("settings") or {}).get("log_file")
        if lf:
            p = Path(lf)
            logs.append({"path": str(p), "origin": "--log-file of process %s" % s.get("pid"), "exists": p.is_file(),
                         "size_bytes": p.stat().st_size if p.is_file() else None})
    if cfg.get("server_log"):
        p = Path(cfg["server_log"])
        logs.append({"path": str(p), "origin": "hotpath.local.json server_log", "exists": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None})

    # under Windows WDDM every desktop program with a GPU context is listed; keep the ones that look like inference
    server_pids = {s.get("pid") for s in servers}
    apps = [a for a in apps if a["pid"] in server_pids or re.search(r"llama|ollama|python|kobold|lm.?studio|vllm|text-gen|comfy", a["name"], re.I)]
    rec = {"gpus": gpus, "driver_cuda_version": driver_cuda, "gpu_compute_apps": apps, "servers": servers, "endpoints": endpoints,
           "installs": installs, "models": models, "launch_scripts": scripts, "tools": tools, "logs": logs,
           "searched": log.searched, "executed": log.executed, "seconds": round(time.time() - t0, 1)}
    rec["issues"] = issues(rec)
    rec["suggested_config"], rec["suggested_config_sources"], rec["ambiguous"] = suggest_config(rec, cfg, prefer)
    rec["unknowns"] = unknowns(rec)
    return rec


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except (TypeError, ValueError):
        return None


def issues(rec):
    """Problems that stop or slow the setup, each with the fix hotpath knows."""
    out = []
    drv = _major(rec.get("driver_cuda_version"))
    for i in rec["installs"]:
        v = i.get("version_check") or {}
        dep = i.get("dependencies") or {}
        if v.get("failure"):
            fix = i.get("needs_path_prepend")
            out.append({"severity": "high", "where": i["binary"],
                        "what": "llama-server does not start: %s%s" % (v["failure"], " (unresolved: %s)" % ", ".join(dep["unresolved"][:6]) if dep.get("unresolved") else ""),
                        "fix": ("put %s first on PATH for this binary: hotpath sets it as path_prepend in hotpath.local.json (verified: --version starts with it)" % fix) if fix
                        else ("install the CUDA toolkit or the runtime DLL package that matches this build, or copy %s next to the binary" % ", ".join(dep.get("cuda_unresolved") or dep.get("unresolved") or ["the missing libraries"]))})
        elif dep.get("cuda_unresolved") and not i.get("needs_path_prepend"):
            out.append({"severity": "medium", "where": i["binary"], "what": "CUDA libraries imported but not found: %s" % ", ".join(dep["cuda_unresolved"]),
                        "fix": "add %s to path_prepend" % dep["path_prepend_fix"] if dep.get("path_prepend_fix") else "install the matching CUDA runtime"})
        need = dep.get("cuda_major_required")
        if need and drv and drv < need:
            out.append({"severity": "high", "where": i["binary"], "what": "build needs CUDA %d but the driver supports CUDA %s" % (need, rec.get("driver_cuda_version")),
                        "fix": "update the NVIDIA driver, or use a build for CUDA %d" % drv})
    for s in user_servers(rec):
        if s.get("exposed"):
            out.append({"severity": "info", "where": "pid %s" % s.get("pid"), "what": "the server listens on %s (all interfaces), reachable from the network" % s.get("host"),
                        "fix": "use --host 127.0.0.1 unless other machines must reach it (and then set --api-key)"})
        h = s.get("http") or {}
        if (h.get("health") or {}).get("status") == "unreachable":
            out.append({"severity": "medium", "where": "pid %s" % s.get("pid"), "what": "llama-server process is running but %s does not answer" % s.get("base_url"),
                        "fix": "it may still be loading the model; run discovery again in a minute"})
        if s.get("settings") and not (s["settings"].get("log_file")):
            out.append({"severity": "info", "where": "pid %s" % s.get("pid"), "what": "no --log-file: per-request timing history is only on the console",
                        "fix": "add --log-file PATH to the launch so `hotpath observe --log` can diagnose real traffic passively (or redirect stderr to a file)"})
        if s.get("settings") and h.get("metrics_enabled") is False:
            out.append({"severity": "info", "where": "pid %s" % s.get("pid"), "what": "/metrics is off",
                        "fix": "add --metrics to the launch: aggregate prompt/generation time and prompt-cache reuse become readable without a log"})
    if not rec["gpus"]:
        out.append({"severity": "high", "where": "nvidia-smi", "what": "no NVIDIA GPU visible (nvidia-smi missing or failing)", "fix": "install the NVIDIA driver; hotpath's GPU features need nvidia-smi"})
    return out


def user_servers(rec):
    """Running servers that are not hotpath's own experiment servers."""
    return [s for s in rec["servers"] if not s.get("launched_by_hotpath")]


def healthy_user_servers(rec):
    return [s for s in user_servers(rec) if ((s.get("http") or {}).get("health") or {}).get("status") == "ok"]


def _bin_dir_tools(install):
    return install["dir"] if "llama-bench" in install.get("siblings", {}) else None


def suggest_config(rec, cfg, prefer=None):
    """hotpath.local.json values that discovery can stand behind, each with where it came from. Nothing ambiguous is
    chosen: when several servers, installs or models qualify, the key is left out and the choices are listed
    (`prefer` holds the user's explicit choice among them)."""
    out, src, amb = {}, {}, {}
    prefer = prefer or {}
    live = [s for s in user_servers(rec) if (s.get("http") or {}).get("props") and not (s["http"]["props"] or {}).get("error")]
    if prefer.get("server_url"):
        live = [s for s in live if s.get("base_url") == prefer["server_url"].rstrip("/")]
        if not live:
            raise ValueError("no running llama-server answers at %s" % prefer["server_url"])
    working = [i for i in rec["installs"] if (i.get("version_check") or {}).get("starts") or i.get("needs_path_prepend")]
    if prefer.get("runtime_binary"):
        working = [i for i in rec["installs"] if _resolved(i["binary"]) == _resolved(prefer["runtime_binary"])]
        if live and working and all(_resolved(s.get("exe") or "") != _resolved(prefer["runtime_binary"]) for s in live):
            live = []      # the user chose a different binary than the running server's: do not mix the two
    if len(live) == 1:
        s = live[0]
        st = s["settings"]
        out["server_url"], src["server_url"] = s["base_url"], "running llama-server pid %s" % s["pid"]
        if s.get("exe"):
            out["runtime_binary"], src["runtime_binary"] = s["exe"], "executable of pid %s" % s["pid"]
        mp = st.get("model") or (s["http"]["props"] or {}).get("model_path")
        if mp and Path(mp).is_file():
            out["model_path"], src["model_path"] = str(Path(mp)), "model of pid %s" % s["pid"]
        if st.get("log_file"):
            out["server_log"], src["server_log"] = st["log_file"], "--log-file of pid %s" % s["pid"]
        from hotpath.propose import strip_runner_owned
        kept, _removed = strip_runner_owned(s["argv"][1:])
        out["server_args"], src["server_args"] = kept, "command line of pid %s (minus model, host, port, log file and other runner-owned flags)" % s["pid"]
    elif len(live) > 1:
        amb["server_url"] = [s["base_url"] for s in live]
    if "runtime_binary" not in out:
        if len(working) == 1:
            out["runtime_binary"], src["runtime_binary"] = working[0]["binary"], working[0]["origin"]
        elif len(working) > 1:
            amb["runtime_binary"] = [i["binary"] for i in working]
    chosen = next((i for i in rec["installs"] if _resolved(i["binary"]) == _resolved(out.get("runtime_binary", ""))), None)
    if chosen:
        if _bin_dir_tools(chosen):
            out["runtime_bin_dir"], src["runtime_bin_dir"] = chosen["dir"], "directory of runtime_binary (has llama-bench)"
        if "test-backend-ops" in chosen.get("siblings", {}):
            out["build_dir"], src["build_dir"] = chosen["dir"], "directory of runtime_binary (has test-backend-ops)"
        if chosen.get("needs_path_prepend"):
            out["path_prepend"] = chosen["needs_path_prepend"].split(os.pathsep)
            src["path_prepend"] = "runtime_binary only starts with this directory first on PATH (verified with --version)"
    if prefer.get("model_path"):
        out["model_path"], src["model_path"] = str(Path(prefer["model_path"])), "chosen with --model"
    if "model_path" not in out:
        gen = [m for m in rec["models"] if m.get("role") == "generative"]
        if len(gen) == 1:
            out["model_path"], src["model_path"] = gen[0]["path"], gen[0]["origin"]
        elif len(gen) > 1:
            amb["model_path"] = [m["path"] for m in gen][:12]
    out["owned_servers"] = list(cfg.get("owned_servers") or ["http://127.0.0.1:8099"])
    src["owned_servers"] = "hotpath's own experiment port; your running server is never added automatically"
    return out, src, amb


def unknowns(rec):
    u = []
    if not rec["servers"] and not rec["endpoints"]:
        u.append("No running llama-server found: live configuration, request history and GPU use under load are unknown until one runs.")
    if any(s.get("settings") for s in rec["servers"]):
        u.append("Settings given through LLAMA_ARG_* environment variables or a launcher's defaults are not visible in a command line; "
                 "`unset` means not on the command line, not off.")
    if not any(l.get("exists") for l in rec["logs"]):
        u.append("No server log known: per-request prompt/generation timings of real traffic are unavailable. Start llama-server with "
                 "--log-file PATH (or redirect its stderr to a file) and pass it to `hotpath observe --log`.")
    if not rec["installs"]:
        u.append("No llama-server binary found. Point hotpath at it: `hotpath discover --root DIR` or runtime_binary in hotpath.local.json.")
    if rec["gpus"] and not rec["gpus"][0].get("bandwidth_gbs_known"):
        u.append("GPU %s is not in hotpath's bandwidth table: pass --bandwidth-gbs to diagnose for decode-ceiling estimates." % rec["gpus"][0].get("name"))
    return u


def discovery_brief(rec):
    """A few lines for `inspect`: the setup and its problems. The full report is the discovery .md file."""
    L = ["# Setup"]
    for g in rec["gpus"][:2]:
        L.append("- GPU %s: %s, %s MiB (%s MiB in use), driver CUDA %s" % (g.get("index"), g.get("name"), g.get("memory.total"), g.get("memory.used"), rec.get("driver_cuda_version")))
    if not rec["gpus"]:
        L.append("- no NVIDIA GPU visible")
    for s in rec["servers"]:
        st = s.get("settings") or {}
        h = s.get("http") or {}
        L.append("- llama-server pid %s on port %s%s: model %s; %s; log %s; /metrics %s" % (
            s.get("pid"), s.get("port"), " (a hotpath experiment's own server)" if s.get("launched_by_hotpath") else "",
            Path(st.get("model") or ((h.get("props") or {}).get("model_path") or "?")).name, launch.describe(st),
            Path(st["log_file"]).name if st.get("log_file") else "none", "on" if h.get("metrics_enabled") else "off"))
    if not rec["servers"]:
        L.append("- no running llama-server")
    ok = sum(1 for i in rec["installs"] if (i.get("version_check") or {}).get("starts") or i.get("needs_path_prepend"))
    L.append("- %d llama-server installation(s) found (%d start), %d GGUF model(s)" % (len(rec["installs"]), ok, len(rec["models"])))
    for x in rec["issues"]:
        if x["severity"] in ("high", "medium"):
            L.append("- problem [%s]: %s. Fix: %s" % (x["severity"], x["what"], x["fix"]))
    return "\n".join(L)


def experiment_vram_note(rec, model_bytes=None):
    """Whether a hotpath experiment (a second copy of the model on its own port) can fit next to what runs now.
    hotpath never stops a server it did not start; this only says what the user has to do first."""
    if not rec.get("gpus"):
        return None
    g = rec["gpus"][0]
    try:
        free = float(g["memory.total"]) - float(g["memory.used"])
    except (KeyError, TypeError, ValueError):
        return None
    if model_bytes is None:
        live = [s for s in rec["servers"] if (s.get("settings") or {}).get("model")]
        sizes = [m.get("size_bytes") for m in rec["models"] for s in live if _resolved(m["path"]) == _resolved(s["settings"]["model"])]
        model_bytes = max([x for x in sizes if x], default=None)
    if not model_bytes:
        return None
    need = model_bytes / (1 << 20)
    if need < free * 0.9:
        return "about %.0f MiB of VRAM is free now; the model's weights need about %.0f MiB, so an experiment server fits next to what runs now" % (free, need)
    holders = ", ".join("llama-server pid %s on port %s" % (s.get("pid"), s.get("port")) for s in rec["servers"]) or "other programs"
    return ("only about %.0f MiB of VRAM is free and the model's weights need about %.0f MiB: an experiment cannot load a second copy while %s holds the GPU. "
            "Stop it yourself before the experiment (hotpath never stops, restarts or reconfigures a server it did not start), or run with --wait-s to wait for the GPU" % (free, need, holders))


def _mib(b):
    return "%.0f MiB" % (b / (1 << 20)) if b else "?"


def discovery_markdown(rec):
    L = ["# Discovery (passive)", ""]
    g = rec["gpus"]
    L += ["## GPU", ""]
    if g:
        for x in g:
            L.append("- GPU %s: %s, %s MiB (%s MiB in use), compute capability %s, driver %s, driver CUDA %s%s" % (
                x.get("index"), x.get("name"), x.get("memory.total"), x.get("memory.used"), x.get("compute_cap"), x.get("driver_version"),
                rec.get("driver_cuda_version"), "" if x.get("bandwidth_gbs_known") else " (not in the bandwidth table)"))
        if rec.get("gpu_compute_apps"):
            L.append("- compute processes on the GPU: %s" % ", ".join("%s (pid %s)" % (a["name"], a["pid"]) for a in rec["gpu_compute_apps"]))
    else:
        L.append("- none visible (nvidia-smi missing)")
    L += ["", "## Running servers", ""]
    if not rec["servers"] and not rec["endpoints"]:
        L.append("- none")
    for s in rec["servers"]:
        h = s.get("http") or {}
        p = h.get("props") or {}
        L.append("- pid %s: %s on %s:%s, health %s%s" % (s.get("pid"), Path(s.get("exe") or "?").name, s.get("host"), s.get("port"), (h.get("health") or {}).get("status"),
                                                      " (launched by a hotpath experiment/benchmark, not your server)" if s.get("launched_by_hotpath") else ""))
        if s.get("settings"):
            st = s["settings"]
            L.append("  - model: %s" % (st.get("model") or p.get("model_path") or ("-hf %s" % st["hf_repo"] if st.get("hf_repo") else "?")))
            L.append("  - launch: %s" % launch.describe(st))
            L.append("  - log file: %s; /metrics: %s; build: %s" % (st.get("log_file") or "none", "on" if h.get("metrics_enabled") else "off", p.get("build_info")))
    for e in rec["endpoints"]:
        L.append("- %s answers (no matching process visible): build %s" % (e["base_url"], (e.get("props") or {}).get("build_info")))
    L += ["", "## llama-server installations", ""]
    if not rec["installs"]:
        L.append("- none found")
    for i in rec["installs"]:
        v = i.get("version_check") or {}
        state = ("starts" if v.get("starts") else ("needs %s on PATH" % i["needs_path_prepend"] if i.get("needs_path_prepend") else
                 ("does not start: %s" % v.get("failure") if v.get("failure") else v.get("skipped", "not run"))))
        L.append("- %s  [%s]" % (i["binary"], i["origin"]))
        L.append("  - %s; version %s; backends %s; tools %s" % (state, v.get("version", "?"), ", ".join(i["backends"]), ", ".join(i["siblings"]) or "none"))
        dep = i.get("dependencies") or {}
        if dep.get("cuda_libraries_imported"):
            L.append("  - CUDA libraries imported: %s%s" % (", ".join(dep["cuda_libraries_imported"]),
                     "; NOT found: %s" % ", ".join(dep["cuda_unresolved"]) if dep.get("cuda_unresolved") else "; all found"))
    L += ["", "## Models (GGUF headers)", ""]
    if not rec["models"]:
        L.append("- none found in the searched locations")
    for m in rec["models"][:30]:
        L.append("- %s: %s %s %s, %s layers, %s%s  [%s]" % (
            m["path"], m.get("architecture") or "?", m.get("size_label") or "", m.get("file_type") or "", m.get("n_layer") or "-",
            _mib(m.get("size_bytes")), "; " + m["vram_fit"] if m.get("vram_fit") else "", m.get("role") or m.get("header_error") or "header not read"))
    if len(rec["models"]) > 30:
        L.append("- ... %d more" % (len(rec["models"]) - 30))
    if rec.get("launch_scripts"):
        L += ["", "## Launch scripts and configs that start llama-server", ""] + ["- %s: `%s`" % (x["file"], x["line"]) for x in rec["launch_scripts"]]
    L += ["", "## Tools", ""] + ["- %s: %s" % (t["tool"], t["path"] if t.get("found") and t.get("path") else ("found" if t.get("found") else "not found")) for t in rec["tools"]]
    if rec["issues"]:
        L += ["", "## Problems and fixes", ""] + ["- [%s] %s: %s. Fix: %s" % (x["severity"], x["where"], x["what"], x["fix"]) for x in rec["issues"]]
    L += ["", "## Suggested hotpath.local.json", ""]
    for k, v in rec["suggested_config"].items():
        L.append("- %s = %s  (%s)" % (k, " ".join(v) if k == "server_args" else v, rec["suggested_config_sources"].get(k)))
    for k, v in rec["ambiguous"].items():
        L.append("- %s: several candidates, not chosen: %s" % (k, "; ".join(v)))
    if rec["unknowns"]:
        L += ["", "## Not known from discovery", ""] + ["- " + u for u in rec["unknowns"]]
    L += ["", "## Where discovery looked", ""] + ["- %s: %s -> %s%s" % (s["what"], s["where"], s["found"], " (%s)" % s["note"] if s.get("note") else "") for s in rec["searched"]]
    if rec["executed"]:
        L.append("- executed: %s" % "; ".join("%s%s -> exit %s" % (e["cmd"], " (with path_prepend)" if e["with_path_prepend"] else "", e["returncode"]) for e in rec["executed"]))
    L.append("- took %.1f s" % rec["seconds"])
    return "\n".join(L)
