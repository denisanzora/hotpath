"""Locating the programs hotpath runs, never from the working directory.

On Windows, a program started by bare name (`nvidia-smi`, `powershell`, `llama-server`) is searched in the current
directory before PATH, and Python 3.10's shutil.which does the same. hotpath runs in a working directory that an agent
may write to, so a planted `nvidia-smi.exe` or `llama-server.bat` there would run from a passive command. Every external
program is therefore resolved here to an absolute path from absolute PATH entries only (no current directory, no relative
entries) and, on Windows, only as an .exe or .com file (never a .bat/.cmd script). A program path with a directory part
comes from the operator's configuration (or the operating system) and is made absolute against the working directory.

A program also never runs in the working directory: Windows loads the libraries a program needs (cudart, ggml backends)
from the current directory before PATH, so a planted DLL there would run inside llama-server or nvidia-smi. Programs
run in their own directory, which they already load libraries from first.
"""
import os
import subprocess
from pathlib import Path


def which(name):
    """Absolute path of program `name`, or None. A name with a directory part (a configured path) is made absolute; a bare
    name is searched on absolute PATH entries only."""
    name = str(name)
    if os.path.dirname(name):
        p = Path(name).absolute()
        return str(p) if p.is_file() else None
    windows = os.name == "nt"
    exts = (".exe", ".com") if windows else ("",)
    here = os.path.normcase(os.path.abspath(os.getcwd()))
    entries = [e.strip().strip('"') for e in os.environ.get("PATH", "").split(os.pathsep)]
    if windows:
        entries.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    for entry in entries:
        if not entry or not os.path.isabs(entry) or os.path.normcase(os.path.abspath(entry)) == here:
            continue
        for ext in exts:
            cand = os.path.join(entry, name if (not windows or name.lower().endswith(exts)) else name + ext)
            if os.path.isfile(cand) and (windows or os.access(cand, os.X_OK)):
                return cand
            if windows and name.lower().endswith(exts):
                break
    return None


def resolve(cmd):
    """`cmd` with its program as an absolute path. Raises FileNotFoundError (an OSError, which callers already treat as
    "not available") when the program is not found outside the working directory."""
    cmd = [str(c) for c in cmd]
    if Path(cmd[0]).is_absolute():
        return cmd
    if os.path.dirname(cmd[0]):
        return [str(Path(cmd[0]).absolute())] + cmd[1:]
    found = which(cmd[0])
    if not found:
        raise FileNotFoundError("%s not found on PATH (the working directory is not searched)" % cmd[0])
    return [found] + cmd[1:]


def absolute_entries(entries):
    """PATH entries made absolute against the working directory: a program runs in its own directory, where a relative
    entry would name another directory (and the libraries it should provide would be missing)."""
    return [str(Path(str(e)).absolute()) if str(e) and not os.path.isabs(str(e)) else str(e) for e in entries or []]


def os_description():
    """platform.platform() without running a program: on Windows, Python 3.10's platform module runs `cmd /c ver` in the
    current directory to read a version that sys.getwindowsversion() already holds."""
    import platform
    import sys
    if os.name != "nt":
        return platform.platform()
    v = sys.getwindowsversion()
    major, minor, build = v.major, v.minor, v.build          # the running build, as `ver` reports it (platform_version is a DLL's)
    return "Windows-%s-%d.%d.%d-SP%d" % (major if minor == 0 else "%d.%d" % (major, minor), major, minor, build, v.service_pack_major)


def program_dir(program):
    """The directory a program runs in: its own, never the working directory."""
    return str(Path(program).absolute().parent)


def run(cmd, **kw):
    """subprocess.run of `cmd` resolved here, in the program's own directory. Arguments that are paths must be absolute."""
    cmd = resolve(cmd)
    kw.setdefault("cwd", program_dir(cmd[0]))
    return subprocess.run(cmd, **kw)
