"""Locating the programs hotpath runs, never from the working directory.

On Windows, a program started by bare name (`nvidia-smi`, `powershell`, `llama-server`) is searched in the current
directory before PATH, and Python 3.10's shutil.which does the same. hotpath runs in a working directory that an agent
may write to, so a planted `nvidia-smi.exe` or `llama-server.bat` there would run from a passive command. Every external
program is therefore resolved here to an absolute path from absolute PATH entries only (no current directory, no relative
entries) and, on Windows, only as an .exe or .com file (never a .bat/.cmd script).
"""
import os
from pathlib import Path


def which(name):
    """Absolute path of program `name`, or None. A name with a directory part is returned only when it is absolute and
    exists (a relative one could point into the working directory)."""
    name = str(name)
    if os.path.dirname(name):
        p = Path(name)
        return str(p) if p.is_absolute() and p.is_file() else None
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
    found = which(cmd[0])
    if not found:
        raise FileNotFoundError("%s not found on PATH (the working directory is not searched)" % cmd[0])
    return [found] + cmd[1:]
