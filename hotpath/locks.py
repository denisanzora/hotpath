"""Ownership of shared things: an improve session, the GPU, and the processes hotpath starts.

- A session lock (results/local/improve-*/running.lock) lets one process drive a session at a time.
- The GPU lock (one file in the system temp directory) lets one hotpath process do GPU work at a time on this machine,
  whichever working directory, client or MCP server it runs from. Two agents (say Claude Code and OpenCode, each with
  its own MCP server) could otherwise both pass the idle-GPU precondition within the same few seconds and measure each
  other. It is reentrant within a process: the improve loop holds it for the whole session.
- Both are created exclusively (O_EXCL) and name their holder by pid and process start time, so a lock whose holder
  died is recognized as stale even when Windows reuses the pid.
- A process the MCP server started watches that server (HOTPATH_PARENT_PID): if the server is killed outright (no stdin
  close, no cleanup), the child stops the llama-servers it launched and exits instead of running on unattended.

None of this stops or signals a process hotpath did not start.
"""
import json
import os
import tempfile
import threading
import time
from pathlib import Path

GPU_LOCK_NAME = "hotpath-gpu.lock"
PARENT_ENV = "HOTPATH_PARENT_PID"


def process_start(pid):
    """An identity for a running process beyond its pid (creation time), or None when it is not running or unknown."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:      # STILL_ACTIVE
                return None
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
                return "alive"
            return str((c.dwHighDateTime << 32) | c.dwLowDateTime)
        finally:
            k32.CloseHandle(h)
    stat = Path("/proc/%d/stat" % pid)
    if stat.is_file():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[19]        # field 22: start time in clock ticks
        except (OSError, IndexError):
            return "alive"
    try:
        os.kill(pid, 0)
        return "alive"
    except OSError:
        return None


def me():
    return {"pid": os.getpid(), "start": process_start(os.getpid())}


def holder_alive(holder):
    """Whether the process named in a lock still runs (same pid AND same start time when both are known)."""
    if not isinstance(holder, dict):
        return False
    now = process_start(holder.get("pid"))
    if now is None:
        return False
    then = holder.get("start")
    return then in (None, "alive") or now == "alive" or now == then


def _read(path):
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw.isdigit():
        return {"pid": int(raw), "start": None}           # a session lock written by 0.3.0.dev0 before this module: a bare pid
    try:
        out = json.loads(raw)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _create(path, content):
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(content))


def acquire_file(path, content):
    """Create `path` exclusively for this process. Returns None on success, or the live holder's record. A lock whose
    holder is gone (or is this very process, from an earlier interrupted call) is replaced."""
    path = Path(path)
    for _ in range(3):
        try:
            _create(path, content)
            return None
        except FileExistsError:
            holder = _read(path)
            mine = isinstance(holder, dict) and holder.get("pid") == os.getpid()
            if holder_alive(holder) and not mine:
                return holder
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return holder or {"pid": None}
    return _read(path) or {"pid": None}


def release_file(path):
    path = Path(path)
    holder = _read(path)
    if isinstance(holder, dict) and holder.get("pid") == os.getpid():
        try:
            path.unlink()
        except OSError:
            pass


# ---- the GPU lock -----------------------------------------------------------------------------------------------------

_held = {"depth": 0, "path": None}
_held_lock = threading.Lock()


def gpu_lock_path():
    return Path(os.environ.get("HOTPATH_GPU_LOCK_DIR") or tempfile.gettempdir()) / GPU_LOCK_NAME


def gpu_holder():
    """The process that holds the GPU lock now, or None."""
    h = _read(gpu_lock_path())
    return h if holder_alive(h) else None


def describe(holder):
    if not holder:
        return "nobody"
    return "hotpath pid %s (%s%s, since %s, working directory %s)" % (
        holder.get("pid"), holder.get("kind") or "?", ": %s" % holder["detail"] if holder.get("detail") else "", holder.get("since") or "?",
        holder.get("cwd") or "?")


class gpu_work:
    """Context manager: hold the machine-wide GPU lock while doing GPU work. Waits up to `wait_s` for another hotpath
    process to finish, then refuses with SystemExit naming it. Reentrant within one process."""

    def __init__(self, kind, detail=None, wait_s=0, log=print):
        self.kind, self.detail, self.wait_s, self.log = kind, detail, float(wait_s or 0), log

    def __enter__(self):
        with _held_lock:
            if _held["depth"]:
                _held["depth"] += 1
                return self
        path = gpu_lock_path()
        content = dict(me(), kind=self.kind, detail=self.detail, cwd=str(Path.cwd()), since=time.strftime("%Y-%m-%d %H:%M:%S"))
        t_end = time.time() + self.wait_s
        while True:
            holder = acquire_file(path, content)
            if holder is None:
                break
            if time.time() >= t_end:
                raise SystemExit("refused: another hotpath process is using the GPU: %s. One hotpath process does GPU work at a time on this "
                                 "machine (experiments would measure each other); wait for it, stop it (hotpath_job_cancel for an MCP job), "
                                 "or pass --wait-s N" % describe(holder))
            self.log("waiting for the GPU: %s" % describe(holder))
            time.sleep(min(30.0, max(1.0, t_end - time.time())))
        with _held_lock:
            _held["depth"], _held["path"] = 1, path
        return self

    def __exit__(self, *exc):
        with _held_lock:
            _held["depth"] -= 1
            if _held["depth"] > 0:
                return False
            path, _held["path"] = _held["path"], None
            _held["depth"] = 0
        if path:
            release_file(path)
        return False


def release_all():
    """Drop every lock this process holds (used when exiting abnormally)."""
    with _held_lock:
        path, _held["path"], _held["depth"] = _held["path"], None, 0
    if path:
        release_file(path)
    for p in list(_session_locks):
        release_file(p)
        _session_locks.discard(p)


# ---- session locks ----------------------------------------------------------------------------------------------------

_session_locks = set()


def take_session(sdir):
    lock = Path(sdir) / "running.lock"
    holder = acquire_file(lock, dict(me(), since=time.strftime("%Y-%m-%d %H:%M:%S")))
    if holder is not None:
        raise SystemExit("error: session %s is running in another process (pid %s); wait for it (hotpath_job_status) or stop that job, then resume" % (
            Path(sdir).name, holder.get("pid")))
    _session_locks.add(lock)
    return lock


def drop_session(lock):
    release_file(lock)
    _session_locks.discard(Path(lock))


# ---- processes hotpath started, and the parent that started hotpath ---------------------------------------------------

_children = set()
_children_lock = threading.Lock()


def track(proc):
    with _children_lock:
        _children.add(proc)


def untrack(proc):
    with _children_lock:
        _children.discard(proc)


def stop_children():
    """Stop every server process this hotpath process launched and still runs (and only those)."""
    with _children_lock:
        procs = list(_children)
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except Exception:  # noqa: BLE001
                    p.kill()
        except OSError:
            pass
        untrack(p)


def watch_parent(env=None, interval=2.0, on_orphan=None):
    """When started by the MCP server: exit (after stopping launched servers and dropping locks) once that server is gone."""
    raw = (env or os.environ).get(PARENT_ENV)
    if not raw:
        return None
    pid, _, start = str(raw).partition(":")
    parent = {"pid": pid, "start": start or None}
    if not holder_alive(parent):
        return None

    def loop():
        while True:
            time.sleep(interval)
            if not holder_alive(parent):
                (on_orphan or _orphaned)()
                return
    t = threading.Thread(target=loop, daemon=True, name="hotpath-parent-watch")
    t.start()
    return t


def _orphaned():
    import sys
    print("hotpath: the MCP server that started this job is gone; stopping the servers this job launched and exiting", file=sys.stderr, flush=True)
    stop_children()
    release_all()
    os._exit(75)
