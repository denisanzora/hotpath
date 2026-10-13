"""Ownership of shared things: an improve session, the GPU, and the processes hotpath starts.

- A session lock (results/local/improve-*/running.lock) lets one process drive a session at a time.
- The GPU lock (one file in the system temp directory) lets one hotpath process do GPU work at a time on this machine,
  whichever working directory, client or MCP server it runs from. Two agents (say Claude Code and OpenCode, each with
  its own MCP server) could otherwise both pass the idle-GPU precondition within the same few seconds and measure each
  other. It is reentrant within a process: the improve loop holds it for the whole session.
- Both are OS file locks (msvcrt.locking on Windows, flock elsewhere): the operating system releases them when the holder
  exits or dies, so a crash leaves no stale lock and two processes can never both hold one. A holder file next to each lock
  names the holder by pid and process start time, for messages.
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


class FileLock:
    """An exclusive OS lock on a file (msvcrt.locking on Windows, flock elsewhere). The operating system releases it when
    the holder exits or dies, so there are no stale locks to detect, and two processes can never both hold it. Who holds
    it is written next to it (`<lock>.holder`), for messages only."""

    def __init__(self, path):
        self.path = Path(path)
        self.fh = None

    def try_acquire(self, holder):
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        try:
            self.holder_path().write_text(json.dumps(holder), encoding="utf-8")
        except OSError:
            pass
        return True

    def holder_path(self):
        return self.path.with_name(self.path.name + ".holder")

    def release(self):
        if self.fh is None:
            return
        try:
            try:
                self.holder_path().unlink()
            except OSError:
                pass
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                try:
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
            self.fh = None

    def holder(self):
        """Who holds the lock, as far as its holder file says (None when that process is gone)."""
        h = _read(self.holder_path())
        return h if holder_alive(h) else None


def legacy_holder(path):
    """A lock file in the format before OS locks (a bare pid or a JSON holder written into the lock file itself) whose
    process still runs: an older hotpath may be driving the session."""
    h = _read(path)
    if isinstance(h, dict) and h.get("pid") != os.getpid() and holder_alive(h) and "kind" not in h:
        return h
    return None


# ---- the GPU lock -----------------------------------------------------------------------------------------------------

_held = {"depth": 0, "lock": None}
_held_lock = threading.Lock()


def gpu_lock_path():
    return Path(os.environ.get("HOTPATH_GPU_LOCK_DIR") or tempfile.gettempdir()) / GPU_LOCK_NAME


def gpu_holder():
    """The process that holds the GPU lock now, by its holder file (read only; the lock itself is not touched)."""
    return FileLock(gpu_lock_path()).holder()


def describe(holder):
    if not holder:
        return "another hotpath process"
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
        lock = FileLock(gpu_lock_path())
        content = dict(me(), kind=self.kind, detail=self.detail, cwd=str(Path.cwd()), since=time.strftime("%Y-%m-%d %H:%M:%S"))
        t_end = time.time() + self.wait_s
        while not lock.try_acquire(content):
            holder = lock.holder()
            if time.time() >= t_end:
                raise SystemExit("refused: another hotpath process is using the GPU: %s. One hotpath process does GPU work at a time on this "
                                 "machine (experiments would measure each other); wait for it, stop it (hotpath_job_cancel for an MCP job), "
                                 "or pass --wait-s N" % describe(holder))
            self.log("waiting for the GPU: %s" % describe(holder))
            time.sleep(min(30.0, max(1.0, t_end - time.time())))
        with _held_lock:
            _held["depth"], _held["lock"] = 1, lock
        return self

    def __exit__(self, *exc):
        with _held_lock:
            _held["depth"] -= 1
            if _held["depth"] > 0:
                return False
            lock, _held["lock"] = _held["lock"], None
            _held["depth"] = 0
        if lock:
            lock.release()
        return False


def release_all():
    """Drop every lock this process holds (used when exiting abnormally; the OS would release them at exit anyway)."""
    with _held_lock:
        lock, _held["lock"], _held["depth"] = _held["lock"], None, 0
    if lock:
        lock.release()
    for lk in list(_session_locks):
        lk.release()
        _session_locks.discard(lk)


# ---- session locks ----------------------------------------------------------------------------------------------------

_session_locks = set()


def take_session(sdir):
    path = Path(sdir) / "running.lock"
    old = legacy_holder(path)
    lock = FileLock(path)
    if old is not None or not lock.try_acquire(dict(me(), since=time.strftime("%Y-%m-%d %H:%M:%S"))):
        holder = old or lock.holder() or {}
        raise SystemExit("error: session %s is running in another process (pid %s); wait for it (hotpath_job_status) or stop that job, then resume" % (
            Path(sdir).name, holder.get("pid", "unknown")))
    _session_locks.add(lock)
    return lock


def drop_session(lock):
    lock.release()
    _session_locks.discard(lock)


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
