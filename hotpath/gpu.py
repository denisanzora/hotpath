"""GPU and driver facts: nvidia-smi plus, on Windows, the WDDM per-process counters.

Nothing here touches the model or submits inference. Memory numbers are reported as the
tooling reports them; hotpath does not sum per-process counters into a paging verdict,
because shared allocations (notably those attributed to dwm.exe) can be counted more than once.
"""
import json
import platform
import shutil
import subprocess
import time

# Peak theoretical memory bandwidth (GB/s), used only for roofline estimates. Keyed by the
# nvidia-smi product name. Add entries or pass --bandwidth-gbs to override.
KNOWN_BANDWIDTH_GBS = {
    "NVIDIA GeForce RTX 2080 SUPER": 496.0,
    "NVIDIA GeForce RTX 2080 Ti": 616.0,
    "NVIDIA GeForce RTX 2070 SUPER": 448.0,
    "NVIDIA GeForce RTX 2060 SUPER": 448.0,
    "NVIDIA GeForce RTX 3060": 360.0,
    "NVIDIA GeForce RTX 3080": 760.0,
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 672.0,
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA GeForce RTX 5090": 1792.0,
    "NVIDIA L4": 300.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
}


def _run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def nvidia_smi_available():
    return shutil.which("nvidia-smi") is not None


def gpu_static():
    """Name, driver, memory size, compute capability, PCIe link. Empty dict if no nvidia-smi."""
    if not nvidia_smi_available():
        return {}
    fields = "name,driver_version,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.max.memory,power.limit"
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        return {}
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    keys = fields.split(",")
    info = dict(zip(keys, vals))
    name = info.get("name")
    info["bandwidth_gbs_known"] = KNOWN_BANDWIDTH_GBS.get(name)
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader"])
    if rc == 0:
        info["driver_model"] = out.strip()
    return info


def gpu_dynamic():
    """Utilization, memory used, power, temperature at this instant."""
    if not nvidia_smi_available():
        return {}
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                       "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        return {}
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    keys = ["util_pct", "mem_used_mib", "mem_total_mib", "power_w", "temp_c", "clock_sm_mhz", "clock_mem_mhz"]
    d = {}
    for k, v in zip(keys, vals):
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d


def sample_background(seconds=10.0, interval=1.0):
    """Sample whole-GPU utilization for a while. Used to judge background load before measuring."""
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        d = gpu_dynamic()
        if d:
            samples.append(d)
        time.sleep(interval)
    if not samples:
        return {"samples": 0}
    utils = [s["util_pct"] for s in samples if isinstance(s.get("util_pct"), float)]
    mems = [s["mem_used_mib"] for s in samples if isinstance(s.get("mem_used_mib"), float)]
    return {
        "samples": len(samples),
        "seconds": seconds,
        "util_pct_max": max(utils) if utils else None,
        "util_pct_mean": round(sum(utils) / len(utils), 1) if utils else None,
        "mem_used_mib_max": max(mems) if mems else None,
        "mem_used_mib_min": min(mems) if mems else None,
    }


def windows_process_gpu_counters(pid):
    """Per-process dedicated/shared/committed GPU memory and 3D-engine utilization (Windows only).

    These are the WDDM counters Task Manager shows. 'shared usage' for a process is the
    best available indicator that *that process* has allocations living in system memory.
    """
    if platform.system() != "Windows" or not pid:
        return {}
    ps = (
        "$p='pid_%d_*';" % int(pid) +
        "$r=@{};"
        "foreach($c in @('Dedicated Usage','Shared Usage','Total Committed')){"
        " $s=(Get-Counter ('\\GPU Process Memory('+$p+')\\'+$c) -ErrorAction SilentlyContinue).CounterSamples;"
        " if($s){$r[$c]=[math]::Round(($s|Measure-Object CookedValue -Sum).Sum/1MB)}};"
        "$u=(Get-Counter ('\\GPU Engine('+$p+'engtype_3D)\\Utilization Percentage') -ErrorAction SilentlyContinue).CounterSamples;"
        "if($u){$r['engine_3d_util_pct']=[math]::Round(($u|Measure-Object CookedValue -Sum).Sum,1)};"
        "$r|ConvertTo-Json -Compress"
    )
    rc, out, _ = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=40)
    if rc != 0 or not out.strip():
        return {}
    try:
        d = json.loads(out.strip())
    except json.JSONDecodeError:
        return {}
    return {
        "dedicated_mib": d.get("Dedicated Usage"),
        "shared_mib": d.get("Shared Usage"),
        "committed_mib": d.get("Total Committed"),
        "engine_3d_util_pct": d.get("engine_3d_util_pct"),
        "source": "Windows GPU Process Memory / GPU Engine counters",
    }


def windows_adapter_memory():
    """Adapter-wide dedicated/shared usage. Reported as-is; not a paging verdict."""
    if platform.system() != "Windows":
        return {}
    ps = (
        "$r=@{};"
        "foreach($c in @('Dedicated Usage','Shared Usage','Total Committed')){"
        " $s=(Get-Counter ('\\GPU Adapter Memory(*)\\'+$c) -ErrorAction SilentlyContinue).CounterSamples|Sort-Object CookedValue -Descending|Select-Object -First 1;"
        " if($s){$r[$c]=[math]::Round($s.CookedValue/1MB)}};"
        "$r|ConvertTo-Json -Compress"
    )
    rc, out, _ = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=40)
    if rc != 0 or not out.strip():
        return {}
    try:
        d = json.loads(out.strip())
    except json.JSONDecodeError:
        return {}
    return {"dedicated_mib": d.get("Dedicated Usage"), "shared_mib": d.get("Shared Usage"),
            "committed_mib": d.get("Total Committed")}


def pid_listening_on(port):
    """PID of the process listening on a local TCP port, or None."""
    if platform.system() == "Windows":
        rc, out, _ = _run(["netstat", "-ano", "-p", "tcp"])
        if rc != 0:
            return None
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(":%d" % port) and parts[3] == "LISTENING":
                try:
                    return int(parts[4])
                except ValueError:
                    return None
        return None
    rc, out, _ = _run(["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"])
    if rc == 0 and out.strip():
        try:
            return int(out.strip().splitlines()[0])
        except ValueError:
            return None
    return None
