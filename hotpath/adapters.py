"""External tool adapters: the extension point for profiling, kernel benchmarks, and builds.

Each adapter locates its tool, says what it can do, and runs a fixed recipe. There is no plugin
registry; adding a tool means adding one class here. When a tool is missing, the adapter says
exactly what to install and how the recipe would run, so the path stays concrete.
"""
import glob
import os
import shutil
import subprocess
from pathlib import Path


def _search(exe, patterns):
    """Find an executable on PATH, then in well-known NVIDIA install locations (glob patterns).

    The install-location patterns are best effort. Only the Windows layout has been exercised;
    on Linux, put the tool on PATH if it is not found.
    """
    p = shutil.which(exe)
    if p:
        return Path(p)
    suffix = ".exe" if os.name == "nt" else ""
    for pat in patterns:
        hits = sorted(glob.glob(pat.format(exe=exe + suffix)), reverse=True)
        if hits:
            return Path(hits[0])
    return None


_NSYS_LOCATIONS = [
    "C:/Program Files/NVIDIA Corporation/Nsight Systems*/target-windows-x64/{exe}",
    "C:/Program Files/NVIDIA Corporation/Nsight Systems*/**/{exe}",
    "/opt/nvidia/nsight-systems/*/bin/{exe}",
    "/usr/local/cuda/bin/{exe}",
]
_NCU_LOCATIONS = [
    "C:/Program Files/NVIDIA Corporation/Nsight Compute*/{exe}",
    "C:/Program Files/NVIDIA Corporation/Nsight Compute*/target/windows-desktop-win7-x64/{exe}",
    "/opt/nvidia/nsight-compute/*/{exe}",
    "/usr/local/cuda/bin/{exe}",
]


class ToolAdapter:
    name = "base"
    purpose = ""
    install_hint = ""

    def locate(self):
        return None

    def status(self):
        p = self.locate()
        return {"tool": self.name, "purpose": self.purpose, "found": bool(p), "path": str(p) if p else None,
                "install_hint": None if p else self.install_hint}


class NsysAdapter(ToolAdapter):
    name = "nsys"
    purpose = "kernel-level timeline of an isolated prefill/decode run (which kernels consume GPU time)"
    install_hint = ("Install NVIDIA Nsight Systems (standalone, free) or the CUDA toolkit; on Windows enable "
                    "'Allow access to GPU performance counters to all users' in the NVIDIA developer settings.")

    def locate(self):
        return _search("nsys", _NSYS_LOCATIONS)

    def recipe(self, llama_bench, model, extra_args, out_stem):
        return [str(self.locate() or "nsys"), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "-o", out_stem,
                "--force-overwrite", "true", llama_bench, "-m", model] + list(extra_args)

    def summarize_cmd(self, report):
        return [str(self.locate() or "nsys"), "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true", report]


class NcuAdapter(ToolAdapter):
    name = "ncu"
    purpose = "per-kernel roofline, occupancy and stall reasons for a named kernel"
    install_hint = "Install NVIDIA Nsight Compute (standalone, free); needs the performance-counter permission on Windows."

    def locate(self):
        return _search("ncu", _NCU_LOCATIONS)

    def recipe(self, kernel_regex, target_cmd, out_stem):
        return [str(self.locate() or "ncu"), "--set", "full", "-k", "regex:" + kernel_regex, "--launch-skip", "10", "--launch-count", "5",
                "-o", out_stem] + list(target_cmd)


class TestBackendOpsAdapter(ToolAdapter):
    name = "test-backend-ops"
    purpose = "isolated correctness (NMSE vs CPU) and perf of one ggml op at fixed shapes/types"
    install_hint = ("Build llama.cpp from source with the CUDA backend (needs CUDA toolkit + CMake + a C++ compiler); "
                    "test-backend-ops is built with the tests and lives in the build tree's bin/ directory. "
                    "Point build_dir in the local config (or --build-dir) at that build tree.")

    def __init__(self, build_dir=None):
        self.build_dir = Path(build_dir) if build_dir else None

    def locate(self):
        if self.build_dir:
            for cand in self.build_dir.glob("**/test-backend-ops*"):
                if cand.is_file() and cand.suffix in ("", ".exe"):
                    return cand
        p = shutil.which("test-backend-ops")
        return Path(p) if p else None

    def recipe(self, mode, op_filter, backend="CUDA0", params_regex=None):
        cmd = [str(self.locate() or "test-backend-ops"), mode, "-b", backend, "-o", op_filter]
        if params_regex:
            cmd += ["-p", params_regex]
        return cmd


class LlamaBenchAdapter(ToolAdapter):
    name = "llama-bench"
    purpose = "whole-model pp/tg throughput with repetitions, JSON output"
    install_hint = "Ships with llama.cpp next to llama-server. Set runtime_bin_dir (or put it on PATH)."

    def __init__(self, bin_dir=None):
        self.bin_dir = Path(bin_dir) if bin_dir else None

    def locate(self):
        if self.bin_dir:
            for name in ("llama-bench.exe", "llama-bench"):
                if (self.bin_dir / name).is_file():
                    return self.bin_dir / name
        p = shutil.which("llama-bench")
        return Path(p) if p else None

    def recipe(self, model, ngl=99, ub=512, pp=512, tg=128, reps=5):
        return [str(self.locate() or "llama-bench"), "-m", model, "-ngl", str(ngl), "-ub", str(ub), "-p", str(pp), "-n", str(tg), "-r", str(reps), "-o", "json"]


def env_with_path(path_prepend=None):
    """Process environment with extra directories (e.g. CUDA runtime DLLs) first on PATH."""
    env = dict(os.environ)
    if path_prepend:
        env["PATH"] = os.pathsep.join(str(x) for x in path_prepend) + os.pathsep + env.get("PATH", "")
    return env


def run_recipe(cmd, timeout=3600, path_prepend=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env_with_path(path_prepend))
    return {"cmd": [str(c) for c in cmd], "returncode": p.returncode, "stdout": p.stdout[-20000:], "stderr": p.stderr[-5000:]}


def all_status(bin_dir=None, build_dir=None):
    return [a.status() for a in (NsysAdapter(), NcuAdapter(), TestBackendOpsAdapter(build_dir), LlamaBenchAdapter(bin_dir))]
