"""hotpath profile: kernel-level time share of an isolated prefill-only or decode-only run.

Launches llama-bench under Nsight Systems (the target must be started by nsys; it cannot attach
to a running server), then summarizes `cuda_gpu_kern_sum`. Profiling runs are never used for
throughput numbers; they answer "which kernels consume the GPU time".
"""
import csv
import io
import subprocess
from pathlib import Path

from hotpath import gpu as gpu_mod
from hotpath.adapters import LlamaBenchAdapter, NsysAdapter
from hotpath.observe import binary_identity
from hotpath.records import redact_args, file_identity, new_record


def _run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def parse_kern_sum_csv(text):
    """nsys stats cuda_gpu_kern_sum CSV -> list of {name, time_pct, total_ns, instances, avg_ns}."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Time (%)") or ln.startswith("Time(%)")), None)
    if start is None:
        return []
    rdr = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows = []
    for r in rdr:
        name = r.get("Name") or r.get("Kernel Name") or ""
        try:
            rows.append({
                "name": name[:160],
                "time_pct": float(r.get("Time (%)") or r.get("Time(%)") or 0),
                "total_ns": float(r.get("Total Time (ns)") or 0),
                "instances": int(float(r.get("Instances") or 0)),
                "avg_ns": float(r.get("Avg (ns)") or 0),
            })
        except ValueError:
            continue
    return rows


def classify(name):
    n = name.lower()
    if "mul_mat_q" in n or "mmq" in n:
        return "matmul_mmq (prefill quantized GEMM)"
    if "mul_mat_vec_q" in n or "mmvq" in n or "dequantize_mul_mat_vec" in n or "mul_mat_vec" in n:
        return "matvec_mmvq (decode quantized GEMV)"
    if "sgemmex" in n and "bfloat16" in n:
        return "cublas_bf16_fallback_gemm"
    if "gated_delta" in n or "gdn" in n or "ssm" in n or "delta_net" in n:
        return "gated_delta_net"
    if "flash_attn" in n or "fattn" in n:
        return "attention"
    if "quantize_q8_1" in n or "quantize" in n:
        return "activation_quantize"
    if "rms_norm" in n or "norm" in n:
        return "norm"
    if "cpy" in n or "copy" in n or "memcpy" in n:
        return "copy"
    if "rope" in n:
        return "rope"
    if "gemm" in n or "cublas" in n or "sgemm" in n or "hgemm" in n:
        return "cublas_gemm"
    return "other"


def profile(bin_dir, model, label, out_dir, pp, tg, ngl=99, ub=512, extra_args=None, path_prepend=None):
    ns = NsysAdapter()
    lb = LlamaBenchAdapter(bin_dir)
    if not ns.locate():
        raise SystemExit("nsys not found: " + ns.install_hint)
    if not lb.locate():
        raise SystemExit("llama-bench not found in %s" % bin_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / ("nsys-%s" % label))
    bench_args = ["-ngl", str(ngl), "-ub", str(ub), "-p", str(pp), "-n", str(tg), "-r", "1"] + list(extra_args or [])
    cmd = ns.recipe(str(lb.locate()), model, bench_args, stem)
    env = None
    if path_prepend:
        import os
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(path_prepend) + os.pathsep + env.get("PATH", "")
    bg = gpu_mod.sample_background(seconds=5.0, interval=1.0)
    rc, so, se = _run(cmd, env=env)
    rec = new_record("profile", "active", label=label, mode={"pp": pp, "tg": tg, "ub": ub, "ngl": ngl},
                     gpu=gpu_mod.gpu_static(), background_before=bg, binary=binary_identity(str(lb.locate()), path_prepend),
                     model=file_identity(model), nsys_cmd=[Path(cmd[0]).name] + redact_args(cmd[1:]),
                     nsys_rc=rc, nsys_stderr=se[-3000:], bench_stdout=so[-3000:])
    report = stem + ".nsys-rep"
    if rc == 0 and Path(report).is_file():
        rc2, so2, se2 = _run(ns.summarize_cmd(report))
        rows = parse_kern_sum_csv(so2)
        rec["kernels"] = rows[:40]
        by_class = {}
        for r in rows:
            c = classify(r["name"])
            by_class[c] = round(by_class.get(c, 0.0) + r["time_pct"], 2)
        rec["time_share_by_class_pct"] = dict(sorted(by_class.items(), key=lambda kv: -kv[1]))
        rec["kernel_count"] = len(rows)
        rec["stats_rc"] = rc2
        if rc2 != 0:
            rec["stats_stderr"] = se2[-2000:]
    else:
        rec["error"] = "nsys profile failed or produced no report"
    return rec


def profile_markdown(rec):
    lines = ["# Profile %s  [active, profiling run; not a throughput measurement]" % rec.get("label"), "",
             "mode: pp=%s tg=%s ub=%s" % (rec["mode"]["pp"], rec["mode"]["tg"], rec["mode"]["ub"]),
             "binary: %s" % (rec.get("binary") or {}).get("version_lines"), ""]
    if rec.get("error"):
        lines += ["ERROR: " + rec["error"], rec.get("nsys_stderr", "")]
        return "\n".join(lines)
    lines += ["## GPU time share by kernel class", "", "| class | % of GPU kernel time |", "|---|---|"]
    for k, v in rec.get("time_share_by_class_pct", {}).items():
        lines.append("| %s | %.1f |" % (k, v))
    lines += ["", "## Top kernels", "", "| % | total ms | instances | avg us | kernel |", "|---|---|---|---|---|"]
    for r in rec.get("kernels", [])[:25]:
        lines.append("| %.1f | %.1f | %d | %.1f | `%s` |" % (r["time_pct"], r["total_ns"] / 1e6, r["instances"], r["avg_ns"] / 1e3, r["name"].replace("|", "/")))
    return "\n".join(lines)
