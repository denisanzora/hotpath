"""hotpath kernels: isolated op correctness and perf via the runtime's own test-backend-ops.

`test` mode compares the backend against the CPU reference with ggml's own NMSE thresholds;
`perf` mode reports us/run and throughput for the built-in cases matching a filter. Both are run
on a named build directory so base and candidate builds can be compared op by op.
"""
import re
import subprocess

from hotpath import gpu as gpu_mod
from hotpath.adapters import TestBackendOpsAdapter
from hotpath.adapters import env_with_path
from hotpath.observe import binary_identity
from hotpath.records import new_record

RE_PERF = re.compile(r"^\s*(?P<case>\S.*?\S)\s*:\s*(?P<runs>\d+) runs -\s*(?P<us>[\d.]+) us/run -\s*(?P<work>[\d.]+ \w+)/run -\s*(?P<rate>[\d.]+ \S+)")
RE_TEST = re.compile(r"^\s*(?P<case>\S.*?\S)\s*:\s*(?P<result>OK|FAIL|not supported)")


def run_tbo(build_dir, mode, op_filter, params_regex=None, backend="CUDA0", timeout=3600, path_prepend=None):
    ad = TestBackendOpsAdapter(build_dir)
    exe = ad.locate()
    if not exe:
        raise SystemExit("test-backend-ops not found under %s" % build_dir)
    cmd = ad.recipe(mode, op_filter, backend=backend, params_regex=params_regex)
    from hotpath import safepath
    p = subprocess.run(safepath.resolve(cmd), capture_output=True, text=True, timeout=timeout, env=env_with_path(path_prepend), stdin=subprocess.DEVNULL)
    text = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)  # strip ANSI color codes
    cases = []
    for line in text.splitlines():
        m = RE_PERF.match(line) if mode == "perf" else RE_TEST.match(line)
        if m:
            d = m.groupdict()
            if mode == "perf":
                d["runs"] = int(d["runs"])
                d["us_per_run"] = float(d.pop("us"))
            cases.append(d)
    summary_line = next((ln for ln in text.splitlines() if "tests passed" in ln or "Backend CUDA0:" in ln), None)
    # the runtime prints a CUDA-graph warmup line per graph; keep the full output minus that noise for debugging
    kept = "\n".join(ln for ln in text.splitlines() if "CUDA graph warmup" not in ln)
    rec = new_record("kernels", "active", mode=mode, op_filter=op_filter, params_regex=params_regex, backend=backend,
                     cmd=[c if i else exe.name for i, c in enumerate(cmd)], returncode=p.returncode,
                     binary=binary_identity(str(exe), path_prepend), gpu=gpu_mod.gpu_static(),
                     cases=cases, n_cases=len(cases), summary_line=summary_line, raw_output=kept[-60000:])
    if mode == "test":
        rec["failures"] = [c for c in cases if c["result"] == "FAIL"]
        rec["not_supported"] = [c["case"] for c in cases if c["result"] == "not supported"]
        rec["passed"] = sum(1 for c in cases if c["result"] == "OK")
    return rec


def kernels_markdown(rec):
    lines = ["# Kernels %s  [%s]" % (rec["mode"], rec["evidence_class"]), "",
             "binary: %s" % (rec.get("binary") or {}).get("version_lines"), "filter: %s %s" % (rec["op_filter"], rec.get("params_regex") or ""),
             "cases: %d, rc %s, %s" % (rec["n_cases"], rec["returncode"], rec.get("summary_line") or ""), ""]
    if rec["mode"] == "perf":
        lines += ["| us/run | work/run | rate | case |", "|---|---|---|---|"]
        for c in rec["cases"]:
            lines.append("| %.1f | %s | %s | `%s` |" % (c["us_per_run"], c["work"], c["rate"], c["case"].replace("|", "/")))
    else:
        lines += ["passed (NMSE within ggml's threshold vs CPU): %d, FAIL: %d, not supported on this backend: %d" % (
            rec.get("passed", 0), len(rec.get("failures", [])), len(rec.get("not_supported", [])))]
        for c in rec.get("failures", [])[:20]:
            lines.append("- FAIL: %s" % c["case"])
    return "\n".join(lines)
