"""hotpath compare: put two or more result sets side by side and say whether they are comparable.

Works on experiment arms (controlled) or separate measurement records (then labeled historical,
because hotpath did not control the conditions between them).
"""
from hotpath.measure import summarize_requests


def _arm_sets(records):
    """Flatten inputs into {label: {requests, meta}}."""
    sets = {}
    classes = set()
    for rec in records:
        if rec["kind"] == "experiment":
            for name, arm in rec["arms"].items():
                sets["%s/%s" % (rec["name"], name)] = {"requests": arm["requests"], "meta": {
                    "binary": arm.get("binary") or rec.get("binary"), "ggml_cuda_dll": arm.get("ggml_cuda_dll"),
                    "model": arm.get("model") or rec.get("model"), "workload": rec.get("workload"),
                    "launch_facts": arm.get("launch_facts"), "failures": arm.get("failures"), "launches": arm.get("launches")}}
            classes.add("controlled")
        elif rec["kind"] == "measurement":
            sets[rec.get("label") or rec["created"]] = {"requests": rec["requests"], "meta": {
                "binary": None, "model": None, "workload": rec.get("workload"), "target": rec.get("target")}}
            classes.add("historical")
    return sets, classes


def comparability(sets):
    issues = []
    wl = {k: (v["meta"].get("workload") or {}).get("set_hash") for k, v in sets.items()}
    if len(set(wl.values())) > 1:
        issues.append("different prompt sets: %s" % wl)
    np_ = {k: (v["meta"].get("workload") or {}).get("n_predict") for k, v in sets.items()}
    if len(set(np_.values())) > 1:
        issues.append("different n_predict: %s" % np_)
    models = {k: ((v["meta"].get("model") or {}).get("basename")) for k, v in sets.items()}
    if len({m for m in models.values() if m}) > 1:
        issues.append("different model files (intended for a model-variant experiment; otherwise not comparable): %s" % models)
    return issues


def output_equivalence(sets):
    """Per prompt, do all arms produce the same output? With token ids (hotpath's own workloads) the first
    divergence index is reported; application-shaped requests carry no token ids, so their outputs are compared
    by content hash (identical or not)."""
    labels = list(sets)
    if len(labels) < 2:
        return {}
    ref = labels[0]
    by_prompt = {}
    for lab in labels:
        for r in sets[lab]["requests"]:
            key = r.get("tokens") if r.get("tokens") else ("sha256:" + str(r.get("output_sha256")))
            by_prompt.setdefault(r["prompt_id"], {}).setdefault(lab, []).append(key)
    result = {}
    for pid, arms in by_prompt.items():
        if ref not in arms:
            continue
        ref_tok = arms[ref][0]
        entry = {"reference": ref, "stable_within_reference": all(t == ref_tok for t in arms[ref]),
                 "basis": "token ids" if isinstance(ref_tok, list) else "output content hash"}
        for lab in labels[1:]:
            if lab not in arms:
                continue
            other = arms[lab][0]
            if isinstance(ref_tok, list) and isinstance(other, list):
                div = next((i for i, (a, b) in enumerate(zip(ref_tok, other)) if a != b), None)
                if div is None and len(ref_tok) != len(other):
                    div = min(len(ref_tok), len(other))
            else:
                div = None if ref_tok == other else -1  # -1: differs, position unknown (hash comparison)
            entry[lab] = {"identical": div is None, "first_divergence_index": div, "stable_within_arm": all(t == other for t in arms[lab])}
        result[pid] = entry
    return result


def compare_kernels(records):
    """Side-by-side of test-backend-ops perf records (one per build): us/run per case and the ratio to the first."""
    labels = [r.get("label") or r["created"] for r in records]
    tables = [{c["case"]: c for c in r["cases"]} for r in records]
    rows = []
    for case in tables[0]:
        if not all(case in t for t in tables):
            continue
        us = [t[case]["us_per_run"] for t in tables]
        rows.append({"case": case, "us_per_run": us, "ratio_to_first": [round(u / us[0], 3) for u in us], "rate": [t[case]["rate"] for t in tables]})
    return {"kind": "kernels_comparison", "evidence_class": "active", "labels": labels,
            "binaries": [{"version": (r.get("binary") or {}).get("version_lines"), "sha256_head64m": (r.get("binary") or {}).get("sha256_head64m")} for r in records],
            "rows": rows, "note": "isolated kernel timings from test-backend-ops perf (one run each, no interleaving); confirm any gain end to end with an experiment"}


def compare_kernels_markdown(rep):
    lines = ["# Kernel perf comparison  [%s]" % rep["evidence_class"], "", "labels: %s" % rep["labels"], "", "| case | " + " | ".join("%s us" % l for l in rep["labels"]) + " | ratio |", "|---|" + "---|" * len(rep["labels"]) + "---|"]
    for r in rep["rows"]:
        short = r["case"].split(",bs=")[0].replace("MUL_MAT(", "")
        lines.append("| `%s` | %s | %s |" % (short, " | ".join("%.1f" % u for u in r["us_per_run"]), " / ".join("%.3f" % x for x in r["ratio_to_first"][1:])))
    lines += ["", rep["note"]]
    return "\n".join(lines)


def compare(records):
    if records and all(r.get("kind") == "kernels" and r.get("mode") == "perf" for r in records):
        return compare_kernels(records)
    sets, classes = _arm_sets(records)
    rows = {}
    for lab, s in sets.items():
        rows[lab] = summarize_requests(s["requests"])
        rows[lab]["launches"] = s["meta"].get("launches")
        rows[lab]["failures"] = s["meta"].get("failures")
        b = s["meta"].get("binary") or {}
        dll = s["meta"].get("ggml_cuda_dll") or {}
        rows[lab]["binary_id"] = {"version": (b.get("version_lines") or [None])[0], "sha256_head64m": b.get("sha256_head64m"),
                                  "ggml_cuda_dll_sha256_head64m": dll.get("sha256_head64m")}
        lf = s["meta"].get("launch_facts") or []
        if lf:
            ded = [x.get("process_counters_after", {}).get("dedicated_mib") for x in lf]
            sh = [x.get("process_counters_after", {}).get("shared_mib") for x in lf]
            rows[lab]["process_dedicated_mib"] = [d for d in ded if d is not None]
            rows[lab]["process_shared_mib"] = [d for d in sh if d is not None]
            rows[lab]["load_s"] = [x.get("load_s") for x in lf]
    labels = list(rows)
    ratios = {}
    if len(labels) >= 2:
        base = rows[labels[0]]
        for lab in labels[1:]:
            r = {}
            for metric in ("prefill_tps", "decode_tps", "wall_ms"):
                a = (base.get(metric) or {}).get("median")
                b = (rows[lab].get(metric) or {}).get("median")
                if a and b:
                    r[metric + "_median_ratio"] = round(b / a, 3)
            ratios["%s vs %s" % (lab, labels[0])] = r
    return {"evidence_class": "controlled" if classes == {"controlled"} else "historical",
            "comparability_issues": comparability(sets), "arms": rows, "median_ratios": ratios,
            "output_equivalence": output_equivalence(sets)}


def compare_markdown(rep):
    if rep.get("kind") == "kernels_comparison":
        return compare_kernels_markdown(rep)
    lines = ["# Comparison  [%s]" % rep["evidence_class"], ""]
    if rep["comparability_issues"]:
        lines += ["**Not directly comparable:**"] + ["- " + i for i in rep["comparability_issues"]] + [""]
    lines += ["| arm | n | prompt_n med | prefill t/s med (p10-p90) | decode t/s med (p10-p90) | gen tok med | wall ms med | valid JSON / schema keys | proc dedicated MiB | proc shared MiB | failures |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in rep["arms"].items():
        pf = r.get("prefill_tps") or {}
        dc = r.get("decode_tps") or {}
        v = r.get("validity")
        vtxt = "%s/%s / %s/%s" % (v["valid_json"], v["judged"], v["schema_required_present"], v["judged"]) if v else "n/a"
        lines.append("| %s | %s | %s | %s (%s-%s) | %s (%s-%s) | %s | %s | %s | %s | %s | %s |" % (
            lab, r["n"], (r.get("prompt_n") or {}).get("median"), pf.get("median"), pf.get("p10"), pf.get("p90"),
            dc.get("median"), dc.get("p10"), dc.get("p90"), (r.get("predicted_n") or {}).get("median"), (r.get("wall_ms") or {}).get("median"), vtxt,
            r.get("process_dedicated_mib"), r.get("process_shared_mib"), len(r.get("failures") or [])))
    lines.append("")
    for k, v in rep["median_ratios"].items():
        lines.append("- %s: %s" % (k, v))
    oe = rep["output_equivalence"]
    if oe:
        lines += ["", "Output equivalence (greedy tokens vs first arm):"]
        for pid, e in oe.items():
            lines.append("- %s: %s" % (pid, {k: v for k, v in e.items() if k != "reference"}))
    return "\n".join(lines)
