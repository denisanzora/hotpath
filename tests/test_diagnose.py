"""Unit tests for the diagnosis and proposal logic on synthetic record structures.

These exercise the rules' arithmetic and shapes only; the numbers are made up and nothing here is a
performance claim. Run: python -m unittest discover -s tests
"""
import json
import os
import tempfile
import unittest

from hotpath import diagnose as dg
from hotpath import propose as pp


def _req(prompt_n, prompt_ms, predicted_n, predicted_ms, wall_ms, tokens=None, prompt_id="p00", cache_n=0):
    return {"prompt_id": prompt_id, "prompt_n": prompt_n, "prompt_ms": prompt_ms, "prompt_tps": 1000.0 * prompt_n / prompt_ms,
            "predicted_n": predicted_n, "predicted_ms": predicted_ms, "predicted_tps": 1000.0 * predicted_n / predicted_ms,
            "wall_ms": wall_ms, "cache_n": cache_n, "tokens": tokens or list(range(predicted_n)), "stop_type": "limit"}


def _measurement(label, reqs, props=None):
    return {"kind": "measurement", "evidence_class": "active", "created": "2026-01-01T00:00:00", "label": label, "requests": reqs,
            "target": {"props": props or {"n_ctx_per_slot": 4096, "total_slots": 1}}, "gpu": {"name": "Test GPU", "memory.total": "8192", "bandwidth_gbs_known": 500.0},
            "gpu_snapshots": [{"gpu": {"mem_used_mib": 6000.0}, "process": {"dedicated_mib": 5500, "shared_mib": 20}}]}


def _experiment(name, arms):
    rec = {"kind": "experiment", "evidence_class": "controlled", "created": "2026-01-01T00:00:00", "name": name, "arms": {},
           "gpu": {"name": "Test GPU", "memory.total": "8192", "bandwidth_gbs_known": 500.0},
           "model": {"summary": {"weight_bytes_streamed_per_decode_token": 5.0e9, "kv_bytes_per_token_f16": 1000}},
           "workload": {"set_hash": "abc", "n_predict": 64}}
    for arm_name, (args, reqs, ded) in arms.items():
        rec["arms"][arm_name] = {"requests": reqs, "args": ["llama-server", "--model", "m.gguf"] + args, "env_ggml": {}, "failures": [],
                                 "launch_facts": [{"process_counters_after": {"dedicated_mib": ded, "shared_mib": 10}, "gpu_after": {"mem_used_mib": 6000.0}}]}
    return rec


class PhaseSplitTests(unittest.TestCase):
    def test_prefill_bound_and_residual(self):
        reqs = [_req(1500, 3000, 64, 1300, 4600, prompt_id="p%02d" % i) for i in range(6)]
        d = dg.diagnose([_measurement("m", reqs)])
        ids = {f["id"]: f for f in d["findings"]}
        self.assertIn("workload_phase_split", ids)
        self.assertIn("prefill-bound", ids["workload_phase_split"]["title"])
        self.assertAlmostEqual(ids["workload_phase_split"]["measurements"]["prefill_share_median"], 3000 / 4600, places=2)
        # residual 300 ms of 4600 = 6.5% > 5% threshold
        self.assertIn("per_request_overhead", ids)
        self.assertEqual(ids["per_request_overhead"]["measurements"]["residual_ms_median"], 300)

    def test_too_few_requests_gives_no_split(self):
        reqs = [_req(1500, 3000, 64, 1300, 4600)]
        d = dg.diagnose([_measurement("m", reqs)])
        self.assertNotIn("workload_phase_split", {f["id"] for f in d["findings"]})
        self.assertTrue(any("Fewer than 6" in s for s in d["insufficient_evidence"]))


class MicrobatchRuleTests(unittest.TestCase):
    def test_small_ubatch_is_a_hypothesis_with_sweep_arms(self):
        reqs = [_req(1500, 30000, 64, 1300, 31400, prompt_id="p%02d" % i) for i in range(6)]
        rec = _experiment("e", {"base": (["--ubatch-size", "32", "--batch-size", "512"], reqs, 5000)})
        d = dg.diagnose([rec])
        f = {x["id"]: x for x in d["findings"]}["microbatch_limits_prefill"]
        self.assertEqual(f["confidence"], "hypothesis")
        self.assertEqual(f["measurements"]["n_ubatch"], 32.0)
        names = [a["name"] for a in f["experiment"]["arms"]]
        self.assertEqual(names, ["ub32", "ub64", "ub128"])
        self.assertEqual(d["recommended_next_experiment"]["finding"], "microbatch_limits_prefill")

    def test_unknown_ubatch_is_marked_insufficient(self):
        reqs = [_req(1500, 3000, 64, 1300, 4400, prompt_id="p%02d" % i) for i in range(6)]
        d = dg.diagnose([_measurement("m", reqs)])
        f = {x["id"]: x for x in d["findings"]}["microbatch_unknown"]
        self.assertEqual(f["confidence"], "insufficient")


class DecodeCeilingTests(unittest.TestCase):
    def test_fraction_against_bandwidth(self):
        # 5 GB per token at 500 GB/s -> ceiling 100 t/s; measured 25 t/s -> 0.25 (below the 0.35 "low" threshold)
        reqs = [_req(1500, 3000, 64, 2560, 5700, prompt_id="p%02d" % i) for i in range(6)]
        rec = _experiment("e", {"base": (["--ubatch-size", "512"], reqs, 5000)})
        d = dg.diagnose([rec])
        f = {x["id"]: x for x in d["findings"]}["decode_below_ceiling"]
        self.assertEqual(f["measurements"]["ceiling_tps_at_peak_bandwidth"], 100.0)
        self.assertAlmostEqual(f["measurements"]["achieved_fraction"], 0.25, places=2)
        self.assertEqual(f["severity"], "high")
        # graphs are not disabled here: toggling them could only confirm the current state, so the next step is a decode profile
        self.assertEqual(f["experiment"]["kind"], "profile_decode")

    def test_graphs_experiment_only_when_graphs_are_off_and_baseline_first(self):
        reqs = [_req(1500, 3000, 64, 2560, 5700, prompt_id="p%02d" % i) for i in range(6)]
        rec = _experiment("e", {"base": (["--ubatch-size", "512"], reqs, 5000)})
        rec["arms"]["base"]["env_ggml"] = {"GGML_CUDA_DISABLE_GRAPHS": "1"}
        f = {x["id"]: x for x in dg.diagnose([rec])["findings"]}["decode_below_ceiling"]
        self.assertEqual([a["name"] for a in f["experiment"]["arms"]], ["graphs-off", "graphs-on"])

    def test_missing_bandwidth_is_insufficient(self):
        reqs = [_req(1500, 3000, 64, 2560, 5700, prompt_id="p%02d" % i) for i in range(6)]
        rec = _experiment("e", {"base": (["--ubatch-size", "512"], reqs, 5000)})
        rec["gpu"]["bandwidth_gbs_known"] = None
        d = dg.diagnose([rec])
        f = {x["id"]: x for x in d["findings"]}["decode_ceiling_unknown"]
        self.assertEqual(f["confidence"], "insufficient")


class ExperimentVerdictTests(unittest.TestCase):
    def test_keep_when_faster_and_identical(self):
        base = [_req(1500, 3000, 64, 1300, 4400 + i, prompt_id="p%02d" % (i % 3)) for i in range(6)]
        cand = [_req(1500, 2000, 64, 1300, 3400 + i, prompt_id="p%02d" % (i % 3)) for i in range(6)]
        rec = _experiment("e", {"base": (["--ubatch-size", "128"], base, 5000), "cand": (["--ubatch-size", "512"], cand, 5200)})
        d = dg.diagnose([rec])
        v = d["experiment_verdicts"][0]["arms"][0]
        self.assertTrue(v["verdict"].startswith("keep"))
        self.assertEqual(v["metrics"]["wall_ms"]["call"], "improvement")
        self.assertEqual(v["outputs"]["identical"], v["outputs"]["prompts_compared"])
        self.assertEqual(v["vram_delta_mib"], 200)

    def test_divergent_outputs_need_explanation(self):
        base = [_req(1500, 3000, 64, 1300, 4400, prompt_id="p%02d" % i) for i in range(6)]
        cand = [_req(1500, 2000, 64, 1300, 3400, prompt_id="p%02d" % i, tokens=[99] + list(range(1, 64))) for i in range(6)]
        rec = _experiment("e", {"base": ([], base, 5000), "cand": ([], cand, 5000)})
        v = dg.diagnose([rec])["experiment_verdicts"][0]["arms"][0]
        self.assertIn("explain the difference", v["verdict"])
        self.assertTrue(v["verdict"].startswith("faster"))
        self.assertEqual(v["outputs"]["identical"], 0)

    def test_within_noise_is_weak(self):
        # the same prompt varies by +-10% between repetitions; a 2.5% shift is inside that noise
        base = [_req(1500, 3000, 64, 1300, 4000 + 400 * (i // 3), prompt_id="p%02d" % (i % 3)) for i in range(9)]
        cand = [_req(1500, 2900, 64, 1300, 3900 + 400 * (i // 3), prompt_id="p%02d" % (i % 3)) for i in range(9)]
        rec = _experiment("e", {"base": ([], base, 5000), "cand": ([], cand, 5000)})
        v = dg.diagnose([rec])["experiment_verdicts"][0]["arms"][0]
        self.assertIn("within noise", v["metrics"]["wall_ms"]["call"])
        self.assertTrue(v["verdict"].startswith("inconclusive"))

    def test_consistent_shift_beats_prompt_length_spread(self):
        # prompts of very different lengths (wall 2 s vs 8 s) and a steady 150 ms saving on each: pooled spreads would
        # overlap, but the paired comparison sees a consistent 2-7% improvement with near-zero within-prompt noise
        base = [_req(500 * (1 + i % 3), 1000 * (1 + i % 3), 64, 1300, 2500 + 2000 * (i % 3), prompt_id="p%02d" % (i % 3)) for i in range(9)]
        cand = [_req(500 * (1 + i % 3), 1000 * (1 + i % 3), 64, 1300, 2350 + 2000 * (i % 3), prompt_id="p%02d" % (i % 3)) for i in range(9)]
        rec = _experiment("e", {"base": ([], base, 5000), "cand": ([], cand, 5000)})
        v = dg.diagnose([rec])["experiment_verdicts"][0]["arms"][0]
        self.assertEqual(v["metrics"]["wall_ms"]["call"], "improvement")
        self.assertEqual(v["metrics"]["residual_ms"]["call"], "improvement")
        self.assertTrue(v["verdict"].startswith("keep"))


class ProposeTests(unittest.TestCase):
    def test_spec_from_finding_keeps_one_variable(self):
        reqs = [_req(1500, 30000, 64, 1300, 31400, prompt_id="p%02d" % i) for i in range(6)]
        rec = _experiment("e", {"base": (["--ctx-size", "4096", "--batch-size", "64", "--ubatch-size", "32"], reqs, 5000)})
        d = dg.diagnose([rec])
        with tempfile.TemporaryDirectory() as td:
            dpath = os.path.join(td, "diag.json")
            with open(dpath, "w", encoding="utf-8") as f:
                json.dump(d, f)
            spec, source = pp.propose(dpath, None, {"path_prepend": ["x"]}, "workloads/w.json", os.path.join(td, "spec.json"))
            self.assertEqual(source, "diagnosed experiment arm")
            self.assertEqual(spec["binary"], "${runtime_binary}")
            self.assertEqual(spec["common_args"], ["--ctx-size", "4096", "--batch-size", "64", "--ubatch-size", "32"])
            self.assertEqual([a["name"] for a in spec["arms"]], ["ub32", "ub64", "ub128"])
            # the sweep raises --batch-size where the microbatch would exceed it; nothing else changes
            self.assertEqual(spec["arms"][2]["args_override"], {"--ubatch-size": 128, "--batch-size": 128})
            self.assertEqual(spec["arms"][0]["args_override"], {"--ubatch-size": 32})
            self.assertEqual(spec["path_prepend"], "${path_prepend}")
            self.assertEqual(spec["provenance"]["finding"], "microbatch_limits_prefill")

    def test_procedural_experiment_cannot_be_proposed(self):
        f = {"id": "x", "experiment": {"kind": "request_shape", "arms": "do it by hand"}}
        with self.assertRaises(ValueError):
            pp.build_spec(f, {}, [], "w.json")


if __name__ == "__main__":
    unittest.main()
