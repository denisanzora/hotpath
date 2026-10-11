"""Cache modes and verification, slot-save-path injection, and the levels of output validation.
Synthetic inputs; nothing here is a performance claim."""
import unittest

from hotpath import diagnose as dg
from hotpath import experiment as ex
from hotpath import measure as ms


def _req(pid, cache_n, prompt_n=1000, wall=3000, pms=2000, gms=900, valid=None, schema_valid=None, req_present=None):
    r = {"prompt_id": pid, "prompt_n": prompt_n, "cache_n": cache_n, "cache_state": ms.cache_state({"cache_n": cache_n}),
         "prompt_ms": pms, "prompt_tps": 1000.0 * prompt_n / pms, "predicted_n": 64, "predicted_ms": gms, "predicted_tps": 64000.0 / gms,
         "wall_ms": wall, "tokens": None, "output_sha256": "h" + pid, "stop_type": "stop"}
    if valid is not None:
        r.update({"valid_json": valid, "schema_required_present": req_present if req_present is not None else valid, "schema_valid": schema_valid,
                  "validation_level": ms.VALIDATION_LEVEL_FULL if schema_valid is not None else ms.VALIDATION_LEVEL_SUBSET})
    return r


def _exp(base_reqs, cand_reqs, base_cache, cand_cache, rounds=3):
    return {"kind": "experiment", "evidence_class": "controlled", "created": "x", "name": "e", "spec_redacted": {"rounds": rounds, "arms": [{"name": "a"}, {"name": "b"}]},
            "gpu": {"memory.total": "8192"}, "workload": {"set_hash": "s", "n_predict": 64},
            "arms": {"a": {"requests": base_reqs, "args": [], "env_ggml": {}, "failures": [], "launch_facts": [], "cache": base_cache},
                     "b": {"requests": cand_reqs, "args": [], "env_ggml": {}, "failures": [], "launch_facts": [], "cache": cand_cache}}}


class CacheStateTests(unittest.TestCase):
    def test_cache_state_from_server_count(self):
        self.assertEqual(ms.cache_state({"cache_n": 0}), "cold")
        self.assertEqual(ms.cache_state({"cache_n": 237}), "warm")
        self.assertEqual(ms.cache_state({}), "unknown")

    def test_cold_fallback_when_erase_unsupported(self):
        from hotpath import gpu as gpu_mod
        from hotpath import server as srv
        saved = (srv.erase_slot, srv.request, gpu_mod.gpu_dynamic)
        sent = []
        try:
            srv.erase_slot = lambda base, slot_id=0: {"error": "erase failed HTTP 501"}
            srv.request = lambda base, ep, body: (sent.append(body) or {"timings": {"prompt_n": 300, "cache_n": 0, "prompt_ms": 500, "prompt_per_second": 600, "predicted_n": 8, "predicted_ms": 160, "predicted_per_second": 50},
                                                                        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "_wall_ms": 700})
            gpu_mod.gpu_dynamic = lambda: {}
            w = {"kind": "requests", "cache_mode": "cold", "items": [{"id": "r00", "sha256": "x", "endpoint": "/v1/chat/completions", "body": {"messages": [], "cache_prompt": True}}]}
            out = ms.run_requests_workload("http://x", w, repetitions=1, warmup=0)
            self.assertFalse(out["cache"]["slot_erase_supported"])
            self.assertIn("cache_prompt=false", out["cache"]["isolation"])
            self.assertFalse(sent[0]["cache_prompt"])              # the fallback overrides the application's cache_prompt
            self.assertEqual(out["requests"][0]["cache_state"], "cold")
        finally:
            srv.erase_slot, srv.request, gpu_mod.gpu_dynamic = saved

    def test_warm_and_as_is_do_not_touch_requests(self):
        from hotpath import gpu as gpu_mod
        from hotpath import server as srv
        saved = (srv.erase_slot, srv.request, gpu_mod.gpu_dynamic)
        try:
            srv.erase_slot = lambda base, slot_id=0: self.fail("erase must not be called in warm/as-is mode")
            srv.request = lambda base, ep, body: {"timings": {"prompt_n": 4, "cache_n": 296, "prompt_ms": 10, "prompt_per_second": 400, "predicted_n": 8, "predicted_ms": 160, "predicted_per_second": 50},
                                                  "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "_wall_ms": 200}
            gpu_mod.gpu_dynamic = lambda: {}
            for mode in ("warm", "as-is"):
                w = {"kind": "requests", "cache_mode": mode, "items": [{"id": "r00", "sha256": "x", "endpoint": "/completion", "body": {"prompt": "p", "cache_prompt": True}}]}
                out = ms.run_requests_workload("http://x", w, repetitions=1, warmup=0)
                self.assertEqual(out["cache"]["isolation"], "none")
                self.assertEqual(out["requests"][0]["cache_state"], "warm")
        finally:
            srv.erase_slot, srv.request, gpu_mod.gpu_dynamic = saved


class SlotSavePathTests(unittest.TestCase):
    def test_auto_injection_and_opt_out(self):
        spec = {"name": "e", "binary": "llama-server", "model": "m.gguf", "common_args": ["--ctx-size", "4096"], "arms": [{"name": "a"}]}
        cmd, _env, _port = ex.arm_launch(spec, spec["arms"][0], out_dir="results/local")
        self.assertNotIn("--slot-save-path", cmd)          # off by default: the launched command stays the user's
        spec["slot_save_path"] = "auto"
        cmd, _env, _port = ex.arm_launch(spec, spec["arms"][0], out_dir="results/local")
        self.assertTrue(cmd[cmd.index("--slot-save-path") + 1].replace("\\", "/").endswith("slot-cache/e"))
        spec["slot_save_path"] = "/tmp/x"
        cmd, _env, _port = ex.arm_launch(spec, spec["arms"][0], out_dir="results/local")
        self.assertEqual(cmd[cmd.index("--slot-save-path") + 1], "/tmp/x")
        spec["common_args"] = ["--slot-save-path", "/mine"]
        spec["slot_save_path"] = "auto"
        cmd, _env, _port = ex.arm_launch(spec, spec["arms"][0], out_dir="results/local")
        self.assertEqual(cmd.count("--slot-save-path"), 1)


class VerdictCacheTests(unittest.TestCase):
    def _cold(self, pid, wall):
        return _req(pid, 0, wall=wall)

    def test_cold_vs_cold_is_comparable_and_kept(self):
        base = [self._cold("p%d" % (i % 3), 3000 + 10 * i) for i in range(9)]
        cand = [self._cold("p%d" % (i % 3), 2700 + 10 * i) for i in range(9)]
        for r in cand:
            r["output_sha256"] = "h" + r["prompt_id"]
        v = dg.experiment_verdict(_exp(base, cand, {"mode": "cold", "isolation": "slot erase before each pass", "slot_erase_supported": True}, {"mode": "cold", "isolation": "slot erase before each pass", "slot_erase_supported": True}))
        row = v["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "cold")
        self.assertTrue(row["verdict"].startswith("keep"))

    def test_mismatch_cannot_yield_a_cold_verdict(self):
        base = [self._cold("p%d" % (i % 3), 3000) for i in range(9)]
        cand = [_req("p%d" % (i % 3), 900, prompt_n=100, wall=1200, pms=200) for i in range(9)]   # warm: much faster, but only because of reuse
        v = dg.experiment_verdict(_exp(base, cand, {"mode": "cold", "isolation": "slot erase before each pass", "slot_erase_supported": True}, {"mode": "cold", "isolation": "slot erase before each pass", "slot_erase_supported": True}))
        row = v["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "mismatch")
        self.assertTrue(row["verdict"].startswith("inconclusive"))
        self.assertTrue(row["metrics"]["wall_ms"]["call"].startswith("not comparable"))
        self.assertTrue(row["metrics"]["prefill_tps"]["call"].startswith("not comparable"))
        self.assertNotIn("not comparable", row["metrics"]["decode_tps"]["call"])

    def test_cold_requested_but_warm_is_not_comparable(self):
        # the real failure mode: slot erase honored, but the host prompt cache restored the prompt
        base = [_req("p%d" % (i % 3), 900, prompt_n=4, wall=3000) for i in range(9)]
        cand = [_req("p%d" % (i % 3), 900, prompt_n=4, wall=2300) for i in range(9)]
        cc = {"mode": "cold", "isolation": "slot erase before each pass", "slot_erase_supported": True}
        row = dg.experiment_verdict(_exp(base, cand, cc, cc))["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "not cold")
        self.assertTrue(row["verdict"].startswith("inconclusive"))

    def test_cold_mode_always_disables_cache_prompt(self):
        from hotpath import gpu as gpu_mod
        from hotpath import server as srv
        saved = (srv.erase_slot, srv.request, gpu_mod.gpu_dynamic)
        sent, erased = [], []
        try:
            srv.erase_slot = lambda base, slot_id=0: (erased.append(1) or {"id_slot": 0, "n_erased": 10})
            srv.request = lambda base, ep, body: (sent.append(body) or {"timings": {"prompt_n": 300, "cache_n": 0, "prompt_ms": 500, "prompt_per_second": 600, "predicted_n": 8, "predicted_ms": 160, "predicted_per_second": 50},
                                                                        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "_wall_ms": 700})
            gpu_mod.gpu_dynamic = lambda: {}
            w = {"kind": "requests", "cache_mode": "cold", "items": [{"id": "r00", "sha256": "x", "endpoint": "/completion", "body": {"prompt": "p"}}]}
            out = ms.run_requests_workload("http://x", w, repetitions=2, warmup=0)
            self.assertTrue(out["cache"]["slot_erase_supported"])
            self.assertTrue(all(b["cache_prompt"] is False for b in sent))   # even when erase works
            self.assertGreaterEqual(len(erased), 3)                          # probe + one per pass
        finally:
            srv.erase_slot, srv.request, gpu_mod.gpu_dynamic = saved

    def test_warm_by_design_is_comparable_but_labelled(self):
        base = [_req("p%d" % (i % 3), 900, prompt_n=100, wall=1500, pms=200) for i in range(9)]
        cand = [_req("p%d" % (i % 3), 900, prompt_n=100, wall=1400, pms=190) for i in range(9)]
        v = dg.experiment_verdict(_exp(base, cand, {"mode": "warm", "isolation": "none"}, {"mode": "warm", "isolation": "none"}))
        row = v["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "warm")
        self.assertIn("warm cache", row["metrics"]["prefill_tps"]["call"])

    def test_unknown_state_blocks_cold_claims(self):
        base = [dict(_req("p%d" % (i % 3), 0, wall=3000), cache_n=None, cache_state="unknown") for i in range(9)]
        cand = [dict(_req("p%d" % (i % 3), 0, wall=2500), cache_n=None, cache_state="unknown") for i in range(9)]
        v = dg.experiment_verdict(_exp(base, cand, {"mode": "as-is", "isolation": "none"}, {"mode": "as-is", "isolation": "none"}))
        self.assertEqual(v["arms"][0]["cache"]["comparability"]["basis"], "unknown")
        self.assertTrue(v["arms"][0]["verdict"].startswith("inconclusive"))


class ValidityLevelTests(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["candidates"], "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": 1,
              "items": {"type": "object", "required": ["draft", "kind"], "properties": {"draft": {"type": "string", "maxLength": 5}, "kind": {"enum": ["a", "b"]}}}}}}

    def test_full_validation_when_validator_installed(self):
        if ms.schema_validator() is None:
            self.skipTest("jsonschema not installed")
        good = ms._validity('{"candidates": [{"draft": "hi", "kind": "a"}]}', self.SCHEMA)
        self.assertEqual((good["valid_json"], good["schema_required_present"], good["schema_valid"]), (True, True, True))
        self.assertEqual(good["validation_level"], ms.VALIDATION_LEVEL_FULL)
        # required top-level key present, but nested constraints violated: the subset check passes, full validation fails
        bad = ms._validity('{"candidates": [{"draft": "far too long", "kind": "z"}]}', self.SCHEMA)
        self.assertEqual((bad["valid_json"], bad["schema_required_present"], bad["schema_valid"]), (True, True, False))
        self.assertTrue(all(set(e) == {"keyword", "path"} for e in bad["schema_errors"]))   # no output text in errors
        self.assertIn("maxLength", [e["keyword"] for e in bad["schema_errors"]])
        self.assertIn("enum", [e["keyword"] for e in bad["schema_errors"]])

    def test_subset_level_without_validator(self):
        saved = ms.schema_validator
        try:
            ms.schema_validator = lambda: None
            r = ms._validity('{"candidates": [{"draft": "far too long", "kind": "z"}]}', self.SCHEMA)
            self.assertEqual((r["valid_json"], r["schema_required_present"], r["schema_valid"]), (True, True, None))
            self.assertEqual(r["validation_level"], ms.VALIDATION_LEVEL_SUBSET)
        finally:
            ms.schema_validator = saved

    def test_verdict_decides_on_strongest_level(self):
        base = [_req("p%d" % (i % 3), 0, valid=True, schema_valid=True) for i in range(9)]
        cand = [_req("p%d" % (i % 3), 0, wall=2500, valid=True, schema_valid=False) for i in range(9)]   # required keys present, nested schema broken
        v = dg.experiment_verdict(_exp(base, cand, {"mode": "cold", "isolation": "x", "slot_erase_supported": True}, {"mode": "cold", "isolation": "x", "slot_erase_supported": True}))
        row = v["arms"][0]
        self.assertEqual(row["validity"]["decided_on"], "schema_valid")
        self.assertTrue(row["verdict"].startswith("reject: output validity dropped on full schema validation"))


if __name__ == "__main__":
    unittest.main()
