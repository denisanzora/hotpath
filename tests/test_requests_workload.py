"""Application-shaped (`requests`) workloads: import, overrides, validity judgement, hash-based equivalence."""
import json
import os
import tempfile
import unittest

from hotpath import compare as cmp_mod
from hotpath import measure as ms
from hotpath import workload as wl


BODY = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "temperature": 0, "seed": 42, "max_tokens": 64,
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}}}}}


class ImportTests(unittest.TestCase):
    def test_jsonl_and_array_import(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "cap.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(BODY) + "\n" + json.dumps({"endpoint": "/completion", "body": {"prompt": "x", "n_predict": 8}}) + "\n")
            w = wl.workload_from_requests(p, name="t")
            self.assertEqual(w["kind"], "requests")
            self.assertEqual([it["endpoint"] for it in w["items"]], ["/v1/chat/completions", "/completion"])
            self.assertTrue(all(it["body"]["stream"] is False for it in w["items"]))
            ident = wl.workload_identity(w)
            self.assertEqual(ident["n_prompts"], 2)
            self.assertIn("response_format", ident["fields_present"])
            self.assertNotIn("messages", ident["fields_present"])
            wl.save_workload(w, os.path.join(td, "w.json"))
            w2 = wl.load_workload(os.path.join(td, "w.json"))
            self.assertEqual(w2["items"][0]["sha256"], w["items"][0]["sha256"])

    def test_rejects_non_request(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps([{"foo": 1}]))
            with self.assertRaises(ValueError):
                wl.workload_from_requests(p)


class OverrideAndValidityTests(unittest.TestCase):
    def test_override_set_and_remove(self):
        b = ms.apply_request_override(BODY, {"remove": ["response_format"], "set": {"temperature": 0.7}})
        self.assertNotIn("response_format", b)
        self.assertEqual(b["temperature"], 0.7)
        self.assertIn("response_format", BODY)  # original untouched

    def test_validity_judgement(self):
        schema = ms._schema_of(BODY)
        self.assertEqual(ms._schema_required(BODY), ["a", "b"])
        pick = lambda d: (d["valid_json"], d["schema_required_present"])
        self.assertEqual(pick(ms._validity('{"a": 1, "b": "x"}', schema)), (True, True))
        self.assertEqual(pick(ms._validity('{"a": 1}', schema)), (True, False))
        self.assertEqual(pick(ms._validity('not json', schema)), (False, False))
        self.assertEqual(pick(ms._validity("anything", None)), (None, None))
        if ms.schema_validator():
            self.assertTrue(ms._validity('{"a": 1, "b": "x"}', schema)["schema_valid"])
            self.assertFalse(ms._validity('{"a": "one", "b": "x"}', schema)["schema_valid"])   # wrong nested type

    def test_validity_judged_against_original_item_schema(self):
        # an arm that removes the schema is still judged on the application's schema
        item_schema = ms._schema_of(BODY)
        overridden = ms.apply_request_override(BODY, {"remove": ["response_format"]})
        self.assertIsNone(ms._schema_of(overridden))
        self.assertFalse(ms._validity('{"draft": "x"}', item_schema)["schema_required_present"])

    def test_summary_counts_validity(self):
        reqs = [{"prompt_n": 10, "cache_n": 0, "prompt_tps": 1, "predicted_tps": 1, "prompt_ms": 1, "predicted_ms": 1, "wall_ms": 3, "predicted_n": 1, "stop_type": "stop",
                 "valid_json": v, "schema_required_present": s, "schema_valid": f, "validation_level": ms.VALIDATION_LEVEL_FULL}
                for v, s, f in ((True, True, True), (True, False, False), (False, False, False))]
        s = ms.summarize_requests(reqs)
        self.assertEqual(s["validity"], {"judged": 3, "valid_json": 2, "schema_required_present": 1, "schema_checked": 3, "schema_valid": 1,
                                         "validation_level": ms.VALIDATION_LEVEL_FULL, "schema_unchecked_keywords": []})


class RunnerTests(unittest.TestCase):
    def test_failed_slot_erase_is_recorded_and_validity_uses_item_schema(self):
        from hotpath import gpu as gpu_mod
        from hotpath import server as srv
        saved = (srv.erase_slot, srv.request, gpu_mod.gpu_dynamic)
        calls = []
        try:
            srv.erase_slot = lambda base, slot_id=0: {"error": "erase failed HTTP 501"}
            srv.request = lambda base, ep, body: (calls.append(body) or {
                "timings": {"prompt_n": 4, "cache_n": 1000, "prompt_ms": 10, "prompt_per_second": 400, "predicted_n": 5, "predicted_ms": 100, "predicted_per_second": 50},
                "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}], "_wall_ms": 150})
            gpu_mod.gpu_dynamic = lambda: {}
            w = {"kind": "requests", "cache_mode": "cold", "items": [{"id": "r00", "sha256": "x", "endpoint": "/v1/chat/completions", "body": BODY}]}
            out = ms.run_requests_workload("http://x", w, repetitions=2, warmup=0, request_override={"remove": ["response_format"]})
            self.assertEqual(out["slot_erase_error"], "erase failed HTTP 501")
            self.assertEqual(len(out["requests"]), 2)
            self.assertNotIn("response_format", calls[0])                       # the override was applied to what was sent
            self.assertEqual(out["requests"][0]["schema_required_present"], False)  # but validity is judged on the item's schema (needs a and b)
            self.assertEqual(out["requests"][0]["validity_schema_from"], "workload item")
            self.assertIsNone(out["requests"][0]["output_text"])                 # outputs are not stored by default
        finally:
            srv.erase_slot, srv.request, gpu_mod.gpu_dynamic = saved


class HashEquivalenceTests(unittest.TestCase):
    def test_hash_based_equivalence_without_tokens(self):
        sets = {"a": {"requests": [{"prompt_id": "r00", "tokens": None, "output_sha256": "abc"}], "meta": {}},
                "b": {"requests": [{"prompt_id": "r00", "tokens": None, "output_sha256": "abc"}], "meta": {}},
                "c": {"requests": [{"prompt_id": "r00", "tokens": None, "output_sha256": "zzz"}], "meta": {}}}
        oe = cmp_mod.output_equivalence(sets)
        self.assertEqual(oe["r00"]["basis"], "output content hash")
        self.assertTrue(oe["r00"]["b"]["identical"])
        self.assertFalse(oe["r00"]["c"]["identical"])
        self.assertEqual(oe["r00"]["c"]["first_divergence_index"], -1)


if __name__ == "__main__":
    unittest.main()
