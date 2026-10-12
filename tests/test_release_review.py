"""Regression tests for defects found in the independent 0.2 release review. Each test names the failure it prevents."""
import json
import os
import tempfile
import unittest

from hotpath import diagnose as dg
from hotpath import experiment as ex
from hotpath import measure as ms
from hotpath import mcp_server as mcp
from hotpath import propose as pp

COLD = {"mode": "cold", "isolation": "cache_prompt=false on every request", "slot_erase_supported": False}
WARM = {"mode": "warm", "isolation": "none"}


def _req(pid, cache_n="absent", wall=3000, pms=2000, **kw):
    r = {"prompt_id": pid, "prompt_n": 1000, "prompt_ms": pms, "prompt_tps": 1000.0 * 1000 / pms, "predicted_n": 64, "predicted_ms": 900,
         "predicted_tps": 71.1, "wall_ms": wall, "tokens": None, "output_sha256": "h" + pid, "stop_type": "stop"}
    if cache_n != "absent":
        r["cache_n"] = cache_n
    r.update(kw)
    return r


def _exp(base, cand, bc=COLD, cc=COLD, rounds=3):
    return {"kind": "experiment", "evidence_class": "controlled", "created": "x", "name": "e", "spec_redacted": {"rounds": rounds},
            "gpu": {"memory.total": "8192"}, "workload": {"set_hash": "s", "n_predict": 64},
            "arms": {"a": {"requests": base, "args": [], "env_ggml": {}, "failures": [], "launch_facts": [], "cache": bc},
                     "b": {"requests": cand, "args": [], "env_ggml": {}, "failures": [], "launch_facts": [], "cache": cc}}}


class CacheEvidenceTests(unittest.TestCase):
    def test_malformed_cache_n_is_unknown_and_never_crashes(self):
        # review: cache_state() raised on "n/a" / NaN / list mid-run, and treated -1 and True as warm reuse
        for bad in ("n/a", "", [], {}, float("nan"), -1, True, False, "12", 1.5, None):
            self.assertEqual(ms.cache_state({"cache_n": bad}), "unknown", bad)
        self.assertEqual(ms.cache_state({}), "unknown")
        self.assertEqual(ms.cache_state({"cache_n": 0}), "cold")
        self.assertEqual(ms.cache_state({"cache_n": 0.0}), "cold")
        self.assertEqual(ms.cache_state({"cache_n": 237}), "warm")

    def test_missing_cache_n_is_never_read_as_cold(self):
        base = [_req("p%d" % (i % 3)) for i in range(9)]            # no cache_n anywhere (older record / other server)
        cand = [_req("p%d" % (i % 3), wall=2500) for i in range(9)]
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "unknown")
        self.assertTrue(row["verdict"].startswith("inconclusive"))

    def test_summary_survives_records_without_cache_n(self):
        s = ms.summarize_requests([{"prompt_n": 5}, {"prompt_n": 6, "cache_n": "oops"}])
        self.assertEqual(s["cache_states"], {"unknown": 2})
        self.assertIsNone(s["cache_n_max"])

    def test_stored_label_is_not_trusted(self):
        r = _req("p0", cache_n=900)
        r["cache_state"] = "cold"                                      # a label that contradicts the server's own count
        self.assertEqual(dg.request_cache_state(r), "warm")


class RecordNameAndConditionsTests(unittest.TestCase):
    def test_same_second_records_never_overwrite(self):
        # found by a real agent: two diagnose calls in one second wrote one file, silently losing the first diagnosis
        from hotpath.records import write_record
        with tempfile.TemporaryDirectory() as td:
            paths = [write_record({"kind": "diagnosis", "n": i}, td, "diagnosis") for i in range(5)]
            self.assertEqual(len({str(p) for p in paths}), 5)
            seen = []
            for p in paths:
                with open(p, encoding="utf-8") as f:
                    seen.append(json.load(f)["n"])
            self.assertEqual(sorted(seen), [0, 1, 2, 3, 4])

    def test_diagnosis_states_cache_conditions_of_measurements(self):
        # found by a real agent: it could not tell what cache state a measurement's numbers were taken under
        cold = {"kind": "measurement", "evidence_class": "active", "label": "baseline", "created": "x", "target": {}, "gpu": {}, "workload": {},
                "settings": {}, "cache": {"mode": "cold", "isolation": "cache_prompt=false on every request"},
                "requests": [_req("p%d" % (i % 3), 0) for i in range(6)]}
        old = dict(cold, label="old-format", cache=None, requests=[_req("p%d" % (i % 3)) for i in range(6)])     # no cache_n at all
        d = dg.diagnose([cold, old])
        by = {c["source"]: c for c in d["cache_conditions"]}
        self.assertEqual(by["measurement baseline"]["state"], "cold")
        self.assertEqual(by["measurement old-format"]["state"], "unknown")
        md = dg.diagnosis_markdown(d)
        self.assertIn("measurement baseline: cold (cache_n = 0 on 6/6 requests", md)
        self.assertIn("not assumed cold", md)


class PairedContaminationTests(unittest.TestCase):
    def _arms(self, warm_prompt_in_cand=True):
        base = [_req("p%d" % (i % 10), 0, wall=3000, pms=2000) for i in range(60)]
        cand = [_req("p%d" % (i % 10), 0, wall=3000, pms=2000) for i in range(60)]
        if warm_prompt_in_cand:
            for r in cand:
                if r["prompt_id"] == "p0":
                    r.update(cache_n=900, wall_ms=600, prompt_ms=100, prompt_tps=10000.0)
        return base, cand

    def test_contaminated_prompt_is_excluded_not_averaged_in(self):
        # review: 54/60 requests cold = 90% passed the arm-level threshold, and the warm prompt's 20x prefill ratio entered the pairing
        base, cand = self._arms()
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        self.assertTrue(row["cache"]["comparability"]["comparable"])
        self.assertEqual(row["cache"]["excluded"]["candidate"], 6)
        self.assertEqual(row["cache"]["excluded"]["dropped_prompts"], ["p0"])
        pf = row["metrics"]["prefill_tps"]
        self.assertEqual(pf["prompts"], 9)
        self.assertNotIn("p0", [r["prompt_id"] for r in pf["per_prompt"]])
        self.assertEqual(pf["call"], "no difference")
        # decode is not cache-sensitive and keeps every prompt
        self.assertEqual(row["metrics"]["decode_tps"]["prompts"], 10)

    def test_keep_verdict_discloses_exclusions(self):
        base, cand = self._arms(warm_prompt_in_cand=False)
        for r in cand:
            r["wall_ms"] = 2500
        cand[0].update(cache_n=800)                                    # one stray warm request in one arm
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        self.assertTrue(row["verdict"].startswith("keep"), row["verdict"])
        self.assertIn("exclude 0 baseline / 1 candidate unverified requests", row["verdict"])

    def test_too_few_verified_prompts_is_not_comparable(self):
        base = [_req("p%d" % (i % 3), 0) for i in range(30)]
        cand = [_req("p%d" % (i % 3), 0) for i in range(30)]
        for r in cand:
            if r["prompt_id"] in ("p0", "p1"):
                r["cache_n"] = 500
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        # candidate is 20/30 = 67% cold: arm-level mismatch already
        self.assertFalse(row["cache"]["comparability"]["comparable"])

    def test_warm_arms_with_different_reuse_are_not_comparable(self):
        base = [_req("p%d" % (i % 3), 900, wall=1500, pms=200) for i in range(9)]
        cand = [_req("p%d" % (i % 3), 300, wall=1400, pms=190) for i in range(9)]      # candidate reuses a third as much
        row = dg.experiment_verdict(_exp(base, cand, WARM, WARM))["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "warm reuse differs")
        self.assertTrue(row["metrics"]["wall_ms"]["call"].startswith("not comparable"))


class ValidityReviewTests(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string", "format": "date"}, "n": {"type": "integer"}}, "additionalProperties": False}

    def setUp(self):
        if ms.schema_validator() is None:
            self.skipTest("jsonschema not installed")

    def test_error_paths_never_contain_output_keys(self):
        # review: the path of an additionalProperties/type error carried a model-chosen key name
        r = ms._validity('{"a": "2020-01-01", "SECRET_MODEL_TEXT": 1}', self.SCHEMA)
        self.assertFalse(r["schema_valid"])
        self.assertNotIn("SECRET_MODEL_TEXT", json.dumps(r))
        r2 = ms._validity('{"a": "2020-01-01", "n": "x"}', {"type": "object", "properties": {"a": {}, "n": {"type": "integer"}}})
        self.assertEqual(r2["schema_errors"][0]["path"], "$.n")           # declared property names are kept
        r3 = ms._validity('{"x": {"SECRET": 1}}', {"type": "object", "additionalProperties": {"type": "string"}})
        self.assertEqual(r3["schema_errors"][0]["path"], "$.*")

    def test_unasserted_keywords_are_reported(self):
        # review: `format` is an annotation in Draft 2020-12, so a bad date still validated while the record said "full"
        r = ms._validity('{"a": "not-a-date"}', self.SCHEMA)
        self.assertTrue(r["schema_valid"])
        self.assertEqual(r["schema_unchecked_keywords"], ["format"])
        s = ms.summarize_requests([dict(_req("p0", 0), **r)])
        self.assertEqual(s["validity"]["schema_unchecked_keywords"], ["format"])

    def test_non_dict_schema_does_not_crash(self):
        for schema in (True, "x", []):
            ms._validity('{"a": 1}', schema)
        self.assertIsNone(ms._schema_of({"json_schema": "not a dict"}))
        self.assertIsNone(ms._schema_of({"response_format": []}))
        self.assertEqual(ms._schema_of({"response_format": {"type": "json_schema", "json_schema": {"schema": "bad"}}}), {})

    def test_unresolvable_ref_is_reported_not_claimed(self):
        r = ms._validity('{"a": 1}', {"$ref": "https://example.invalid/s.json"})
        self.assertIsNone(r["schema_valid"])
        self.assertTrue(r["validation_level"].startswith("validator could not run"))

    def test_rates_not_counts(self):
        # review: a candidate with fewer judged requests (a failed round) but 100% valid output was rejected as "dropped"
        def v(pid):
            return dict(_req(pid, 0), valid_json=True, schema_required_present=True, schema_valid=True, validation_level=ms.VALIDATION_LEVEL_FULL)
        base = [v("p%d" % (i % 3)) for i in range(9)]
        cand = [v("p%d" % (i % 3)) for i in range(6)]
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        self.assertFalse(row["validity"]["dropped"])
        cand_bad = [dict(v("p%d" % (i % 3)), schema_valid=(i % 3 != 0)) for i in range(6)]
        self.assertTrue(dg.experiment_verdict(_exp(base, cand_bad))["arms"][0]["validity"]["dropped"])

    def test_mixed_levels_downgrade_the_decision(self):
        def v(pid, sv):
            return dict(_req(pid, 0), valid_json=True, schema_required_present=True, schema_valid=sv, validation_level=ms.VALIDATION_LEVEL_FULL)
        base = [v("p%d" % (i % 3), True) for i in range(9)]
        cand = [v("p%d" % (i % 3), None if i == 0 else True) for i in range(9)]
        row = dg.experiment_verdict(_exp(base, cand))["arms"][0]
        self.assertEqual(row["validity"]["decided_on"], "schema_required_present")
        self.assertTrue(row["validity"]["downgraded_from_full"])


class ExplicitValidationTests(unittest.TestCase):
    def test_full_without_validator_is_an_actionable_error(self):
        saved = ms.schema_validator
        try:
            ms.schema_validator = lambda: None
            with self.assertRaises(SystemExit) as cm:
                ms.require_schema_validation("full")
            msg = str(cm.exception)
            self.assertIn('pip install "hotpath[schema]"', msg)
            self.assertIn("not installed", msg)
            ms.require_schema_validation("auto")              # allowed: records state the subset
            ms.require_schema_validation("required-keys")
            r = ms._validity('{"a": 1}', {"required": ["a"]}, "required-keys")
            self.assertEqual(r["validation_level"], ms.VALIDATION_LEVEL_CHOSEN_SUBSET)
            with self.assertRaises(SystemExit):
                ms.require_schema_validation("strict")
        finally:
            ms.schema_validator = saved

    def test_required_keys_mode_skips_full_validation_even_if_installed(self):
        if ms.schema_validator() is None:
            self.skipTest("jsonschema not installed")
        r = ms._validity('{"a": 1}', {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}, "required-keys")
        self.assertIsNone(r["schema_valid"])
        self.assertTrue(r["schema_required_present"])

    def test_experiment_fails_before_launching_anything(self):
        saved = ms.schema_validator
        with tempfile.TemporaryDirectory() as td:
            wl = os.path.join(td, "w.json")
            with open(wl, "w", encoding="utf-8") as f:
                json.dump({"name": "t", "kind": "requests", "items": [{"id": "r00", "endpoint": "/completion", "body": {"prompt": "x"}, "sha256": "x"}]}, f)
            spec = {"name": "t", "binary": "/nonexistent/llama-server", "model": "m.gguf", "workload": wl, "schema_validation": "full",
                    "arms": [{"name": "a"}, {"name": "b"}], "rounds": 1}
            try:
                ms.schema_validator = lambda: None
                with self.assertRaises(SystemExit) as cm:
                    ex.run(spec, td)
                self.assertIn("jsonschema", str(cm.exception))
            finally:
                ms.schema_validator = saved

    def test_spec_errors_surface_before_waiting_for_the_gpu(self):
        # review: with --wait-s a spec that could never run waited for an idle GPU first and failed afterwards
        saved_pre, saved_val = ex.check_preconditions, ms.schema_validator
        calls = []
        with tempfile.TemporaryDirectory() as td:
            wl = os.path.join(td, "w.json")
            with open(wl, "w", encoding="utf-8") as f:
                json.dump({"name": "t", "kind": "requests", "items": [{"id": "r00", "endpoint": "/completion", "body": {"prompt": "x"}, "sha256": "x"}]}, f)
            spec = {"name": "t", "binary": "/nonexistent", "model": "m.gguf", "workload": wl, "schema_validation": "full", "arms": [{"name": "a"}, {"name": "b"}],
                    "rounds": 1, "preconditions": {"max_gpu_util_pct": 10}}
            try:
                ex.check_preconditions = lambda pre: (calls.append(1) or {"checked": True, "ok": False, "reasons": ["busy"]})
                ms.schema_validator = lambda: None
                with self.assertRaises(SystemExit) as cm:
                    ex.run(spec, td, wait_for_preconditions_s=600)
                self.assertIn("jsonschema", str(cm.exception))
                self.assertEqual(calls, [])                        # the GPU was never even consulted
            finally:
                ex.check_preconditions, ms.schema_validator = saved_pre, saved_val

    def test_request_override_on_synthetic_workload_is_refused(self):
        # review: the arms would have run identical requests while the record claimed a request-shape comparison
        with tempfile.TemporaryDirectory() as td:
            wl = os.path.join(td, "w.json")
            with open(wl, "w", encoding="utf-8") as f:
                json.dump({"name": "t", "n_predict": 8, "prompts": [{"id": "p00", "text": "x", "sha256": "x"}]}, f)
            spec = {"name": "t", "binary": "/nonexistent", "model": "m.gguf", "workload": wl,
                    "arms": [{"name": "a"}, {"name": "b", "request_override": {"remove": ["response_format"]}}], "rounds": 1}
            with self.assertRaises(SystemExit) as cm:
                ex.run(spec, td)
            self.assertIn("request_override", str(cm.exception))


class ProposeGuardTests(unittest.TestCase):
    def _diag(self, td, workload_kind="requests"):
        finding = {"id": "request_shape_x", "title": "t", "experiment": {"kind": "request_shape", "variable": "v", "metric": "m",
                   "arms": [{"name": "as-captured"}, {"name": "no-schema", "request_override": {"remove": ["response_format"]}}]}}
        d = os.path.join(td, "diag.json")
        with open(d, "w", encoding="utf-8") as f:
            json.dump({"findings": [finding], "recommended_next_experiment": {"finding": "request_shape_x"}}, f)
        w = os.path.join(td, "w.json")
        with open(w, "w", encoding="utf-8") as f:
            json.dump({"kind": workload_kind, "name": "w", "items": [], "prompts": []}, f)
        return d, w

    def test_request_shape_needs_a_requests_workload(self):
        with tempfile.TemporaryDirectory() as td:
            d, w = self._diag(td, workload_kind="synthetic")
            with self.assertRaises(ValueError):
                pp.propose(d, None, {}, w, os.path.join(td, "spec.json"), common_args=["--ctx-size", "4096"])
            d, w = self._diag(td, workload_kind="requests")
            spec, _ = pp.propose(d, None, {}, w, os.path.join(td, "spec.json"), common_args=["--ctx-size", "4096"])
            self.assertEqual(spec["arms"][1]["request_override"], {"remove": ["response_format"]})

    def test_never_overwrites_existing_or_local_config(self):
        with tempfile.TemporaryDirectory() as td:
            d, w = self._diag(td)
            existing = os.path.join(td, "spec.json")
            with open(existing, "w", encoding="utf-8") as f:
                f.write("{}")
            with self.assertRaises(ValueError):
                pp.propose(d, None, {}, w, existing, common_args=[])
            with open(existing, encoding="utf-8") as f:
                self.assertEqual(f.read(), "{}")
            with self.assertRaises(ValueError):
                pp.propose(d, None, {}, w, os.path.join(td, "hotpath.local.json"), common_args=[], force=True)
            with self.assertRaises(ValueError):
                pp.propose(d, None, {}, w, os.path.join(td, "spec.txt"), common_args=[])
            pp.propose(d, None, {}, w, existing, common_args=[], force=True)       # CLI --force may replace a spec


class _InDir:
    """chdir into a directory for the duration of a block and always restore (Windows cannot delete the cwd)."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.old)


class ConsoleEncodingTests(unittest.TestCase):
    def test_unencodable_characters_do_not_abort_a_command(self):
        # found by exercising diagnose under a cp437 console: UnicodeEncodeError on an em dash after the record was written
        import subprocess
        import sys
        env = dict(os.environ, PYTHONIOENCODING="cp437")
        code = "from hotpath import cli; cli._tolerant_console(); print('a \\u2014 b')"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("a ", r.stdout)


class RunnerOwnedFlagTests(unittest.TestCase):
    def test_propose_strips_flags_that_would_share_the_users_server_resources(self):
        # review: a user's server_args with --port / --log-file would make the experiment server bind the production port
        # (the runner's own --port is overridden by a later duplicate) and write to the production log
        args = ["--ctx-size", "8192", "--port", "8080", "--host=0.0.0.0", "--log-file", "prod.log", "--slot-save-path", "D:/slots",
                "-m", "x.gguf", "--api-key", "k", "--jinja", "--cache-ram", "0", "--no-webui"]
        kept, removed = pp.strip_runner_owned(args)
        self.assertEqual(kept, ["--ctx-size", "8192", "--jinja", "--cache-ram", "0", "--no-webui"])
        self.assertEqual(sorted(removed), ["--api-key", "--host", "--log-file", "--port", "--slot-save-path", "-m"])
        self.assertEqual(pp.strip_runner_owned(["--ctx-size", "4096", "--port"])[0], ["--ctx-size", "4096"])    # flag with no value at the end
        self.assertEqual(pp.strip_runner_owned(["--no-webui", "--port", "--jinja"])[0], ["--no-webui", "--jinja"])  # next token is a flag, not a value

    def test_propose_end_to_end_and_provenance_lists_names_not_values(self):
        with tempfile.TemporaryDirectory() as td:
            finding = {"id": "f", "title": "t", "experiment": {"kind": "server_overhead_flags", "variable": "v", "metric": "m",
                       "arms": [{"name": "default"}, {"name": "cache-ram-0", "args_override": {"--cache-ram": 0}}]}}
            d = os.path.join(td, "diag.json")
            with open(d, "w", encoding="utf-8") as f:
                json.dump({"findings": [finding], "recommended_next_experiment": {"finding": "f"}}, f)
            spec, _ = pp.propose(d, None, {"server_args": ["--ctx-size", "8192", "--port", "8080", "--log-file", "C:/private/prod.log"]}, "w.json",
                                 os.path.join(td, "s.json"))
            self.assertEqual(spec["common_args"], ["--ctx-size", "8192"])
            self.assertEqual(spec["provenance"]["removed_server_args"], ["--log-file", "--port"])
            self.assertNotIn("private", json.dumps(spec))
            self.assertNotIn("8080", json.dumps(spec))

    def test_mcp_refuses_these_flags_including_equals_form(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            base = {"name": "s", "binary": "${runtime_binary}", "model": "${model_path}", "port": 8099, "arms": [{"name": "a"}, {"name": "b"}], "workload": "w.json"}
            for common in (["--log-file", "x.log"], ["--host=0.0.0.0"], ["--port", "8080"], ["--slot-save-path", "D:/x"], ["--lora", "a.gguf"]):
                with open("s.json", "w", encoding="utf-8") as f:
                    json.dump(dict(base, common_args=common), f)
                with self.assertRaises(ValueError, msg=str(common)):
                    mcp.check_spec_for_agent("s.json")
            with open("s.json", "w", encoding="utf-8") as f:
                json.dump(dict(base, common_args=["--ctx-size", "4096", "--jinja"], arms=[{"name": "a"}, {"name": "b", "args_override": {"--log-file=x": 1}}]), f)
            with self.assertRaises(ValueError):
                mcp.check_spec_for_agent("s.json")
            with open("s.json", "w", encoding="utf-8") as f:
                json.dump(dict(base, common_args=["--ctx-size", "4096", "--jinja", "--cache-ram", "0"]), f)
            mcp.check_spec_for_agent("s.json")                                  # ordinary tuning flags pass


class McpSpecGateTests(unittest.TestCase):
    def _spec(self, td, **over):
        spec = {"name": "s", "binary": "${runtime_binary}", "model": "${model_path}", "path_prepend": "${path_prepend}", "port": 8099,
                "common_args": ["--ctx-size", "4096"], "env": {}, "arms": [{"name": "a"}, {"name": "b", "args_override": {"--ubatch-size": 128}}], "workload": "w.json"}
        spec.update(over)
        p = os.path.join(td, "spec.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        return p

    def test_proposed_style_spec_passes_and_agent_authored_programs_do_not(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            mcp.check_spec_for_agent(self._spec(td))
            for over, needle in (({"binary": "C:/Windows/System32/calc.exe"}, "binary"),
                                 ({"model": "/etc/passwd"}, "model"),
                                 ({"path_prepend": ["C:/evil"]}, "path_prepend"),
                                 ({"env": {"PATH": "/evil", "LD_PRELOAD": "/x.so"}}, "PATH"),
                                 ({"common_args": ["--host", "0.0.0.0"]}, "--host"),
                                 ({"common_args": ["--hf-repo", "someone/model"]}, "--hf-repo"),
                                 ({"arms": [{"name": "a", "binary": "/tmp/x"}, {"name": "b"}]}, "binary"),
                                 ({"arms": [{"name": "a", "env_override": {"PYTHONPATH": "/x"}}, {"name": "b"}]}, "PYTHONPATH"),
                                 ({"arms": [{"name": "a"}, {"name": "b", "args_override": {"--model": "x.gguf"}}]}, "--model")):
                with self.assertRaises(ValueError, msg=str(over)) as cm:
                    mcp.check_spec_for_agent(self._spec(td, **over))
                self.assertIn(needle, str(cm.exception))
            mcp.check_spec_for_agent(self._spec(td, env={"GGML_CUDA_DISABLE_GRAPHS": "1"}))
            with self.assertRaises(ValueError):                       # workload outside the server's root
                mcp.check_spec_for_agent(self._spec(td, workload=os.path.join(os.path.abspath(os.sep), "Windows", "win.ini")))

    def test_active_gate_and_spec_gate_are_independent(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            bad = self._spec(td, binary="/tmp/evil")
            os.environ.pop("HOTPATH_MCP_ALLOW_ACTIVE", None)
            text, is_err = mcp.call_tool("hotpath_experiment_run", {"spec": bad, "confirm": True})
            self.assertTrue(is_err)
            self.assertIn("active tools are disabled", text)           # gate 1 fires before the spec is even read
            os.environ["HOTPATH_MCP_ALLOW_ACTIVE"] = "1"
            try:
                with self.assertRaises(ValueError):
                    mcp.call_tool("hotpath_experiment_run", {"spec": bad, "confirm": True})   # gate 2 (spec contents) with the operator flag on
            finally:
                os.environ.pop("HOTPATH_MCP_ALLOW_ACTIVE", None)

    def test_read_only_tools_cannot_reach_remote_hosts_or_local_config(self):
        with self.assertRaises(ValueError):
            mcp.call_tool("hotpath_observe", {"server": "http://169.254.169.254:80"})
        with self.assertRaises(ValueError):
            mcp.call_tool("hotpath_observe", {"server": "http://example.com:8080"})
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            with self.assertRaises(ValueError):
                mcp.call_tool("hotpath_propose", {"diagnosis": "d.json", "out": "hotpath.local.json"})
            with self.assertRaises(ValueError):
                mcp.call_tool("hotpath_workload", {"out_file": "x.local.json"})

    def test_no_read_only_tool_is_active_and_every_active_tool_is_gated(self):
        for name, (_d, _s, active, _f) in mcp.TOOLS.items():
            if name in ("hotpath_experiment_run", "hotpath_measure", "hotpath_bench", "hotpath_kernels", "hotpath_profile", "hotpath_improve"):
                self.assertTrue(active, name)
                self.assertIn("confirm", _s["required"], name)
            else:
                self.assertFalse(active, name)


if __name__ == "__main__":
    unittest.main()
