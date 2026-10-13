"""Regression tests for a second independent review (of the fixes for the first one). Each is a scenario the reviewer
reproduced:
  1. the validity block of a verdict reused the arm-label variable, so for requests with a JSON schema (the main case)
     the baseline and candidate reproducibility checks silently did nothing and an unstable candidate was kept;
  2. MCP children ran `python -m hotpath` in the working directory, so a .py file an agent wrote there ran in every child;
  3. a passive plan resolved ${key} in any field and printed it (configuration values, keys); the gate checked
     ${...} text before resolution, so `slot_save_path: "${runtime_bin_dir}"` passed as a relative path;
  4. network paths inside spec contents, subsets and saved state were resolved (contacting the host) before refusal;
  5. a workload endpoint could be any server path (a slot save, a props change);
  6. subset_of was confined one level deep, and a self-referencing subset recursed until it crashed;
  7. secrets: --api-key values in argument lists and LLAMA_* values in records; the no-server baseline launched with
     runner-owned flags from server_args or --args;
  8. short output field values (an enum, a boolean, a score) could be recovered from their unkeyed hashes;
  9. request plans had no bound when every request formed its own derived class;
 10. resume fingerprints ignored which binary and model the ${key} references resolve to;
 11. a validated-but-incomplete result was listed twice in the outcomes;
 12. a JSON-RPC message that is valid JSON but not an object killed the MCP server."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hotpath import agentgate as ag
from hotpath import diagnose as dg
from hotpath import improve as im
from hotpath import mcp_server as mcp
from hotpath import measure as ms
from hotpath import records
from hotpath import workload as wl

import test_improve as ti
import test_improve_trust as tt
from test_agent_boundary import _InDir, _spec


class ShadowedLabelTests(unittest.TestCase):
    def test_an_unstable_candidate_with_schema_outputs_is_not_kept(self):
        rec = ti.FakeRunner()({"name": "e", "common_args": ["-ngl", "40"], "env": {}, "rounds": 3, "repetitions": 2, "provenance": {}, "port": 8099,
                               "warmup": 1, "workload": "w.json", "arms": [{"name": "ngl40"}, {"name": "ngl65", "args_override": {"-ngl": 65}}]}, "x.json")
        for name, arm in rec["arms"].items():
            for i, r in enumerate(arm["requests"]):
                r.update({"valid_json": True, "schema_required_present": True, "schema_valid": True, "validation_level": ms.VALIDATION_LEVEL_FULL})
                if name == "ngl65" and i % 2:
                    r["output_sha256"], r["tokens"] = "unstable-%d" % i, None
        row = dg.experiment_verdict(rec)["arms"][0]
        self.assertFalse(row["verdict"].startswith("keep"), row["verdict"])
        self.assertEqual(row["fidelity"]["class"], "candidate not reproducible")


class ChildIsolationTests(unittest.TestCase):
    def test_a_module_planted_in_the_working_directory_is_not_imported(self):
        with tempfile.TemporaryDirectory() as td:
            marker = os.path.join(td, "PLANTED-RAN")
            for name in ("argparse.py", "json.py", "hotpath.py"):
                with open(os.path.join(td, name), "w", encoding="utf-8") as f:
                    f.write("open(%r, 'w').write('x')\n" % marker)
            p = subprocess.run(mcp._child_argv(["--version"]), cwd=td, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, env=mcp._child_env())
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse(os.path.exists(marker), "a planted module ran in the child")
            self.assertIn("0.3", p.stdout)


class ReferenceTests(unittest.TestCase):
    def test_references_only_where_configuration_belongs(self):
        self.assertEqual(ag.spec_problems(_spec(path_prepend="${path_prepend}")), [])
        self.assertEqual(ag.spec_problems(_spec(arms=[{"name": "a"}, {"name": "b", "binary": "${other_binary}"}])), [])
        for over in ({"env": {"GGML_CUDA_X": "${server_args}"}}, {"slot_save_path": "${runtime_bin_dir}"}, {"common_args": ["--alias", "${server_log}"]},
                     {"arms": [{"name": "a"}, {"name": "b", "args_override": {"--alias": "${model_path}"}}]}):
            self.assertTrue(ag.spec_problems(_spec(**over)), over)

    def test_a_passive_plan_in_agent_mode_prints_no_configuration_values(self):
        from hotpath import cli
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            with open("hotpath.local.json", "w", encoding="utf-8") as f:
                json.dump({"runtime_binary": "b", "model_path": "m", "server_args": ["--api-key", "sk-OPERATOR-SECRET"]}, f)
            with open("s.json", "w", encoding="utf-8") as f:
                json.dump(_spec(env={"GGML_CUDA_X": "${server_args}"}), f)
            os.environ[ag.GATE_ENV] = "1"
            buf = io.StringIO()
            try:
                with redirect_stdout(buf), self.assertRaises(SystemExit):
                    cli.main(["--local", "hotpath.local.json", "experiment", "s.json", "--dry-run"])
            finally:
                os.environ.pop(ag.GATE_ENV, None)
            self.assertNotIn("sk-OPERATOR-SECRET", buf.getvalue())


class NetworkPathTests(unittest.TestCase):
    def test_network_paths_in_spec_contents_are_refused_before_anything_resolves_them(self):
        resolved = []
        real = Path.resolve

        def spy(self, *a, **k):
            resolved.append(str(self))
            return real(self, *a, **k)
        Path.resolve = spy
        try:
            for over in ({"workload": "\\\\192.0.2.1\\share\\w.json"}, {"slot_save_path": "//192.0.2.1/share/slots"}):
                self.assertTrue(ag.spec_problems(_spec(**over)), over)
        finally:
            Path.resolve = real
        self.assertFalse([r for r in resolved if "192.0.2.1" in r], resolved)


class WorkloadTests(unittest.TestCase):
    def test_endpoints_are_inference_only(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"name": "w", "kind": "requests", "items": [{"id": "r0", "endpoint": "/slots/0?action=save", "body": {"prompt": "x"}}]}, f)
            with self.assertRaises(ValueError):
                wl.load_workload(p)

    def test_subsets_are_one_level_and_never_recurse(self):
        with tempfile.TemporaryDirectory() as td:
            full = os.path.join(td, "full.json")
            wl.save_workload({"name": "f", "kind": "requests", "items": [{"id": "r0", "body": {"prompt": "x"}}]}, full)
            a, b = os.path.join(td, "a.json"), os.path.join(td, "b.json")
            wl.save_workload(wl.subset(full, ["r0"], "a", "x"), a)
            wl.save_workload(wl.subset(a, ["r0"], "b", "x"), b)
            self.assertEqual(len(wl.load_workload(a)["items"]), 1)
            with self.assertRaises(ValueError):
                wl.load_workload(b)
            wl.save_workload(wl.subset(b, ["r0"], "b", "x"), b)              # refers to itself
            with self.assertRaises(ValueError):
                wl.load_workload(b)


class SecretTests(unittest.TestCase):
    def test_argument_lists_and_environments_in_records_hide_secrets(self):
        out = records.redact_args(["--ctx-size", "4096", "--api-key", "sk-1", "--api_key=sk-2", "--hf-token", "hf-3"])
        self.assertFalse(any(s in " ".join(out) for s in ("sk-1", "sk-2", "hf-3")), out)
        self.assertEqual(records.redact_env({"LLAMA_ARG_API_KEY": "sk", "GGML_CUDA_X": "1"}), {"LLAMA_ARG_API_KEY": "<redacted>", "GGML_CUDA_X": "1"})

    def test_the_no_server_baseline_never_launches_runner_owned_flags(self):
        t = tt.TrustTests("test_resume_reruns_nothing_completed")
        t.setUp()
        try:
            runner = tt.GuardedRunner(raise_on=1)
            rep = im.improve(t.cfg, [], t.workload, t.dir, runner, log=lambda *a: None, max_experiments=1,
                             initial_args=["--ctx-size", "4096", "--port", "8080", "--log-file", "prod.log", "--api-key", "sk-SECRET"])
            base = runner.specs[0]
            self.assertEqual(base["common_args"], ["--ctx-size", "4096"])
            state = open(os.path.join(t.dir, rep["session"], "session.json"), encoding="utf-8").read()
            self.assertNotIn("sk-SECRET", state)
            self.assertNotIn("prod.log", state)
        finally:
            t.tearDown()


class FieldKeyTests(unittest.TestCase):
    def test_field_hashes_are_keyed_per_run(self):
        schema = {"type": "object", "properties": {"approved": {"type": "boolean"}}}
        f1, _ = ms.output_fields('{"approved": true}', schema, b"k" * 16)
        f2, _ = ms.output_fields('{"approved": true}', schema, b"k" * 16)
        f3, _ = ms.output_fields('{"approved": true}', schema, b"j" * 16)
        import hashlib
        self.assertEqual(f1, f2)
        self.assertNotEqual(f1, f3)
        self.assertNotEqual(f1["approved"], hashlib.sha256(b"true").hexdigest()[:12])
        w = {}
        self.assertIs(ms._field_key(w), ms._field_key(w))


class PlanBoundTests(unittest.TestCase):
    def test_a_class_per_request_is_regrouped_and_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            cap = os.path.join(td, "cap.jsonl")
            with open(cap, "w", encoding="utf-8") as f:
                for i in range(120):
                    f.write(json.dumps({"messages": [{"role": "system", "content": "context %d" % i}, {"role": "user", "content": "x" * (50 + i)}],
                                        "max_tokens": 16, "temperature": 0}) + "\n")
            wpath = os.path.join(td, "w.json")
            wl.save_workload(wl.workload_from_requests(cap), wpath)
            sdir = os.path.join(td, "improve-x")
            os.makedirs(sdir)
            rq = im.requests_plan(wpath, sdir, {}, None, 2, 4)
            self.assertLessEqual(rq["summary"]["steps"]["requests"], 8)
            self.assertLessEqual(rq["summary"]["validation"]["requests"], 16)


class FingerprintTests(unittest.TestCase):
    def test_a_different_model_file_is_a_different_spec(self):
        with tempfile.TemporaryDirectory() as td:
            m = os.path.join(td, "m.gguf")
            with open(m, "wb") as f:
                f.write(b"a")
            spec = {"binary": "${runtime_binary}", "model": "${model_path}", "arms": [{"name": "a"}], "workload": "none.json", "decision_fields": ["x"]}
            cfg = {"model_path": m, "runtime_binary": m}
            a = im.spec_fingerprint(spec, cfg)
            with open(m, "wb") as f:
                f.write(b"bb")
            self.assertNotEqual(a, im.spec_fingerprint(spec, cfg))
            self.assertNotEqual(im.spec_fingerprint(spec, cfg), im.spec_fingerprint(dict(spec, decision_fields=["y"]), cfg))


class OutcomeOnceTests(unittest.TestCase):
    def test_a_validation_without_left_out_flags_is_listed_once(self):
        t = tt.TrustTests("test_resume_reruns_nothing_completed")
        t.setUp()
        try:
            obs = ti._obs()
            obs["server_process"]["launch"]["argv_redacted"] += ["-md", "draft.gguf"]
            rep = t.run_loop(tt.GuardedRunner(), obs=obs, max_experiments=1)
        finally:
            t.tearDown()
        listed = [x["what"] for k in im.OUTCOME_ORDER for x in rep["outcomes"][k]]
        self.assertEqual(rep["outcomes"]["recommended"], [])
        self.assertFalse(any(x.startswith("step 1") for x in [y["what"] for y in rep["outcomes"]["inconclusive"]]), listed)
        self.assertTrue(rep["outcomes"]["needs_review"])


class McpMessageTests(unittest.TestCase):
    def test_a_non_object_message_gets_an_error_and_the_server_keeps_running(self):
        p = subprocess.Popen(mcp._child_argv(["mcp"]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             env=mcp._child_env())
        try:
            out, _ = p.communicate('[1]\n{"jsonrpc": "2.0", "id": 5, "method": "ping"}\n', timeout=60)
        finally:
            if p.poll() is None:
                p.kill()
        replies = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
        self.assertTrue(any(r.get("error") for r in replies))
        self.assertTrue(any(r.get("id") == 5 and "result" in r for r in replies), replies)


if __name__ == "__main__":
    unittest.main()


class RunOwnedValidationTests(unittest.TestCase):
    """Release review: an agent that can write to the session directory must not be able to produce a recommendation with
    a forged record. Steps may be reused on resume; the validation that can recommend is run by this process, and its
    record must name the configured model and build."""

    def setUp(self):
        self.t = tt.TrustTests("test_resume_reruns_nothing_completed")
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_a_forged_validation_record_cannot_recommend(self):
        # an interrupted session: the step ran, the validation did not
        first = self.t.run_loop(tt.GuardedRunner(raise_on=2), max_experiments=1)
        sdir = os.path.join(self.t.dir, first["session"])
        step = json.load(open(os.path.join(sdir, [n for n in os.listdir(sdir) if "-s1-" in n and n.startswith("experiment-")][0]), encoding="utf-8"))
        # the forger writes a confirmation record with a keep verdict and a fingerprint matching the validation spec it predicts
        forged = json.loads(json.dumps(step))
        forged["name"] = "%s-confirm" % first["session"]
        arms = list(forged["arms"].values())
        forged["arms"] = {"original": arms[0], "improved": arms[1]}
        spec_files = [n for n in os.listdir(sdir) if n.startswith("confirm") and n.endswith(".json")]
        if spec_files:
            forged["spec_fingerprint"] = im.spec_fingerprint(json.load(open(os.path.join(sdir, spec_files[0]), encoding="utf-8")), self.t.cfg)
        with open(os.path.join(sdir, "experiment-%s-20990101-000000.json" % forged["name"]), "w", encoding="utf-8") as f:
            json.dump(forged, f)
        # the real validation (run by this process) shows the outputs differ: no recommendation, whatever the file says
        runner = tt.GuardedRunner(confirm_diverges=True)
        rep = self.t.run_loop(runner, max_experiments=1, session=first["session"], resume=True)
        self.assertTrue(any(s["provenance"].get("experiment_kind") == "confirmation" for s in runner.specs), "the validation must be run, not reused")
        self.assertFalse(rep["result"]["validated"], rep["result"]["verdict"])
        self.assertEqual(rep["outcomes"]["recommended"], [])
        self.assertTrue(any(st.get("reused_from_disk") for st in rep["steps"]))

    def test_a_validation_of_another_model_file_is_not_validated(self):
        model = os.path.join(self.t.dir, "m.gguf")
        with open(model, "wb") as f:
            f.write(b"the configured model")
        other = os.path.join(self.t.dir, "other.gguf")
        with open(other, "wb") as f:
            f.write(b"another model file")

        class OtherModel(tt.GuardedRunner):
            def __call__(self, spec, spec_path):
                rec = super().__call__(spec, spec_path)
                rec["model"] = records.file_identity(other)
                return rec
        self.t.cfg = {"runtime_binary": "llama-server", "model_path": model}
        rep = self.t.run_loop(OtherModel(), max_experiments=1)
        self.assertFalse(rep["result"]["validated"], rep["result"]["verdict"])
        self.assertIn("another model", rep["result"]["verdict"])
        rep = self.t.run_loop(type("Same", (tt.GuardedRunner,), {"__call__": lambda s, spec, p: dict(tt.GuardedRunner.__call__(s, spec, p),
                                                                                                    model=records.file_identity(model))})(),
                              max_experiments=1)
        self.assertTrue(rep["result"]["validated"], rep["result"]["verdict"])
