"""Regression tests for an independent review of the discover-autopilot -> improve-loop -> improve-trust stack. Each test is a
scenario the reviewer reproduced (or traced) against the code before the fixes:
  1. args_override values became argv tokens, so a value could carry a flag past the key check;
  2. llama.cpp reads `--log_file` as `--log-file`; the runner-owned strip and the agent check did not;
  3. an agent spec with the user's port and no preconditions would send the workload and slot erases to the user's server;
  4. a spec or arm name with path components wrote server logs and records outside the results directory;
  5. --api-key / --hf-token values and LLAMA_API_KEY reached records, reports and the agent;
  6. the recommended launch re-added flags the experiments never ran with (a draft model, a vision projector);
  7. a candidate whose own repeated outputs differ was judged identical from its first output;
  8. the "different llama-server build" refusal compared the configured binary with itself;
  9. two active MCP calls arriving together both started a job; NaN budgets passed; wrong-typed arguments left the
     client without a reply; network (UNC) paths made Windows contact another host;
 10. a dry run of a running session rewrote spec files the running process reads; a killed write left half a record."""
import json
import os
import tempfile
import threading
import time
import unittest

from hotpath import agentgate as ag
from hotpath import diagnose as dg
from hotpath import improve as im
from hotpath import launch
from hotpath import mcp_server as mcp
from hotpath import propose as prop
from hotpath import records

import test_improve as ti
import test_improve_trust as tt
from test_agent_boundary import PRODUCTION_LAUNCH, _InDir, _spec


class SmuggledArgsTests(unittest.TestCase):
    def test_override_values_cannot_carry_flags(self):
        # 1: {"--no-warmup": "--host", "0.0.0.0": "--no-perf"} materializes to --no-warmup --host 0.0.0.0 --no-perf
        p = ag.spec_problems(_spec(arms=[{"name": "a"}, {"name": "b", "args_override": {"--no-warmup": "--host", "0.0.0.0": "--no-perf"}}]))
        self.assertTrue(any("--host" in x for x in p), p)
        p = ag.spec_problems(_spec(arms=[{"name": "a"}, {"name": "b", "args_override": {"--alias": "x --api-key k"}}]))
        self.assertEqual(p, [], "a value with spaces is one argv token, not a flag")

    def test_underscore_spellings_are_the_same_argument(self):
        # 2
        kept, removed = prop.strip_runner_owned(["--log_file", "prod.log", "--api_key", "k", "--ctx-size", "4096"])
        self.assertEqual(kept, ["--ctx-size", "4096"])
        self.assertEqual(sorted(removed), ["--api-key", "--log-file"])
        for flag in ("--log_file", "--hf_repo", "--slot_save_path", "--mcp_servers_config"):
            self.assertTrue(ag.spec_problems(_spec(common_args=PRODUCTION_LAUNCH + [flag, "x"])), flag)
        self.assertEqual(launch.get(["--gpu_layers", "40"], "n_gpu_layers"), "40")


class PortTests(unittest.TestCase):
    def test_a_port_someone_already_listens_on_is_refused_before_any_launch(self):
        # 3
        from hotpath import experiment as ex
        from hotpath import gpu as gpu_mod
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            with open("w.json", "w", encoding="utf-8") as f:
                json.dump({"name": "w", "n_predict": 4, "prompts": [{"id": "p0", "text": "x"}]}, f)
            os.environ["HOTPATH_GPU_LOCK_DIR"] = td
            saved, launched = (gpu_mod.pid_listening_on, ex._launch), []
            gpu_mod.pid_listening_on = lambda port: 4242
            ex._launch = lambda *a, **k: launched.append(a)
            try:
                with self.assertRaises(SystemExit) as cm:
                    ex.run({"name": "s", "binary": "b", "model": "m", "port": 8080, "arms": [{"name": "a"}], "workload": "w.json", "preconditions": {}}, td)
            finally:
                gpu_mod.pid_listening_on, ex._launch = saved
                os.environ.pop("HOTPATH_GPU_LOCK_DIR", None)
            self.assertIn("4242", str(cm.exception))
            self.assertEqual(launched, [])

    def test_agent_specs_keep_the_idle_gpu_and_port_preconditions(self):
        spec = ag.harden(_spec(port=8123, preconditions={}))
        self.assertIn(8123, spec["preconditions"]["ports_must_be_free"])
        self.assertLessEqual(spec["preconditions"]["max_gpu_util_pct"], 10)
        self.assertGreaterEqual(spec["preconditions"]["sample_seconds"], 5)
        self.assertTrue(ag.spec_problems(_spec(port=80)))


class NameTests(unittest.TestCase):
    def test_names_with_path_components_are_refused(self):
        # 4
        for bad in ("x/../../ESCAPED", "..", "a\\b", "", "../x"):
            with self.assertRaises(ValueError, msg=bad):
                records.check_name(bad)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                records.write_record({"kind": "x"}, td, "experiment-x/../../../ESCAPED")
        self.assertTrue(ag.spec_problems(_spec(name="x/../../ESCAPED")))
        with self.assertRaises(ValueError):
            mcp.call_tool("hotpath_profile", {"label": "../../x", "confirm": True}) if os.environ.get("HOTPATH_MCP_ALLOW_ACTIVE") == "1" else mcp._profile({"label": "../../x"})


class SecretTests(unittest.TestCase):
    def test_secret_values_never_reach_records(self):
        # 5
        red = launch.redact(["--api-key", "sk-SECRET123", "--hf-token", "hf_abc", "--ctx-size", "4096"])
        self.assertNotIn("sk-SECRET123", red)
        self.assertNotIn("hf_abc", red)
        os.environ["LLAMA_API_KEY"] = "sk-SECRET123"
        try:
            self.assertEqual(records.env_snapshot().get("LLAMA_API_KEY"), "<redacted>")
        finally:
            os.environ.pop("LLAMA_API_KEY", None)


class MeasuredLaunchTests(unittest.TestCase):
    def test_flags_the_experiments_could_not_run_block_an_automatic_recommendation(self):
        # 6: the user's server runs with a draft model and a vision projector; experiments run without them
        plumbing, missing = im.left_out_of_measurement(["-m", "model.gguf", "--port", "8080", "-md", "draft.gguf", "--mmproj", "v.gguf", "-ngl", "40"],
                                                       ["-ngl", "40"])
        self.assertEqual(sorted(plumbing), ["--port", "-m"])
        self.assertEqual(sorted(missing), ["--mmproj", "-md"])
        t = tt.TrustTests("test_resume_reruns_nothing_completed")
        t.setUp()
        try:
            obs = ti._obs()
            obs["server_process"]["launch"]["argv_redacted"] += ["-md", "draft.gguf"]
            rep = t.run_loop(tt.GuardedRunner(), obs=obs, max_experiments=1)
        finally:
            t.tearDown()
        self.assertFalse(rep["result"]["validated"], rep["result"]["verdict"])
        self.assertIn("-md", rep["result"]["verdict"])
        self.assertEqual(rep["recommended"]["changes"], [])
        self.assertTrue(any("-md" in d["needs"] for d in rep["decisions"]))


class ReproducibilityTests(unittest.TestCase):
    def test_a_candidate_whose_repeats_differ_is_not_identical(self):
        # 7
        runner = ti.FakeRunner()
        rec = runner({"name": "e", "common_args": ["-ngl", "40"], "env": {}, "rounds": 3, "repetitions": 2, "provenance": {}, "port": 8099,
                      "warmup": 1, "workload": "w.json",
                      "arms": [{"name": "ngl40"}, {"name": "ngl65", "args_override": {"-ngl": 65}}]}, "x.json")
        reqs = rec["arms"]["ngl65"]["requests"]
        for i, r in enumerate(reqs):
            if i % 2 == 1:
                r["output_sha256"] = "different-%d" % i
                r["tokens"] = None
        v = dg.experiment_verdict(rec)
        row = v["arms"][0]
        self.assertFalse(row["verdict"].startswith("keep"), row["verdict"])
        self.assertEqual(row["fidelity"]["class"], "candidate not reproducible")


class BuildIdentityTests(unittest.TestCase):
    def test_another_build_than_the_observed_servers_is_refused(self):
        # 8
        with tempfile.TemporaryDirectory() as td:
            mine, theirs = os.path.join(td, "mine"), os.path.join(td, "theirs")
            for d, dll in ((mine, b"cuda-A"), (theirs, b"cuda-B")):
                os.makedirs(d)
                with open(os.path.join(d, "llama-server.exe"), "wb") as f:
                    f.write(b"same exe")
                with open(os.path.join(d, "ggml-cuda.dll"), "wb") as f:
                    f.write(dll)
            from hotpath.observe import build_identity
            obs = {"kind": "observation", "server_build": build_identity(os.path.join(theirs, "llama-server.exe"))}
            cfg = {"runtime_binary": os.path.join(mine, "llama-server.exe")}
            problem = im.identity_problem([obs], cfg, records.file_identity)
            self.assertIn("ggml-cuda.dll", problem or "")
            obs["server_build"] = build_identity(os.path.join(mine, "llama-server.exe"))
            self.assertIsNone(im.identity_problem([obs], cfg, records.file_identity))
            self.assertIsNone(im.identity_problem([{"kind": "observation"}], cfg, records.file_identity), "no observed build: not judged")


class McpRobustnessTests(unittest.TestCase):
    def test_wrong_types_nan_and_network_paths_get_an_error_reply(self):
        # 9
        text, err = mcp.call_tool("hotpath_job_status", {"job_id": ["x"]})
        self.assertTrue(err)
        self.assertIn("job_id", text)
        text, err = mcp.call_tool("hotpath_improve_plan", {"max_minutes": float("nan")})
        self.assertTrue(err)
        text, err = mcp.call_tool("hotpath_improve_plan", {"max_experiments": [3]})
        self.assertTrue(err)
        for args in ({"log": "\\\\attacker\\share\\x.log"}, {"roots": ["//attacker/share"]}):
            with self.assertRaises(ValueError):
                mcp.call_tool("hotpath_inspect", args)
        with self.assertRaises(ValueError):
            mcp.call_tool("hotpath_gguf_summary", {"path": "\\\\attacker\\share\\m.gguf"})

    def test_an_exception_inside_a_tool_still_gets_a_reply(self):
        import io
        from contextlib import redirect_stdout
        saved = mcp.call_tool
        mcp.call_tool = lambda n, a: (_ for _ in ()).throw(TypeError("boom"))
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mcp._tool_call(7, {"name": "x", "arguments": {}})
        finally:
            mcp.call_tool = saved
        reply = json.loads(buf.getvalue())
        self.assertEqual(reply["id"], 7)
        self.assertTrue(reply["result"]["isError"])

    def test_two_calls_arriving_together_start_one_job(self):
        started = []

        class StubJob:
            def __init__(self, name, argv, timeout_s):
                time.sleep(0.2)                                    # the window in which a second call used to pass the check
                self.id, self.tool, self.done = "job-%d" % len(started), name, threading.Event()
                started.append(self)

            def report(self):
                return "started", False
        saved = mcp.Job
        mcp.Job = StubJob
        results = []
        try:
            ts = [threading.Thread(target=lambda: results.append(mcp.start_job("hotpath_bench", ["bench"], 10, 0))) for _ in range(2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(10)
        finally:
            mcp.Job = saved
            for j in started:
                mcp.JOBS.pop(j.id, None)
        self.assertEqual(len(started), 1, "only one job may start")
        self.assertTrue(any(r[1] and "refused" in r[0] for r in results))


class SessionFileTests(unittest.TestCase):
    def test_a_dry_run_never_rewrites_a_running_sessions_spec_files(self):
        # 10
        t = tt.TrustTests("test_resume_reruns_nothing_completed")
        t.setUp()
        try:
            first = t.run_loop(tt.GuardedRunner(raise_on=2), max_experiments=1)
            sdir = os.path.join(t.dir, first["session"])
            before = {n: os.path.getmtime(os.path.join(sdir, n)) for n in os.listdir(sdir) if n.endswith(".json")}
            time.sleep(0.05)
            t.run_loop(tt.GuardedRunner(), max_experiments=3, session=first["session"], resume=True, dry_run=True)
            after = {n: os.path.getmtime(os.path.join(sdir, n)) for n in os.listdir(sdir) if n.endswith(".json") and n in before}
            self.assertEqual(before, after)
        finally:
            t.tearDown()

    def test_records_are_written_whole(self):
        with tempfile.TemporaryDirectory() as td:
            p = records.write_record({"kind": "x", "big": "y" * 100000}, td, "experiment-x")
            self.assertEqual(json.load(open(p, encoding="utf-8"))["big"], "y" * 100000)
            self.assertEqual([n for n in os.listdir(td) if n.endswith(".tmp")], [])


if __name__ == "__main__":
    unittest.main()
