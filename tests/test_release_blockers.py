"""Release blockers found by the final independent reviews of the whole branch stack, one test per reproduced scenario:
  T2 passive discovery ran a llama-server.exe/.bat (and any command started by bare name) planted in the working directory;
  T3 flags read from an on-disk observation (--rpc, --mcp-servers-config) reached the operator's CLI launches; a
     future-dated observation won as "the last observation";
  T4 an agent able to write hotpath.local.json changed what MCP-started processes launch;
  B1 the recommended launch was built from the observed line while the run measured another one (--args, or an earlier
     experiment's winner whose outputs differed);
  B2 warm mode excluded exactly the requests a cache setting made cold, hiding its cost;
  B3 a baseline whose own outputs differ still allowed `keep`;
  B4 decision fields (and the output policies) compared only the first repetition;
  S1 outcomes misfiled class regressions and preliminary validations and left measured arms out;
  S2 a faster-but-different option skipped the regression and spill checks;
  S3 rejected requests were compared by count, not by request;
  S4 output hashes of application requests were unkeyed;
  A  programs ran in the working directory, where Windows looks for the libraries they load (a planted cudart DLL);
  B  the config's digest was checked on one read and the config parsed from another;
  C  a configured relative program path with a directory part could not be run;
  D  arms of a finding read from a diagnosis file reached a launch with any argument;
  E  outputs of two runs compared as different because their hashes are keyed per run;
  F  the loop, the validation and the recommended line each accumulated the adopted changes their own way."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from hotpath import config
from hotpath import diagnose as dg
from hotpath import improve as im
from hotpath import measure as ms
from hotpath import propose as prop
from hotpath import safepath

import test_improve as ti
import test_improve_trust as tt
import test_release_review as rr
from test_agent_boundary import _InDir


class PlantedProgramTests(unittest.TestCase):
    """T2"""

    def test_programs_in_the_working_directory_are_never_found_or_run(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            marker = os.path.join(td, "RAN")
            for name in ("llama-server.bat", "llama-server.cmd", "nvidia-smi.bat", "netstat.cmd"):
                with open(name, "w") as f:
                    f.write("@echo off\necho x > \"%s\"\n" % marker)
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = "." + os.pathsep + td + os.pathsep + old       # even when PATH names the working directory
            try:
                for name in ("llama-server", "nvidia-smi", "netstat"):
                    found = safepath.which(name)
                    self.assertTrue(found is None or os.path.dirname(os.path.abspath(found)) != os.path.abspath(td), found)
                with self.assertRaises(FileNotFoundError):
                    safepath.resolve(["llama-server", "--version"])
                from hotpath import gpu
                gpu.gpu_static()                                                 # runs nvidia-smi by bare name internally
            finally:
                os.environ["PATH"] = old
            self.assertFalse(os.path.exists(marker), "a program planted in the working directory ran")


class RecordDerivedArgsTests(unittest.TestCase):
    """T3"""

    def test_flags_read_from_an_observation_never_reach_a_launch(self):
        d = {"observed_launch": {"args_redacted": ["-m", "x.gguf", "--port", "8080", "-ngl", "40", "--rpc", "10.1.2.3:50052",
                                                    "--mcp-servers-config", "mcp.json", "--log-prompts-dir", "p", "--jinja"], "source": "a record"}}
        args, source = prop.baseline_args(d, {})
        for bad in ("--rpc", "--mcp-servers-config", "--log-prompts-dir", "--port", "-m"):
            self.assertNotIn(bad, args)
        self.assertIn("-ngl", args)
        self.assertIn("--jinja", args)
        self.assertIn("--rpc", source)

    def test_a_future_dated_observation_is_not_the_last_observation(self):
        from hotpath import cli
        with tempfile.TemporaryDirectory() as td:
            obs = {"kind": "observation", "created": "2099-12-31T23:59:59+00:00", "server": {"base_url": "http://127.0.0.1:8080"},
                   "server_process": {"launch": {"available": True}}}
            with open(os.path.join(td, "observation-20991231-235959.json"), "w", encoding="utf-8") as f:
                json.dump(obs, f)
            self.assertIsNone(cli._last_observation(td, "http://127.0.0.1:8080"))


class ConfigSnapshotTests(unittest.TestCase):
    """T4"""

    def test_a_config_changed_after_the_mcp_server_started_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "hotpath.local.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"runtime_binary": "llama-server.exe"}, f)
            os.environ[config.SNAPSHOT_ENV] = config.config_digest(p)
            try:
                self.assertEqual(config.load_local(p)["runtime_binary"], "llama-server.exe")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"runtime_binary": "C:/evil/llama-server.exe"}, f)
                with self.assertRaises(SystemExit):
                    config.load_local(p)
            finally:
                os.environ.pop(config.SNAPSHOT_ENV, None)


class RecommendedIsMeasuredTests(unittest.TestCase):
    """B1"""

    def setUp(self):
        self.t = tt.TrustTests("test_resume_reruns_nothing_completed")
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_the_apply_line_is_the_validated_configuration(self):
        # the last observation said -ngl 40 -c 4096; the user started the run from other arguments
        rep = im.improve(self.t.cfg, [ti._obs()], self.t.workload, self.t.dir, tt.GuardedRunner(), log=lambda *a: None, max_experiments=1,
                         max_minutes=600, initial_args=["-ngl", "40", "-c", "16384", "--flash-attn", "on"])
        self.assertTrue(rep["result"]["validated"], rep["result"]["verdict"])
        line = rep["recommended"]["args"]
        self.assertIn("16384", line)
        self.assertIn("--flash-attn", line)
        self.assertNotIn("4096", line)
        self.assertIn("-m", line, "the user's own plumbing (model, port, log file) is kept")
        self.assertTrue(rep["starting_point"].get("differs_from_observed"))

    def test_improve_starts_from_the_users_launch_not_an_earlier_winner(self):
        # an earlier experiment whose faster arm changed outputs, among the records the run starts from
        earlier = ti.FakeRunner(diverge={"ngl65"})({"name": "earlier", "common_args": ["-ngl", "40", "-c", "4096"], "env": {}, "rounds": 3,
                                                    "repetitions": 2, "warmup": 1, "port": 8099, "workload": self.t.workload, "provenance": {},
                                                    "arms": [{"name": "ngl40"}, {"name": "ngl65", "args_override": {"-ngl": 65}}]}, "x.json")
        rep = im.improve(self.t.cfg, [ti._obs(), earlier], self.t.workload, self.t.dir, tt.GuardedRunner(), log=lambda *a: None, max_experiments=1,
                         max_minutes=600)
        from hotpath import launch
        self.assertEqual(str(launch.get(rep["starting_point"]["args"], "n_gpu_layers")), "40", rep["starting_point"])
        self.assertEqual(str(launch.get(rep["recommended"]["args"], "n_gpu_layers")), "40", "nothing unvalidated is recommended")


class WarmCacheSettingTests(unittest.TestCase):
    """B2"""

    def test_requests_a_cache_setting_made_cold_count_against_it(self):
        base = [rr._req("p%d" % (i % 5), 900, wall=1500, pms=200) for i in range(15)]
        cand = []
        for i in range(15):
            pid = "p%d" % (i % 5)
            if pid in ("p3", "p4"):
                cand.append(rr._req(pid, 0, wall=4500, pms=3200))           # lost all reuse: three times slower
            else:
                cand.append(rr._req(pid, 900, wall=1400, pms=180))
        rec = rr._exp(base, cand, rr.WARM, rr.WARM)
        rec["spec_redacted"]["arms"] = [{"name": "a"}, {"name": "b", "args_override": {"--cache-ram": 0}}]
        row = dg.experiment_verdict(rec)["arms"][0]
        self.assertFalse(row["verdict"].startswith("keep"), row["verdict"])
        ok, _why, _ = im.adoptable(row, {"arms": [row]}, {"name": "b", "args_override": {"--cache-ram": 0}}, rec["arms"], "a", {})
        self.assertFalse(ok)


class ReproducibilityTests(unittest.TestCase):
    """B3"""

    def test_an_unreproducible_baseline_allows_no_keep(self):
        rec = ti.FakeRunner()({"name": "e", "common_args": ["-ngl", "40"], "env": {}, "rounds": 3, "repetitions": 2, "warmup": 1, "port": 8099,
                               "workload": "w.json", "provenance": {}, "arms": [{"name": "ngl40"}, {"name": "ngl65", "args_override": {"-ngl": 65}}]}, "x.json")
        for i, r in enumerate(rec["arms"]["ngl40"]["requests"]):
            if r["prompt_id"] == "p00" and i % 2:
                r["output_sha256"], r["tokens"] = "baseline-differs-%d" % i, None
        row = dg.experiment_verdict(rec)["arms"][0]
        self.assertFalse(row["verdict"].startswith("keep"), row["verdict"])
        self.assertFalse(im.adoptable(row, {"arms": [row]}, {}, rec["arms"], "ngl40", {})[0])


class DecisionRepetitionTests(unittest.TestCase):
    """B4"""

    def test_every_repetition_must_decide_alike(self):
        base, cand = [], []
        for rep in range(4):
            for pid in ("p0", "p1"):
                base.append({"prompt_id": pid, "fields": {"decision": "A"}, "field_kinds": {"decision": "categorical"}})
                cand.append({"prompt_id": pid, "fields": {"decision": "A" if rep == 0 else "B"}, "field_kinds": {"decision": "categorical"}})
        fa = dg.field_agreement(base, cand, ["decision"])
        self.assertEqual(fa["decisions_equal"], 0, fa)
        row = {"validity": {"candidate": {"judged": 8, "schema_valid": 8}, "decided_on": "schema_valid", "dropped": False}, "fields": fa}
        self.assertFalse(im._policy_check(row, {"output_policy": "decisions_equal", "decision_fields": ["decision"]})[1]["passed"])


class OutcomeTests(unittest.TestCase):
    """S1, S2"""

    def setUp(self):
        self.t = tt.TrustTests("test_resume_reruns_nothing_completed")
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_a_preliminary_validation_is_inconclusive_not_needs_review(self):
        rep = self.t.run_loop(tt.GuardedRunner(), max_experiments=1, rounds=2)
        self.assertEqual(rep["outcomes"]["needs_review"], [], rep["outcomes"])
        self.assertTrue(rep["outcomes"]["inconclusive"])

    def test_a_faster_but_different_option_that_also_spills_is_rejected(self):
        rep = self.t.run_loop(tt.GuardedRunner(diverge={"ngl65"}, spill={"ngl65"}), max_experiments=1)
        self.assertFalse(any("ngl65" in x["what"] for x in rep["outcomes"]["needs_review"]), rep["outcomes"])
        self.assertTrue(any("ngl65" in x["what"] for x in rep["outcomes"]["rejected"]), rep["outcomes"])

    def test_every_measured_arm_is_in_exactly_one_outcome(self):
        # a microbatch sweep after the offload step: three arms, one taken, one measured and not taken
        obs = ti._obs()
        obs["server_process"]["launch"]["argv_redacted"] += ["-ub", "128"]
        obs["server_process"]["launch"]["settings"]["ubatch_size"] = "128"
        rep = self.t.run_loop(tt.GuardedRunner(), obs=obs, max_experiments=3)
        self.assertTrue(any(len(st.get("other_arms") or []) for st in rep["steps"]), "the scenario must have an arm that was not taken")
        listed = [x["what"] for k in im.OUTCOME_ORDER for x in rep["outcomes"][k]]
        self.assertEqual(len(listed), len(set(listed)), listed)
        for st in rep["steps"]:
            for o in st.get("other_arms") or []:
                if not o.get("control"):
                    self.assertTrue(any(o["arm"] in w for w in listed), (o["arm"], listed))

    def test_a_class_regression_is_rejected(self):
        self.assertEqual(im.outcome_bucket("keep on ...", "not adopted: rejected because the request class draft got slower", False), "rejected")


class RejectedRequestTests(unittest.TestCase):
    """S3"""

    def test_rejections_are_compared_request_by_request(self):
        base = [rr._req("p%d" % (i % 3), 0, wall=1500) for i in range(9)]
        cand = [rr._req("p%d" % (i % 3), 0, wall=1000) for i in range(9)]
        for r in base:
            if r["prompt_id"] == "p0":
                r["error"] = {"http": 400, "type": "exceed_context_size_error"}
        for r in cand:
            if r["prompt_id"] == "p1":
                r["error"] = {"http": 400, "type": "exceed_context_size_error"}
        row = dg.experiment_verdict(rr._exp(base, cand))["arms"][0]
        self.assertTrue(row["verdict"].startswith("reject"), row["verdict"])


class OutputHashTests(unittest.TestCase):
    """S4"""

    def test_output_hashes_are_keyed_per_run(self):
        import hashlib
        w1, w2 = {}, {}
        self.assertEqual(ms.output_digest('{"ok": true}', w1), ms.output_digest('{"ok": true}', w1))
        self.assertNotEqual(ms.output_digest('{"ok": true}', w1), ms.output_digest('{"ok": true}', w2))
        self.assertNotEqual(ms.output_digest('{"ok": true}', w1), hashlib.sha256(b'{"ok": true}').hexdigest()[:16])



class ProgramDirectoryTests(unittest.TestCase):
    """A"""

    def test_a_program_runs_in_its_own_directory(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            p = safepath.run([sys.executable, "-I", "-c", "import os; print(os.getcwd())"], capture_output=True, text=True)
            self.assertEqual(os.path.normcase(p.stdout.strip()), os.path.normcase(os.path.dirname(os.path.abspath(sys.executable))))

    def test_the_server_runs_in_its_own_directory_with_absolute_paths(self):
        from hotpath import experiment as ex
        seen = {}

        class P:
            pid = 1

        saved = (ex.subprocess.Popen, ex.locks.track)
        ex.subprocess.Popen = lambda cmd, **kw: seen.update(cmd=cmd, cwd=kw.get("cwd")) or P()
        ex.locks.track = lambda proc: None
        try:
            with tempfile.TemporaryDirectory() as td, _InDir(td):
                _proc, log = ex._launch(["bin/llama-server", "--model", "models/m.gguf", "--slot-save-path=cache", "--ctx-size", "4096"], {}, "s.log")
                log.close()
                here = os.getcwd()
        finally:
            ex.subprocess.Popen, ex.locks.track = saved
        self.assertEqual(seen["cwd"], os.path.join(here, "bin"))
        self.assertEqual(seen["cmd"], [os.path.join(here, "bin", "llama-server"), "--model", os.path.join(here, "models", "m.gguf"),
                                       "--slot-save-path=" + os.path.join(here, "cache"), "--ctx-size", "4096"])

    def test_every_program_launch_names_its_directory(self):
        # the package's process launches: through safepath.run, the server launch with its own cwd, or the MCP server's own
        # isolated Python children (python -I does not import from the working directory)
        import re
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hotpath")
        for name in os.listdir(root):
            if not name.endswith(".py") or name == "safepath.py":
                continue
            with open(os.path.join(root, name), encoding="utf-8") as f:
                src = f.read()
            for m in re.finditer(r"subprocess\.(run|Popen|call|check_output|check_call)\(", src):
                line = src[src.rfind("\n", 0, m.start()) + 1:src.find("\n", m.end())]
                ok = (name == "experiment.py" and "cwd=safepath.program_dir" in line) or (name == "mcp_server.py" and "_child_argv(" in line)
                self.assertTrue(ok, "%s launches a program outside safepath: %s" % (name, line.strip()))


class ConfigReadOnceTests(unittest.TestCase):
    """B"""

    def test_the_checked_bytes_are_the_parsed_bytes(self):
        import builtins
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "hotpath.local.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"runtime_binary": "llama-server.exe"}, f)
            os.environ[config.SNAPSHOT_ENV] = config.config_digest(p)
            real_open = builtins.open

            def no_second_read(*a, **k):
                raise AssertionError("the config was read a second time")
            builtins.open = no_second_read
            try:
                self.assertEqual(config.load_local(p)["runtime_binary"], "llama-server.exe")
            finally:
                builtins.open = real_open
                os.environ.pop(config.SNAPSHOT_ENV, None)


class RelativeProgramPathTests(unittest.TestCase):
    """C"""

    def test_a_configured_relative_path_is_made_absolute(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            os.mkdir("bin")
            exe = os.path.join("bin", "tool.exe")
            open(exe, "w").close()
            self.assertEqual(safepath.which(exe), os.path.join(os.getcwd(), exe))
            self.assertEqual(safepath.resolve([exe, "-x"]), [os.path.join(os.getcwd(), exe), "-x"])
            self.assertIsNone(safepath.which(os.path.join("bin", "missing.exe")))


class DiagnosisArmTests(unittest.TestCase):
    """D"""

    def _finding(self, arm):
        return {"id": "f", "experiment": {"kind": "x", "variable": "v", "arms": [{"name": "a"}, dict(arm, name="b")]}}

    def test_a_forged_finding_cannot_add_a_launch_argument(self):
        for arm in ({"args_override": {"--rpc": "10.0.0.1:50052"}}, {"args_override": {"--mcp-servers-config": "x.json"}},
                    {"env_override": {"GGML_BACKEND_PATH": "x.dll"}}):
            with self.assertRaises(ValueError):
                prop.build_spec(self._finding(arm), {}, ["--ctx-size", "4096"], "w.json")

    def test_tuning_arms_are_proposed(self):
        spec = prop.build_spec(self._finding({"args_override": {"--ubatch-size": 256}, "env_override": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}), {},
                               ["--ctx-size", "4096"], "w.json")
        self.assertEqual(spec["arms"][1]["args_override"], {"--ubatch-size": 256})


class CrossRunOutputTests(unittest.TestCase):
    """E"""

    def test_keyed_outputs_of_two_runs_are_not_compared(self):
        from hotpath import compare as cmp_mod

        def rec(name, h1, h2):
            req = lambda h: {"prompt_id": "r1", "output_sha256": h, "output_hash": "keyed per run"}
            return {"kind": "experiment", "name": name, "arms": {"a": {"requests": [req(h1)]}, "b": {"requests": [req(h2)]}}}
        oe = cmp_mod.output_equivalence(cmp_mod._arm_sets([rec("one", "k1", "k1"), rec("two", "k2", "k2")])[0])["r1"]
        self.assertTrue(oe["one/b"]["identical"])
        self.assertIsNone(oe["two/a"]["identical"])
        self.assertFalse(oe["two/a"]["comparable"])


class OneAccumulationTests(unittest.TestCase):
    """F"""

    def test_later_steps_replace_earlier_ones_under_any_spelling(self):
        steps = [{"change": {"args_remove": ["--cache-ram"], "args_override": {"-ngl": 40}}},
                 {"change": {"args_override": {"--cache-ram": 0, "--gpu-layers": 65}, "env_remove": ["GGML_CUDA_DISABLE_GRAPHS"]}},
                 {"change": {"env_override": {"GGML_CUDA_DISABLE_GRAPHS": 1}}}]
        ch = im._cum_change(steps)
        self.assertEqual(ch["args_override"], {"--cache-ram": 0, "-ngl": 65})
        self.assertEqual(ch["args_remove"], [])
        self.assertEqual((ch["env_override"], ch["env_remove"]), ({"GGML_CUDA_DISABLE_GRAPHS": "1"}, []))

    def test_the_apply_line_is_what_the_validated_arm_launched(self):
        original = {"args": ["--ctx-size", "8192", "--cache-ram", "4096", "-ngl", "30"], "env": {}}
        steps = [{"change": {"args_remove": ["--cache-ram"]}}, {"change": {"args_override": {"--cache-ram": 0, "--gpu-layers": 65}}}]
        ch = im._cum_change(steps)
        arm = {"name": "improved", "args_override": ch["args_override"], "args_remove": ch["args_remove"]}
        measured = im.materialize(original["args"], original["env"], arm)[0]
        self.assertEqual(im.recommended_launch(None, original, im._as_cum(ch))["args"], measured)
        self.assertIn("--cache-ram", measured)

    def test_the_no_server_baseline_runs_with_the_declared_environment(self):
        spec = im.baseline_spec({}, ["--ctx-size", "4096"], "w.json", 8099, 2, 1, None, "b", env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
        self.assertEqual(spec["env"], {"GGML_CUDA_DISABLE_GRAPHS": "1"})


if __name__ == "__main__":
    unittest.main()
