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
  S4 output hashes of application requests were unkeyed."""
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


if __name__ == "__main__":
    unittest.main()
