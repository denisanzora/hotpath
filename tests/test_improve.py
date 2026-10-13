"""The improve loop (hotpath improve): diagnose -> propose -> experiment -> verdict -> build on the winner -> confirm.

Written before the implementation, from the ways the loop could mislead or harm a user:
  1. adopting a candidate whose verdict is not `keep` (faster but outputs differ, inconclusive, reject, launch failures);
  2. adopting a cache/overhead flag whose production gate says real traffic would lose prefix reuse, or cannot tell;
  3. testing the same finding twice, or a finding with no executable arms (profiling procedures, request shape without
     captured requests);
  4. building the next experiment on the replaced configuration instead of the adopted one (arguments AND environment);
  5. running past the experiment or time budget;
  6. claiming a cumulative improvement that no single paired run measured (two adopted changes need a final
     original-vs-improved confirmation; one adopted change is already that comparison);
  7. continuing to launch after a precondition failure (the user's own server holds the GPU);
  8. changing the workload between steps, which makes steps incomparable;
  9. recommending a launch line with runner-owned flags (port, host, model, log) or without the adopted changes.
The experiment runner is replaced by a deterministic synthetic world; diagnosis and verdicts are the real code.
GPU behavior is validated on real runs (see docs)."""
import json
import os
import tempfile
import unittest

from hotpath import improve as im
from hotpath import launch

COLD = {"mode": "cold", "isolation": "cache_prompt=false on every request", "slot_erase_supported": False}


def _obs(ngl="40"):
    """A user's running server, observed passively: -ngl 40 of a 64-block model, /metrics with real traffic."""
    m = {"available": True, "since_server_start": {"prompt_seconds_total": 10.0, "tokens_predicted_seconds_total": 30.0, "prompt_tokens_total": 5000,
                                                   "tokens_predicted_total": 600, "prompt_tokens_cached_total": 0}}
    return {"kind": "observation", "evidence_class": "passive", "created": "t", "gpu": {"memory.total": "8192", "name": "X", "bandwidth_gbs_known": 500.0},
            "model": {"summary": {"n_layer": 64, "total_bytes": 4 << 30, "weight_bytes_streamed_per_decode_token": 4 << 30, "kv_bytes_per_token_f16": 16384}},
            "server_process": {"pid_found": True, "launch": {"available": True, "settings": {"n_gpu_layers": ngl, "ctx_size": "4096"},
                                                             "argv_redacted": ["-m", "model.gguf", "--port", "8080", "-ngl", ngl, "-c", "4096", "--log-file", "s.log"],
                                                             "source": "command line of pid 1 (observed)"}},
            "log": {"sessions": 0, "session_table": [], "requests": []}, "metrics": m}


def world(args, env, diverge=False):
    """Synthetic speeds: full offload is much faster; -ub 512 helps prefill; --cache-ram 0 trims the residual.
    Returns per-request (prompt_ms, predicted_ms, residual_ms)."""
    ngl = launch.int_or_none(launch.get(args, "n_gpu_layers")) or 0
    ub = launch.int_or_none(launch.get(args, "ubatch_size")) or 512
    pf = 250.0 if ngl < 65 else (500.0 if ub >= 512 else 350.0)
    dc = 8.0 if ngl < 65 else 48.0
    resid = 250.0 if str(launch.get(args, "cache_ram")) == "0" else 450.0
    return pf, dc, resid


class FakeRunner:
    """Stands in for the experiment runner: one record per spec, timings from `world`. Arms named in `diverge` produce
    different outputs; arms named in `fail` fail every launch; `raise_on` makes a call raise like a failed precondition."""

    def __init__(self, diverge=(), fail=(), raise_on=None):
        self.specs, self.diverge, self.fail, self.raise_on = [], set(diverge), set(fail), raise_on

    def __call__(self, spec, spec_path):
        self.specs.append(json.loads(json.dumps(spec)))
        if self.raise_on is not None and len(self.specs) == self.raise_on:
            raise SystemExit("preconditions not met: free VRAM 900 MiB < 4505. Running now: llama-server pid 7 (port 8080). "
                             "hotpath does not stop, restart or reconfigure servers it did not start")
        rec = {"kind": "experiment", "evidence_class": "controlled", "created": "x", "name": spec["name"],
               "spec_redacted": {"rounds": spec["rounds"], "arms": spec["arms"], "provenance": spec.get("provenance") or {}},
               "gpu": {"memory.total": "8192", "bandwidth_gbs_known": 500.0}, "workload": {"set_hash": "s", "n_predict": 48},
               "model": {"summary": _obs()["model"]["summary"]}, "arms": {}}
        for a in spec["arms"]:
            args = list(spec["common_args"])
            for k, v in (a.get("args_override") or {}).items():
                args = launch.set_arg(args, k, v)
            for k in a.get("args_remove") or []:
                args = launch.remove_arg(args, k)
            env = dict(spec.get("env") or {})
            env.update(a.get("env_override") or {})
            for k in a.get("env_remove") or []:
                env.pop(k, None)
            pf, dc, resid = world(args, env)
            reqs = []
            for rnd in range(spec["rounds"]):
                for rep in range(spec["repetitions"]):
                    for i, pn in enumerate((240, 320, 420)):
                        j = 1 + 0.01 * ((rnd + rep + i) % 3 - 1)
                        pms, gms = 1000.0 * pn / pf * j, 1000.0 * 48 / dc * j
                        reqs.append({"prompt_id": "p%02d" % i, "round": rnd, "prompt_n": pn, "prompt_ms": pms, "prompt_tps": 1000.0 * pn / pms,
                                     "predicted_n": 48, "predicted_ms": gms, "predicted_tps": 48000.0 / gms, "wall_ms": pms + gms + resid * j,
                                     "cache_n": 0, "tokens": None, "output_sha256": "h%d%s" % (i, "x" if a["name"] in self.diverge else ""), "stop_type": "length"})
            failed = a["name"] in self.fail
            rec["arms"][a["name"]] = {"requests": [] if failed else reqs, "warmup_requests": [], "launch_facts": [],
                                      "failures": [{"round": 0, "error": "exit 1"}] if failed else [], "cache": COLD, "env_ggml": env,
                                      "args": ["llama-server", "--model", "m.gguf", "--host", "127.0.0.1", "--port", str(spec["port"])] + args}
        return rec


class ImproveLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.workload = os.path.join(self.dir, "w.json")
        with open(self.workload, "w", encoding="utf-8") as f:
            json.dump({"kind": "prompts", "name": "w", "n_predict": 48, "prompts": [{"id": "p%02d" % i, "text": "x " * (200 + 100 * i), "words": 200 + 100 * i}
                                                                                   for i in range(3)]}, f)
        self.cfg = {"runtime_binary": "llama-server", "model_path": "m.gguf"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_loop(self, runner, **kw):
        kw.setdefault("max_experiments", 4)
        kw.setdefault("max_minutes", 600)
        return im.improve(self.cfg, [_obs()], self.workload, self.dir, runner, log=lambda *a: None, **kw)

    def test_adopts_a_keep_and_builds_the_next_step_on_it(self):
        runner = FakeRunner()
        rep = self.run_loop(runner)
        first = runner.specs[0]
        self.assertEqual(first["provenance"]["finding"], "gpu_offload_partial")
        self.assertEqual(rep["steps"][0]["adopted"], True)
        # failure 4: the second step starts from -ngl 65, not from the replaced -ngl 40 (a second finding: a small microbatch)
        obs = _obs()
        obs["server_process"]["launch"]["argv_redacted"] += ["-ub", "128"]
        obs["server_process"]["launch"]["settings"]["ubatch_size"] = "128"
        runner2 = FakeRunner()
        im.improve(self.cfg, [obs], self.workload, self.dir, runner2, log=lambda *a: None, max_experiments=4, max_minutes=600)
        second = runner2.specs[1]
        self.assertEqual(launch.get(second["common_args"], "n_gpu_layers"), "65")
        # failure 9: no runner-owned flag is copied into a spec
        for s in runner.specs:
            for flag in ("--port", "-m", "--log-file", "--host"):
                self.assertNotIn(flag, s["common_args"])

    def test_never_tests_a_finding_twice_and_skips_procedures(self):
        runner = FakeRunner()
        rep = self.run_loop(runner)
        tested = [s["provenance"]["finding"] for s in runner.specs if s["provenance"].get("finding")]
        self.assertEqual(len(tested), len(set(tested)))
        for s in runner.specs:
            self.assertIn(s["provenance"].get("experiment_kind"), im.EXECUTABLE_KINDS | {"confirmation"})
        self.assertIn(rep["stopped_because"], ("no further executable finding", "experiment budget reached"))

    def test_gated_cache_flag_is_reported_not_adopted(self):
        # failure 2: --cache-ram 0 would be faster on isolated requests, but nothing shows real traffic's reuse on this record
        # set, so a keep could not be adopted: the loop does not spend GPU time on it and says why
        runner = FakeRunner()
        rep = self.run_loop(runner)
        self.assertFalse([s for s in rep["steps"] if s.get("experiment_kind") == "server_overhead_flags"])
        skipped = {s["finding"]: s["why_not_run"] for s in rep["skipped_findings"]}
        self.assertIn("per_request_overhead", skipped)
        self.assertIn("real-traffic evidence", skipped["per_request_overhead"])
        self.assertNotIn("--cache-ram", " ".join(str(x) for x in rep["recommended"]["args"]))
        # and when a verdict does carry a gate that does not support applying it, the adoption check refuses it
        ok, why, _ = im.adoptable({"verdict": "keep on this machine: 9% wall improvement"}, {"production_gate": "production applicability unknown: no evidence"})
        self.assertFalse(ok)
        self.assertIn("production", why)

    def test_faster_but_different_outputs_is_not_adopted(self):
        # failure 1: the full-offload arm diverges; the user must judge it, the loop must not
        rep = self.run_loop(FakeRunner(diverge={"ngl65"}))
        self.assertFalse(rep["steps"][0]["adopted"])
        self.assertIn("outputs differ", rep["steps"][0]["verdict"])
        self.assertIsNone(rep.get("confirmation"))
        self.assertEqual(rep["recommended"]["changes"], [])

    def test_launch_failures_are_rejected(self):
        rep = self.run_loop(FakeRunner(fail={"ngl65"}))
        self.assertFalse(rep["steps"][0]["adopted"])
        self.assertTrue(rep["steps"][0]["verdict"].startswith("reject"))

    def test_experiment_budget(self):
        # failure 5
        runner = FakeRunner()
        rep = self.run_loop(runner, max_experiments=1)
        self.assertEqual(len([s for s in runner.specs if s["provenance"].get("experiment_kind") != "confirmation"]), 1)
        self.assertEqual(rep["stopped_because"], "experiment budget reached")

    def test_time_budget_stops_before_launching(self):
        runner = FakeRunner()
        rep = self.run_loop(runner, max_minutes=0.01)
        self.assertEqual(runner.specs, [])
        self.assertTrue(rep["stopped_because"].startswith("time budget"))

    def test_precondition_failure_stops_the_loop_and_says_why(self):
        # failure 7: nothing more is launched after the user's server is found holding the GPU
        runner = FakeRunner(raise_on=1)
        rep = self.run_loop(runner)
        self.assertEqual(len(runner.specs), 1)
        self.assertIn("preconditions not met", rep["stopped_because"])
        self.assertIn("does not stop", rep["stopped_because"])

    def test_same_workload_every_step(self):
        # failure 8
        runner = FakeRunner()
        self.run_loop(runner)
        self.assertEqual({s["workload"] for s in runner.specs if s["provenance"].get("experiment_kind") != "confirmation"}, {self.workload})

    def test_one_adopted_change_is_still_validated(self):
        # failure 6 (revised in 0.3: a three-prompt step is not enough to recommend a change; see test_improve_trust)
        runner = FakeRunner()
        rep = self.run_loop(runner)
        conf = [s for s in runner.specs if s["provenance"].get("experiment_kind") == "confirmation"]
        self.assertEqual(len(conf), 1)
        self.assertEqual(rep["result"]["measured_by"], conf[0]["name"])
        self.assertTrue(rep["result"]["verdict"].startswith("keep"))

    def test_two_adopted_changes_are_confirmed_jointly(self):
        # an observation at -ngl 40 with a small microbatch: offload first, then the microbatch; both adopted
        obs = _obs()
        obs["server_process"]["launch"]["argv_redacted"] += ["-ub", "128"]
        obs["server_process"]["launch"]["settings"]["ubatch_size"] = "128"
        runner = FakeRunner()
        rep = im.improve(self.cfg, [obs], self.workload, self.dir, runner, log=lambda *a: None, max_experiments=4, max_minutes=600)
        adopted = [s for s in rep["steps"] if s["adopted"]]
        self.assertGreaterEqual(len(adopted), 2, [(s["finding"], s["verdict"]) for s in rep["steps"]])
        conf = runner.specs[-1]
        self.assertEqual(conf["provenance"]["experiment_kind"], "confirmation")
        self.assertEqual([a["name"] for a in conf["arms"]], ["original", "improved"])
        self.assertEqual(launch.get(conf["common_args"], "n_gpu_layers"), "40")          # the original launch
        improved = launch.set_arg(launch.set_arg(list(conf["common_args"]), "--gpu-layers", conf["arms"][1]["args_override"]["--gpu-layers"]),
                                  "--ubatch-size", conf["arms"][1]["args_override"]["--ubatch-size"])
        self.assertEqual(launch.get(improved, "n_gpu_layers"), "65")
        self.assertEqual(rep["result"]["measured_by"], conf["name"])

    def test_recommended_launch_is_the_users_line_with_the_changes(self):
        rep = self.run_loop(FakeRunner())
        rec = rep["recommended"]
        self.assertEqual(launch.get(rec["args"], "n_gpu_layers"), "65")
        self.assertEqual(launch.get(rec["args"], "port"), "8080")           # the user's own port and log stay as they launched them
        self.assertEqual(rec["changes"][0]["flag"], "--gpu-layers")
        self.assertIn("hotpath did not change", rec["note"])

    def test_dry_run_launches_nothing_and_names_the_first_step(self):
        runner = FakeRunner()
        rep = self.run_loop(runner, dry_run=True)
        self.assertEqual(runner.specs, [])
        self.assertEqual(rep["next_step"]["finding"], "gpu_offload_partial")
        self.assertTrue(os.path.isfile(rep["next_step"]["spec"]))


if __name__ == "__main__":
    unittest.main()
