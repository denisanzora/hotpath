"""Integrity of the improve loop beyond step verdicts. Written before the implementation, from the ways the loop could
still mislead or harm:
  1. a gain on short prompts hiding a regression on the longest prompt (adopted on the median);
  2. adopting a configuration that spills into system memory, or leaves no VRAM headroom, as if it were a clean speedup;
  3. recommending a change that only a three-prompt step measured: every adopted change must survive a final validation
     against the original launch on a broader workload (more prompts, the largest observed size, the longest generation),
     and when it does not, the original launch stays the recommendation;
  4. running arms that repeat the configuration already measured;
  5. carrying evidence from one model or runtime build into experiments on another;
  6. losing completed experiments when a session is interrupted (resume re-runs nothing it already has);
  7. a context cap that ignores parallel slots, growing conversations or an operator's binding limit, and a context arm whose
     speedup cannot be told apart from the extra GPU layers it enables."""
import json
import os
import tempfile
import unittest

from hotpath import diagnose as dg
from hotpath import improve as im
from hotpath import launch

import test_improve as ti


class GuardedRunner(ti.FakeRunner):
    """FakeRunner plus: arms in `slow_long` are 40% slower on the longest prompt; arms in `spill` report shared
    (system-memory) use; arms in `full` leave almost no free VRAM."""

    def __init__(self, slow_long=(), spill=(), full=(), confirm_diverges=False, **kw):
        super().__init__(**kw)
        self.slow_long, self.spill, self.full, self.confirm_diverges = set(slow_long), set(spill), set(full), confirm_diverges

    def __call__(self, spec, spec_path):
        rec = super().__call__(spec, spec_path)
        longest = max({r["prompt_id"]: r["prompt_n"] for a in rec["arms"].values() for r in a["requests"]}.items(), key=lambda kv: kv[1])[0] \
            if any(a["requests"] for a in rec["arms"].values()) else None
        first = next(iter(rec["arms"]))
        ref = {(r["prompt_id"], r["round"], i % 2): r["wall_ms"] for i, r in enumerate(rec["arms"][first]["requests"])}
        for name, arm in rec["arms"].items():
            for i, r in enumerate(arm["requests"]):
                if name in self.slow_long and r["prompt_id"] == longest:
                    r["wall_ms"] = 1.4 * ref[(r["prompt_id"], r["round"], i % 2)]      # slower than the baseline on the longest prompt
                if self.confirm_diverges and spec["provenance"].get("experiment_kind") == "confirmation" and name != "original":
                    r["output_sha256"] += "z"
            arm["launch_facts"] = [{"round": 0, "load_s": 5.0, "process_counters_after": {"dedicated_mib": 6000 if name != "ngl40" else 5000,
                                                                                         "shared_mib": 900 if name in self.spill else 40},
                                    "gpu_after": {"mem_total_mib": 8192, "mem_used_mib": 8100 if name in self.full else 6800}}]
        return rec


class TrustTests(unittest.TestCase):
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

    def run_loop(self, runner, obs=None, **kw):
        kw.setdefault("max_experiments", 4)
        kw.setdefault("max_minutes", 600)
        return im.improve(self.cfg, [obs or ti._obs()], self.workload, self.dir, runner, log=lambda *a: None, **kw)

    def test_regression_on_the_longest_prompt_blocks_adoption(self):
        # failure 1
        rep = self.run_loop(GuardedRunner(slow_long={"ngl65"}))
        s = rep["steps"][0]
        self.assertFalse(s["adopted"], s["why"])
        self.assertIn("regress", s["why"])

    def test_spill_into_system_memory_blocks_adoption(self):
        # failure 2
        rep = self.run_loop(GuardedRunner(spill={"ngl65"}))
        self.assertFalse(rep["steps"][0]["adopted"])
        self.assertIn("system memory", rep["steps"][0]["why"])

    def test_no_headroom_needs_a_decision(self):
        rep = self.run_loop(GuardedRunner(full={"ngl65"}))
        self.assertFalse(rep["steps"][0]["adopted"])
        self.assertTrue(any("headroom" in d["needs"] for d in rep["decisions"]))

    def test_single_adopted_change_is_validated_on_a_broader_workload(self):
        # failure 3
        runner = GuardedRunner()
        rep = self.run_loop(runner, max_experiments=1)
        conf = [s for s in runner.specs if s["provenance"].get("experiment_kind") == "confirmation"]
        self.assertEqual(len(conf), 1)
        with open(conf[0]["workload"], encoding="utf-8") as f:
            vw = json.load(f)
        self.assertGreater(len(vw["prompts"]), 3)
        self.assertNotEqual(conf[0]["workload"], self.workload)
        self.assertEqual(rep["result"]["measured_by"], conf[0]["name"])

    def test_failed_validation_keeps_the_original_launch(self):
        runner = GuardedRunner(confirm_diverges=True)
        rep = self.run_loop(runner, max_experiments=1)
        self.assertFalse(rep["result"]["validated"])
        self.assertEqual(rep["recommended"]["changes"], [])
        self.assertIn("original", rep["recommended"]["note"])

    def test_redundant_arms_are_not_run(self):
        # failure 4: the current configuration appears once
        runner = GuardedRunner()
        self.run_loop(runner, max_experiments=1)
        for s in runner.specs:
            configs = [json.dumps(launch.parse(im.materialize(s["common_args"], s.get("env"), a)[0])["settings"], sort_keys=True) for a in s["arms"]]
            self.assertEqual(len(configs), len(set(configs)), [a["name"] for a in s["arms"]])

    def test_different_model_identity_refuses_before_launching(self):
        # failure 5
        obs = ti._obs()
        obs["model"]["identity"] = {"basename": "other.gguf", "size_bytes": 123, "sha256_head64m": "aa"}
        runner = GuardedRunner()
        cfg_id = {"basename": "m.gguf", "size_bytes": 999, "sha256_head64m": "bb"}
        rep = im.improve(self.cfg, [obs], self.workload, self.dir, runner, log=lambda *a: None, identity=lambda path: cfg_id)
        self.assertEqual(runner.specs, [])
        self.assertIn("different model", rep["stopped_because"])

    def test_resume_reruns_nothing_completed(self):
        # failure 6
        runner = GuardedRunner()
        first = self.run_loop(runner, max_experiments=2, session="s-test")
        n = len(runner.specs)
        runner2 = GuardedRunner()
        second = self.run_loop(runner2, max_experiments=2, session=first["session"], resume=True)
        self.assertEqual(runner2.specs, [], "completed experiments must be reused, not re-run")
        self.assertEqual([s["adopted"] for s in first["steps"]], [s["adopted"] for s in second["steps"]])
        self.assertGreater(n, 0)

    def test_dry_run_resume_never_launches(self):
        # found in a real OpenCode session: the passive plan tool with `resume` ran a pending step actively
        runner = GuardedRunner()
        first = self.run_loop(runner, max_experiments=1)
        runner2 = GuardedRunner()
        plan = self.run_loop(runner2, max_experiments=3, session=first["session"], resume=True, dry_run=True)
        self.assertEqual(runner2.specs, [], "a dry run must never launch, also when resuming")
        self.assertTrue(plan["stopped_because"].startswith(("dry run", "no further executable finding")), plan["stopped_because"])

    def test_cli_dry_run_resume_never_launches(self):
        # the MCP plan tool calls `improve --dry-run --resume NAME`; the CLI must not run anything for it
        from hotpath import cli
        from hotpath import experiment as ex
        # an interrupted session: step 1 recorded, the validation run never happened
        first = self.run_loop(GuardedRunner(raise_on=2), max_experiments=1)
        cfg_path = os.path.join(self.dir, "hotpath.local.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
        saved, saved_runner = ex.run, cli._experiment_runner
        launched = []
        ex.run = lambda *a, **k: launched.append("experiment.run")
        cli._experiment_runner = lambda cfg, wait_s: (lambda spec, path: launched.append(spec["name"]) or (_ for _ in ()).throw(SystemExit("blocked")))
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)
            cli.main(["--local", cfg_path, "--out", self.dir, "improve", "--resume", first["session"], "--dry-run", "--max-experiments", "3"])
        finally:
            ex.run, cli._experiment_runner = saved, saved_runner
            os.chdir(cwd)
        self.assertEqual(launched, [], "a dry run must never reach the experiment runner")

    def test_a_running_session_cannot_be_resumed_by_a_second_process(self):
        import os as _os
        runner = GuardedRunner()
        first = self.run_loop(runner, max_experiments=1)
        lock = _os.path.join(self.dir, first["session"], "running.lock")
        with open(lock, "w", encoding="utf-8") as f:
            f.write(str(_os.getppid()))                    # another live process holds the session
        with self.assertRaises(SystemExit) as cm:
            self.run_loop(GuardedRunner(), max_experiments=1, session=first["session"], resume=True)
        self.assertIn("running", str(cm.exception))


def _ctx_ev(n_ctx=49152, parallel=1, reqs=None, flags=None):
    model = {"n_layer": 64, "total_bytes": 5935 << 20, "weight_bytes_streamed_per_decode_token": 5935 << 20, "kv_bytes_per_token_f16": 65536}
    reqs = reqs if reqs is not None else [{"prompt_n": 300 + 10 * i, "cache_n": 54, "predicted_n": 48} for i in range(30)]
    arm = {"evidence_class": "passive", "requests": reqs, "settings": {"n_ctx": n_ctx, "n_parallel": parallel, "flags": flags or []}}
    est = {"weights_mib": 5935, "kv_mib": 3072.0 * n_ctx / 49152, "compute_mib": 512, "free_mib": 7592}
    return {"model": model, "server_props": {}, "log": {"rows": reqs}}, arm, est


class ContextTests(unittest.TestCase):
    def test_parallel_slots_scale_the_cap(self):
        ev, arm, est = _ctx_ev(parallel=4)
        a = dg.smaller_context_arm(ev, arm, est, arm["settings"], 40, 65)
        self.assertIsNotNone(a)
        self.assertEqual(a["args_override"]["--ctx-size"] % 4, 0)
        self.assertGreaterEqual(a["args_override"]["--ctx-size"] // 4, 2 * 384)

    def test_growing_conversations_get_no_cap_without_a_policy(self):
        # requests that reuse most of their prompt are later turns of growing conversations
        reqs = [{"prompt_n": 40, "cache_n": 300 + 120 * i, "predicted_n": 48} for i in range(30)]
        ev, arm, est = _ctx_ev(reqs=reqs)
        self.assertIsNone(dg.smaller_context_arm(ev, arm, est, arm["settings"], 40, 65))
        a = dg.smaller_context_arm(ev, arm, est, arm["settings"], 40, 65, policy_max_tokens=8000)
        self.assertIsNotNone(a)
        self.assertNotIn("capacity_change", a)              # the operator's binding limit makes it adoptable
        self.assertEqual(a["context_policy"], 8000)

    def test_context_arm_comes_with_a_context_only_control(self):
        # failure 7: the speedup of ctx+ngl must be separable from the context change alone
        ev, arm, est = _ctx_ev()
        arms = dg.context_arms(ev, arm, est, arm["settings"], 40, 65)
        names = [a["name"] for a in arms]
        self.assertEqual(len(arms), 2, names)
        ctrl = [a for a in arms if a["args_override"].get("--gpu-layers") == 40]
        self.assertEqual(len(ctrl), 1)
        self.assertIn("control", ctrl[0])

    def test_consequence_of_exceeding_the_cap_is_stated(self):
        ev, arm, est = _ctx_ev(flags=["--context-shift"])
        a = dg.smaller_context_arm(ev, arm, est, arm["settings"], 40, 65)
        self.assertIn("context shift", a["capacity_change"])


if __name__ == "__main__":
    unittest.main()
