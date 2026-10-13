"""Outcomes and real-request plans, written from the ways they could mislead:
  1. a change validated on a run of exactly that set must be the only kind of "recommended";
  2. a faster change with differing outputs, or one that limits what the server accepts, must be "needs review", never
     recommended and never silently dropped;
  3. a slower change or one that regresses on a prompt must be "rejected", one within the noise "inconclusive";
  4. a request plan must keep request bodies out of the session directory, cover every request class, include each
     class's largest request in the validation and validate mostly on requests the steps did not see;
  5. a preliminary validation run (too few prompts or rounds) must not produce a recommendation."""
import json
import os
import tempfile
import unittest

from hotpath import improve as im
from hotpath import workload as wl

import test_improve_trust as tt


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.t = tt.TrustTests("test_resume_reruns_nothing_completed")
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_validated_is_recommended_and_nothing_else_is(self):
        rep = self.t.run_loop(tt.GuardedRunner(), max_experiments=1)
        self.assertTrue(rep["result"]["validated"])
        self.assertEqual(len(rep["outcomes"]["recommended"]), 1)
        self.assertIn("validated", rep["outcomes"]["recommended"][0]["why"])

    def test_differing_outputs_need_review(self):
        rep = self.t.run_loop(tt.GuardedRunner(diverge={"ngl65"}), max_experiments=1)
        self.assertEqual(rep["outcomes"]["recommended"], [])
        self.assertTrue(rep["outcomes"]["needs_review"], rep["outcomes"])

    def test_failed_validation_is_not_recommended(self):
        rep = self.t.run_loop(tt.GuardedRunner(confirm_diverges=True), max_experiments=1)
        self.assertEqual(rep["outcomes"]["recommended"], [])
        self.assertTrue(rep["outcomes"]["needs_review"] or rep["outcomes"]["rejected"] or rep["outcomes"]["inconclusive"])

    def test_regression_is_rejected(self):
        rep = self.t.run_loop(tt.GuardedRunner(slow_long={"ngl65"}), max_experiments=1)
        self.assertTrue(any("step 1" in x["what"] for x in rep["outcomes"]["rejected"]), rep["outcomes"])

    def test_markdown_states_the_outcomes(self):
        rep = self.t.run_loop(tt.GuardedRunner(diverge={"ngl65"}), max_experiments=1)
        md = im.improvement_markdown(rep)
        self.assertIn("## Outcome", md)
        self.assertIn("Needs review", md)


class RequestPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        cap = os.path.join(self.dir, "cap.jsonl")
        with open(cap, "w", encoding="utf-8") as f:
            for i in range(30):
                stage = ("score", "draft", "critic")[i % 3]
                body = {"messages": [{"role": "system", "content": "you are"}, {"role": "user", "content": "SECRET-PROMPT " + "x" * (100 + 37 * i)}],
                        "max_tokens": 64, "temperature": 0}
                f.write(json.dumps({"stage": stage, "body": body}) + "\n")
        self.wpath = os.path.join(self.dir, "w.json")
        wl.save_workload(wl.workload_from_requests(cap, name="cap"), self.wpath)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_covers_classes_includes_the_largest_and_holds_out(self):
        sdir = os.path.join(self.dir, "improve-x")
        os.makedirs(sdir)
        rq = im.requests_plan(self.wpath, sdir, {}, None, 2, 4)
        steps, val = wl.load_workload(rq["steps"]), wl.load_workload(rq["validation"])
        full = wl.load_workload(self.wpath)
        for c in ("score", "draft", "critic"):
            mine = sorted((it for it in full["items"] if it["class"] == c), key=wl.request_size)
            self.assertEqual(sum(1 for it in steps["items"] if it["class"] == c), 2)
            self.assertEqual(sum(1 for it in val["items"] if it["class"] == c), 4)
            self.assertIn(mine[-1]["id"], [it["id"] for it in val["items"]], "the largest request of each class is validated")
        self.assertGreaterEqual(rq["summary"]["validation"]["held_out"], 9)
        for p in (rq["steps"], rq["validation"]):
            self.assertNotIn("SECRET-PROMPT", open(p, encoding="utf-8").read(), "a session file must not hold request bodies")
        order = [it["id"] for it in full["items"]]
        self.assertEqual([it["id"] for it in val["items"]], sorted((it["id"] for it in val["items"]), key=order.index), "application order is kept")

    def test_the_plan_is_kept_for_the_session(self):
        sdir = os.path.join(self.dir, "improve-y")
        os.makedirs(sdir)
        a = im.requests_plan(self.wpath, sdir, {}, None, 2, 4)
        b = im.requests_plan(self.wpath, sdir, {}, None, 3, 6)            # a resumed run with other sizes measures the same requests
        self.assertEqual(open(a["validation"]).read(), open(b["validation"]).read())

    def test_sampled_requests_are_detected(self):
        self.assertTrue(wl.is_sampled({"messages": [], "temperature": 0.7}))
        self.assertTrue(wl.is_sampled({"messages": []}))                    # llama-server's default temperature samples
        self.assertFalse(wl.is_sampled({"messages": [], "temperature": 0}))
        self.assertFalse(wl.is_sampled({"messages": [], "temperature": 0.7, "top_k": 1}))


class PreliminaryValidationTests(unittest.TestCase):
    def test_a_preliminary_validation_recommends_nothing(self):
        t = tt.TrustTests("test_resume_reruns_nothing_completed")
        t.setUp()
        try:
            rep = t.run_loop(tt.GuardedRunner(), max_experiments=1, rounds=2)
        finally:
            t.tearDown()
        self.assertFalse(rep["result"]["validated"], rep["result"]["verdict"])
        self.assertIn("insufficient evidence", rep["result"]["verdict"])
        self.assertEqual(rep["outcomes"]["recommended"], [])


if __name__ == "__main__":
    unittest.main()


class CacheSettingWarmTests(unittest.TestCase):
    """In warm mode a different reuse is not comparable, except when the change under test is a prompt-cache setting:
    then the reuse it loses or gains is its effect, and wall time in the application's order is the comparison."""

    def test_reuse_difference_is_the_effect_of_a_cache_setting(self):
        import test_release_review as rr
        from hotpath import diagnose as dg
        base = [rr._req("p%d" % (i % 3), 900, wall=1500, pms=200) for i in range(9)]
        cand = [rr._req("p%d" % (i % 3), 300, wall=1700, pms=400) for i in range(9)]      # loses reuse, gets slower
        rec = rr._exp(base, cand, rr.WARM, rr.WARM)
        rec["spec_redacted"]["arms"] = [{"name": "a"}, {"name": "b", "args_override": {"--cache-ram": 0}}]
        row = dg.experiment_verdict(rec)["arms"][0]
        self.assertTrue(row["cache"]["comparability"].get("reuse_is_the_effect"), row["cache"]["comparability"])
        self.assertTrue(row["verdict"].startswith("reject"), row["verdict"])
        rec["spec_redacted"]["arms"] = [{"name": "a"}, {"name": "b", "args_override": {"--ubatch-size": 512}}]
        row = dg.experiment_verdict(rec)["arms"][0]
        self.assertEqual(row["cache"]["comparability"]["basis"], "warm reuse differs")
