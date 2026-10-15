"""A candidate that changes a prompt-cache setting together with something else (a build, a microbatch) was inconclusive
whenever its reuse differed from the baseline's, even when the same experiment had an arm changing only that cache setting
with the same reuse; an arm that swapped the binary or model counted as cache-setting-only; and the production gate said a
warm run was measured "with the prompt cache off"."""
import unittest

from hotpath import diagnose as dg

from test_release_review import WARM, _exp, _req


def _three_arms(sib_reuse_on_p3, cand_reuse_on_p3):
    """Baseline a reuses 900 tokens everywhere; on p3 the cache-setting arm loses reuse; the combined candidate is faster."""
    base, sib, cand = [], [], []
    for i in range(15):
        pid = "p%d" % (i % 5)
        base.append(_req(pid, 900, wall=2000, pms=300))
        sib.append(_req(pid, sib_reuse_on_p3 if pid == "p3" else 900, wall=1950, pms=300))
        cand.append(_req(pid, cand_reuse_on_p3 if pid == "p3" else 900, wall=1600, pms=250))
    rec = _exp(base, cand, WARM, WARM)
    rec["arms"] = {"a": rec["arms"]["a"], "s": dict(rec["arms"]["b"], requests=sib), "b": rec["arms"]["b"]}
    rec["cache_mode"] = "warm"
    rec["spec_redacted"]["arms"] = [{"name": "a"}, {"name": "s", "args_override": {"--cache-ram": 0}},
                                    {"name": "b", "binary": "llama-server.exe", "args_override": {"--ubatch-size": 512, "--cache-ram": 0}}]
    return rec


class CacheSiblingAttributionTests(unittest.TestCase):
    def test_reuse_matched_by_a_cache_only_sibling_is_comparable(self):
        row = next(r for r in dg.experiment_verdict(_three_arms(300, 300))["arms"] if r["arm"] == "b")
        comp = row["cache"]["comparability"]
        self.assertTrue(comp["comparable"], comp)
        self.assertEqual(comp.get("reuse_matched_by"), "s")
        self.assertEqual(row["metrics"]["wall_ms"]["call"].split(" ")[0], "improvement")

    def test_reuse_the_sibling_does_not_explain_stays_not_comparable(self):
        row = next(r for r in dg.experiment_verdict(_three_arms(300, 0))["arms"] if r["arm"] == "b")
        self.assertFalse(row["cache"]["comparability"]["comparable"])
        self.assertTrue(row["verdict"].startswith("inconclusive"), row["verdict"])

    def test_an_arm_that_also_swaps_the_binary_is_not_cache_only(self):
        rec = _three_arms(300, 300)
        self.assertTrue(dg._cache_setting_arm(rec, "s"))
        self.assertFalse(dg._cache_setting_arm(rec, "b"))
        rec["spec_redacted"]["arms"][2] = {"name": "b", "binary": "llama-server.exe", "args_override": {"--cache-ram": 0}}
        self.assertFalse(dg._cache_setting_arm(rec, "b"))                # same flag as the sibling, different build

    def test_without_a_sibling_a_combined_change_stays_not_comparable(self):
        rec = _three_arms(300, 300)
        rec["spec_redacted"]["arms"][1] = {"name": "s", "args_override": {"--ubatch-size": 512}}
        row = next(r for r in dg.experiment_verdict(rec)["arms"] if r["arm"] == "b")
        self.assertFalse(row["cache"]["comparability"]["comparable"])


class CacheSiblingEvidenceTests(unittest.TestCase):
    """Missing or non-explaining sibling evidence must never turn a reuse difference into a comparable result."""

    def _row(self, rec):
        return next(r for r in dg.experiment_verdict(rec)["arms"] if r["arm"] == "b")

    def test_a_sibling_with_no_usable_requests_explains_nothing(self):
        rec = _three_arms(300, 0)
        for r in rec["arms"]["s"]["requests"]:
            r["error"] = "rejected"
        row = self._row(rec)
        self.assertFalse(row["cache"]["comparability"]["comparable"])
        self.assertTrue(row["verdict"].startswith("inconclusive"), row["verdict"])

    def test_a_sibling_that_never_ran_the_differing_prompt_explains_nothing(self):
        rec = _three_arms(300, 0)
        rec["arms"]["s"]["requests"] = [r for r in rec["arms"]["s"]["requests"] if r["prompt_id"] != "p3"]
        row = self._row(rec)
        self.assertFalse(row["cache"]["comparability"]["comparable"])
        self.assertTrue(row["verdict"].startswith("inconclusive"), row["verdict"])

    def test_a_sibling_that_reuses_like_the_baseline_does_not_explain_the_candidates_reuse(self):
        rec = _three_arms(900, 300)      # the cache-only arm keeps all its reuse; only the combined candidate loses it
        row = self._row(rec)
        self.assertFalse(row["cache"]["comparability"]["comparable"])
        self.assertIsNone(row["cache"]["comparability"].get("reuse_matched_by"))

    def test_a_record_without_a_recorded_spec_has_no_sibling(self):
        rec = _three_arms(300, 300)
        del rec["spec_redacted"]
        self.assertIsNone(dg._cache_sibling(rec, "b", ["a", "s", "b"]))
        self.assertFalse(self._row(rec)["cache"]["comparability"]["comparable"])


class ProductionGateTextTests(unittest.TestCase):
    def test_a_warm_run_is_not_described_as_cache_off(self):
        rec = _three_arms(300, 300)
        out = dg.diagnose([rec])
        gates = [v.get("production_gate") for v in out.get("experiment_verdicts") or [] if v.get("production_gate")]
        self.assertTrue(gates)
        self.assertNotIn("prompt cache off", gates[0])
        self.assertIn("warm", gates[0])


if __name__ == "__main__":
    unittest.main()
