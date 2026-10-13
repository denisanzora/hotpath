"""Output fidelity between experiment arms, from untimed probe requests that return top-k log-probabilities.

Written before the implementation, from the ways a comparison could mislead:
  1. calling different outputs equivalent because the first divergence is a near-tie (a near-tie is evidence of numerical
     sensitivity; the outputs still differ from that token on);
  2. calling a top-k comparison a full-distribution comparison (the KL here is over the top-k tokens both arms report);
  3. hiding numerical drift behind identical tokens (identical tokens with shifted log-probabilities is a separate class
     from identical log-probabilities);
  4. treating a candidate whose token was outside the baseline's top-k as a near-tie (its margin is unknown: at least the
     gap to the k-th token);
  5. trusting identity when the baseline itself is not reproducible;
  6. letting an arm-level class hide one diverging prompt among identical ones;
  7. treating a validity drop as anything but a regression, whatever the tokens;
  8. accepting a difference as verified without an explicit, operator-configured policy that was actually checked."""
import unittest

from hotpath import fidelity as fd


def _probe(tokens, tops):
    """tops: per position, list of (token_id, logprob), best first."""
    return {"tokens": list(tokens), "top": [[[t, lp] for t, lp in pos] for pos in tops]}


A = [(1, -0.01), (2, -5.0), (3, -6.0)]
B = [(2, -0.02), (1, -4.5), (3, -6.0)]


class PromptComparisonTests(unittest.TestCase):
    def test_identical_logprobs(self):
        p = _probe([1, 2], [A, B])
        r = fd.compare_probes(p, _probe([1, 2], [A, B]))
        self.assertEqual(r["class"], "logprob-identical")
        self.assertEqual(r["max_logprob_shift"], 0.0)

    def test_identical_tokens_with_drift_is_its_own_class(self):
        # failure 3
        c = _probe([1, 2], [[(1, -0.03), (2, -4.9), (3, -6.1)], B])
        r = fd.compare_probes(_probe([1, 2], [A, B]), c)
        self.assertEqual(r["class"], "token-identical")
        self.assertGreater(r["max_logprob_shift"], 0.0)

    def test_near_tie_divergence_is_still_a_difference(self):
        # failure 1: at position 1 the baseline chose 2 over 5 by 0.05 nats; the candidate chose 5
        base = _probe([1, 2, 9], [A, [(2, -0.67), (5, -0.72), (3, -4.0)], A])
        cand = _probe([1, 5, 9], [A, [(5, -0.66), (2, -0.73), (3, -4.0)], A])
        r = fd.compare_probes(base, cand)
        self.assertEqual(r["class"], "diverged-near-tie")
        self.assertEqual(r["first_divergence_index"], 1)
        self.assertLess(r["baseline_margin_nats"], 0.1)
        self.assertFalse(fd.is_equivalent(r))

    def test_clear_divergence(self):
        base = _probe([1, 2], [A, [(2, -0.01), (5, -6.0)]])
        cand = _probe([1, 5], [A, [(5, -0.01), (2, -6.0)]])
        r = fd.compare_probes(base, cand)
        self.assertEqual(r["class"], "diverged")
        self.assertGreater(r["baseline_margin_nats"], 1.0)

    def test_token_outside_top_k_is_not_a_near_tie(self):
        # failure 4: the candidate's token 7 is not in the baseline's top 3 at the divergence; the margin is at least the gap to
        # the 3rd token, so it cannot be called a near-tie
        base = _probe([1, 2], [A, [(2, -0.30), (5, -0.35), (3, -0.40)]])
        cand = _probe([1, 7], [A, [(7, -0.2), (2, -0.5), (5, -0.6)]])
        r = fd.compare_probes(base, cand)
        self.assertEqual(r["class"], "diverged")
        self.assertIsNone(r["baseline_margin_nats"])
        self.assertTrue(r["baseline_margin_at_least_nats"] >= 0.1)

    def test_kl_is_labelled_partial(self):
        # failure 2
        c = _probe([1, 2], [[(1, -0.05), (2, -3.5), (3, -6.0)], B])
        r = fd.compare_probes(_probe([1, 2], [A, B]), c)
        self.assertIn("top", r["kl_basis"])
        self.assertIn("not the full distribution", r["kl_basis"])
        self.assertGreater(r["max_partial_kl"], 0.0)


class ArmClassificationTests(unittest.TestCase):
    def _prompts(self, classes):
        return {"p%d" % i: {"class": c, "first_divergence_index": 3 if c.startswith("diverged") else None} for i, c in enumerate(classes)}

    def test_one_divergent_prompt_decides_the_arm(self):
        # failure 6
        a = fd.classify_arm(self._prompts(["logprob-identical", "token-identical", "diverged-near-tie"]))
        self.assertEqual(a["class"], "numerically close, behavior differs")
        self.assertFalse(a["equivalent"])

    def test_identical_arm(self):
        a = fd.classify_arm(self._prompts(["logprob-identical", "token-identical"]))
        self.assertEqual(a["class"], "token-identical")
        self.assertTrue(a["equivalent"])

    def test_clear_divergence_is_unverified(self):
        a = fd.classify_arm(self._prompts(["token-identical", "diverged"]))
        self.assertEqual(a["class"], "unverified differences")

    def test_validity_drop_is_a_regression(self):
        # failure 7
        a = fd.classify_arm(self._prompts(["token-identical"]), validity_dropped=True)
        self.assertEqual(a["class"], "invalid or regressed")
        self.assertFalse(a["equivalent"])

    def test_unstable_baseline_voids_identity(self):
        # failure 5
        a = fd.classify_arm(self._prompts(["token-identical", "token-identical"]), baseline_reproducible=False)
        self.assertEqual(a["class"], "baseline not reproducible")
        self.assertFalse(a["equivalent"])

    def test_policy_verified_difference_needs_a_configured_and_checked_policy(self):
        # failure 8
        prompts = self._prompts(["token-identical", "diverged"])
        self.assertEqual(fd.classify_arm(prompts, policy=None, policy_check=None)["class"], "unverified differences")
        self.assertEqual(fd.classify_arm(prompts, policy="schema_valid", policy_check=None)["class"], "unverified differences")
        a = fd.classify_arm(prompts, policy="schema_valid", policy_check={"passed": True, "basis": "full JSON Schema validation, 18/18"})
        self.assertEqual(a["class"], "different, verified by policy")
        self.assertTrue(a["equivalent"])
        self.assertFalse(fd.classify_arm(prompts, policy="schema_valid", policy_check={"passed": False, "basis": "x"})["equivalent"])


if __name__ == "__main__":
    unittest.main()
