"""Passive (server-log) diagnosis on synthetic observation records. Numbers are made up; this checks shapes,
thresholds and that nothing is manufactured where the log cannot support it."""
import unittest

from hotpath import diagnose as dg


def _row(session, prompt_n, prompt_ms, gen_n, gen_ms, reused=None, by="LRU", f_sim=None):
    r = {"session": session, "prompt_n": prompt_n, "prompt_ms": prompt_ms, "prompt_tps": 1000.0 * prompt_n / prompt_ms,
         "predicted_n": gen_n, "predicted_ms": gen_ms, "predicted_tps": 1000.0 * gen_n / gen_ms, "total_ms": prompt_ms + gen_ms, "slot_by": by}
    if f_sim is not None:
        r["f_sim_best"] = f_sim
    if reused is not None:
        r["reused_tokens"] = reused
    return r


def _obs(rows, sessions, cache=None, coverage=None, model=True):
    from hotpath import logparse
    return {"kind": "observation", "evidence_class": "passive", "created": "2026-01-01T00:00:00",
            "gpu": {"name": "Test GPU", "memory.total": "8192", "bandwidth_gbs_known": 500.0}, "gpu_now": {"mem_used_mib": 6000.0},
            "launcher_settings": {"available": False}, "launcher_env_documented": {},
            "model": {"summary": {"weight_bytes_streamed_per_decode_token": 5.0e9, "kv_bytes_per_token_f16": 1000}} if model else {"available": False},
            "log": {"sessions": len(sessions), "requests": rows, "session_table": sessions, "cache_reuse": cache or logparse.cache_reuse_stats(rows),
                    "parse_coverage": coverage or {"lines": 1000, "timing_lines": len(rows) * 3, "sessions": len(sessions), "requests_parsed": len(rows)},
                    "cpu_mode_prefill_tps_below": 10.0, "gpu_like_sessions_stats": {}}}


def _sess(i, n, pf, dc, ub="256", device="CUDA0"):
    return {"session": i, "requests": n, "prefill_median_tps": pf, "decode_median_tps": dc, "n_ubatch": ub, "n_batch": "512", "device": device, "warnings": 0, "errors": 0}


class PassiveSplitTests(unittest.TestCase):
    def test_log_only_split_uses_server_basis_and_no_residual(self):
        rows = [_row(1, 1500, 3000, 70, 1500, reused=120) for _ in range(30)]
        d = dg.diagnose([_obs(rows, [_sess(1, 30, 500.0, 46.0)])])
        ids = {f["id"]: f for f in d["findings"]}
        self.assertIn("workload_phase_split", ids)
        self.assertEqual(ids["workload_phase_split"]["measurements"]["time_basis"], "server")
        self.assertIsNone(ids["workload_phase_split"]["measurements"]["residual_share_median"])
        self.assertNotIn("per_request_overhead", ids)      # never manufactured from a log
        self.assertEqual(ids["workload_phase_split"]["evidence_class"], "passive")
        self.assertTrue(any("Single best next measurement" in s for s in d["insufficient_evidence"]))
        self.assertIn("log/session-1", d["diagnosed_arm"])

    def test_microbatch_from_session_table(self):
        rows = [_row(2, 1500, 40000, 70, 1500) for _ in range(30)]
        d = dg.diagnose([_obs(rows, [_sess(1, 10, 500.0, 46.0), _sess(2, 30, 37.0, 46.0, ub="1")])])
        f = {x["id"]: x for x in d["findings"]}["microbatch_limits_prefill"]
        self.assertEqual(f["measurements"]["n_ubatch"], 1.0)
        self.assertIn("session 2", f["measurements"]["source"])


class CacheReuseTests(unittest.TestCase):
    def test_low_reuse_is_reported_and_gates_cache_flags(self):
        rows = [_row(1, 1400, 2800, 70, 1500, reused=100, by="LRU") for _ in range(40)]
        d = dg.diagnose([_obs(rows, [_sess(1, 40, 500.0, 46.0)])])
        ids = {f["id"]: f for f in d["findings"]}
        self.assertIn("prefix_reuse_low", ids)
        self.assertAlmostEqual(ids["prefix_reuse_low"]["measurements"]["reused_share_of_prompt_median"], 100 / 1500, places=3)
        cp = dg.cache_pattern_evidence(dg.collect([_obs(rows, [_sess(1, 40, 500.0, 46.0)])]))
        self.assertEqual(cp["level"], "low")

    def test_high_reuse_warns_against_disabling(self):
        rows = [_row(1, 300, 600, 70, 1500, reused=1500, by="LCP similarity", f_sim=0.8) for _ in range(40)]
        d = dg.diagnose([_obs(rows, [_sess(1, 40, 500.0, 46.0)])])
        ids = {f["id"]: f for f in d["findings"]}
        self.assertIn("prefix_reuse_high", ids)
        self.assertTrue(any("Do not disable" in r for r in ids["prefix_reuse_high"]["risks"]))

    def test_no_selection_lines_is_insufficient(self):
        rows = [{"session": 1, "prompt_n": 1500, "prompt_ms": 3000, "prompt_tps": 500.0, "predicted_n": 70, "predicted_ms": 1500, "predicted_tps": 46.7, "total_ms": 4500} for _ in range(30)]
        obs = _obs(rows, [_sess(1, 30, 500.0, 46.0)], cache={"available": False})
        d = dg.diagnose([obs])
        self.assertIn("prefix_reuse_unknown", {f["id"] for f in d["findings"]})


class SessionRuleTests(unittest.TestCase):
    def test_cpu_fallback_latest_session_is_high(self):
        rows = [_row(2, 1500, 300000, 70, 3000) for _ in range(10)] + [_row(1, 1500, 3000, 70, 1500) for _ in range(30)]
        d = dg.diagnose([_obs(rows, [_sess(1, 30, 500.0, 46.0), _sess(2, 10, 5.0, 23.0, device="CPU")])])
        f = {x["id"]: x for x in d["findings"]}["cpu_fallback_sessions"]
        self.assertEqual(f["severity"], "high")
        self.assertEqual(f["measurements"]["sessions"][0]["session"], 2)
        # the GPU session, not the CPU one, is the diagnosed baseline
        self.assertIn("session-1", d["diagnosed_arm"])

    def test_session_drift_with_changed_ubatch_proposes_sweep(self):
        rows = [_row(2, 1500, 15000, 70, 1500) for _ in range(30)]
        d = dg.diagnose([_obs(rows, [_sess(1, 30, 500.0, 46.0, ub="512"), _sess(2, 30, 100.0, 46.0, ub="32")])])
        f = {x["id"]: x for x in d["findings"]}["session_drift"]
        self.assertEqual(f["evidence_class"], "historical")
        self.assertEqual([a["name"] for a in f["experiment"]["arms"]], ["ub32", "ub512"])

    def test_format_problem_is_a_finding(self):
        obs = _obs([], [], cache={"available": False}, coverage={"lines": 500, "timing_lines": 90, "sessions": 1, "requests_parsed": 0, "problem": "timing lines are present but no request could be assembled"})
        obs["log"]["sessions"] = 1
        d = dg.diagnose([obs])
        f = {x["id"]: x for x in d["findings"]}["log_format_problem"]
        self.assertEqual(f["confidence"], "insufficient")


if __name__ == "__main__":
    unittest.main()
