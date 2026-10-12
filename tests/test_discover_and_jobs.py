"""0.3 additions: command-line parsing, alias-safe argument overrides, offload evidence and impact ranking, discovery bounds,
and the MCP job lifecycle. Shapes and arithmetic on synthetic inputs; GPU behavior is validated on real runs (see docs)."""
import os
import struct
import sys
import tempfile
import time
import unittest

from hotpath import diagnose as dg
from hotpath import discover as disc
from hotpath import experiment as ex
from hotpath import launch
from hotpath import mcp_server as mcp
from hotpath import propose as pp


class LaunchParsingTests(unittest.TestCase):
    def test_aliases_values_and_booleans(self):
        s = launch.parse(["-m", "C:/m/x.gguf", "-ngl", "40", "--ctx-size=8192", "-ub", "256", "-fa", "-c", "4096", "--metrics", "--seed", "-1",
                          "--log-file", "/tmp/l.log"])["settings"]
        self.assertEqual((s["model"], s["n_gpu_layers"], s["ubatch_size"], s["flash_attn"], s["metrics"]), ("C:/m/x.gguf", "40", "256", "on", True))
        self.assertEqual(s["ctx_size"], "4096")                       # the later spelling wins, as in llama.cpp
        self.assertEqual(s["log_file"], "/tmp/l.log")
        self.assertEqual(launch.parse(["--flash-attn", "off"])["settings"]["flash_attn"], "off")

    def test_set_arg_replaces_every_spelling(self):
        # a user's -ngl 50 overridden by an arm's --gpu-layers 65 must not reach the server twice (diagnosis read the first one)
        args = launch.set_arg(["-ngl", "50", "-c", "4096", "--n-gpu-layers=30"], "--gpu-layers", 65)
        self.assertEqual(args, ["-c", "4096", "--gpu-layers", "65"])
        self.assertEqual(launch.remove_arg(["-ub", "64", "--slots"], "--ubatch-size"), ["--slots"])

    def test_arm_launch_is_alias_safe(self):
        spec = {"binary": "b", "model": "m", "common_args": ["-ngl", "40", "--slots"], "port": 8099}
        cmd, _env, _port = ex.arm_launch(spec, {"name": "a", "args_override": {"--gpu-layers": 65}})
        self.assertEqual(cmd.count("-ngl") + cmd.count("--gpu-layers"), 1)
        self.assertEqual(cmd[cmd.index("--gpu-layers") + 1], "65")

    def test_paths_never_reach_records_or_specs(self):
        argv = ["-m", "C:/Users/me/My Models/x.gguf", "--chat-template-file", "/home/me/t.jinja", "-c", "4096"]
        self.assertEqual(launch.redact(argv), ["-m", "x.gguf", "--chat-template-file", "t.jinja", "-c", "4096"])
        kept, dropped = launch.drop_path_args(launch.redact(argv))
        self.assertEqual(kept, ["-c", "4096"])
        self.assertEqual(dropped, ["--chat-template-file", "-m"])


def _obs(ngl="40", n_layer=64, log_rows=None, metrics=None, session_table=None):
    return {"kind": "observation", "evidence_class": "passive", "created": "t", "gpu": {"memory.total": "8192", "name": "X", "bandwidth_gbs_known": 500.0},
            "model": {"summary": {"n_layer": n_layer, "total_bytes": 6 << 30, "weight_bytes_streamed_per_decode_token": 6 << 30, "kv_bytes_per_token_f16": 65536}},
            "server_process": {"pid_found": True, "launch": {"available": True, "settings": {"n_gpu_layers": ngl, "ctx_size": "8192"},
                                                             "argv_redacted": ["-ngl", ngl, "-c", "8192"], "source": "command line of pid 1 (observed)"}},
            "log": {"sessions": 1 if session_table else 0, "session_table": session_table or [], "requests": log_rows or []},
            "metrics": metrics or {"available": False}}


class OffloadAndRankingTests(unittest.TestCase):
    def test_partial_offload_from_command_line_ranks_first(self):
        m = {"available": True, "since_server_start": {"prompt_seconds_total": 10.0, "tokens_predicted_seconds_total": 30.0, "prompt_tokens_total": 500,
                                                       "tokens_predicted_total": 60, "prompt_tokens_cached_total": 0}}
        d = dg.diagnose([_obs(metrics=m)])
        f = d["findings"][0]
        self.assertEqual(f["id"], "gpu_offload_partial")
        self.assertEqual(f["confidence"], "likely")
        self.assertEqual([a["name"] for a in f["experiment"]["arms"]], ["ngl40", "ngl65"])
        self.assertEqual(d["recommended_next_experiment"]["finding"], "gpu_offload_partial")
        ceiling = next(x for x in d["findings"] if x["id"] == "decode_below_ceiling")
        self.assertEqual(ceiling["superseded_by"], "gpu_offload_partial")
        self.assertIn("not measured", f["measured"]["status"])

    def test_full_offload_and_log_line_take_precedence(self):
        self.assertFalse(any(f["id"] == "gpu_offload_partial" for f in dg.diagnose([_obs(ngl="99")])["findings"]))
        rows = [{"session": 1, "offloaded": "65/65"}]
        self.assertFalse(any(f["id"] == "gpu_offload_partial" for f in dg.diagnose([_obs(ngl="40", session_table=rows)])["findings"]))

    def test_impact_bound_comes_from_measured_shares_only(self):
        f = {"id": "microbatch_limits_prefill"}
        self.assertEqual(dg.impact(f, {"prefill": 0.6, "decode": 0.3, "residual": 0.1, "basis": "b"})["bound_share_of_request"], 0.6)
        self.assertIsNone(dg.impact({"id": "per_request_overhead"}, {"prefill": 0.6, "decode": 0.4, "residual": None, "basis": "log"})["bound_share_of_request"])
        self.assertIsNone(dg.impact({"id": "prefix_reuse_low"}, None))

    def test_thin_traffic_is_not_a_reuse_claim(self):
        rows = [{"session": 1, "prompt_n": 100, "prompt_ms": 50, "prompt_tps": 2000, "predicted_n": 10, "predicted_ms": 100, "predicted_tps": 100,
                 "slot_by": "LRU", "reused_tokens": 0}]
        obs = _obs(ngl="99", log_rows=rows)
        obs["log"]["cache_reuse"] = {"available": True, "selected_by": {"LRU": 1}, "reused_share_of_prompt": {"median": 0.0, "p90": 0.0}, "requests_with_reuse_data": 1}
        ids = [f["id"] for f in dg.diagnose([obs])["findings"]]
        self.assertIn("prefix_reuse_thin", ids)
        self.assertNotIn("prefix_reuse_low", ids)

    def test_propose_uses_the_observed_launch_and_a_vram_floor_from_the_model(self):
        with tempfile.TemporaryDirectory() as td:
            d = {"observed_launch": {"args_redacted": ["-ngl", "40", "-c", "8192", "--chat-template-file", "t.jinja"], "source": "pid 1"}}
            args, src = pp.baseline_args(d, {})
            self.assertEqual(args, ["-ngl", "40", "-c", "8192"])
            self.assertIn("--chat-template-file", src)
            full = ["-ngl", "40", "-c", "8192", "--chat-template-file", os.path.join(td, "t.jinja")]
            args, src = pp.baseline_args(d, {"server_args": full})
            self.assertEqual(args, full)                                     # same settings: the config's full-fidelity args win
            args, src = pp.baseline_args(d, {"server_args": ["-ngl", "99"]})
            self.assertIn("differ", src)


class HardwareAndShapeTests(unittest.TestCase):
    def _obs(self, cc="7.5", bf16_bytes=64 << 20):
        o = _obs(ngl="99")
        o["gpu"]["compute_cap"] = cc
        o["model"]["summary"]["tensor_types"] = {"BF16": {"count": 96, "bytes": bf16_bytes}, "Q4_K": {"count": 300, "bytes": 6 << 30}}
        return o

    def _profile(self, names):
        return {"kind": "profile", "evidence_class": "active", "created": "t", "label": "pf", "mode": {"pp": 512, "tg": 0},
                "time_share_by_class_pct": {"cublas_gemm": 12.5},     # an older record's coarse class; names decide
                "kernels": [{"name": n, "time_pct": p, "instances": 10, "avg_ns": 1e5} for n, p in names]}

    def test_bf16_rule_states(self):
        f = lambda recs: next((x for x in dg.diagnose(recs)["findings"] if x["id"] == "bf16_without_tensor_cores"), None)
        self.assertIsNone(f([self._obs(cc="8.6")]))                                  # BF16 tensor cores present
        self.assertIsNone(f([self._obs(bf16_bytes=1 << 20)]))                         # norm vectors only
        self.assertEqual(f([self._obs()])["confidence"], "hypothesis")
        hit = f([self._obs(), self._profile([("magma_sgemmEx_kernel<float, __nv_bfloat16, float>", 12.5), ("mul_mat_q", 70.0)])])
        self.assertEqual((hit["confidence"], hit["measurements"]["profile_bf16_share_pct"]), ("likely", 12.5))
        clean = f([self._obs(), self._profile([("mul_mat_q", 80.0)])])
        self.assertEqual(clean["severity"], "info")

    def test_traffic_shape_counts_cached_prompt_tokens_and_ignores_experiments(self):
        arm = {"evidence_class": "passive", "requests": [{"prompt_n": 100, "cache_n": 133, "predicted_n": 48}] * 5}
        s = dg.traffic_shape(arm, "log")
        self.assertEqual((s["prompt_tokens_median"], s["generated_tokens_median"]), (233, 48))
        self.assertIsNone(dg.traffic_shape(dict(arm, evidence_class="controlled")))


class ImproveLoopTests(unittest.TestCase):
    def _exp(self, cand_wall):
        def reqs(wall, dec_ms):
            return [{"prompt_id": "p%d" % (i % 3), "prompt_n": 1000, "prompt_ms": 2000, "prompt_tps": 500, "predicted_n": 64, "predicted_ms": dec_ms,
                     "predicted_tps": 64000.0 / dec_ms, "wall_ms": wall + i, "cache_n": 0, "tokens": [1, 2, 3]} for i in range(9)]
        arm = lambda ngl, r: {"requests": r, "args": ["llama-server", "--model", "m.gguf", "--host", "127.0.0.1", "--port", "8099", "-c", "4096", "--gpu-layers", str(ngl)],
                              "env_ggml": {}, "failures": [], "launch_facts": [], "cache": {"mode": "cold"}}
        return {"kind": "experiment", "evidence_class": "controlled", "created": "t", "name": "gl", "spec_redacted": {"rounds": 3, "provenance": {"finding": "gpu_offload_partial"}},
                "gpu": {"memory.total": "8192", "bandwidth_gbs_known": 500.0}, "workload": {"set_hash": "s", "n_predict": 64},
                "model": {"summary": {"n_layer": 64, "total_bytes": 6 << 30, "weight_bytes_streamed_per_decode_token": 6 << 30}},
                "arms": {"ngl56": arm(56, reqs(10000, 7800)), "ngl65": arm(65, reqs(cand_wall, 1300))}}

    def test_next_diagnosis_and_baseline_build_on_the_faster_arm(self):
        d = dg.diagnose([self._exp(3500)])
        self.assertEqual(d["diagnosed_arm"], "gl/ngl65")
        self.assertFalse(any(f["id"] == "gpu_offload_partial" for f in d["findings"]))     # fixed on the candidate
        self.assertEqual(d["experiment_verdicts"][0]["tests_finding"], "gpu_offload_partial")
        args, src = pp.baseline_args(d, {"server_args": ["-ngl", "56"]})
        args, _removed = pp.strip_runner_owned(pp._strip_launch_prefix(args))
        self.assertEqual(args, ["-c", "4096", "--gpu-layers", "65"])
        self.assertIn("faster", src)

    def test_a_slower_candidate_leaves_the_baseline_in_place(self):
        d = dg.diagnose([self._exp(14000)])
        self.assertEqual(d["diagnosed_arm"], "gl/ngl56")
        self.assertIsNone(d["baseline_from_verdict"])


class DiscoveryBoundsTests(unittest.TestCase):
    def test_drive_roots_are_refused_and_walk_is_bounded(self):
        with self.assertRaises(ValueError):
            disc.check_root(os.path.abspath(os.sep))
        with tempfile.TemporaryDirectory() as td:
            deep = os.path.join(td, *["d%d" % i for i in range(8)])
            os.makedirs(deep)
            for d in (td, deep):
                open(os.path.join(d, "m.gguf"), "wb").close()
            os.makedirs(os.path.join(td, "node_modules"))
            open(os.path.join(td, "node_modules", "x.gguf"), "wb").close()
            found, _seen, _trunc = disc.walk(td, max_depth=4, want=disc._is_gguf)
            self.assertEqual([os.path.basename(os.path.dirname(str(f))) for f in found], [os.path.basename(td)])
            _f, _s, trunc = disc.walk(td, max_depth=8, max_entries=3)
            self.assertTrue(trunc)

    def test_metrics_text(self):
        m = disc.parse_metrics("# HELP x\nllamacpp:prompt_tokens_total 7\nllamacpp:n_busy_slots_per_decode 1\nother 3\n")
        self.assertEqual(m, {"prompt_tokens_total": 7.0, "n_busy_slots_per_decode": 1.0})

    def test_suggested_config_never_owns_the_users_server_and_skips_hotpath_servers(self):
        srv = {"pid": 5, "exe": "C:/b/llama-server.exe", "argv": ["llama-server", "-m", "C:/m.gguf", "--port", "8080", "-ngl", "99"],
               "settings": {"model": "C:/m.gguf"}, "base_url": "http://127.0.0.1:8080", "http": {"props": {"model_path": "C:/m.gguf"}, "health": {"status": "ok"}}}
        rec = {"servers": [srv, dict(srv, pid=6, launched_by_hotpath=True, base_url="http://127.0.0.1:8099")], "installs": [], "models": []}
        out, src, amb = disc.suggest_config(rec, {})
        self.assertEqual(out["server_url"], "http://127.0.0.1:8080")
        self.assertEqual(out["owned_servers"], ["http://127.0.0.1:8099"])
        self.assertEqual(out["server_args"], ["-ngl", "99"])
        self.assertFalse(amb)

    @unittest.skipUnless(os.name == "nt", "PE images exist only on Windows")
    def test_pe_imports_of_a_python_extension_module(self):
        import _ctypes                                  # a real PE image that imports the Python DLL
        names = [n.lower() for n in disc.pe_imports(_ctypes.__file__) or []]
        self.assertIn("python%d%d.dll" % sys.version_info[:2], names)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dll") as f:
            f.write(b"MZ" + b"\0" * 100)
        try:
            self.assertIsNone(disc.pe_imports(f.name))
        finally:
            os.unlink(f.name)


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.td.name)

    def tearDown(self):
        mcp.stop_all_jobs()
        mcp.JOBS.clear()
        os.chdir(self.cwd)
        self.td.cleanup()

    def test_quick_job_returns_its_result_in_the_same_call(self):
        text, err = mcp.start_job("hotpath_bench", ["--version"], timeout_s=60, wait_s=30)
        self.assertFalse(err, text)
        self.assertIn("finished", text)

    def test_long_job_returns_an_id_refuses_a_second_and_cancels_its_tree(self):
        text, err = mcp.start_job("hotpath_profile", ["observe", "--sample-seconds", "60"], timeout_s=120, wait_s=1)
        self.assertIn("Still running", text)
        jid = next(w for w in text.split() if w.startswith("job-"))
        text2, err2 = mcp.start_job("hotpath_bench", ["--version"], timeout_s=60, wait_s=1)
        self.assertTrue(err2)
        self.assertIn("still running", text2)
        t0 = time.time()
        text3, err3 = mcp.call_tool("hotpath_job_cancel", {"job_id": jid})
        self.assertIn("cancelled", text3)
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(mcp.JOBS[jid].state(), "cancelled")
        self.assertIn(jid, mcp.call_tool("hotpath_job_status", {})[0])

    def test_unknown_job_and_timeout(self):
        self.assertTrue(mcp.call_tool("hotpath_job_status", {"job_id": "job-nope"})[1])
        text, err = mcp.start_job("hotpath_profile", ["observe", "--sample-seconds", "60"], timeout_s=1, wait_s=20)
        self.assertTrue(err)
        self.assertIn("timeout", text)


if __name__ == "__main__":
    unittest.main()
