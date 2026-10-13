"""The authorization boundary between an agent and the GPU, written before the fixes from the ways it failed or could fail:
  1. agent-run specs were checked against a denylist, and this llama-server build has arguments and LLAMA_ARG_* variables it
     did not list that start programs (--mcp-servers-config, LLAMA_ARG_MCP_SERVERS_JSON), enable built-in tools and a CORS
     proxy (--tools, --agent), reach other hosts (--rpc) or write prompts to disk (--log-prompts-dir); GGML_BACKEND_PATH and
     CUDA_INJECTION64_PATH load a library, and the GGML_/CUDA_ prefixes let them through;
  2. only hotpath_experiment_run checked spec contents: the improve loop launched specs built from saved session state
     (session.json's starting arguments) and observed command lines without any check, so an edited session could launch
     what an agent-written spec could not;
  3. the spec was checked in the MCP server and read again by the child: a file swapped in between was launched unchecked;
  4. resume reused any record whose file name matched the step, even one written for a different spec;
  5. the session lock was check-then-write (two processes could both take it) and trusted a bare pid (reused by Windows);
  6. nothing stopped two hotpath processes (two MCP servers, or the CLI and an MCP job) from running GPU work at once;
  7. a job whose MCP server was killed outright kept its llama-server running with nobody to report to;
  8. hotpath_workload overwrote any existing file inside the working directory, and it and hotpath_propose were advertised as
     read-only.
GPU behavior is validated on real MCP runs (docs/agents.md)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from hotpath import agentgate as ag
from hotpath import improve as im
from hotpath import locks
from hotpath import mcp_server as mcp

import test_improve_trust as tt


class _InDir:
    def __init__(self, d):
        self.d, self.old = d, None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.d)

    def __exit__(self, *a):
        os.chdir(self.old)


PRODUCTION_LAUNCH = ["--alias", "my-model", "--ctx-size", "8192", "--gpu-layers", "99", "--batch-size", "256", "--ubatch-size", "256", "--parallel", "1",
                "--slots", "--jinja", "--reasoning-format", "deepseek", "--no-webui", "--offline"]


def _spec(**over):
    s = {"name": "s", "binary": "${runtime_binary}", "model": "${model_path}", "port": 8099, "common_args": list(PRODUCTION_LAUNCH), "env": {},
         "arms": [{"name": "a"}, {"name": "b", "args_override": {"--ubatch-size": 512}}], "workload": "w.json"}
    s.update(over)
    return s


class AllowlistTests(unittest.TestCase):
    def test_a_real_production_launch_passes(self):
        self.assertEqual(ag.spec_problems(_spec()), [])
        self.assertEqual(ag.spec_problems(_spec(env={"GGML_CUDA_DISABLE_GRAPHS": "1", "LLAMA_ARG_N_GPU_LAYERS": "40"},
                                                arms=[{"name": "a"}, {"name": "b", "args_override": {"-ctk": "q8_0", "-ctv": "q8_0", "-fa": "on"}},
                                                      {"name": "c", "args_remove": ["--offline"]}])), [])

    def test_flags_that_start_programs_or_reach_out_are_refused_in_every_form(self):
        # failure 1
        for common in (["--mcp-servers-config", "m.json"], ["--tools", "all"], ["--agent"], ["-ag"], ["--rpc", "10.0.0.2:50052"],
                       ["--log-prompts-dir", "p"], ["-lcd", "x.bin"], ["--chat-template-file", "t.jinja"], ["--props"], ["--models-dir", "m"],
                       ["--gpt-oss-20b-default"], ["--webui-mcp-proxy"], ["--tools=all"], ["--some-future-flag", "1"]):
            p = ag.spec_problems(_spec(common_args=PRODUCTION_LAUNCH + common))
            self.assertTrue(p, common)
        p = ag.spec_problems(_spec(arms=[{"name": "a"}, {"name": "b", "args_override": {"--mcp-servers-config": "m.json"}}]))
        self.assertTrue(any("--mcp-servers-config" in x for x in p))

    def test_environment_is_an_allowlist_too(self):
        for k in ("LLAMA_ARG_MCP_SERVERS_JSON", "LLAMA_ARG_MCP_SERVERS_CONFIG", "LLAMA_ARG_TOOLS", "LLAMA_ARG_AGENT", "LLAMA_ARG_RPC", "LLAMA_ARG_LOG_FILE",
                  "LLAMA_ARG_HOST", "LLAMA_ARG_MODEL_URL", "LLAMA_CACHE", "LLAMA_API_KEY", "GGML_BACKEND_PATH", "CUDA_INJECTION64_PATH", "CUDA_CACHE_PATH",
                  "PATH", "PYTHONPATH"):
            for where in ("env", "env_override"):
                spec = _spec(env={k: "x"}) if where == "env" else _spec(arms=[{"name": "a", "env_override": {k: "x"}}, {"name": "b"}])
                self.assertTrue(ag.spec_problems(spec), (k, where))

    def test_mcp_check_uses_the_allowlist(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            with open("s.json", "w", encoding="utf-8") as f:
                json.dump(_spec(env={"LLAMA_ARG_MCP_SERVERS_JSON": "{}"}), f)
            with self.assertRaises(ValueError) as cm:
                mcp.check_spec_for_agent("s.json")
            self.assertIn("LLAMA_ARG_MCP_SERVERS_JSON", str(cm.exception))

    def test_improve_args_parameter_uses_the_allowlist(self):
        with self.assertRaises(ValueError) as cm:
            mcp._improve_argv({"args": "--ctx-size 4096 --tools all"}, True)
        self.assertIn("--tools", str(cm.exception))
        self.assertIn("--args=--ctx-size 4096 --jinja", mcp._improve_argv({"args": "--ctx-size 4096 --jinja"}, True))


class AgentModeLaunchTests(unittest.TestCase):
    """failures 2 and 3: in a process the MCP server started, every launch is checked on the content that is launched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.environ[ag.GATE_ENV] = "1"

    def tearDown(self):
        os.environ.pop(ag.GATE_ENV, None)
        self.tmp.cleanup()

    def test_children_of_the_mcp_server_run_gated_and_watch_their_parent(self):
        env = mcp._child_env()
        self.assertEqual(env.get(ag.GATE_ENV), "1")
        self.assertTrue(env.get(locks.PARENT_ENV, "").startswith("%d:" % os.getpid()))

    def test_runner_refuses_a_spec_the_gate_refuses_before_resolving_or_launching(self):
        from hotpath import cli
        from hotpath import experiment as ex
        with _InDir(self.dir):
            with open("w.json", "w", encoding="utf-8") as f:
                json.dump({"name": "w", "n_predict": 8, "prompts": [{"id": "p0", "text": "x"}]}, f)
            with open("s.json", "w", encoding="utf-8") as f:
                json.dump(_spec(common_args=PRODUCTION_LAUNCH + ["--rpc", "10.0.0.2:50052"]), f)
            launched = []
            saved = ex.run
            ex.run = lambda *a, **k: launched.append(a)
            try:
                with self.assertRaises(SystemExit) as cm:
                    cli._experiment_runner({"runtime_binary": sys.executable, "model_path": sys.executable}, 0)(json.load(open("s.json")), "s.json")
            finally:
                ex.run = saved
            self.assertIn("--rpc", str(cm.exception))
            self.assertEqual(launched, [])

    def test_an_edited_session_cannot_launch_what_a_spec_could_not(self):
        # the starting arguments of a session come from session.json on resume; an agent can edit that file
        runner = tt.GuardedRunner()
        first = im.improve({"runtime_binary": "b", "model_path": "m"}, [], self._workload(), self.dir, tt.GuardedRunner(raise_on=1), log=lambda *a: None,
                           initial_args=["--ctx-size", "4096"], max_experiments=1)
        state = os.path.join(self.dir, first["session"], "session.json")
        saved = json.load(open(state, encoding="utf-8"))
        saved["initial_args"] = ["--ctx-size", "4096", "--tools", "all"]
        with open(state, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        with self.assertRaises(SystemExit) as cm:
            im.improve({"runtime_binary": "b", "model_path": "m"}, [], "", self.dir, runner, log=lambda *a: None, session=first["session"], resume=True)
        self.assertIn("--tools", str(cm.exception))
        self.assertEqual(runner.specs, [])

    def test_a_resumed_workload_outside_the_roots_is_refused(self):
        first = im.improve({"runtime_binary": "b", "model_path": "m"}, [], self._workload(), self.dir, tt.GuardedRunner(raise_on=1), log=lambda *a: None,
                           initial_args=["--ctx-size", "4096"], max_experiments=1)
        state = os.path.join(self.dir, first["session"], "session.json")
        saved = json.load(open(state, encoding="utf-8"))
        saved["workload"] = os.path.join(os.path.abspath(os.sep), "Windows", "win.ini")
        with open(state, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        runner = tt.GuardedRunner()
        with _InDir(self.dir), self.assertRaises(SystemExit):
            im.improve({"runtime_binary": "b", "model_path": "m"}, [], "", self.dir, runner, log=lambda *a: None, session=first["session"], resume=True)
        self.assertEqual(runner.specs, [])

    def _workload(self):
        p = os.path.join(self.dir, "w.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"name": "w", "n_predict": 16, "prompts": [{"id": "p%02d" % i, "text": "x " * (100 + 50 * i), "words": 100 + 50 * i} for i in range(3)]}, f)
        return p


class ResumeIntegrityTests(unittest.TestCase):
    """failure 4"""

    def setUp(self):
        self.t = tt.TrustTests("test_resume_reruns_nothing_completed")
        self.t.setUp()

    def tearDown(self):
        self.t.tearDown()

    def test_a_record_written_for_another_spec_is_not_reused(self):
        first = self.t.run_loop(tt.GuardedRunner(), max_experiments=1)
        sdir = os.path.join(self.t.dir, first["session"])
        rec_name = [n for n in os.listdir(sdir) if n.startswith("experiment-") and "-s1-" in n][0]
        p = os.path.join(sdir, rec_name)
        rec = json.load(open(p, encoding="utf-8"))
        rec["spec_fingerprint"] = "not-this-spec"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        runner = tt.GuardedRunner()
        self.t.run_loop(runner, max_experiments=1, session=first["session"], resume=True)
        self.assertTrue(any("-s1-" in s["name"] for s in runner.specs), "a mismatched record must be measured again, not reused")

    def test_a_matching_record_is_reused_and_a_dry_run_still_launches_nothing(self):
        first = self.t.run_loop(tt.GuardedRunner(), max_experiments=1)
        runner = tt.GuardedRunner()
        self.t.run_loop(runner, max_experiments=1, session=first["session"], resume=True)
        self.assertFalse(any("-s1-" in s["name"] for s in runner.specs), "a matching step record is reused")
        sdir = os.path.join(self.t.dir, first["session"])
        for n in os.listdir(sdir):
            if n.startswith("experiment-") and "-s1-" in n:
                p = os.path.join(sdir, n)
                rec = json.load(open(p, encoding="utf-8"))
                rec["spec_fingerprint"] = "changed"
                json.dump(rec, open(p, "w", encoding="utf-8"))
        dry = tt.GuardedRunner()
        plan = self.t.run_loop(dry, max_experiments=1, session=first["session"], resume=True, dry_run=True)
        self.assertEqual(dry.specs, [])
        self.assertTrue(plan["stopped_because"].startswith("dry run"), plan["stopped_because"])


HOLDER = """
import sys, time
sys.path.insert(0, %r)
from hotpath import locks
if sys.argv[1] == "gpu":
    with locks.gpu_work("experiment", "other-run"):
        print("held", flush=True)
        time.sleep(60)
else:
    locks.take_session(sys.argv[2])
    print("held", flush=True)
    time.sleep(60)
"""


class LockTests(unittest.TestCase):
    """failures 5 and 6, with a real second process holding the lock"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HOTPATH_GPU_LOCK_DIR"] = self.tmp.name
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
            p.stdout.close()
        os.environ.pop("HOTPATH_GPU_LOCK_DIR", None)
        locks.release_all()
        self.tmp.cleanup()

    def _holder(self, *args):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = subprocess.Popen([sys.executable, "-c", HOLDER % here] + list(args), stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
                             env=dict(os.environ))
        self.procs.append(p)
        self.assertEqual(p.stdout.readline().strip(), "held")
        return p

    def test_gpu_lock_refuses_while_another_process_holds_it_and_names_it(self):
        p = self._holder("gpu")
        with self.assertRaises(SystemExit) as cm:
            with locks.gpu_work("experiment", "mine"):
                pass
        self.assertIn("other-run", str(cm.exception))
        self.assertIn("hotpath pid", str(cm.exception))       # the holder's own pid (in a venv, Popen.pid is the launcher's)

    def test_a_dead_holder_leaves_no_lock_behind(self):
        p = self._holder("gpu")
        p.kill()
        p.wait(10)
        with locks.gpu_work("experiment", "mine"):
            self.assertEqual(locks.gpu_holder()["pid"], os.getpid())
        self.assertIsNone(locks.gpu_holder())

    def test_a_holder_file_without_a_lock_is_not_a_holder(self):
        with open(str(locks.gpu_lock_path()) + ".holder", "w", encoding="utf-8") as f:
            json.dump({"pid": os.getppid(), "start": locks.process_start(os.getppid()), "kind": "experiment"}, f)
        with locks.gpu_work("experiment", "mine"):
            pass

    def test_reentrant_within_one_process(self):
        with locks.gpu_work("improve", "session"):
            with locks.gpu_work("experiment", "step"):
                pass
            self.assertEqual((locks.gpu_holder() or {}).get("pid"), os.getpid(), "the inner release must not drop the outer hold")
        self.assertIsNone(locks.gpu_holder())

    def test_session_lock_is_exclusive_and_released_by_a_dying_holder(self):
        d = self.tmp.name
        p = self._holder("session", d)
        with self.assertRaises(SystemExit) as cm:
            locks.take_session(d)
        self.assertIn("running in another process", str(cm.exception))
        p.kill()
        p.wait(10)
        lk = locks.take_session(d)
        locks.drop_session(lk)


class OrphanTests(unittest.TestCase):
    """failure 7: the launched server is stopped when the MCP server that started the job is gone."""

    def test_children_are_stopped_when_the_parent_disappears(self):
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
        locks.track(child)
        fired = []
        try:
            t = locks.watch_parent({locks.PARENT_ENV: "%d:%s" % (parent.pid, locks.process_start(parent.pid))}, interval=0.2,
                                   on_orphan=lambda: (locks.stop_children(), fired.append(1)))
            self.assertIsNotNone(t)
            time.sleep(0.6)
            self.assertEqual(fired, [], "a live parent is not an orphaning")
            parent.kill()
            parent.wait(10)
            t.join(10)
            self.assertEqual(fired, [1])
            self.assertIsNotNone(child.poll(), "the launched process must have been stopped")
        finally:
            for p in (parent, child):
                if p.poll() is None:
                    p.kill()
                    p.wait(10)


class WriteBoundaryTests(unittest.TestCase):
    """failure 8"""

    def test_workload_tool_never_overwrites_and_writing_tools_are_not_advertised_read_only(self):
        with tempfile.TemporaryDirectory() as td, _InDir(td):
            open("app.py", "w").close()
            with self.assertRaises(ValueError):
                mcp.call_tool("hotpath_workload", {"out_file": "app.py"})
            with open("existing.json", "w", encoding="utf-8") as f:
                f.write("{}")
            with self.assertRaises(ValueError):
                mcp.call_tool("hotpath_workload", {"out_file": "existing.json"})
        hints = {}
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        for t in json.loads(buf.getvalue())["result"]["tools"]:
            hints[t["name"]] = t["annotations"]["readOnlyHint"]
        for name in ("hotpath_workload", "hotpath_propose", "hotpath_job_cancel", "hotpath_improve", "hotpath_experiment_run"):
            self.assertFalse(hints[name], name)
        for name in ("hotpath_inspect", "hotpath_job_status", "hotpath_list_results", "hotpath_experiment_plan"):
            self.assertTrue(hints[name], name)


if __name__ == "__main__":
    unittest.main()
