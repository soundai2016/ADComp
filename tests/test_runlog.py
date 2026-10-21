"""Check run recording, subprocess failures, and comparison metadata."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runlog


class RunLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="acoustic-record-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "model.py").write_text("order = 32\n")
        (self.source / "config.json").write_text('{"order":32,"seed":73,"cpu_affinity":[1,2]}')
        self.environment = dict(os.environ, OPENBLAS_NUM_THREADS="2", PRIVATE_TEST_TOKEN="do-not-record-this")

    def tearDown(self):
        self.temp.cleanup()

    def record(self, folder="logs", **kwargs):
        return runlog.RunLog(self.root / folder, mode="test", command=[sys.executable, "model.py"],
                            source_root=self.source, environment=self.environment, **kwargs)

    def test_output_metrics_and_environment(self):
        with self.record() as record:
            stage = record.run_stage("solver", [sys.executable, "-c",
                "import sys,time; print('first',flush=True); time.sleep(.3); print('second',file=sys.stderr); print('last')"],
                cwd=self.source, console=False)
            record.metrics("convergence", {"error": 0.125, "iterations": 4}, step=4,
                           units={"error": "relative", "iterations": "count"})
        log = (record.log_dir / stage["log"]).read_text()
        self.assertIn("[stdout] first", log)
        self.assertIn("[stderr] second", log)
        self.assertIn("[stdout] last", log)
        summary = json.loads((record.log_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["stages"][0]["exit_code"], 0)
        self.assertEqual(summary["stages"][0]["timing_scope"], "subprocess_wall")
        metric = json.loads((record.log_dir / "metrics.jsonl").read_text())
        self.assertEqual(metric["values"], {"error": 0.125, "iterations": 4})
        self.assertEqual(metric["step"], 4)
        snapshot = (record.log_dir / "environment.json").read_text()
        self.assertNotIn("do-not-record-this", snapshot)
        snapshot = json.loads(snapshot)
        self.assertEqual(snapshot["thread_environment"]["OPENBLAS_NUM_THREADS"], "2")
        self.assertEqual(snapshot["numerical_backend"]["status"], "ok")
        events = [json.loads(line)["record"]["extra"] for line in (record.log_dir / "events.jsonl").read_text().splitlines()]
        self.assertTrue(all(event["run_id"] == record.run_id for event in events))
        self.assertTrue(any(event.get("stream") == "stderr" for event in events))

    def test_nonzero_exit_is_preserved(self):
        record = self.record()
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            with record:
                record.run_stage("failure", [sys.executable, "-c", "import sys; print('failure detail',file=sys.stderr); sys.exit(7)"],
                                 cwd=self.source, console=False)
        self.assertEqual(failure.exception.returncode, 7)
        summary = json.loads((record.log_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"][0]["exit_code"], 7)
        self.assertIn("failure detail", (record.log_dir / summary["stages"][0]["log"]).read_text())

    def test_fingerprints_ignore_run_paths_and_timestamps(self):
        with self.record("one", configuration={"order": 32, "output_dir": "/tmp/a"}) as first:
            pass
        (self.source / "config.json").write_text('{"order":32,"seed":73,"cpu_affinity":[5,6]}')
        with self.record("two", configuration={"order": 32, "output_dir": "/tmp/b"}) as second:
            pass
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.record["configuration_fingerprint"], second.record["configuration_fingerprint"])
        self.assertNotEqual(first.record["source_fingerprint"], second.record["source_fingerprint"])
        self.assertEqual(first.record["environment_fingerprint"], second.record["environment_fingerprint"])
        with self.record("three", configuration={"order": 64, "output_dir": "/tmp/b"}) as third:
            pass
        self.assertNotEqual(first.record["configuration_fingerprint"], third.record["configuration_fingerprint"])

    def test_generated_models_do_not_change_source_fingerprints(self):
        custom = self.source / "computed"
        configuration = {"order": 32, "models_dir": str(custom)}
        with self.record("before_models", configuration=configuration) as first:
            pass
        for base in (self.source / "models", custom):
            old = base / "old_run"
            old.mkdir(parents=True)
            (old / "solver.py").write_text("raise RuntimeError('historical source')\n")
            (old / "config.json").write_text("not a source configuration\n")
        with self.record("after_models", configuration=configuration) as second:
            pass
        self.assertEqual(first.record["source_fingerprint"], second.record["source_fingerprint"])
        self.assertEqual(first.record["configuration_fingerprint"], second.record["configuration_fingerprint"])
        child = custom / "current_run"
        child.mkdir()
        (child / "solver.py").write_text("order = 32\n")
        with runlog.RunLog(self.root / "child_log", mode="test", command=[sys.executable, "solver.py"],
                           source_root=child, environment=self.environment,
                           configuration={"models_dir": str(custom), "workdir": str(child)}) as run:
            pass
        manifest = json.loads((run.log_dir / "manifest.json").read_text())
        self.assertEqual(list(manifest["source"]), ["solver.py"])

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup")
    def test_interrupt_terminates_child(self):
        record = self.record()
        timer = None
        try:
            with self.assertRaises(KeyboardInterrupt):
                with record:
                    timer = threading.Timer(.5, lambda: os.kill(os.getpid(), signal.SIGINT))
                    timer.start()
                    record.run_stage("long_stage", [sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(30)"],
                                     cwd=self.source, console=False)
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
        summary = json.loads((record.log_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["stages"][0]["status"], "interrupted")
        with self.assertRaises(ProcessLookupError):
            os.kill(summary["stages"][0]["pid"], 0)


    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup")
    def test_nested_sigterm_stops_solver_and_records_both_runs(self):
        solver = self.source / "solver.py"
        pidfile = self.root / "solver.pid"
        solver.write_text(f"import os,time\nfrom pathlib import Path\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n")
        child = self.source / "child.py"
        child.write_text(
            "import sys\nfrom pathlib import Path\nfrom runlog import RunLog\n"
            f"with RunLog(Path({str(self.root / 'inner')!r}), mode='test', command=sys.argv, source_root=Path({str(self.source)!r})) as run:\n"
            f"    run.run_stage('solver', [sys.executable, {str(solver)!r}], cwd={str(self.source)!r})\n")
        manager = self.source / "manager.py"
        manager.write_text(
            "import sys\nfrom pathlib import Path\nfrom runlog import RunLog\n"
            f"with RunLog(Path({str(self.root / 'outer')!r}), mode='test', command=sys.argv, source_root=Path({str(self.source)!r})) as run:\n"
            f"    run.run_stage('manager', [sys.executable, {str(child)!r}], cwd={str(self.source)!r})\n")
        environment = dict(self.environment, PYTHONPATH=str(Path(runlog.__file__).parent))
        process = subprocess.Popen([sys.executable, str(manager)], env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 12
            while not pidfile.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(pidfile.exists(), "Nested solver did not start")
            solver_pid = int(pidfile.read_text())
            process.send_signal(signal.SIGTERM)
            self.assertNotEqual(process.wait(timeout=10), 0)
            for folder in ("outer", "inner"):
                summary = json.loads((self.root / folder / "summary.json").read_text())
                self.assertEqual(summary["status"], "interrupted")
                self.assertEqual(summary["termination_signal"], signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                os.kill(solver_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
