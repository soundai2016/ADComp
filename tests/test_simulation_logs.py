"""Check report collection and validation in the simulation runner."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acoustic_simulation_runner", SOURCE / "simulation" / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
HANDOFF = {"ACOUSTIC_RUN_ID", "ACOUSTIC_LOG_DIR", "ACOUSTIC_WORKDIR", "ACOUSTIC_REQUESTED_WORKDIR"}


class SimulationLogTests(unittest.TestCase):
    def test_reports_are_attributed_only_to_writing_stages(self):
        with tempfile.TemporaryDirectory(prefix="acoustic-simulation-log-test-") as temporary:
            root = Path(temporary)
            source, workdir = root / "simulation", root / "work"
            case = source / "case"
            case.mkdir(parents=True)
            shutil.copyfile(SOURCE / "runlog.py", source / "runlog.py")
            payload = '{"threads":2}\n'
            setup = "from pathlib import Path\nimport os\np = Path('results/environment.json')\n"
            (case / "producer.py").write_text(setup +
                f"p.write_text({payload!r})\n" +
                "os.utime(p, ns=(1600000000000000000, 1600000000000000000))\n")
            (case / "reuser.py").write_text(setup + "print(p.read_text())\n")
            (case / "rewriter.py").write_text(setup +
                "p.write_bytes(p.read_bytes())\n" +
                "os.utime(p, ns=(1600000002000000000, 1600000002000000000))\n")
            steps = tuple((f"case/{name}.py",) for name in ("producer", "reuser", "rewriter"))
            environment = {key: value for key, value in os.environ.items()
                           if key not in HANDOFF}
            with patch.object(runner, "HERE", source), patch.object(runner, "KERNEL_STEPS", steps), \
                    patch.dict(os.environ, environment, clear=True), patch.object(sys, "path", sys.path.copy()):
                actual = runner.execute(workdir, smoke=True)

            status = json.loads((actual / "run_status.json").read_text())
            self.assertTrue(status["passed"])
            self.assertEqual(status["stages_completed"], 3)
            logs = Path(status["log_dir"])
            summary = json.loads((logs / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual([stage["name"] for stage in summary["stages"]], [step[0] for step in steps])
            self.assertTrue(all(stage["exit_code"] == 0 for stage in summary["stages"]))
            events = [json.loads(line)["record"]["extra"]
                      for line in (logs / "events.jsonl").read_text().splitlines()]
            reports = [event for event in events if event.get("event") == "kernel_environment"]
            self.assertEqual([event["stage"] for event in reports], [steps[0][0], steps[2][0]])
            self.assertNotEqual(reports[0]["snapshot"], reports[1]["snapshot"])
            expected_hash = hashlib.sha256(payload.encode()).hexdigest()
            for event in reports:
                self.assertEqual(event["sha256"], expected_hash)
                self.assertEqual((logs / event["snapshot"]).read_text(), payload)

    def test_nested_environment_and_numerical_summary_are_archived(self):
        with tempfile.TemporaryDirectory(prefix="acoustic-simulation-reports-") as temporary:
            root = Path(temporary)
            source = root / "simulation"
            case = source / "case"
            case.mkdir(parents=True)
            shutil.copyfile(SOURCE / "runlog.py", source / "runlog.py")
            payloads = {
                "results/smooth/environment.json": '{"threads":2}\n',
                "results/numerical_summary.json": '{"selection_count":24}\n',
            }
            (case / "report.py").write_text(
                "from pathlib import Path\n"
                f"for name, text in {payloads!r}.items():\n"
                "    path = Path(name)\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_text(text)\n")
            step = ("case/report.py",)
            summaries = {step: ("case/results/numerical_summary.json",)}
            environment = {key: value for key, value in os.environ.items() if key not in HANDOFF}
            with patch.object(runner, "HERE", source), patch.object(runner, "KERNEL_STEPS", (step,)), \
                    patch.object(runner, "NUMERICAL_SUMMARIES", summaries), \
                    patch.dict(os.environ, environment, clear=True), patch.object(sys, "path", sys.path.copy()):
                actual = runner.execute(root / "models", smoke=True)

            status = json.loads((actual / "run_status.json").read_text())
            self.assertTrue(status["passed"])
            logs = Path(status["log_dir"])
            events = [json.loads(line)["record"]["extra"]
                      for line in (logs / "events.jsonl").read_text().splitlines()]
            reports = [event for event in events
                       if event.get("event") in {"kernel_environment", "numerical_summary"}]
            self.assertEqual({event["event"] for event in reports}, {"kernel_environment", "numerical_summary"})
            self.assertEqual(len(reports), 2)
            for event in reports:
                payload = payloads[event["report"].removeprefix("case/")]
                self.assertEqual(event["stage"], step[0])
                self.assertEqual(event["sha256"], hashlib.sha256(payload.encode()).hexdigest())
                self.assertEqual((logs / event["snapshot"]).read_text(), payload)

    def test_noop_verifier_fails_the_simulation(self):
        with tempfile.TemporaryDirectory(prefix="acoustic-simulation-noop-") as temporary:
            root = Path(temporary)
            source = root / "simulation"
            case = source / "case"
            case.mkdir(parents=True)
            shutil.copyfile(SOURCE / "runlog.py", source / "runlog.py")
            (case / "verify.py").write_text(
                "from pathlib import Path\n"
                "def main():\n"
                "    Path('results/verification.json').write_text('{\"passed\":true}')\n")
            step = ("case/verify.py",)
            contracts = {step: ({"path": "case/results/verification.json", "equals": {"passed": True}},)}
            environment = {key: value for key, value in os.environ.items() if key not in HANDOFF}
            with patch.object(runner, "HERE", source), patch.object(runner, "KERNEL_STEPS", (step,)), \
                    patch.object(runner, "VERIFICATION_REPORTS", contracts), \
                    patch.dict(os.environ, environment, clear=True), patch.object(sys, "path", sys.path.copy()):
                with self.assertRaisesRegex(RuntimeError, "Verification report"):
                    runner.execute(root / "models", smoke=True)

            actual, = (root / "models").iterdir()
            status = json.loads((actual / "run_status.json").read_text())
            self.assertFalse(status["passed"])
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stages_completed"], 0)
            summary = json.loads((Path(status["log_dir"]) / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["stages"][0]["exit_code"], 0)
            self.assertEqual(summary["stages"][0]["status"], "failed")
            self.assertEqual(summary["verification_reports"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
