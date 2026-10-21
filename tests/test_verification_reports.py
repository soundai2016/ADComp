"""Require fresh, successful reports from commands that exit normally."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runlog import RunLog


class VerificationReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="acoustic-verification-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "solver.py").write_text("order = 4\n")
        self.report = self.source / "results" / "verification.json"
        self.report.parent.mkdir()
        self.environment = dict(os.environ, OPENBLAS_NUM_THREADS="2")

    def record(self, name):
        return RunLog(self.root / name, mode="test", command=[sys.executable, "solver.py"],
                      source_root=self.source, environment=self.environment)

    def command(self, payload, timestamp=1600000000000000000):
        code = ("from pathlib import Path\nimport os\n"
                f"p = Path({str(self.report)!r})\np.write_text({payload!r})\n"
                f"os.utime(p, ns=({timestamp}, {timestamp}))\n")
        return [sys.executable, "-c", code]

    def test_identical_rewrites_are_fresh_and_archived_separately(self):
        payload = '{"passed":true,"defect":1e-12,"verification":{"identity":true}}\n'
        specification = {"path": self.report, "equals": {"passed": True},
                         "below": {"defect": 1e-10}, "all_true": ["verification"]}
        with self.record("fresh") as run:
            for index in range(2):
                run.run_stage(f"check_{index}", self.command(payload, 1600000000000000000 + index * 2000000000),
                              cwd=self.source, console=False, verification_reports=[specification])
        reports = run.record["verification_reports"]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["status"] == "passed" for report in reports))
        self.assertEqual(reports[0]["sha256"], reports[1]["sha256"])
        self.assertNotEqual(reports[0]["snapshot"], reports[1]["snapshot"])
        for report in reports:
            self.assertEqual((run.log_dir / report["snapshot"]).read_text(), payload)
        self.assertTrue(all(stage["verification_elapsed_seconds"] >= 0 for stage in run.record["stages"]))

    def test_missing_and_stale_reports_fail_a_zero_exit_stage(self):
        for name in ("missing", "stale"):
            with self.subTest(name=name):
                if name == "stale":
                    self.report.write_text('{"passed":true}\n')
                run = self.record(name)
                with self.assertRaises(RuntimeError):
                    with run:
                        run.run_stage("verifier", [sys.executable, "-c", "def main(): pass"],
                                      cwd=self.source, console=False,
                                      verification_reports=[{"path": self.report, "equals": {"passed": True}}])
                summary = json.loads((run.log_dir / "summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["stages"][0]["exit_code"], 0)
                self.assertEqual(summary["stages"][0]["status"], "failed")
                self.assertEqual(summary["verification_reports"][0]["status"], "failed")

    def test_failed_and_malformed_reports_are_preserved(self):
        cases = (
            ("false_flag", '{"passed":false}', {"equals": {"passed": True}}),
            ("numeric_flag", '{"passed":1}', {"equals": {"passed": True}}),
            ("malformed", '{', {"equals": {"passed": True}}),
            ("large_defect", '{"defect":1e-4}', {"below": {"defect": 1e-10}}),
            ("empty_checks", '{"verification":{}}', {"all_true": ["verification"]}),
            ("failed_check", '{"verification":{"identity":false}}', {"all_true": ["verification"]}),
        )
        for name, payload, checks in cases:
            with self.subTest(name=name):
                run = self.record(name)
                with self.assertRaises(RuntimeError):
                    with run:
                        run.run_stage("verifier", self.command(payload), cwd=self.source, console=False,
                                      verification_reports=[dict(checks, path=self.report)])
                report = run.record["verification_reports"][0]
                self.assertEqual(report["status"], "failed")
                self.assertEqual((run.log_dir / report["snapshot"]).read_text(), payload)


if __name__ == "__main__":
    unittest.main()
