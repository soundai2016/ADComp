"""Check fresh work directories and full-pipeline path propagation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]


def load_runner(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


simulation = load_runner("workdir_simulation_runner", SOURCE / "simulation" / "run.py")
reproduction = load_runner("workdir_reproduction_runner", SOURCE / "reproduce.py")


def files(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def events(log_dir):
    return [json.loads(line)["record"]["extra"]
            for line in (log_dir / "events.jsonl").read_text().splitlines()]


class WorkdirTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="acoustic-workdir-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "sourcecode"
        self.simulation = self.source / "simulation"
        case = self.simulation / "case"
        case.mkdir(parents=True)
        shutil.copyfile(SOURCE / "runlog.py", self.source / "runlog.py")
        self.steps = (("case/compute.py",),)
        (case / "compute.py").write_text(
            "from pathlib import Path\n"
            "result = Path('results/fresh_result.txt')\n"
            "assert not result.exists(), 'Expected a fresh computation directory'\n"
            "result.write_text('computed\\n')\n")
        (self.simulation / "run.py").write_text(
            "import importlib.util\nfrom pathlib import Path\n"
            f"spec = importlib.util.spec_from_file_location('fixture_simulation', {str(SOURCE / 'simulation' / 'run.py')!r})\n"
            "runner = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(runner)\n"
            "runner.HERE = Path(__file__).resolve().parent\n"
            "runner.KERNEL_STEPS = runner.FULL_STEPS = (('case/compute.py',),)\n"
            "if __name__ == '__main__':\n    runner.main()\n")
        analysis, figures = self.source / "analysis", self.source / "figures"
        analysis.mkdir()
        figures.mkdir()
        (analysis / "core.py").write_text("GRIDS = (4,)\n")
        downstream = (
            "import argparse,json\nfrom pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "for name in ('simulation-root', 'output-dir', 'data-dir', 'analysis-root'):\n"
            "    parser.add_argument('--' + name, type=Path)\n"
            "args = parser.parse_args()\n"
            "if args.simulation_root is not None:\n"
            "    result = args.simulation_root / 'case/results/fresh_result.txt'\n"
            "    assert result.read_text() == 'computed\\n', str(result)\n"
            "args.output_dir.mkdir(parents=True, exist_ok=True)\n"
            "(args.output_dir / 'used_workdir.txt').write_text(str(args.simulation_root))\n"
            "if Path(__file__).name == 'run.py':\n"
            "    (args.output_dir / 'environment.json').write_text(json.dumps({'fixture': True}))\n"
            "    (args.output_dir / 'verification.json').write_text(json.dumps({'passed': True}))\n"
            "if Path(__file__).name == 'selection_audit.py':\n"
            "    (args.output_dir / 'selection_verification.json').write_text(json.dumps({'passed': True}))\n"
            "if Path(__file__).name == 'tables.py':\n"
            "    origin = 'simulation' if args.simulation_root else 'bundled'\n"
            "    (args.output_dir / 'verification.json').write_text(json.dumps({'table_count': 13, 'input_source': origin}))\n")
        for name in ("extract_inputs.py", "run.py", "selection_audit.py"):
            (analysis / name).write_text(downstream)
        for name in ("prepare.py", "render.py", "tables.py"):
            (figures / name).write_text(downstream)
        handoff = {"ACOUSTIC_RUN_ID", "ACOUSTIC_LOG_DIR", "ACOUSTIC_WORKDIR", "ACOUSTIC_REQUESTED_WORKDIR"}
        self.environment = {key: value for key, value in os.environ.items() if key not in handoff}

    def execute(self, requested=None):
        with (
            patch.object(simulation, "HERE", self.simulation),
            patch.object(simulation, "KERNEL_STEPS", self.steps),
            patch.dict(os.environ, self.environment, clear=True),
            patch.object(sys, "path", [str(SOURCE), *sys.path]),
        ):
            return simulation.execute(requested, smoke=True)

    def reproduce(self, requested, *extra):
        arguments = ["reproduce.py", "--mode", "full", "--output-dir", str(self.root / "output"), *extra]
        if requested is not None:
            arguments.extend(["--workdir", str(requested)])
        with (
            patch.object(reproduction, "HERE", self.source),
            patch.object(reproduction, "record_metrics"),
            patch.dict(os.environ, self.environment, clear=True),
            patch.object(sys, "path", [str(SOURCE), *sys.path]),
            patch.object(sys, "argv", arguments),
        ):
            reproduction.main()

    def test_absent_and_empty_bases_get_fresh_run_children(self):
        for name, existing in (("absent", False), ("empty", True)):
            with self.subTest(name=name):
                requested = self.root / name
                if existing:
                    requested.mkdir()
                actual = self.execute(requested)
                self.assertEqual(actual.parent, requested)
                status = json.loads((actual / "run_status.json").read_text())
                self.assertTrue(status["passed"])
                self.assertEqual(actual.name, status["run_id"])
                self.assertEqual(status["models_dir"], str(requested))
                self.assertEqual(status["workdir"], str(actual))
                self.assertEqual(status["requested_workdir"], str(requested))
                self.assertFalse((actual / ".prepare.lock").exists())

    def test_repeated_standalone_runs_preserve_existing_contents(self):
        requested = self.root / "models"
        first = self.execute(requested)
        original = files(first)
        second = self.execute(requested)
        third = self.execute(requested)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(files(first), original)
        for actual in (second, third):
            self.assertEqual(actual.parent, requested)
            status = json.loads((actual / "run_status.json").read_text())
            self.assertTrue(status["passed"])
            self.assertEqual(status["requested_workdir"], str(requested))
            selected = [event for event in events(Path(status["log_dir"]))
                        if event.get("event") == "workdir_selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["workdir"], str(actual))
            self.assertEqual(selected[0]["requested_workdir"], str(requested))

    def test_full_pipeline_uses_the_selected_directory(self):
        requested = self.source / "models"
        (requested / "old_run/case/results").mkdir(parents=True)
        (requested / "old_run/case/results/fresh_result.txt").write_text("old result\n")
        (requested / "old_run/config.json").write_text("historical data, not a source configuration\n")
        (requested / "keep.txt").write_text("existing contents\n")
        original = files(requested)
        self.reproduce(None)
        logs = list((self.root / "output/logs").glob("*/summary.json"))
        self.assertEqual(len(logs), 1)
        summary = json.loads(logs[0].read_text())
        self.assertEqual(summary["status"], "completed")
        first_command = summary["stages"][0]["command"]
        actual = Path(first_command[first_command.index("--workdir") + 1])
        self.assertNotEqual(actual, requested)
        self.assertEqual(actual.parent, requested)
        self.assertEqual(actual.name, summary["run_id"])
        downstream = [stage["command"] for stage in summary["stages"]
                      if "--simulation-root" in stage["command"]]
        self.assertEqual(len(downstream), 5)
        self.assertTrue(all(Path(command[command.index("--simulation-root") + 1]) == actual
                            for command in downstream))
        for name, data in original.items():
            self.assertEqual((requested / name).read_bytes(), data)
        child = json.loads((actual / "run_status.json").read_text())
        self.assertEqual(child["workdir"], str(actual))
        self.assertEqual(child["requested_workdir"], str(requested))
        manifest = json.loads((logs[0].parent / "manifest.json").read_text())
        self.assertEqual(manifest["configuration"]["workdir"], str(actual))
        self.assertFalse(any(name.startswith("models/") for name in manifest["source"]))
        self.assertEqual(len(summary["verification_reports"]), 3)
        self.assertTrue(all(report["status"] == "passed" for report in summary["verification_reports"]))
        for log_dir in (logs[0].parent, Path(child["log_dir"])):
            selected = [event for event in events(log_dir) if event.get("event") == "workdir_selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["requested_workdir"], str(requested))
            self.assertEqual(selected[0]["workdir"], str(actual))

    def test_selection_and_plan_do_not_create_files(self):
        requested = self.root / "models"
        requested.mkdir()
        (requested / "keep.txt").write_text("keep\n")
        before = files(self.root)
        before_directories = {path.relative_to(self.root) for path in self.root.rglob("*") if path.is_dir()}
        with patch.object(simulation, "HERE", self.simulation):
            selected = simulation.select_workdir(requested)
        self.assertNotEqual(selected, requested)
        self.assertEqual(selected.parent, requested)
        self.assertFalse(selected.exists())
        planned = []
        original_run = subprocess.run

        def capture_plan(command, **kwargs):
            result = original_run(command, **kwargs, text=True, capture_output=True)
            planned.append((kwargs["env"], json.loads(result.stdout)))
            return result

        with patch.object(reproduction.subprocess, "run", side_effect=capture_plan):
            self.reproduce(requested, "--plan")
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0][1]["workdir"], planned[0][0]["ACOUSTIC_WORKDIR"])
        self.assertEqual(Path(planned[0][1]["workdir"]).parent, requested)
        self.assertEqual(files(self.root), before)
        self.assertEqual({path.relative_to(self.root) for path in self.root.rglob("*") if path.is_dir()}, before_directories)
        self.assertFalse((self.root / "output").exists())

    def test_historical_run_metadata_is_not_a_replay_input(self):
        for name in ("analysis", "figures", "tables", "run"):
            directory = self.source / "data" / name
            directory.mkdir(parents=True)
            (directory / "input.json").write_text('{"value":1}\n')
        self.reproduce(None, "--mode", "data")
        history = self.source / "data/run/input.json"
        history.write_text('{"hardware":"another CPU","timestamp":"later"}\n')
        self.reproduce(None, "--mode", "data")
        manifests = [json.loads(path.read_text())
                     for path in (self.root / "output/logs").glob("*/manifest.json")]
        self.assertEqual(len(manifests), 2)
        expected = {f"data/{name}/input.json" for name in ("analysis", "figures", "tables")}
        for manifest in manifests:
            self.assertEqual(set(manifest["inputs"]), expected)
        self.assertEqual(manifests[0]["configuration_fingerprint"], manifests[1]["configuration_fingerprint"])
        self.assertFalse((self.source / "models").exists())

    def test_copied_runner_keeps_the_same_default_base(self):
        first = self.execute()
        self.assertEqual(first.parent, self.source / "models")
        original = files(first)
        subprocess.run([sys.executable, "-B", str(first / "run.py"), "--smoke"],
                       env=dict(self.environment, PYTHONDONTWRITEBYTECODE="1"),
                       check=True, text=True, capture_output=True)
        runs = sorted(path for path in first.parent.iterdir() if path.is_dir())
        self.assertEqual(len(runs), 2)
        self.assertEqual(files(first), original)
        second = next(path for path in runs if path != first)
        self.assertTrue(json.loads((second / "run_status.json").read_text())["passed"])
        self.assertFalse((first / "models").exists())

    def test_files_and_source_directories_are_rejected(self):
        regular_file = self.root / "regular_file"
        regular_file.write_text("keep\n")
        before = files(self.root)
        with patch.object(simulation, "HERE", self.simulation):
            for requested in (regular_file, self.simulation, self.simulation / "case"):
                with self.subTest(path=requested), self.assertRaises((ValueError, NotADirectoryError)):
                    simulation.select_workdir(requested)
        self.assertEqual(files(self.root), before)

    def test_existing_run_ids_and_nested_output_directories_are_rejected(self):
        base = self.root / "models"
        (base / "existing_run").mkdir(parents=True)
        (base / "existing_run/keep.txt").write_text("keep\n")
        before = files(self.root)
        with patch.object(simulation, "HERE", self.simulation):
            with self.assertRaises(FileExistsError):
                simulation.select_workdir(base, "existing_run")
            for name in ("../escape", "bad/name"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    simulation.select_workdir(base, name)
        with self.assertRaises(SystemExit) as error:
            self.reproduce(base, "--output-dir", str(base / "outputs"))
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(files(self.root), before)


if __name__ == "__main__":
    unittest.main()
