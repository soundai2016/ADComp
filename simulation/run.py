#!/usr/bin/env python3
"""Compute the acoustic studies in a fresh model-run directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4

HERE = Path(__file__).resolve().parent
KERNEL_STEPS = (
    ("propagation/run_propagation.py", "--verify"),
    ("accuracy_cost/verify.py",),
    ("main_residual/verify.py",),
    ("interface_case/verify.py",),
    ("total_error_cost/verify.py",),
    ("total_error_cost/verify_interface.py",),
    ("unequal_speed_interface/verify.py",),
)
FULL_STEPS = (
    *KERNEL_STEPS,
    ("studies/stage1_1d/run_experiment.py",),
    ("studies/stage1_1d/refinement_diagnostic.py",),
    ("studies/stage1_1d/direct_baseline_check.py",),
    ("studies/stage2_2d/run_2d.py",),
    ("studies/stage2_2d/capacity_and_sampling.py",),
    ("studies/stage2_2d/energy_defect_certificate.py",),
    ("studies/patch_2d/run_patch.py",),
    ("studies/patch_2d/capacity_followup.py",),
    ("studies/patch_2d/audit_patch.py",),
    ("propagation/assemble_reduced_responses.py",),
    ("dispersion/run_dispersion.py",),
    ("propagation/run_propagation.py",),
    ("propagation/reference_checks.py",),
    ("propagation/chebyshev_reference.py", "--grids", "384", "--materials", "rotated_stripes", "--check-suffix"),
    ("propagation/analyze_propagation.py",),
    ("propagation/analyze_asymptotic.py",),
    ("propagation/analyze_reduction.py", "--comparison-grid", "512"),
    ("accuracy_cost/run.py",),
    ("accuracy_cost/analyze.py",),
    ("modal_diagnostics/run.py",),
    ("modal_diagnostics/evaluate_reference.py",),
    ("modal_diagnostics/export_modes.py",),
    ("interface_case/run.py",),
    ("interface_case/analyze.py",),
    ("main_residual/run.py",),
    ("main_residual/run_interface.py",),
    ("main_residual/analyze.py",),
    ("total_error_cost/run.py",),
    ("total_error_cost/run_interface.py",),
    ("total_error_cost/analyze.py",),
    ("total_error_cost/analyze_interface.py",),
    ("observation_diagnostics/analyze.py",),
    ("observation_diagnostics/verify.py",),
    ("grid_cost_diagnostics/verify.py",),
    ("grid_cost_diagnostics/run.py",),
    ("grid_cost_diagnostics/analyze.py",),
    ("grid_cost_diagnostics/reference_sensitivity.py",),
    ("modal_robustness/run.py",),
    ("modal_robustness/verify.py",),
    ("unequal_speed_interface/run.py",),
    ("unequal_speed_interface/modal.py",),
    ("unequal_speed_interface/analyze.py",),
    ("source_receiver_validation/run.py",),
    ("source_receiver_validation/analyze.py",),
    ("source_receiver_validation/verify.py",),
    ("frequency_diagnostics/run.py",),
    ("frequency_diagnostics/richardson.py",),
    ("propagation/run_propagation.py", "--verify"),
    ("propagation/assemble_reduced_responses.py", "--verify"),
    ("studies/verify_outputs.py",),
    ("propagation/verify_reference.py",),
    ("accuracy_cost/verify_outputs.py",),
    ("modal_diagnostics/verify.py",),
    ("total_error_cost/verify_outputs.py",),
    ("grid_cost_diagnostics/verify.py", "--results"),
    ("grid_cost_diagnostics/reference_sensitivity.py", "--verify"),
    ("unequal_speed_interface/verify_outputs.py",),
)


def check_report(path, **equals):
    return {"path": path, "equals": equals}


VERIFICATION_REPORTS = {
    ("propagation/run_propagation.py", "--verify"): (
        check_report("propagation/results/small_grid_verification.json", status="pass"),),
    ("accuracy_cost/verify.py",): (
        check_report("accuracy_cost/results/small_grid_verification.json", status="pass"),),
    ("main_residual/verify.py",): ({
        **check_report("main_residual/results/verification.json", all_small_system_bounds_hold=True),
        "below": {"analytic_vs_original_gram_relative_difference": 1e-13,
                  "analytic_vs_gauss_legendre_gram_relative_difference": 1e-10,
                  "full_state_sparse_vs_modal_relative_difference": 1e-10,
                  "modal_defect_identity_relative_difference": 1e-10}},),
    ("interface_case/verify.py",): (check_report("interface_case/results/verification.json", passed=True),),
    ("total_error_cost/verify.py",): (check_report("total_error_cost/results/verification.json", status="pass"),),
    ("total_error_cost/verify_interface.py",): (
        check_report("total_error_cost/results/interface_verification.json", status="pass"),),
    ("unequal_speed_interface/verify.py",): (
        check_report("unequal_speed_interface/results/verification.json", passed=True),),
    ("propagation/assemble_reduced_responses.py",): (
        check_report("propagation/results/capacity_trace_linkage.json", passed=True),),
    ("propagation/assemble_reduced_responses.py", "--verify"): (
        check_report("propagation/results/capacity_trace_linkage.json", passed=True),),
    ("propagation/reference_checks.py",): (
        check_report("propagation/results/reference_checks.json", passed=True),),
    ("observation_diagnostics/verify.py",): (
        check_report("observation_diagnostics/results/verification.json", passed=True),),
    ("grid_cost_diagnostics/verify.py",): (
        check_report("grid_cost_diagnostics/results/preflight.json", status="pass"),),
    ("grid_cost_diagnostics/verify.py", "--results"): (
        check_report("grid_cost_diagnostics/results/output_verification.json", status="pass"),),
    ("grid_cost_diagnostics/reference_sensitivity.py",): (
        check_report("grid_cost_diagnostics/results/richardson_verification.json", status="pass"),),
    ("grid_cost_diagnostics/reference_sensitivity.py", "--verify"): (
        check_report("grid_cost_diagnostics/results/richardson_verification.json", status="pass"),),
    ("modal_robustness/verify.py",): (
        {"path": "modal_robustness/results/summary.json", "all_true": ["verification"]},),
    ("source_receiver_validation/verify.py",): (
        check_report("source_receiver_validation/results/verification.json", passed=True),),
    ("frequency_diagnostics/run.py",): (
        check_report("frequency_diagnostics/results/verification.json", passed=True),),
    ("frequency_diagnostics/richardson.py",): (
        check_report("frequency_diagnostics/results/richardson_verification.json", passed=True),),
    ("studies/verify_outputs.py",): (check_report("studies/checks/verification_recomputed.json", passed=True),),
    ("propagation/verify_reference.py",): (
        check_report("propagation/results/reference_verification.json", status="pass", numerical_checks_pass=True),),
    ("accuracy_cost/verify_outputs.py",): (
        check_report("accuracy_cost/results/data_integrity.json", status="pass"),),
    ("modal_diagnostics/verify.py",): (check_report("modal_diagnostics/results/verification.json", passed=True),),
    ("total_error_cost/verify_outputs.py",): (
        check_report("total_error_cost/results/output_verification.json", status="pass"),),
    ("unequal_speed_interface/verify_outputs.py",): (
        check_report("unequal_speed_interface/results/output_verification.json", passed=True),),
}
NUMERICAL_SUMMARIES = {
    ("grid_cost_diagnostics/reference_sensitivity.py",): (
        "grid_cost_diagnostics/results/richardson_summary.json",),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment_signature(path):
    if not path.is_file():
        return None
    return path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def default_models_dir():
    return HERE.parent if (HERE / "runlog.py").is_file() else HERE.parent / "models"


def validate_models_dir(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.name or directory == HERE or HERE in directory.parents:
        raise ValueError("Choose a model base outside the simulation source directory.")
    if not (HERE / "runlog.py").is_file():
        source = HERE.parent
        protected = [source / name for name in ("data", "analysis", "figures", "simulation")]
        if directory == source or any(directory == path or path in directory.parents for path in protected):
            raise ValueError("Choose a model base separate from the source files and bundled data.")
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"Model base is not a directory: {directory}")
    return directory


def select_workdir(directory, run_id=None):
    """Select a new child of the model base without creating files."""
    directory = validate_models_dir(directory)
    if run_id is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id) is None or ".." in run_id:
            raise ValueError("Invalid model run identifier.")
        candidate = directory / run_id
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"Model run already exists: {candidate}")
        return candidate
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = directory / f"{stamp}-{uuid4().hex[:8]}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate


def resolve_workdir(requested, run_id=None):
    requested = Path(requested or default_models_dir()).expanduser().resolve()
    managed = bool(os.environ.get("ACOUSTIC_RUN_ID")) and os.environ.get("ACOUSTIC_WORKDIR") == str(requested)
    if managed:
        directory = validate_models_dir(os.environ.get("ACOUSTIC_REQUESTED_WORKDIR", requested.parent))
        if requested.parent != directory:
            raise ValueError("The managed model run must be a direct child of its base directory.")
        return directory, requested
    directory = validate_models_dir(requested)
    return directory, select_workdir(directory, run_id)


def prepare(workdir):
    if workdir == HERE or HERE in workdir.parents:
        raise ValueError("Choose a working directory outside simulation/.")
    workdir.mkdir(parents=True, exist_ok=False)
    source_files = [path for path in HERE.rglob("*")
                    if path.is_file() and path.suffix in {".py", ".json"}
                    and (path.suffix == ".py" or path.name in {
                        "config.json", "layout_config.json", "evaluation_config.json",
                        "experiment_config.json", "protocol.json"})
                    and not any(part in {"results", "cache", "checks", "__pycache__", "logs", "models"}
                                for part in path.relative_to(HERE).parts)]
    for source in source_files:
        destination = workdir / source.relative_to(HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    recording = HERE / "runlog.py"
    if not recording.is_file():
        recording = HERE.parent / "runlog.py"
    shutil.copyfile(recording, workdir / "runlog.py")
    for directory in sorted({path.parent for path in workdir.rglob("*.py")}):
        (directory / "results").mkdir(exist_ok=True)
    (workdir / "studies/checks").mkdir(parents=True, exist_ok=True)
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    adjustments = []
    for path in sorted(workdir.rglob("*.json")):
        config = json.loads(path.read_text())
        changed = False
        for key, requested in config.items():
            if key.endswith("cpu_affinity") and requested and available and not set(requested).issubset(available):
                actual = available[:len(requested)]
                adjustments.append({"file": path.relative_to(workdir).as_posix(), "key": key,
                                    "requested": requested, "actual": actual})
                config[key] = actual
                changed = True
        if changed:
            write_json(path, config)
    return adjustments


def execute(workdir=None, smoke=False):
    """Compute the selected stages and retain their run records."""
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or later is required.")
    if not smoke and not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("The complete timing suite requires Linux CPU affinity support.")
    runtime = HERE if (HERE / "runlog.py").is_file() else HERE.parent
    sys.path.insert(0, str(runtime))
    from runlog import RunLog, atomic_json, new_run_id

    run_id = new_run_id("simulation")
    models_dir, workdir = resolve_workdir(workdir, run_id)
    adjustments = prepare(workdir)
    steps = KERNEL_STEPS if smoke else FULL_STEPS
    env = os.environ.copy()
    env.update({name: "2" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    parent_run_id = env.get("ACOUSTIC_RUN_ID")
    parent_logs = env.get("ACOUSTIC_LOG_DIR") if parent_run_id else None
    log_dir = Path(parent_logs) / "simulation" if parent_logs else workdir / "logs" / run_id
    mode = "smoke" if smoke else "full"
    run = RunLog(log_dir, mode=f"simulation_{mode}", command=[sys.executable, *sys.argv],
                 source_root=workdir, run_id=run_id, parent_run_id=parent_run_id,
                 configuration={"steps": steps, "models_dir": str(models_dir), "workdir": str(workdir)},
                 environment=env)

    def status():
        atomic_json(workdir / "run_status.json", {
            "run_id": run_id, "mode": mode, "status": run.record["status"],
            "passed": run.record["status"] == "completed", "log_dir": str(log_dir),
            "requested_workdir": str(models_dir), "models_dir": str(models_dir), "workdir": str(workdir),
            "stages_completed": sum(stage["status"] == "completed" for stage in run.record["stages"]),
            "stages_total": len(steps),
        })

    try:
        with run:
            status()
            run.event("workdir_selected", requested_workdir=str(models_dir), models_dir=str(models_dir), workdir=str(workdir))
            run.logger.info("Model run: {}", workdir)
            run.event("cpu_affinity_adjustments", adjustments=adjustments)
            for step in steps:
                script, *arguments = step
                target = workdir / script
                results = target.parent / "results"
                previous_reports = {path: environment_signature(path)
                                    for pattern in ("environment.json", "*/environment.json") for path in results.glob(pattern)}
                summaries = [workdir / path for path in NUMERICAL_SUMMARIES.get(tuple(step), ())]
                previous_summaries = {path: environment_signature(path) for path in summaries}
                checks = [dict(specification, path=workdir / specification["path"])
                          for specification in VERIFICATION_REPORTS.get(tuple(step), ())]
                entry = run.run_stage(script, [sys.executable, str(target), *arguments],
                                      cwd=target.parent, env=env, verification_reports=checks)
                for pattern in ("environment.json", "*/environment.json"):
                    for reported in sorted(results.glob(pattern)):
                        current = environment_signature(reported)
                        if current == previous_reports.get(reported):
                            continue
                        name = "_".join(reported.relative_to(workdir).parts[:-1])
                        snapshot = run.log_dir / "environments" / f"{entry['stage_id']:02d}_{name}.json"
                        snapshot.parent.mkdir(exist_ok=True)
                        shutil.copyfile(reported, snapshot)
                        run.event("kernel_environment", stage=script,
                                  report=reported.relative_to(workdir).as_posix(),
                                  snapshot=snapshot.relative_to(run.log_dir).as_posix(), sha256=current[1])
                for reported in summaries:
                    current = environment_signature(reported)
                    if current is None or current == previous_summaries[reported]:
                        raise RuntimeError(f"Numerical summary was not produced by {script}: {reported}")
                    json.loads(reported.read_text())
                    snapshot = run.log_dir / "numerical_summaries" / f"{entry['stage_id']:02d}_{reported.name}"
                    snapshot.parent.mkdir(exist_ok=True)
                    shutil.copyfile(reported, snapshot)
                    run.event("numerical_summary", stage=script, report=reported.relative_to(workdir).as_posix(),
                              snapshot=snapshot.relative_to(run.log_dir).as_posix(), sha256=current[1])
                status()
    finally:
        status()
    return workdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path,
                        help="Base for fresh model runs; defaults to sourcecode/models.")
    parser.add_argument("--plan", action="store_true", help="List commands without creating files or computing.")
    parser.add_argument("--smoke", action="store_true", help="Run seven independent small-system kernel checks.")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.plan:
        steps = KERNEL_STEPS if args.smoke else FULL_STEPS
        try:
            models_dir, workdir = resolve_workdir(args.workdir)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps({"mode": "smoke" if args.smoke else "full", "models_dir": str(models_dir),
                          "workdir": str(workdir), "steps": [["python", *step] for step in steps]}, indent=2))
        return
    execute(args.workdir, args.smoke)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode >= 0 else 128 - error.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)
