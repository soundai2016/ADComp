#!/usr/bin/env python3
"""Check table coverage and intervention identities without rerunning propagation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    rows = list(csv.DictReader((HERE / "results" / "channel_errors.csv").open()))
    groups = list(csv.DictReader((HERE / "results" / "group_errors.csv").open()))
    assert len(rows) == 3 * 3 * 2 * 8 * 12 == summary["channel_rows"]
    assert len(groups) == 3 * 3 * 2 * 8 * 8 == summary["group_rows"]
    assert summary["config_sha256"] == digest(HERE / "config.json")
    assert summary["analysis_code_sha256"] == digest(HERE / "analyze.py")
    assert all(digest(HERE.parent / filename) == value for filename, value in summary["input_sha256"].items())
    assert summary["max_normalized_squared_error_identity_residual"] < 1e-12
    assert summary["max_relative_remove_ritz_trajectory_identity"] < 1e-12
    max_sum_residual = 0.0
    for group in groups:
        if group["group"] != "all":
            continue
        keys = ("material", "method", "modes_per_patch", "reference_grid", "window_kind", "start", "end")
        subset = [r for r in rows if all(r[k] == group[k] for k in keys)]
        assert len(subset) == 12
        np.testing.assert_allclose(sum(float(r["reference_energy_share"]) for r in subset), 1, atol=1e-14)
        for intervention in ("remove_ritz", "fine_frequency"):
            contribution = sum(float(r[f"{intervention}_squared_error_change_contribution"]) for r in subset)
            target = float(group[f"{intervention}_relative_error"])**2 - float(group["original_relative_error"])**2
            max_sum_residual = max(max_sum_residual, abs(contribution - target))
            assert abs(contribution - target) < 1e-12
    max_window_partition_residual = 0.0
    for full in rows:
        if full["window_kind"] != "cumulative" or float(full["end"]) != 8:
            continue
        keys = ("material", "method", "modes_per_patch", "reference_grid", "source", "receiver")
        parts = [r for r in rows if r["window_kind"] == "disjoint" and all(r[k] == full[k] for k in keys)]
        assert len(parts) == 4
        den = float(full["reference_energy"])
        np.testing.assert_allclose(sum(float(r["reference_energy"]) for r in parts), den, rtol=2e-14)
        assert sum(int(r["samples"]) for r in parts) == int(full["samples"])
        for metric in ("original_relative_error", "remove_ritz_relative_error", "fine_frequency_relative_error"):
            reconstructed = sum(float(r[metric]) ** 2 * float(r["reference_energy"]) for r in parts) / den
            difference = abs(reconstructed - float(full[metric]) ** 2)
            max_window_partition_residual = max(max_window_partition_residual, difference)
            assert difference < 1e-12
    previous = list(csv.DictReader((HERE.parent / "modal_diagnostics" / "results" / "reference_evaluation.csv").open()))
    for old in previous:
        new = next(r for r in groups if r["reference_grid"] == "512" and r["window_kind"] == "cumulative"
                   and float(r["end"]) == 8 and r["group"] == "all"
                   and r["material"] == old["material"] and r["method"] == old["method"]
                   and r["modes_per_patch"] == old["m"])
        for current, earlier in (("original_relative_error", "total_error_vs_512"),
                                 ("remove_ritz_relative_error", "error_without_ritz_shift_vs_512"),
                                 ("fine_frequency_relative_error", "error_with_fine_frequencies_vs_512")):
            np.testing.assert_allclose(float(new[current]), float(old[earlier]), rtol=2e-14, atol=2e-14)
    result = dict(passed=True, channel_rows=len(rows), group_rows=len(groups),
                  input_files_checked=len(summary["input_sha256"]), max_additive_contribution_residual=max_sum_residual,
                  max_disjoint_window_partition_residual=max_window_partition_residual,
                  previous_aggregate_comparisons_checked=3 * len(previous))
    (HERE / "results" / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
