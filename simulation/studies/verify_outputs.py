#!/usr/bin/env python3
"""Check freshly computed study metrics against the generated arrays."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    with np.load(ROOT / "patch_2d/receiver_and_transfer_responses.npz") as reference, np.load(ROOT / "patch_2d/capacity_followup_responses.npz") as reduced:
        with (ROOT / "patch_2d/capacity_followup_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        differences = []
        active = reference["time"] > 0
        for row in rows:
            prefix = row["material"] + "_" + row["method"] + "_r" + row["modes_per_patch"]
            pairs = (("same_grid_receiver_error", reduced[prefix + "_y"][active], reference[row["material"] + "_n32_full_y"][active]),
                     ("same_grid_transfer_error", reduced[prefix + "_H"], reference[row["material"] + "_n32_full_H"]),
                     ("receiver_error_vs_48", reduced[prefix + "_y"][active], reference[row["material"] + "_n48_full_y"][active]))
            for key, value, target in pairs:
                error = float(np.linalg.norm(value - target) / np.linalg.norm(target))
                differences.append(abs(error - float(row[key])))
    with (ROOT / "stage2_2d/energy_certificate_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    margins = [float(row["guarded_certificate_relative"]) - float(row["actual_relative_full_energy_error"]) for row in rows]
    passed = bool(differences and margins and max(differences) < 2e-6 and min(margins) >= 0)
    result = {"passed": passed, "mode": "fresh-computation", "capacity_metrics_checked": len(differences), "maximum_metric_difference": max(differences),
              "residual_rows_checked": len(margins), "minimum_residual_margin": min(margins)}
    destination = ROOT / "checks/verification_recomputed.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Recomputed supporting-study metrics failed validation.")


if __name__ == "__main__":
    main()
