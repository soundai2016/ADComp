#!/usr/bin/env python3
"""Summarize unequal-speed interface errors and record numerical outputs."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from unequal_speed_interface.kernel import CFG
OUT = HERE/"results"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_summaries():
    """Recompute window-specific error cosines from saved receiver trajectories."""
    reference = np.load(OUT/"continuous_reference.npz")
    times, yf = reference["time"], reference["receiver_pressure"]
    rows = []
    for path in sorted(OUT.glob("n*_*.json")):
        record = json.loads(path.read_text())
        if record["method"] != "full_separable":
            raw = np.load(path.with_suffix(".npz"))["receiver_pressure"]
            full = np.load(OUT/f"n{record['grid']}_full_receiver.npz")["receiver_pressure"]
            record["receiver_reduction_grid_error_cosine"] = {}
            for end in (2,8):
                active = times <= end+1e-12
                local, grid = raw[active]-full[active], full[active]-yf[active]
                record["receiver_reduction_grid_error_cosine"][f"T{end}"] = float(np.sum(local*grid)/max(np.linalg.norm(local)*np.linalg.norm(grid), 1e-300))
            path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        for end in (2,8):
            window = f"T{end}"
            rows.append({"grid": record["grid"], "method": record["method"],
                "modes_per_patch": record.get("modes_per_patch", ""), "dimension": record["dimension"], "time_end": end,
                "receiver_total_error": record["receiver_total_error"][window]["relative_error"],
                "continuous_pressure_total_error": record["pressure_total_error"][window]["relative_error"],
                "receiver_same_grid_error": record.get("receiver_same_grid_error", {}).get(window, {}).get("relative_error", ""),
                "pressure_same_grid_error": record.get("pressure_same_grid_error", {}).get(window, {}).get("relative_error", ""),
                "pressure_cell_projection_floor": record["pressure_cell_projection_floor"][window]["relative_error"],
                "receiver_error_vector_cosine": record.get("receiver_reduction_grid_error_cosine", {}).get(window, "")})
    write_csv(OUT/"summary.csv", rows)
    (OUT/"summary.json").write_text(json.dumps({"configuration": CFG, "rows": rows}, indent=2, allow_nan=False)+"\n")
    return rows


def main():
    rows = refresh_summaries()
    manifest = {
        "run_scope": "Three grids, nine CB models, two n=32 RK controls; modal attribution at n=32.",
        "reference_scope": CFG["initial_condition"],
        "verification_passed": json.loads((OUT/"verification.json").read_text())["passed"],
        "modal_validation_passed": json.loads((OUT/"modal_validation.json").read_text())["passed"],
        "source_hashes": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in list(HERE.glob("*.py")) + [HERE/"config.json"]},
    }
    paths = list(OUT.glob("n*.npz")) + list(OUT.glob("n*.json")) + list(OUT.glob("*.csv")) + [
        OUT/"continuous_reference.npz", OUT/"continuous_catalogue.npz", OUT/"modal_summary.json",
        OUT/"modal_validation.json", OUT/"summary.json", OUT/"verification.json"]
    paths += list(OUT.glob("*_decomposition.npz"))
    manifest["artifact_hashes"] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in paths}
    (OUT/"output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"summary_rows": len(rows), "numeric_outputs": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
