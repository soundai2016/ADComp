#!/usr/bin/env python3
"""Summarize receiver and continuous-field errors for the impedance interface."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import csv
import json
from pathlib import Path
import numpy as np
from kernel import (CFG, exact_mass_weighted_pressure, exact_pressure_norm_squared,
                    cell_projection_norm_squared)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available & set(CFG["cpu_affinity"]) or available)
    records = [json.loads(path.read_text()) for path in sorted(OUT.glob("aligned_impedance_step_*.json"))]
    records.sort(key=lambda a: (a["grid"], a["method"], a.get("modes_per_patch", 0)))
    times = np.linspace(0., CFG["time_end"], CFG["samples"])
    rows = []
    for record in records:
        n = record["grid"]
        for end in (2, 8):
            tt = times[times <= end + 1e-12]
            norm2 = exact_pressure_norm_squared(tt)
            projected_norm2 = cell_projection_norm_squared(n, tt)
            floor = float(np.sqrt(np.sum(norm2 - projected_norm2) / np.sum(norm2)))
            rows.append({"grid": n, "method": record["method"], "modes_per_patch": record.get("modes_per_patch", ""),
                "dimension": record["dimension"], "time_end": end,
                "receiver_total_error": record["receiver_total_error"][f"T{end}"]["relative_error"],
                "continuous_pressure_total_error": record["pressure_total_error"][f"T{end}"]["relative_error"],
                "receiver_same_grid_error": record.get("receiver_same_grid_error", {}).get(f"T{end}", {}).get("relative_error", ""),
                "pressure_same_grid_error": record.get("pressure_same_grid_error", {}).get(f"T{end}", {}).get("relative_error", ""),
                "pressure_cell_projection_floor": floor,
                "receiver_error_vector_cosine": record.get("receiver_error_vector_cosine", "")})
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"configuration": CFG, "settings": len(records), "rows": rows,
        "interpretation": "The receiver reference is the continuous Gaussian observation. The field norm includes the exact within-cell projection error. Equal wave speed and a centred planar interface allow exact continuum modes and a complete separated full-order solver; these assumptions are specific to this benchmark."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Summarized {len(records)} settings into {len(rows)} time-window rows.")


if __name__ == "__main__":
    main()
