#!/usr/bin/env python3
"""Tabulate measured costs at the prescribed acoustic accuracy targets."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CFG = json.loads((HERE / "config.json").read_text())
METHODS = ["craig_bampton", "rational_krylov"]
LABELS = {"craig_bampton": "Craig–Bampton", "rational_krylov": "Source-conditioned rational Krylov"}
COLORS = {"craig_bampton": "#16567a", "rational_krylov": "#c56d1a"}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    records = [json.loads(path.read_text()) for path in sorted(OUT.glob("*_timing.json"))]
    full = [r for r in records if r["method"] == "full_sparse_exponential"]
    reduced = [r for r in records if r["method"] in METHODS]
    assert len(full) == len(CFG["grids"]) * len(CFG["materials"])
    assert len(reduced) == len(full) * len(METHODS) * len(CFG["modes_per_patch"])
    full_by_case = {(r["grid"], r["material"]): r for r in full}
    rows = []
    for r in records:
        row = {k: r[k] for k in ["material", "grid", "method", "dimension"]}
        row.update(r["median"])
        row.update({key: r.get("errors", {}).get(key, 0.0) for key in CFG["accuracy_thresholds"]})
        row["threshold_pass"] = r.get("threshold_pass", True)
        row["total_speedup_both_outputs"] = full_by_case[(r["grid"], r["material"])]["median"]["both_total_seconds"] / r["median"]["both_total_seconds"]
        rows.append(row)
    with (OUT / "accuracy_cost.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregated = []
    for n in CFG["grids"]:
        for method in METHODS:
            for m in CFG["modes_per_patch"]:
                selected = [r for r in reduced if r["grid"] == n and r["method"] == method and r["modes_per_patch"] == m]
                entry = {"grid": n, "method": method, "modes_per_patch": m, "dimension": n + 2 * m,
                         "cases_meeting_both_thresholds": sum(r["threshold_pass"] for r in selected)}
                for key in CFG["accuracy_thresholds"]:
                    values = [r["errors"][key] for r in selected]
                    entry[key + "_mean"] = float(np.mean(values))
                    entry[key + "_min"] = min(values)
                    entry[key + "_max"] = max(values)
                for key in selected[0]["median"]:
                    entry[key + "_mean"] = float(np.mean([r["median"][key] for r in selected]))
                aggregated.append(entry)
    selections = []
    for baseline in full:
        for method in METHODS:
            candidates = [r for r in reduced if r["grid"] == baseline["grid"] and r["material"] == baseline["material"] and r["method"] == method and r["threshold_pass"]]
            best = min(candidates, key=lambda r: r["median"]["both_total_seconds"]) if candidates else None
            selections.append({"grid": baseline["grid"], "material": baseline["material"], "method": method,
                               "selected_dimension": best["dimension"] if best else None,
                               "both_total_seconds": best["median"]["both_total_seconds"] if best else None,
                               "full_both_total_seconds": baseline["median"]["both_total_seconds"],
                               "both_total_speedup": baseline["median"]["both_total_seconds"] / best["median"]["both_total_seconds"] if best else None,
                               "receiver_relative_error": best["errors"]["receiver_relative_error"] if best else None,
                               "mass_weighted_pressure_relative_error": best["errors"]["mass_weighted_pressure_relative_error"] if best else None})
    timing_variability = []
    for record in records:
        values = [sum(rep[key] for key in CFG["timing_stages"]) for rep in record["repetitions"]]
        timing_variability.append({"grid": record["grid"], "material": record["material"],
                                   "method": record["method"], "dimension": record["dimension"],
                                   "total_seconds_each_repetition": values, "minimum_seconds": min(values),
                                   "median_seconds": float(np.median(values)), "maximum_seconds": max(values),
                                   "range_over_median": (max(values) - min(values)) / np.median(values)})
    summary = {"number_of_full_settings": len(full), "number_of_reduced_settings": len(reduced),
               "repetitions": CFG["repetitions"], "thresholds": CFG["accuracy_thresholds"],
               "aggregation": "Arithmetic means of three material errors and of the per-material median times; thresholds applied to each individual material, not to the mean.",
               "source_conditioning": CFG["krylov_source_conditioning"],
               "aggregated": aggregated, "fastest_threshold_admissible_per_case": selections,
               "timing_variability": timing_variability,
               "structure": {"max_full_energy_relative_drift": max(r["structure"]["energy_relative_drift"] for r in full),
                             "max_reduced_orthonormality_frobenius": max(r["structure"]["orthonormality_frobenius"] for r in reduced),
                             "max_constant_mode_projection_error": max(r["structure"]["constant_mode_projection_error"] for r in reduced)},
               "median_timing_sum_seconds": float(sum(r["median"]["both_total_seconds"] for r in records))}
    dump(OUT / "summary.json", summary)
    print(json.dumps({"settings": len(records), "threshold_selections": selections, "structure": summary["structure"]}, indent=2))


if __name__ == "__main__":
    main()
