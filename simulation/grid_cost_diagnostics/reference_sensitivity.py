#!/usr/bin/env python3
"""Conditional h^2 Richardson comparators for the fixed receiver-cost sweep.

The extrapolated responses assume leading h^2 error; they do not certify
continuum error. Candidate waveforms and construction times are unchanged.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
LABELS = ("grid512", "grid384", "richardson512", "richardson384")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def relative(value, reference):
    return float(np.sqrt(np.sum((value-reference)**2)/np.sum(reference**2)))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def key(row):
    return (row["material"], row["grid"], row["method"], row["requested_cfl"])


def winner(rows, label, tolerance, family):
    feasible = [r for r in rows if r[label+"_error"] <= tolerance
                and r["family"] == family]
    return min(feasible, key=lambda r: r["median_seconds"]) if feasible else None


def calculate():
    cfg = json.loads((HERE/"config.json").read_text())
    inputs = {}
    references, differences = {}, []
    formula_defects = []
    times = np.linspace(0, cfg["time_end"], cfg["samples"])
    for material in cfg["materials"]:
        raw = {}
        for n in (256, 384, 512):
            path = ROOT/"propagation"/"results"/f"{material}_n{n}.npz"
            with np.load(path) as data:
                np.testing.assert_array_equal(data["time"], times)
                raw[n] = data["receiver_pressure"].copy()
            assert raw[n].shape == (len(times), 4, 3)
            inputs[str(path.relative_to(ROOT))] = digest(path)
        ext512 = raw[512]+(9/7)*(raw[512]-raw[384])
        ext384 = raw[384]+(4/5)*(raw[384]-raw[256])
        references[material] = {"grid512": raw[512], "grid384": raw[384],
                               "richardson512": ext512, "richardson384": ext384}
        formula_defects.extend([relative(ext512, (16*raw[512]-9*raw[384])/7),
                                relative(ext384, (9*raw[384]-4*raw[256])/5)])
        differences.append({
            "material": material,
            "extrapolant_difference_relative_to_richardson512": relative(ext384, ext512),
            "grid512_difference_relative_to_richardson512": relative(raw[512], ext512),
            "grid384_difference_relative_to_richardson384": relative(raw[384], ext384),
            "largest_source_extrapolant_difference": max(relative(ext384[:,:,s], ext512[:,:,s]) for s in range(3)),
            "largest_channel_extrapolant_difference": max(relative(ext384[:,r,s], ext512[:,r,s]) for r in range(4) for s in range(3)),
        })
    records = sorted((OUT/"candidates").glob("*_timing.json"))
    prescribed = {(m,n,"full_stormer_verlet",c) for m in cfg["materials"]
                  for n in cfg["grids"] for c in cfg["explicit_cfl_values"]}
    prescribed |= {(m,n,"craig_bampton",None) for m in cfg["materials"] for n in cfg["grids"]}
    candidates = []
    for path in records:
        timing = json.loads(path.read_text())
        rawpath = path.with_name(path.name.replace("_timing.json", ".npz"))
        assert digest(rawpath) == timing["raw_sha256"]
        inputs[str(path.relative_to(ROOT))] = digest(path)
        inputs[str(rawpath.relative_to(ROOT))] = timing["raw_sha256"]
        with np.load(rawpath) as data:
            np.testing.assert_array_equal(data["time"], times)
            y = data["receiver_pressure"].copy()
        totals = [sum(r.values()) for r in timing["repetitions"]]
        assert float(np.median(totals)) == timing["median"]["total_seconds"]
        row = {name: timing[name] for name in ("material", "grid", "method", "modes_per_patch", "requested_cfl", "dimension")}
        row.update(family="full" if timing["method"].startswith("full_") else "reduced",
                   median_seconds=float(np.median(totals)),
                   minimum_seconds=min(totals), maximum_seconds=max(totals))
        for label in LABELS:
            row[label+"_error"] = relative(y, references[row["material"]][label])
        candidates.append(row)
    assert {key(row) for row in candidates} == prescribed
    assert len(candidates) == len(prescribed)
    selections = []
    for material in cfg["materials"]:
        rows = [r for r in candidates if r["material"] == material]
        for tolerance in cfg["receiver_tolerances"]:
            primary = {f: winner(rows, "grid512", tolerance, f) for f in ("full", "reduced")}
            for label in LABELS:
                selected = {f: winner(rows, label, tolerance, f) for f in ("full", "reduced")}
                row = {"material": material, "tolerance": tolerance, "comparator": label}
                for family, candidate in selected.items():
                    row[family+"_feasible"] = candidate is not None
                    for attr in ("grid", "method", "dimension", "requested_cfl", "median_seconds", "minimum_seconds", "maximum_seconds"):
                        row[family+"_"+attr] = None if candidate is None else candidate[attr]
                    row[family+"_error"] = None if candidate is None else candidate[label+"_error"]
                    row[family+"_selection_changed_from_grid512"] = (
                        (None if candidate is None else key(candidate)) !=
                        (None if primary[family] is None else key(primary[family])))
                    samefamily = [r for r in rows if r["family"] == family]
                    row[family+"_pass_to_fail_from_grid512"] = sum(
                        r["grid512_error"] <= tolerance < r[label+"_error"] for r in samefamily)
                    row[family+"_fail_to_pass_from_grid512"] = sum(
                        r[label+"_error"] <= tolerance < r["grid512_error"] for r in samefamily)
                full, reduced = selected["full"], selected["reduced"]
                both = full is not None and reduced is not None
                row["median_full_over_reduced"] = full["median_seconds"]/reduced["median_seconds"] if both else None
                row["fastest_full_over_slowest_reduced"] = full["minimum_seconds"]/reduced["maximum_seconds"] if both else None
                row["slowest_full_over_fastest_reduced"] = full["maximum_seconds"]/reduced["minimum_seconds"] if both else None
                selections.append(row)
    old = list(csv.DictReader((OUT/"receiver_errors.csv").open()))
    old_by_key = {(r["material"], int(r["grid"]), r["method"], None if r["requested_cfl"] in ("", "None") else float(r["requested_cfl"])):r for r in old}
    replay = max(abs(r[label+"_error"]-float(old_by_key[key(r)][column]))
                 for r in candidates for label,column in (("grid512","total_receiver_error"), ("grid384","alternative_reference_total_receiver_error")))
    assert replay < 1e-13 and max(formula_defects) < 1e-14
    inputs[str((OUT/"receiver_errors.csv").relative_to(ROOT))] = digest(OUT/"receiver_errors.csv")
    summary = {
        "scope": "Conditional Richardson comparators under a leading h^2 waveform-error expansion. Neither extrapolant is a known exact solution or a rigorous error bound.",
        "formulas": {"richardson512": "Y512 + (9/7)*(Y512-Y384)", "richardson384": "Y384 + (4/5)*(Y384-Y256)"},
        "norm": "Equal-weight Frobenius norm over all 641 times including t=0, four receivers and three sources. Each candidate error uses the corresponding comparator norm.",
        "cost": "Unchanged median, minimum and maximum of the three original setup-inclusive stage sums. No construction or timing is repeated.",
        "candidate_count": len(candidates), "reference_differences": differences,
        "mean_extrapolant_difference": float(np.mean([r["extrapolant_difference_relative_to_richardson512"] for r in differences])),
        "selections": selections, "input_sha256": inputs,
        "config_sha256": digest(HERE/"config.json"), "analysis_code_sha256": digest(Path(__file__)),
        "maximum_previous_error_replay_difference": replay,
        "maximum_alternative_formula_difference": max(formula_defects),
    }
    return candidates, differences, selections, summary


def verify(candidates, differences, selections, summary):
    previous = json.loads((OUT/"richardson_summary.json").read_text())
    assert previous == summary
    maximum = 0.0
    for name, rows in (("candidate_errors",candidates), ("reference_differences",differences), ("selections",selections)):
        saved = list(csv.DictReader((OUT/f"richardson_{name}.csv").open()))
        assert len(saved) == len(rows)
        for old, new in zip(saved, rows):
            for k,v in new.items():
                if isinstance(v, float):
                    maximum = max(maximum, abs(float(old[k])-v))
                else:
                    assert old[k] == ("" if v is None else str(v))
    assert maximum < 1e-13
    # The two Richardson weight pairs preserve constants and cancel h^2.
    h2_defect = max(abs(16/7/512**2-9/7/384**2), abs(9/5/384**2-4/5/256**2))
    assert h2_defect < 1e-20
    result = {"status":"pass", "candidate_errors_recomputed":len(candidates)*len(LABELS),
        "least_cost_selections_recomputed":len(selections)*2,
        "input_hashes_recomputed":len(summary["input_sha256"]),
        "maximum_saved_numeric_difference":maximum,
        "maximum_previous_error_replay_difference":summary["maximum_previous_error_replay_difference"],
        "maximum_alternative_formula_difference":summary["maximum_alternative_formula_difference"],
        "leading_h_squared_cancellation_defect":h2_defect,
        "no_timing_repeated":True}
    dump(OUT/"richardson_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Recompute and compare the saved postprocessing outputs")
    args = parser.parse_args()
    candidates, differences, selections, summary = calculate()
    if not args.verify:
        for name,rows in (("candidate_errors",candidates), ("reference_differences",differences), ("selections",selections)):
            write_csv(OUT/f"richardson_{name}.csv", rows)
        dump(OUT/"richardson_summary.json", summary)
    print(json.dumps(verify(candidates, differences, selections, summary), indent=2))


if __name__ == "__main__":
    main()
