#!/usr/bin/env python3
"""Evaluate the fixed design over nine modal-matching thresholds."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from core import (GRIDS, RANKS, WINDOWS, MACS, GAPS, DEFAULT_ID, SIGN_TOLERANCE,
                  analysis_rows, chain, get_assignment, modal_response, sha256)
from source_subspace import source_subspace_rows, summarize as summarize_subspace

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def budget_key(row):
    return (row["grid"], row["modes_per_patch"], row["source_layout"], row["receiver_layout"],
            row["time_end"], 0 if row["group"] == "aggregate" else 1, row["source"], row["receiver"])


def position_contrasts(rows):
    index = {(row["selection_id"], *budget_key(row)): row for row in rows}
    positions = [("source", (0, 0), (1, 0)), ("source", (0, 1), (1, 1)),
                 ("receiver", (0, 0), (0, 1)), ("receiver", (1, 0), (1, 1))]
    selections = list(dict.fromkeys(row["selection_id"] for row in rows))
    output = []
    for selection_id in selections:
        for n in GRIDS:
            for m in RANKS:
                for end in WINDOWS:
                    for changed, before, after in positions:
                        aggregates = [index[(selection_id, n, m, *layout, end, 0, 0, 0)] for layout in (before, after)]
                        common_den = np.sqrt(aggregates[0]["reference_squared_norm"] * aggregates[1]["reference_squared_norm"])
                        groups = [(0, 0, 0)] + [(1, s, r) for s in range(1, 4) for r in range(1, 5)]
                        for group, s, r in groups:
                            first, second = [index[(selection_id, n, m, *layout, end, group, s, r)] for layout in (before, after)]
                            d0, d1 = first["delta_squared_error"], second["delta_squared_error"]
                            reversal = d0 * d1 < 0 and min(abs(d0), abs(d1)) > SIGN_TOLERANCE
                            output.append(dict(grid=n, modes_per_patch=m, time_end=end,
                                selection_id=selection_id, selection_type=first["selection_type"],
                                mac_threshold=first["mac_threshold"], relative_gap=first["relative_gap"],
                                matched_count=first["matched_count"], changed_positions=changed,
                                before_source_layout=before[0], before_receiver_layout=before[1],
                                after_source_layout=after[0], after_receiver_layout=after[1],
                                group="aggregate" if group == 0 else "channel", source=s, receiver=r,
                                sign_reversal=int(reversal),
                                before_change_percentage_points=first["change_percentage_points"],
                                after_change_percentage_points=second["change_percentage_points"],
                                before_delta_squared_error=d0, after_delta_squared_error=d1,
                                before_reference_energy_fraction=first["reference_energy_fraction"],
                                after_reference_energy_fraction=second["reference_energy_fraction"],
                                before_common_normalized_increment=first["raw_squared_error_change"] / common_den,
                                after_common_normalized_increment=second["raw_squared_error_change"] / common_den,
                                min_absolute_change_percentage_points=min(abs(first["change_percentage_points"]), abs(second["change_percentage_points"])),
                                display_strength=min(abs(first["raw_squared_error_change"]), abs(second["raw_squared_error_change"])) / common_den))
    return output


PAIR_FIELDS = ("grid", "modes_per_patch", "time_end", "changed_positions", "before_source_layout",
               "before_receiver_layout", "after_source_layout", "after_receiver_layout", "group", "source", "receiver")


def paired_robustness(contrasts):
    grouped = {}
    for row in contrasts:
        grouped.setdefault(tuple(row[key] for key in PAIR_FIELDS), []).append(row)
    output = []
    for key, cases in grouped.items():
        threshold = [row for row in cases if row["selection_type"] == "threshold"]
        assert len(threshold) == 9
        default = next(row for row in threshold if row["selection_id"] == DEFAULT_ID)
        intersection = next(row for row in cases if row["selection_type"] == "intersection")
        count = sum(row["sign_reversal"] for row in threshold)
        before_signs = {int(np.sign(row["before_delta_squared_error"]))
                        if abs(row["before_delta_squared_error"]) > SIGN_TOLERANCE else 0 for row in threshold}
        after_signs = {int(np.sign(row["after_delta_squared_error"]))
                       if abs(row["after_delta_squared_error"]) > SIGN_TOLERANCE else 0 for row in threshold}
        fixed_direction = count == 9 and len(before_signs) == len(after_signs) == 1
        output.append(dict(zip(PAIR_FIELDS, key)) | dict(
            threshold_count=9, reversal_count=count, default_reversal=default["sign_reversal"],
            all_nine_reversal=int(count == 9), any_threshold_reversal=int(count > 0),
            fixed_direction_all_nine=int(fixed_direction),
            all_nine_with_direction_swap=int(count == 9 and not fixed_direction),
            before_signs_across_thresholds="|".join(map(str, sorted(before_signs))),
            after_signs_across_thresholds="|".join(map(str, sorted(after_signs))),
            intersection_reversal=intersection["sign_reversal"],
            default_before_change_percentage_points=default["before_change_percentage_points"],
            default_after_change_percentage_points=default["after_change_percentage_points"],
            before_change_percentage_points_min=min(row["before_change_percentage_points"] for row in threshold),
            before_change_percentage_points_max=max(row["before_change_percentage_points"] for row in threshold),
            after_change_percentage_points_min=min(row["after_change_percentage_points"] for row in threshold),
            after_change_percentage_points_max=max(row["after_change_percentage_points"] for row in threshold),
            minimum_absolute_paired_change_over_thresholds=min(row["min_absolute_change_percentage_points"] for row in threshold),
            minimum_display_strength_over_thresholds=min(row["display_strength"] for row in threshold),
            maximum_display_strength_over_thresholds=max(row["display_strength"] for row in threshold)))
    return output


def count_tables(contrasts, robustness):
    threshold_rows, robust_rows = [], []
    settings = list(dict.fromkeys(row["selection_id"] for row in contrasts if row["selection_type"] == "threshold"))
    for end in WINDOWS:
        for changed in ("source", "receiver"):
            for group in ("channel", "aggregate"):
                subset = [row for row in robustness if row["time_end"] == end and row["changed_positions"] == changed and row["group"] == group]
                stable = [row["minimum_absolute_paired_change_over_thresholds"] for row in subset if row["all_nine_reversal"]]
                robust_rows.append(dict(time_end=end, changed_positions=changed, group=group,
                    comparisons=len(subset), default_reversals=sum(row["default_reversal"] for row in subset),
                    any_threshold_reversals=sum(row["any_threshold_reversal"] for row in subset),
                    all_nine_reversals=sum(row["all_nine_reversal"] for row in subset),
                    fixed_direction_all_nine_reversals=sum(row["fixed_direction_all_nine"] for row in subset),
                    all_nine_with_direction_swap=sum(row["all_nine_with_direction_swap"] for row in subset),
                    intersection_reversals=sum(row["intersection_reversal"] for row in subset),
                    stable_minimum_absolute_pp_min=float(np.min(stable)) if stable else "",
                    stable_minimum_absolute_pp_median=float(np.median(stable)) if stable else "",
                    stable_minimum_absolute_pp_max=float(np.max(stable)) if stable else ""))
                for selection_id in settings:
                    rows = [row for row in contrasts if row["time_end"] == end and row["changed_positions"] == changed
                            and row["group"] == group and row["selection_id"] == selection_id]
                    threshold_rows.append(dict(selection_id=selection_id, mac_threshold=rows[0]["mac_threshold"], relative_gap=rows[0]["relative_gap"],
                        time_end=end, changed_positions=changed, group=group,
                        comparisons=len(rows), sign_reversals=sum(row["sign_reversal"] for row in rows)))
    return threshold_rows, robust_rows


def verify_default(data, rows):
    defaults = {budget_key(row): row for row in rows if row["selection_id"] == DEFAULT_ID}
    fields = list(data["baseline_expected_fields"])
    maximum = 0.
    metric_max = {}
    for saved in data["baseline_expected"]:
        previous = dict(zip(fields, saved))
        key = tuple(previous[name] for name in fields[:8])
        actual = defaults[key]
        for name in fields[8:]:
            difference = abs(actual[name] - previous[name])
            scaled = difference / max(1., abs(previous[name]))
            maximum = max(maximum, scaled)
            metric_max[name] = max(metric_max.get(name, 0.), difference)
    assert len(defaults) == len(data["baseline_expected"]) == 936
    assert maximum < 1e-10, metric_max
    return dict(default_budget_rows=936, maximum_scaled_default_replay_difference=maximum,
                default_replay_maximum_absolute_difference_by_metric=metric_max)


def run(data_dir, output_dir, simulation_root=None):
    metadata = json.loads((data_dir / "provenance.json").read_text())
    compact = data_dir / "compact_inputs.npz"
    if sha256(compact) != metadata["compact_sha256"]:
        raise RuntimeError("Compact input hash differs from provenance.json")
    source_verified = False
    if simulation_root is not None:
        for name, expected in metadata["source_sha256"].items():
            if sha256(simulation_root / name) != expected:
                raise RuntimeError(f"Simulation input hash differs: {name}")
        source_verified = True
    with np.load(compact, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    times, omega_f = data["time"], data["omega_f"]
    bf, cf = data["bf"], data["cf"]
    yf = np.stack([np.stack([modal_response(times, omega_f, bf[a], cf[b]) for b in range(2)]) for a in range(2)])
    records, selections, subspaces, subspace_matrices = [], [], [], []
    default_matching = True
    for n in GRIDS:
        af = get_assignment(data, f"n{n}_af")
        omega_h = data[f"n{n}_omega_h"]
        norms = data[f"n{n}_continuous_mode_norms"]
        for m in RANKS:
            prefix = f"n{n}_m{m}"
            ar = get_assignment(data, prefix + "_ar")
            omega_r, br, cr = (data[prefix + suffix] for suffix in ("_omega_r", "_br", "_cr"))
            yr = np.stack([np.stack([modal_response(times, omega_r, br[a], cr[b]) for b in range(2)]) for a in range(2)])
            sets = [(f"mac{mac:.2f}_gap{gap:.4f}", "threshold", mac, gap,
                     chain(ar, af, omega_f, float(data["active_max"]), norms, mac, gap)) for mac in MACS for gap in GAPS]
            intersection = sorted(set.intersection(*(set(row[4]) for row in sets)))
            sets.append(("intersection", "intersection", "", "", intersection))
            for selection_id, selection_type, mac, gap, triples in sets:
                if selection_id == DEFAULT_ID:
                    default_matching &= bool(np.array_equal(np.asarray(triples)[:, :3], data[prefix + "_default_triples"]))
                selections.append(dict(grid=n, modes_per_patch=m, selection_id=selection_id,
                    selection_type=selection_type, mac_threshold=mac, relative_gap=gap, matched_count=len(triples)))
                records.extend(analysis_rows(n, m, times, omega_r, omega_h, omega_f,
                    br, cr, bf, cf, yr, yf, triples, ar, af, selection_id, selection_type, mac, gap))
                source_rows, source_matrices = source_subspace_rows(n, m, times, omega_r,
                    omega_h, omega_f, br, cr, bf, cf, yr, yf, triples, selection_id, selection_type)
                subspaces.extend(source_rows)
                subspace_matrices.extend(source_matrices)
            print(f"n={n}, m={m}: all layouts/windows/channels, 9 thresholds + intersection", flush=True)
    checks = verify_default(data, records)
    checks.update(default_matching_indices_identical=default_matching,
        maximum_budget_absolute_defect=max(row["budget_absolute_defect"] for row in records),
        maximum_decomposition_relative_defect=max(row["decomposition_relative_defect"] for row in records),
        budget_rows=len(records))
    assert default_matching and len(records) == 9360
    assert checks["maximum_budget_absolute_defect"] < 1e-10
    assert checks["maximum_decomposition_relative_defect"] < 1e-10
    contrasts = position_contrasts(records)
    robustness = paired_robustness(contrasts)
    counts, stable_counts = count_tables(contrasts, robustness)
    assert len(contrasts) == 9360 and len(robustness) == 936
    for changed in ("source", "receiver"):
        subset = [r for r in contrasts if r["selection_id"] == DEFAULT_ID and r["changed_positions"] == changed]
        assert sum(r["group"] == "channel" for r in subset) == 432
        assert sum(r["group"] == "aggregate" for r in subset) == 36
    checks.update(contrast_rows=len(contrasts), unique_contrasts=len(robustness),
                  default_channel_pairs_per_position_factor=432, default_aggregate_pairs_per_position_factor=36)
    source_summary = summarize_subspace(subspaces, records)
    checks["source_subspace"] = source_summary["verification"]
    t8checks = []
    for n in GRIDS:
        for m in RANKS:
            for choice in [r for r in selections if r["grid"] == n and r["modes_per_patch"] == m]:
                active = [r for r in records if r["grid"] == n and r["modes_per_patch"] == m
                          and r["selection_id"] == choice["selection_id"] and r["time_end"] == 8 and r["group"] == "aggregate"]
                signs = [int(np.sign(r["delta_squared_error"])) if abs(r["delta_squared_error"]) > SIGN_TOLERANCE else 0 for r in active]
                t8checks.append(choice | dict(all_four_layouts_same_sign=len(set(signs)) == 1,
                    change_percentage_points_00=active[0]["change_percentage_points"],
                    change_percentage_points_01=active[1]["change_percentage_points"],
                    change_percentage_points_10=active[2]["change_percentage_points"],
                    change_percentage_points_11=active[3]["change_percentage_points"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_budgets.csv", records)
    write_csv(output_dir / "all_position_contrasts.csv", contrasts)
    write_csv(output_dir / "contrast_robustness.csv", robustness)
    write_csv(output_dir / "threshold_counts.csv", counts)
    write_csv(output_dir / "robustness_counts.csv", stable_counts)
    write_csv(output_dir / "matched_counts.csv", selections)
    write_csv(output_dir / "T8_aggregate_layout_checks.csv", t8checks)
    write_csv(output_dir / "source_subspace.csv", subspaces)
    np.savez_compressed(output_dir / "source_subspace_matrices.npz",
        reference_gram=np.array([pair[0] for pair in subspace_matrices]),
        error_change=np.array([pair[1] for pair in subspace_matrices]))
    source_summary["matrix_rows"] = "source_subspace.csv row order"
    source_summary["compact_sha256"] = metadata["compact_sha256"]
    (output_dir / "source_subspace_summary.json").write_text(
        json.dumps(source_summary, indent=2, allow_nan=False) + "\n")
    summary = dict(design=dict(grids=list(GRIDS), internal_modes_per_patch=list(RANKS),
        source_layouts=2, receiver_layouts=2, channel_labels=12, window_ends=list(WINDOWS),
        mac_thresholds=list(MACS), relative_frequency_gaps=list(GAPS), residue_path="ROM residues",
        threshold_choices=9, additional_intersection_selection=True,
        sign_tolerance_squared_relative_error=SIGN_TOLERANCE,
        ordinary_channel_pairs_per_position_factor=432, aggregate_pairs_per_position_factor=36,
        interpretation="Finite deterministic design with dependent windows and shared models. Each threshold defines a distinct selected-frequency intervention.",
        direction_definition="all_nine_reversal: opposite endpoint signs at every threshold. fixed_direction_all_nine: each endpoint also retains its sign across thresholds."),
        robustness_counts=stable_counts, threshold_counts=counts,
        direction_totals={changed: dict(
            channel_comparisons=sum(r["changed_positions"] == changed and r["group"] == "channel" for r in robustness),
            all_nine_reversals=sum(r["all_nine_reversal"] for r in robustness if r["changed_positions"] == changed and r["group"] == "channel"),
            fixed_direction_all_nine=sum(r["fixed_direction_all_nine"] for r in robustness if r["changed_positions"] == changed and r["group"] == "channel"),
            all_nine_with_direction_swap=sum(r["all_nine_with_direction_swap"] for r in robustness if r["changed_positions"] == changed and r["group"] == "channel"))
            for changed in ("source", "receiver")},
        T8_aggregate_layout_failures=[r for r in t8checks if not r["all_four_layouts_same_sign"]],
        T8_aggregate_threshold_cases=54, T8_aggregate_intersection_cases=6,
        source_subspace=source_summary,
        verification=checks)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_dir / "verification.json").write_text(json.dumps(dict(passed=True, checks=checks), indent=2) + "\n")
    environment = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        thread_environment={key: os.environ[key] for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if key in os.environ},
        threadpools=threadpool_info(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        compact_sha256=metadata["compact_sha256"], simulation_input_hashes_verified=source_verified)
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    for changed, totals in summary["direction_totals"].items():
        print(f"{changed}: {totals['all_nine_reversals']} all-threshold reversals, "
              f"{totals['fixed_direction_all_nine']} with fixed direction")
    print(f"Verified {len(records)} budgets and {len(robustness)} position contrasts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data" / "analysis")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "results" / "analysis")
    parser.add_argument("--simulation-root", type=Path, help="Verify numerical input hashes against this simulation directory")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.simulation_root)


if __name__ == "__main__":
    main()
