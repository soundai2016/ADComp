#!/usr/bin/env python3
"""Tabulate prescribed channels and source/receiver position contrasts."""
from common import CFG, HERE, OUT, config_digest, csv_write, dump, response
import json
import numpy as np


def groups(source_layout, receiver_layout, layout_config):
    sources = layout_config["original_source_centres"] if source_layout == 0 else CFG["source_centres"]
    receivers = layout_config["original_receiver_centres"] if receiver_layout == 0 else CFG["receiver_centres"]
    same = np.asarray([[(r[0] < .5) == (s[0] < .5) for s in sources] for r in receivers])
    answer = [("aggregate", 0, 0, np.ones((4, 3), dtype=bool)),
              ("same_side", 0, 0, same), ("opposite_sides", 0, 0, ~same)]
    for s in range(3):
        for r in range(4):
            mask = np.zeros((4, 3), dtype=bool)
            mask[r, s] = True
            answer.append(("channel", s + 1, r + 1, mask))
    return answer


def summarize():
    layout_config = json.loads((HERE / "layout_config.json").read_text())
    reference = np.load(OUT / "factorial_reference.npz")
    times = reference["time"]
    reference = reference["receiver_pressure"]
    accuracy, budgets, pairs = [], [], []
    for n in CFG["grids"]:
        full = np.load(OUT / f"n{n}_full.npz")
        field = np.load(OUT / f"n{n}_pressure_errors.npz")
        for a in range(2):
            for b in range(2):
                yr, yf = full["factorial_receiver_pressure"][a, b], reference[a, b]
                for end in CFG["window_ends"]:
                    active = times <= end + 1e-12
                    for group, s, r, channels in groups(a, b, layout_config):
                        ref = yf[active][:, channels]
                        err = yr[active][:, channels] - ref
                        accuracy.append({"grid": n, "method": "full", "modes_per_patch": 0, "dimension": n * n,
                            "source_layout": a, "receiver_layout": b, "time_end": end,
                            "group": group, "source": s, "receiver": r,
                            "total_receiver_error": float(np.linalg.norm(err) / np.linalg.norm(ref)),
                            "same_grid_receiver_error": 0., "reference_squared_norm": float(np.sum(ref ** 2)),
                            "continuous_pressure_error": float(np.sqrt(field["full_continuous_error_squared"][active].sum() /
                                field["continuous_norm_squared"][active].sum())) if a == 1 and group == "aggregate" else ""})
        for m in CFG["modes_per_patch"]:
            model = np.load(OUT / f"n{n}_cb_m{m}.npz")
            wr, wh, wf = (model[key] for key in ("matched_reduced_omega", "matched_full_omega", "matched_continuous_omega"))
            for j, (ri, hi, fi) in enumerate(zip(model["matched_reduced_indices"], model["matched_full_indices"], model["matched_continuous_indices"])):
                pairs.append({"grid": n, "modes_per_patch": m, "dimension": len(model["omega"]),
                    "reduced_index": int(ri), "full_index": int(hi), "continuous_index": int(fi),
                    "mac_reduced_full": float(model["mac_reduced_full"][j]),
                    "mac_full_continuous": float(model["mac_full_continuous"][j]),
                    "omega_reduced": float(wr[j]), "omega_full": float(wh[j]), "omega_continuous": float(wf[j]),
                    "relative_ritz_shift": float((wr[j] - wh[j]) / wf[j]),
                    "relative_grid_shift": float((wh[j] - wf[j]) / wf[j])})
            ritz_phase = np.cos(times[:, None] * wr[None]) - np.cos(times[:, None] * wh[None])
            grid_phase = np.cos(times[:, None] * wh[None]) - np.cos(times[:, None] * wf[None])
            for a in range(2):
                for b in range(2):
                    yf, yh = reference[a, b], full["factorial_receiver_pressure"][a, b]
                    yr = model["factorial_receiver_pressure"][a, b]
                    A, B = model["factorial_ritz_frequency"][a, b], model["factorial_grid_frequency"][a, b]
                    source, receiver, unmatched = (model["factorial_" + key][a, b] for key in ("source_coupling", "receiver_coupling", "unmatched"))
                    q = source + receiver + unmatched
                    Hr = model["factorial_reduced_residues"][a, b]
                    Hf = model["factorial_continuous_residues"][a, b]
                    omitted = yf - response(times, wf, Hf)
                    for end in CFG["window_ends"]:
                        active = times <= end + 1e-12
                        single_mode = np.sum(-ritz_phase[active] ** 2 - 2 * ritz_phase[active] * grid_phase[active], axis=0)
                        for group, s, r, channels in groups(a, b, layout_config):
                            ref = yf[active][:, channels]
                            er = (yr - yf)[active][:, channels]
                            er0 = (yr - A - yf)[active][:, channels]
                            er_cont = (yr - A - B - yf)[active][:, channels]
                            aa, bb, qq = (item[active][:, channels] for item in (A, B, q))
                            den = float(np.sum(ref ** 2))
                            exact = float((np.sum(er0 ** 2) - np.sum(er ** 2)) / den)
                            diagonal = float(single_mode @ np.sum(Hr[:, channels] ** 2, axis=1) / den)
                            frequency = float((-np.sum(aa ** 2) - 2 * np.sum(aa * bb)) / den)
                            cross_mode = frequency - diagonal
                            remainder_cross = float(-2 * np.sum(aa * qq) / den)
                            total = float(np.linalg.norm(er) / np.sqrt(den))
                            replaced = float(np.linalg.norm(er0) / np.sqrt(den))
                            key = {"grid": n, "method": "craig_bampton", "modes_per_patch": m, "dimension": len(model["omega"]),
                                   "source_layout": a, "receiver_layout": b, "time_end": end,
                                   "group": group, "source": s, "receiver": r}
                            same_grid = float(np.linalg.norm((yr - yh)[active][:, channels]) / np.linalg.norm(yh[active][:, channels]))
                            accuracy.append(key | {"total_receiver_error": total, "same_grid_receiver_error": same_grid,
                                "reference_squared_norm": den,
                                "continuous_pressure_error": float(np.sqrt(field[f"m{m}_continuous_error_squared"][active].sum() /
                                    field["continuous_norm_squared"][active].sum())) if a == 1 and group == "aggregate" else ""})
                            weights = np.linalg.norm(Hf[:, channels], axis=1)
                            weights /= max(weights.sum(), 1e-300)
                            budgets.append(key | {"accepted_triples": len(wr), "total_receiver_error": total,
                                "same_grid_receiver_error": same_grid, "coarse_replacement_error": replaced,
                                "continuous_replacement_error": float(np.linalg.norm(er_cont) / np.sqrt(den)),
                                "change_percentage_points": 100 * (replaced - total), "delta_squared_error": exact,
                                "same_mode_term": diagonal, "cross_mode_term": cross_mode, "remainder_cross_term": remainder_cross,
                                "source_cross_term": float(-2 * np.sum(aa * source[active][:, channels]) / den),
                                "receiver_cross_term": float(-2 * np.sum(aa * receiver[active][:, channels]) / den),
                                "unmatched_cross_term": float(-2 * np.sum(aa * unmatched[active][:, channels]) / den),
                                "budget_absolute_defect": abs(exact - diagonal - cross_mode - remainder_cross),
                                "ritz_relative_norm": float(np.linalg.norm(aa) / np.sqrt(den)),
                                "grid_relative_norm": float(np.linalg.norm(bb) / np.sqrt(den)),
                                "remainder_relative_norm": float(np.linalg.norm(qq) / np.sqrt(den)),
                                "matched_reference_omission": float(np.linalg.norm(omitted[active][:, channels]) / np.sqrt(den)),
                                "weighted_relative_grid_shift": float(weights @ ((wh - wf) / wf)),
                                "weighted_relative_ritz_shift": float(weights @ ((wr - wh) / wf)),
                                "reference_squared_norm": den})
    csv_write(OUT / "accuracy.csv", accuracy)
    csv_write(OUT / "finite_time_budgets.csv", budgets)
    csv_write(OUT / "modal_pairs.csv", pairs)
    primary = [row for row in budgets if row["source_layout"] == row["receiver_layout"] == 1]
    csv_write(OUT / "second_layout_budgets.csv", primary)
    cases = []
    for n in CFG["grids"]:
        for m in CFG["modes_per_patch"]:
            for end in CFG["window_ends"]:
                selected = [row for row in primary if row["grid"] == n and row["modes_per_patch"] == m and row["time_end"] == end]
                agg = next(row for row in selected if row["group"] == "aggregate")
                channels = [row for row in selected if row["group"] == "channel"]
                sides = {row["group"]: row["change_percentage_points"] for row in selected if row["group"] in ("same_side", "opposite_sides")}
                cases.append({"grid": n, "modes_per_patch": m, "time_end": end,
                    "total_receiver_error": agg["total_receiver_error"], "coarse_replacement_error": agg["coarse_replacement_error"],
                    "change_percentage_points": agg["change_percentage_points"], "accepted_triples": agg["accepted_triples"],
                    "beneficial_ritz_channels": sum(row["change_percentage_points"] > 1e-10 for row in channels),
                    "harmful_ritz_channels": sum(row["change_percentage_points"] < -1e-10 for row in channels),
                    "same_side_change_percentage_points": sides["same_side"],
                    "opposite_sides_change_percentage_points": sides["opposite_sides"]})
    summary = {"config_sha256": config_digest(), "accuracy_rows": len(accuracy), "budget_rows": len(budgets),
        "matched_mode_rows": len(pairs), "maximum_budget_absolute_defect": max(row["budget_absolute_defect"] for row in budgets),
        "second_layout_cases": cases,
        "factorial_T8_aggregate": [row for row in budgets if row["group"] == "aggregate" and row["time_end"] == 8.]}
    contrasts, example = position_contrasts(budgets)
    csv_write(OUT / "position_contrasts.csv", contrasts)
    summary["position_contrast_rows"] = len(contrasts)
    summary["channel_position_sign_reversals"] = sum(row["sign_reversal"] for row in contrasts if row["group"] == "channel")
    summary["aggregate_position_sign_reversals"] = [row for row in contrasts if row["group"] == "aggregate" and row["sign_reversal"]]
    summary["illustrative_position_reversal"] = example
    summary["figure_position_examples"] = figure_examples(budgets)
    dump(OUT / "summary.json", summary)
    return accuracy, budgets, summary


def position_contrasts(budgets):
    cases = [("source", (0, 0), (1, 0)), ("source", (0, 1), (1, 1)),
             ("receiver", (0, 0), (0, 1)), ("receiver", (1, 0), (1, 1))]
    index = {(row["grid"], row["modes_per_patch"], row["time_end"], row["source_layout"], row["receiver_layout"],
              row["group"], row["source"], row["receiver"]): row for row in budgets}
    rows = []
    for n in CFG["grids"]:
        for m in CFG["modes_per_patch"]:
            for end in CFG["window_ends"]:
                for changed, before, after in cases:
                    before_aggregate = index[(n, m, end, *before, "aggregate", 0, 0)]
                    after_aggregate = index[(n, m, end, *after, "aggregate", 0, 0)]
                    norm = np.sqrt(before_aggregate["reference_squared_norm"] * after_aggregate["reference_squared_norm"])
                    for group, s, r in [("aggregate", 0, 0)] + [("channel", s, r) for s in range(1, 4) for r in range(1, 5)]:
                        first, second = index[(n, m, end, *before, group, s, r)], index[(n, m, end, *after, group, s, r)]
                        d0, d1 = first["delta_squared_error"], second["delta_squared_error"]
                        reversal = d0 * d1 < 0 and min(abs(d0), abs(d1)) > 1e-12
                        strength = min(abs(d0 * first["reference_squared_norm"]), abs(d1 * second["reference_squared_norm"])) / norm
                        rows.append({"grid": n, "modes_per_patch": m, "time_end": end, "changed_positions": changed,
                            "before_source_layout": before[0], "before_receiver_layout": before[1],
                            "after_source_layout": after[0], "after_receiver_layout": after[1],
                            "group": group, "source": s, "receiver": r, "sign_reversal": int(reversal),
                            "before_change_percentage_points": first["change_percentage_points"],
                            "after_change_percentage_points": second["change_percentage_points"],
                            "before_delta_squared_error": d0, "after_delta_squared_error": d1,
                            "before_reference_energy_fraction": first["reference_squared_norm"] / before_aggregate["reference_squared_norm"],
                            "after_reference_energy_fraction": second["reference_squared_norm"] / after_aggregate["reference_squared_norm"],
                            "display_strength": float(strength)})
    candidates = [row for row in rows if row["group"] == "channel" and row["sign_reversal"]]
    example = max(candidates, key=lambda row: row["display_strength"]) if candidates else None
    if example is not None:
        example = example.copy()
        base = (example["grid"], example["modes_per_patch"], example["time_end"])
        example["before_budget"] = index[(*base, example["before_source_layout"], example["before_receiver_layout"],
            "channel", example["source"], example["receiver"])]
        example["after_budget"] = index[(*base, example["after_source_layout"], example["after_receiver_layout"],
            "channel", example["source"], example["receiver"])]
    return rows, example


def figure_examples(budgets):
    contrasts, _ = position_contrasts(budgets)
    selected = []
    for changed in ("source", "receiver"):
        candidates = [row for row in contrasts if row["group"] == "channel" and row["sign_reversal"]
            and row["changed_positions"] == changed]
        chosen = max(candidates, key=lambda row: row["display_strength"])
        first = next(row for row in budgets if row["group"] == "channel" and row["grid"] == chosen["grid"]
            and row["modes_per_patch"] == chosen["modes_per_patch"] and row["time_end"] == chosen["time_end"]
            and row["source"] == chosen["source"] and row["receiver"] == chosen["receiver"]
            and row["source_layout"] == chosen["before_source_layout"] and row["receiver_layout"] == chosen["before_receiver_layout"])
        second = next(row for row in budgets if row["group"] == "channel" and row["grid"] == chosen["grid"]
            and row["modes_per_patch"] == chosen["modes_per_patch"] and row["time_end"] == chosen["time_end"]
            and row["source"] == chosen["source"] and row["receiver"] == chosen["receiver"]
            and row["source_layout"] == chosen["after_source_layout"] and row["receiver_layout"] == chosen["after_receiver_layout"])
        selected.append(chosen | {"before_budget": first, "after_budget": second})
    return selected



def main():
    _, _, summary = summarize()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
