#!/usr/bin/env python3
"""Compare fixed frequency interventions under h^2-extrapolated references."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
MATERIALS = ("rotated_stripes", "double_inclusion", "smooth_checker")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, replay, inputs, formula_defects, margin_absolute, margin_scaled = [], [], [], [], [], []
    for material in MATERIALS:
        print(f"Richardson reference comparison: {material}", flush=True)
        reference = {}
        for n in (256, 384, 512):
            path = SOURCE / "propagation/results" / f"{material}_n{n}.npz"
            data = load_npz(path)
            if reference:
                np.testing.assert_array_equal(data["time"], times)
            times = data["time"]
            reference[f"grid{n}"] = data["receiver_pressure"]
            inputs.append(path)
        reference["richardson512"] = reference["grid512"] + 9 / 7 * (reference["grid512"] - reference["grid384"])
        reference["richardson384"] = reference["grid384"] + 4 / 5 * (reference["grid384"] - reference["grid256"])
        formula_defects.extend([
            float(np.max(abs(reference["richardson512"] - (16 * reference["grid512"] - 9 * reference["grid384"]) / 7))),
            float(np.max(abs(reference["richardson384"] - (9 * reference["grid384"] - 4 * reference["grid256"]) / 5)))])
        rompath = SOURCE / "accuracy_cost/results" / f"{material}_n32_craig_bampton_m32.npz"
        coarsepath = SOURCE / "modal_diagnostics/results" / f"{material}_n32_spectrum.npz"
        rowpath = SOURCE / "modal_robustness/results" / f"{material}_finite_time.csv"
        model, coarse = load_npz(rompath), load_npz(coarsepath)
        np.testing.assert_array_equal(model["time"], times)
        inputs.extend((rompath, coarsepath, rowpath))
        w = np.sqrt(np.maximum(model["reduced_eigenvalues"], 0.))
        wh = np.sqrt(np.maximum(coarse["eigenvalues"], 0.))
        modes = model["orthonormal_basis"] @ model["reduced_eigenvectors"]
        transform = coarse["eigenvectors"].T @ modes
        b = transform.T @ coarse["source_coupling"]
        c = coarse["receiver_coupling"] @ transform
        h = np.einsum("rj,js->jrs", c, b)
        yr = model["receiver_pressure"]
        direct = np.einsum("tj,jrs->trs", np.cos(np.outer(times, w)), h)
        model_defect = float(np.linalg.norm(direct - yr) / np.linalg.norm(yr))
        old = list(csv.DictReader(rowpath.open()))
        for fine_grid in (128, 192):
            selection_path = SOURCE / "modal_robustness/results" / f"{material}_n{fine_grid}_craig_bampton_m32_gram.npz"
            selected = load_npz(selection_path)
            inputs.append(selection_path)
            ri, ci = selected["rom_modes"], selected["coarse_modes"]
            aa = np.einsum("tj,jrs->trs", np.cos(np.outer(times, w[ri])) - np.cos(np.outer(times, wh[ci])), h[ri])
            own = [r for r in old if int(r["fine_grid"]) == fine_grid and r["method"] == "craig_bampton" and int(r["m"]) == 32]
            for prior in own:
                start, end = float(prior["start"]), float(prior["end"])
                active = (times >= start) & (times <= end)
                if prior["window_kind"] == "successive" and start != 0:
                    active = (times > start) & (times <= end)
                s, r = int(prior["source"]), int(prior["receiver"])
                channels = np.ones((4, 3), dtype=bool)
                if prior["group"] == "channel":
                    channels[:] = False
                    channels[r - 1, s - 1] = True
                a = aa[active][:, channels]
                y = yr[active][:, channels]
                zbase = reference["grid512"][active][:, channels]
                denbase = float(np.sum(zbase ** 2))
                ebase = y - zbase
                dbase = float((np.sum((ebase - a) ** 2) - np.sum(ebase ** 2)) / denbase)
                replay.append(abs(dbase - float(prior["exact_squared_error_change"])))
                for comparator in ("grid512", "grid384", "richardson512", "richardson384"):
                    z = reference[comparator][active][:, channels]
                    den = float(np.sum(z ** 2))
                    e = y - z
                    delta_numerator = float(np.sum((e - a) ** 2) - np.sum(e ** 2))
                    actual = delta_numerator / den
                    original_error = float(np.linalg.norm(e) / np.sqrt(den))
                    replaced_error = float(np.linalg.norm(e - a) / np.sqrt(den))
                    difference = z - zbase
                    cross = float(2 * np.sum(a * difference) / denbase)
                    bound = float(2 * np.linalg.norm(a) * np.linalg.norm(difference) / denbase)
                    same_denominator = delta_numerator / denbase
                    margin = abs(dbase) / bound if bound > 0 else None
                    if comparator == "grid384":
                        old_margin = float(prior["reference_sign_margin"])
                        margin_absolute.append(abs(margin - old_margin))
                        # Use relative and absolute drift for the unbounded margin ratios.
                        margin_scaled.append(abs(margin - old_margin) / max(1., abs(old_margin)))
                    rows.append(dict(material=material, fine_grid=fine_grid, method="craig_bampton", m=32,
                        matched_count=len(ri), window_kind=prior["window_kind"], start=start, end=end,
                        group=prior["group"], source=s, receiver=r, comparator=comparator,
                        original_error=original_error, replaced_error=replaced_error,
                        change_percentage_points=100 * (replaced_error - original_error),
                        delta_squared_error=actual,
                        delta_squared_error_baseline_denominator=same_denominator,
                        baseline_delta_squared_error=dbase,
                        sign_agrees_with_grid512=bool(actual * dbase > 0) if abs(dbase) > 1e-12 else None,
                        baseline_sign_margin_for_this_perturbation=margin,
                        reference_change_exact_cross=cross, reference_change_cauchy_bound=bound,
                        reference_change_identity_defect=abs((same_denominator - dbase) - cross),
                        reference_relative_difference=float(np.linalg.norm(difference) / np.sqrt(denbase)),
                        saved_model_modal_replay_relative_difference=model_defect,
                        original_grid384_margin=float(prior["reference_sign_margin"])))
    write_csv(out / "richardson_sign_checks.csv", rows)
    relevant = [r for r in rows if r["group"] == "channel" and r["window_kind"] != "short_cumulative"]
    summary = []
    weakest = []
    for material in MATERIALS:
        for fine_grid in (128, 192):
            own = [r for r in relevant if r["material"] == material and r["fine_grid"] == fine_grid]
            weakest_base = min((r for r in own if r["comparator"] == "grid512"),
                               key=lambda r: r["original_grid384_margin"])
            key = {k: weakest_base[k] for k in ("material", "fine_grid", "window_kind", "start", "end", "source", "receiver")}
            weaker = [r for r in own if all(r[k] == v for k, v in key.items())]
            weakest.extend(weaker)
            for comparator in ("grid384", "richardson512", "richardson384"):
                selected = [r for r in own if r["comparator"] == comparator]
                summary.append(dict(material=material, fine_grid=fine_grid, comparator=comparator,
                    comparisons=len(selected), sign_agreements=sum(r["sign_agrees_with_grid512"] is True for r in selected),
                    margins_greater_than_one=sum(r["baseline_sign_margin_for_this_perturbation"] > 1 for r in selected),
                    minimum_margin=min(r["baseline_sign_margin_for_this_perturbation"] for r in selected),
                    sign_changes=[r for r in selected if r["sign_agrees_with_grid512"] is False]))
    write_csv(out / "richardson_weakest_cases.csv", weakest)
    checks = dict(all_expected_rows=len(rows) == 3432,
                  maximum_saved_budget_replay_absolute_difference=max(replay),
                  maximum_saved_margin_scaled_difference=max(margin_scaled),
                  maximum_reference_identity_defect=max(r["reference_change_identity_defect"] for r in rows),
                  maximum_model_modal_replay_relative_difference=max(r["saved_model_modal_replay_relative_difference"] for r in rows),
                  maximum_equivalent_richardson_formula_absolute_difference=max(formula_defects),
                  all_reference_changes_within_cauchy_bound=all(abs(r["reference_change_exact_cross"]) <= r["reference_change_cauchy_bound"] + 1e-12 for r in rows))
    passed = all(v if isinstance(v, bool) else v < 1e-10 for v in checks.values())
    write_json(out / "richardson_verification.json", dict(passed=passed, checks=checks,
        maximum_saved_margin_absolute_difference=max(margin_absolute),
        margin_replay_metric="abs(new-old)/max(1,abs(old)); dimensionless margins are unbounded ratios",
        no_success_condition_on_scientific_signs=True, no_propagation_or_timing_repeated=True))
    write_json(out / "richardson_summary.json", dict(passed=passed,
        formulas={"richardson512": "(16*Y512-9*Y384)/7", "richardson384": "(9*Y384-4*Y256)/5"},
        scope="Fixed m32 Craig-Bampton responses on the 32-square grid, baseline matched sets against 128-square or 192-square modal references, all 13 output groups and 11 observation windows. Only the waveform error reference is changed. Extrapolants are conditional h^2 comparators, not exact solutions or certified continuum-error bounds.",
        counts=dict(all_comparator_rows=len(rows), ordinary_channel_windows=len(relevant) // 4),
        ordinary_channel_window_summary=summary, weakest_original_margin_cases=weakest,
        input_sha256={str(p.relative_to(SOURCE)): digest(p) for p in sorted(set(inputs))},
        code_sha256=digest(Path(__file__).resolve())))
    print(json.dumps(dict(passed=passed, checks=checks, ordinary_channel_window_summary=summary), indent=2))
    if not passed:
        raise RuntimeError("Richardson replay verification failed; see richardson_verification.json")


if __name__ == "__main__":
    main()
