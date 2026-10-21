#!/usr/bin/env python3
"""Evaluate relocation examples and matching/path sensitivity from spectra."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ORIGINAL = SOURCE / "source_receiver_validation"
CFG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(1, str(SOURCE))
from common import acoustic, relative_gaps


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def project_separated(full, vectors):
    """Tensor projection against the complete separated full eigenbasis."""
    n = len(full["y_eigenvectors"])
    by_y = np.einsum("ijk,jl->lik", vectors.reshape(n, n, -1), full["y_eigenvectors"], optimize=True)
    return np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), by_y).reshape(n * n, -1)


def continuous_overlap(full, fine):
    """Cell-average continuous modes, normalized in discrete mass coordinates."""
    n = len(full["y_eigenvectors"])
    x = (np.arange(n) + 0.5) / n
    ell = fine["ell"]
    xavg = acoustic.x_values(x, fine["q_squared"], fine["amplitudes"], 1 / n)
    yavg = acoustic.y_values(x, ell.max())[:, ell] * np.sinc(ell / (2 * n))[None]
    xweighted = xavg * np.sqrt(full["mass"].reshape(n, n)[:, 0])[:, None]
    norms = np.sqrt(np.sum(xweighted ** 2, axis=0) * np.sum(yavg ** 2, axis=0))
    ox = np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), xweighted[None])
    oy = full["y_eigenvectors"].T @ yavg
    return (ox * oy[:, None, :]).reshape(n * n, -1) / norms[None], norms


def assignment(overlap, omega_left, omega_right):
    left, right = linear_sum_assignment(-(overlap ** 2))
    return dict(left=left, right=right, mac=overlap[left, right] ** 2,
                sign=np.sign(overlap[left, right]),
                gap_left=relative_gaps(omega_left)[left],
                gap_right=relative_gaps(omega_right)[right],
                omega_left=omega_left[left], omega_right=omega_right[right])


def accepted(a, mac, gap):
    return ((a["mac"] >= mac) & (a["gap_left"] >= gap) & (a["gap_right"] >= gap)
            & (a["omega_left"] > 1e-7) & (a["omega_right"] > 1e-7))


def chain(reduced_assignment, continuum_assignment, fine, norms, mac, gap):
    ok = accepted(continuum_assignment, mac, gap)
    jj = continuum_assignment["right"]
    ok &= (fine["omega"][jj] <= float(fine["active_max"]) + 1e-9) & (norms[jj] > 1e-8)
    continuum = {int(i): (int(j), k) for k, (i, j, yes) in enumerate(zip(
        continuum_assignment["left"], jj, ok)) if yes}
    triples = []
    for k, (i, j, yes) in enumerate(zip(reduced_assignment["left"], reduced_assignment["right"],
                                      accepted(reduced_assignment, mac, gap))):
        if yes and int(j) in continuum:
            f, q = continuum[int(j)]
            triples.append((int(i), int(j), f, k, q))
    return triples


def budget(case, position, layout, times, yf, yr, model, fine, reference,
           triples, ar, af, omega_h, theta):
    s, r = case["source"] - 1, case["receiver"] - 1
    a, b = layout
    ri, hi, fi, kr, kf = np.asarray(triples, dtype=int).T
    sr, sf = ar["sign"][kr], af["sign"][kf]
    br = model["factorial_source_coefficients"][a, ri, s] * sr
    cr = model["factorial_receiver_modes"][b, r, ri] * sr
    bf_full = (reference["original_catalogue_coefficients"] if a == 0 else fine["coefficients"])
    cf_full = (reference["original_catalogue_receiver_modes"] if b == 0 else fine["receiver_modes"])
    bf, cf = bf_full[fi, s] * sf, cf_full[r, fi] * sf
    hr, hf = cr * br, cf * bf
    wr, wh, wf = model["omega"][ri], omega_h[hi], fine["omega"][fi]
    zr, zh, zf = (np.cos(np.outer(times, w)) for w in (wr, wh, wf))
    htheta = (1 - theta) * hr + theta * hf
    zcoupling = (1 - theta) * zf + theta * zr
    aj, bj = (zr - zh) * htheta[None], (zh - zf) * htheta[None]
    aa, bb = aj.sum(axis=1), bj.sum(axis=1)
    source = zcoupling @ ((cr + cf) * (br - bf) / 2)
    receiver = zcoupling @ ((cr - cf) * (br + bf) / 2)
    selected_difference = zr @ hr - zf @ hf
    error = yr - yf
    unmatched = error - selected_difference
    qq = source + receiver + unmatched
    den = float(yf @ yf)
    err0 = error - aa
    total = float(np.linalg.norm(error) / np.sqrt(den))
    replaced = float(np.linalg.norm(err0) / np.sqrt(den))
    exact = float((err0 @ err0 - error @ error) / den)
    same = float((-np.sum(aj * aj) - 2 * np.sum(aj * bj)) / den)
    frequency = float((-aa @ aa - 2 * aa @ bb) / den)
    remainder = float(-2 * aa @ qq / den)
    omitted = yf - zf @ hf
    independent = float((aa @ aa - 2 * aa @ error) / den)
    row = dict(case_id=case["case_id"], grid=case["grid"], modes_per_patch=case["modes_per_patch"],
               time_end=case["time_end"], source=case["source"], receiver=case["receiver"],
               position=position, source_layout=a, receiver_layout=b,
               matched_count=len(ri), total_receiver_error=total,
               coarse_replacement_error=replaced, change_percentage_points=100 * (replaced - total),
               delta_squared_error=exact, same_mode_term=same, cross_mode_term=frequency - same,
               remainder_cross_term=remainder,
               source_cross_term=float(-2 * aa @ source / den),
               receiver_cross_term=float(-2 * aa @ receiver / den),
               unmatched_cross_term=float(-2 * aa @ unmatched / den),
               matched_reference_omission=float(np.linalg.norm(omitted) / np.sqrt(den)),
               matched_reference_response_norm=float(np.linalg.norm(zf @ hf) / np.sqrt(den)),
               reference_squared_norm=den,
               ritz_relative_norm=float(np.linalg.norm(aa) / np.sqrt(den)),
               grid_relative_norm=float(np.linalg.norm(bb) / np.sqrt(den)),
               decomposition_relative_defect=float(np.linalg.norm(error - aa - bb - qq) / np.sqrt(den)),
               budget_absolute_defect=abs(exact - frequency - remainder),
               independent_budget_absolute_defect=abs(exact - independent))
    waves = dict(y_reference=yf, y_rom=yr, y_no_ritz=yr - aa, ritz_frequency=aa,
                 grid_frequency=bb, q_source=source, q_receiver=receiver, q_unmatched=unmatched)
    return row, waves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = ORIGINAL / "results"
    reference = load_npz(src / "factorial_reference.npz")
    fine = load_npz(src / "continuous_catalogue.npz")
    times_all = reference["time"]
    input_paths = [src / "factorial_reference.npz", src / "continuous_catalogue.npz",
                   ORIGINAL / "config.json", ORIGINAL / "layout_config.json"]
    records, pair_rows, baseline, wave_rows, contrast_rows = [], [], [], [], []
    verification = {"baseline_matching_indices_identical": True,
                    "baseline_modal_assignment_mac_max_absolute_difference": 0.,
                    "baseline_waveform_relative_difference": 0.,
                    "reference_modal_replay_relative_difference": 0.,
                    "rom_modal_replay_relative_difference": 0.,
                    "baseline_budget_max_absolute_difference": 0.}
    old_rows = list(csv.DictReader((src / "finite_time_budgets.csv").open()))
    settings = [(m, g) for m in CFG["mac_thresholds"] for g in CFG["relative_frequency_gaps"]]
    for case in CFG["cases"]:
        print(f"Replaying {case['case_id']} and recomputing global assignments", flush=True)
        n, m = case["grid"], case["modes_per_patch"]
        fullpath, rompath = src / f"n{n}_full.npz", src / f"n{n}_cb_m{m}.npz"
        input_paths.extend((fullpath, rompath))
        full, model = load_npz(fullpath), load_npz(rompath)
        order = np.argsort(full["omega"].ravel())
        omega_h = full["omega"].ravel()[order]
        overlap, norms = continuous_overlap(full, fine)
        af = assignment(overlap[order], omega_h, fine["omega"])
        del overlap
        overlap = project_separated(full, model["eigenvectors"])[order].T
        ar = assignment(overlap, model["omega"], omega_h)
        del overlap
        selections = {(mac, gap): chain(ar, af, fine, norms, mac, gap) for mac, gap in settings}
        intersection = sorted(set.intersection(*(set(v) for v in selections.values())))
        base = selections[(CFG["baseline_mac"], CFG["baseline_gap"])]
        ri, hi, fi, kr, kf = np.asarray(base).T
        for actual, key in ((ri, "matched_reduced_indices"), (hi, "matched_full_indices"), (fi, "matched_continuous_indices")):
            verification["baseline_matching_indices_identical"] &= bool(np.array_equal(actual, model[key]))
        verification["baseline_modal_assignment_mac_max_absolute_difference"] = max(
            verification["baseline_modal_assignment_mac_max_absolute_difference"],
            float(np.max(abs(ar["mac"][kr] - model["mac_reduced_full"]))),
            float(np.max(abs(af["mac"][kf] - model["mac_full_continuous"]))))
        active = times_all <= case["time_end"] + 1e-12
        times = times_all[active]
        s, r = case["source"] - 1, case["receiver"] - 1
        own_selections = [(f"mac{mac:.2f}_gap{gap:.4f}", "threshold", mac, gap, triples)
                          for (mac, gap), triples in selections.items()]
        scopes = own_selections + [("intersection", "intersection", "", "", intersection)]
        case_records = []
        for selection_id, selection, mac, gap, triples in scopes:
            for i, j, k, qi, qj in triples:
                pair_rows.append(dict(case_id=case["case_id"], selection_id=selection_id,
                    selection=selection, mac_threshold=mac, relative_gap=gap,
                    reduced_index=i, full_index=j, continuous_index=k,
                    mac_reduced_full=float(ar["mac"][qi]), mac_full_continuous=float(af["mac"][qj]),
                    gap_reduced=float(ar["gap_left"][qi]), gap_full=float(ar["gap_right"][qi]),
                    gap_continuous=float(af["gap_right"][qj]),
                    omega_reduced=float(model["omega"][i]), omega_full=float(omega_h[j]),
                    omega_continuous=float(fine["omega"][k]),
                    in_intersection=(i, j, k, qi, qj) in intersection))
            for path, theta in CFG["residue_paths"].items():
                comparison = []
                for position in ("before", "after"):
                    layout = case[position]
                    a, b = layout
                    yf = reference["receiver_pressure"][a, b, active, r, s]
                    yr = model["factorial_receiver_pressure"][a, b, active, r, s]
                    row, waves = budget(case, position, layout, times, yf, yr, model, fine,
                                        reference, triples, ar, af, omega_h, theta)
                    row.update(selection_id=selection_id, selection=selection,
                               mac_threshold=mac, relative_gap=gap, residue_path=path, theta=theta,
                               intersection_count=len(intersection))
                    records.append(row)
                    case_records.append(row)
                    comparison.append(row)
                    if mac == CFG["baseline_mac"] and gap == CFG["baseline_gap"] and path == "rom":
                        baseline.append(row.copy())
                        for idx, time in enumerate(times):
                            wave_rows.append(dict(case_id=case["case_id"], position=position, source_layout=a,
                                receiver_layout=b, time=float(time), **{k: float(v[idx]) for k, v in waves.items()}))
                        for new, old in (("ritz_frequency", "factorial_ritz_frequency"),
                                         ("grid_frequency", "factorial_grid_frequency"),
                                         ("q_source", "factorial_source_coupling"),
                                         ("q_receiver", "factorial_receiver_coupling"),
                                         ("q_unmatched", "factorial_unmatched")):
                            verification["baseline_waveform_relative_difference"] = max(
                                verification["baseline_waveform_relative_difference"],
                                float(np.linalg.norm(waves[new] - model[old][a, b, active, r, s]) / np.linalg.norm(yf)))
                        old = next(z for z in old_rows if int(z["grid"]) == n and int(z["modes_per_patch"]) == m
                            and float(z["time_end"]) == case["time_end"] and int(z["source_layout"]) == a
                            and int(z["receiver_layout"]) == b and z["group"] == "channel"
                            and int(z["source"]) == s + 1 and int(z["receiver"]) == r + 1)
                        for key in ("delta_squared_error", "change_percentage_points", "same_mode_term",
                                    "cross_mode_term", "source_cross_term", "receiver_cross_term", "unmatched_cross_term"):
                            verification["baseline_budget_max_absolute_difference"] = max(
                                verification["baseline_budget_max_absolute_difference"], abs(row[key] - float(old[key])))
                        bf_all = (reference["original_catalogue_coefficients"] if a == 0 else fine["coefficients"])[:, s]
                        cf_all = (reference["original_catalogue_receiver_modes"] if b == 0 else fine["receiver_modes"])[r]
                        # Independently sum modes to verify the computed trajectories.
                        direct_ref = np.zeros_like(times)
                        for w, hh in zip(fine["omega"], bf_all * cf_all):
                            direct_ref += np.cos(times * w) * hh
                        direct_rom = np.zeros_like(times)
                        hrom = model["factorial_source_coefficients"][a, :, s] * model["factorial_receiver_modes"][b, r]
                        for w, hh in zip(model["omega"], hrom):
                            direct_rom += np.cos(times * w) * hh
                        verification["reference_modal_replay_relative_difference"] = max(
                            verification["reference_modal_replay_relative_difference"], relative(direct_ref, yf))
                        verification["rom_modal_replay_relative_difference"] = max(
                            verification["rom_modal_replay_relative_difference"], relative(direct_rom, yr))
                before, after = comparison
                contrast_rows.append(dict(case_id=case["case_id"], selection_id=selection_id,
                    selection=selection, mac_threshold=mac, relative_gap=gap, residue_path=path, theta=theta,
                    matched_count=len(triples), before_change_percentage_points=before["change_percentage_points"],
                    after_change_percentage_points=after["change_percentage_points"],
                    before_delta_squared_error=before["delta_squared_error"],
                    after_delta_squared_error=after["delta_squared_error"],
                    sign_reversal=(before["delta_squared_error"] * after["delta_squared_error"] < 0
                        and min(abs(before["delta_squared_error"]), abs(after["delta_squared_error"])) > CFG["sign_tolerance"])))
        write_csv(out / "baseline_waveforms.csv", wave_rows)
        write_csv(out / "baseline_budgets.csv", baseline)
        print(f"{case['case_id']}: {len(base)} baseline triples, {len(intersection)} intersection triples; waveforms written", flush=True)
        del full, model

    write_csv(out / "threshold_budgets.csv", records)
    write_csv(out / "threshold_contrasts.csv", contrast_rows)
    write_csv(out / "matched_sets.csv", pair_rows)
    verification["maximum_decomposition_relative_defect"] = max(r["decomposition_relative_defect"] for r in records)
    verification["maximum_budget_absolute_defect"] = max(r["budget_absolute_defect"] for r in records)
    verification["maximum_independent_budget_absolute_defect"] = max(r["independent_budget_absolute_defect"] for r in records)
    verification["all_expected_rows_present"] = len(records) == 120 and len(contrast_rows) == 60 and len(baseline) == 4
    verification["q_subterms_sum_defect"] = max(abs(r["remainder_cross_term"] - r["source_cross_term"]
                                                    - r["receiver_cross_term"] - r["unmatched_cross_term"]) for r in records)
    numeric = {k: v for k, v in verification.items() if not isinstance(v, bool)}
    gates = {k: (bool(v) if isinstance(v, bool) else v < 1e-10) for k, v in verification.items()}
    passed = all(gates.values())
    write_json(out / "verification.json", dict(passed=passed, checks=verification, gates=gates,
        numerical_tolerance=1e-10, budget_rows=len(records), comparison_rows=len(contrast_rows),
        no_success_condition_on_scientific_signs=True))
    case_summaries = []
    for case in CFG["cases"]:
        rows = [r for r in contrast_rows if r["case_id"] == case["case_id"]]
        base_rows = [r for r in baseline if r["case_id"] == case["case_id"]]
        path_summaries = []
        for path in CFG["residue_paths"]:
            selected = [r for r in rows if r["residue_path"] == path and r["selection"] == "threshold"]
            common = next(r for r in rows if r["residue_path"] == path and r["selection"] == "intersection")
            path_summaries.append(dict(residue_path=path, threshold_comparisons=len(selected),
                sign_reversals=sum(r["sign_reversal"] for r in selected),
                matched_count_range=[min(r["matched_count"] for r in selected), max(r["matched_count"] for r in selected)],
                before_change_percentage_points_range=[min(r["before_change_percentage_points"] for r in selected),
                                                       max(r["before_change_percentage_points"] for r in selected)],
                after_change_percentage_points_range=[min(r["after_change_percentage_points"] for r in selected),
                                                      max(r["after_change_percentage_points"] for r in selected)],
                intersection=common))
        case_summaries.append(dict(case=case, baseline=base_rows, path_sensitivity=path_summaries))
    write_json(out / "summary.json", dict(config=CFG, cases=case_summaries, verification_passed=passed,
        interpretation="Signs describe particular matched-set interventions. The theta=0 ROM-residue intervention is the main manuscript definition; theta=1 and 0.5 redistribute phase/residue interaction and therefore define different interventions. A change of sign across these paths is reported as attribution sensitivity, not a numerical failure. The sample count is a deterministic sensitivity grid, not independent trials or an estimated occurrence probability.",
        counts=dict(budgets=len(records), contrasts=len(contrast_rows), mode_rows=len(pair_rows), waveform_rows=len(wave_rows))))
    write_json(out / "environment.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
        input_sha256={str(p.relative_to(SOURCE)): sha256(p) for p in sorted(set(input_paths))},
        analysis_sha256=sha256(Path(__file__).resolve()), config_sha256=sha256(HERE / "config.json")))
    print(json.dumps(dict(passed=passed, counts=dict(budgets=len(records), comparisons=len(contrast_rows)),
                          sensitivity=[dict(case=c["case"]["case_id"], paths=c["path_sensitivity"]) for c in case_summaries]), indent=2))
    if not passed:
        raise RuntimeError("Relocation sensitivity reproducibility checks failed; see verification.json")


if __name__ == "__main__":
    main()
