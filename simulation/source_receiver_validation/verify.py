#!/usr/bin/env python3
"""Check the continuous reference, projections, saved responses and budgets."""
from common import CFG, HERE, OUT, acoustic, config_digest, dump
import csv
import hashlib
import json
import zipfile
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import expm_multiply


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def modal_response(times, omega, coefficients, observation):
    phases = np.cos(np.outer(times, omega))
    output = np.zeros((len(times), observation.shape[0], coefficients.shape[1]))
    for j in range(len(omega)):
        output += phases[:, j, None, None] * np.outer(observation[:, j], coefficients[j])[None]
    return output


def quadrature_ports(n, initial, receiver_centres):
    points, weights = leggauss(12)
    x = ((np.arange(n)[:, None] + .5 + points[None] / 2) / n).ravel()
    xx = np.stack([acoustic.x_values(x, initial["q_squared"][ell], initial["amplitudes"][ell])
                   .reshape(n, 12, -1).transpose(0, 2, 1) @ (weights / 2)
                   for ell in range(CFG["max_y_mode"] + 1)])
    yy = acoustic.y_values(x, CFG["max_y_mode"]).reshape(n, 12, -1)
    yy = np.einsum("ijl,j->il", yy, weights / 2)
    B = np.einsum("lim,jl,lms->ijs", xx, yy, initial["coefficients"], optimize=True).reshape(n * n, 3)
    mass = acoustic.mass_vector(n)
    B *= np.sqrt(mass)[:, None]
    receivers = []
    for cx, cy in receiver_centres:
        gx = np.exp(-.5 * ((x - cx) / CFG["sigma"]) ** 2).reshape(n, 12) @ (weights / (2 * n))
        gy = np.exp(-.5 * ((x - cy) / CFG["sigma"]) ** 2).reshape(n, 12) @ (weights / (2 * n))
        receivers.append((np.outer(gx, gy) / (gx.sum() * gy.sum())).ravel() / np.sqrt(mass))
    return B, np.stack(receivers, axis=1)


def selected_full_vectors(full, indices):
    n = len(full["y_eigenvectors"])
    raw = full["full_sort_order"][indices]
    ell, longitudinal = np.divmod(raw, n)
    return np.column_stack([np.outer(full["x_eigenvectors"][l, :, m], full["y_eigenvectors"][:, l]).ravel()
                            for l, m in zip(ell, longitudinal)])


def selected_continuous_vectors(n, fine, indices):
    x = (np.arange(n) + .5) / n
    ell = fine["ell"][indices]
    xx = acoustic.x_values(x, fine["q_squared"][indices], fine["amplitudes"][indices], 1 / n)
    yy = acoustic.y_values(x, ell.max())[:, ell] * np.sinc(ell / (2 * n))[None]
    vectors = np.einsum("ik,jk->ijk", xx, yy).reshape(n * n, -1) * np.sqrt(acoustic.mass_vector(n))[:, None]
    return vectors / np.linalg.norm(vectors, axis=0)[None]


def channel_mask(row, layout_config):
    group = row["group"]
    if group == "aggregate":
        return np.ones((4, 3), dtype=bool)
    if group == "channel":
        mask = np.zeros((4, 3), dtype=bool)
        mask[int(row["receiver"]) - 1, int(row["source"]) - 1] = True
        return mask
    sources = layout_config["original_source_centres"] if int(row["source_layout"]) == 0 else CFG["source_centres"]
    receivers = layout_config["original_receiver_centres"] if int(row["receiver_layout"]) == 0 else CFG["receiver_centres"]
    mask = np.asarray([[(r[0] < .5) == (s[0] < .5) for s in sources] for r in receivers])
    return mask if group == "same_side" else ~mask


def main():
    layout_config = json.loads((HERE / "layout_config.json").read_text())
    environment = json.loads((OUT / "environment.json").read_text())
    checks = {"configuration_matches": config_digest() == environment["config_sha256"],
              "layout_configuration_matches": hashlib.sha256((HERE / "layout_config.json").read_bytes()).hexdigest() == environment["layout_config_sha256"]}
    archives = sorted(OUT.glob("*.npz"))
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise IOError(f"Corrupt NPZ: {path.name}")
    reference = np.load(OUT / "continuous_reference.npz")
    factorial = np.load(OUT / "factorial_reference.npz")
    fine = np.load(OUT / "continuous_catalogue.npz")
    t = reference["time"]
    yf = factorial["receiver_pressure"]
    quadrature = acoustic.exact_receiver(t, order=384)
    refined_roots = acoustic.exact_receiver(t, scan_step=.01)
    checks["new_reference_quadrature_relative_difference"] = relative(quadrature, reference["receiver_pressure"])
    checks["new_reference_root_scan_relative_difference"] = relative(refined_roots, reference["receiver_pressure"])
    checks["continuous_initial_energy_max_error"] = float(np.max(abs(.5 * np.sum(reference["coefficients"] ** 2, axis=(0, 1)) - 1)))
    checks["primary_factorial_reference_difference"] = relative(yf[1, 1], reference["receiver_pressure"])
    checks["original_reference_regression_difference"] = relative(yf[0, 0], np.load(HERE.parent / "unequal_speed_interface/results/continuous_reference.npz")["receiver_pressure"])
    source_coefficients = [factorial["original_catalogue_coefficients"], fine["coefficients"]]
    receiver_modes = [factorial["original_catalogue_receiver_modes"], fine["receiver_modes"]]
    checks["factorial_reference_modal_replay_difference"] = max(relative(modal_response(t, fine["omega"], source_coefficients[a], receiver_modes[b]), yf[a, b]) for a in range(2) for b in range(2))
    initial = {key: reference[key] for key in ("omega", "q_squared", "amplitudes", "coefficients")}
    original_initial = initial | {"coefficients": factorial["original_coefficients"]}
    max_values = {key: 0. for key in ("cell_average_initialization_difference", "receiver_cell_quadrature_difference",
        "original_full_response_regression_difference", "original_reduced_response_regression_difference", "original_reduced_frequency_regression_difference",
        "reduced_orthonormality", "projected_eigenproblem_relative_residual", "full_modal_eigenproblem_relative_residual",
        "reduced_full_mac_absolute_difference", "full_continuous_mac_absolute_difference",
        "receiver_modal_replay_difference", "direct_frequency_terms_difference", "independent_budget_absolute_difference",
        "diagonal_budget_absolute_difference", "tabulated_receiver_error_difference", "pressure_error_replay_difference")}
    budget_rows = list(csv.DictReader((OUT / "finite_time_budgets.csv").open()))
    accuracy_rows = list(csv.DictReader((OUT / "accuracy.csv").open()))
    replayed_rows = 0
    for n in CFG["grids"]:
        full = np.load(OUT / f"n{n}_full.npz")
        field = np.load(OUT / f"n{n}_pressure_errors.npz")
        _, A, _, _, _, _, _ = acoustic.assembly(n)
        for data, receiver_positions, bname, rname in ((initial, CFG["receiver_centres"], "B", "R"),
            (original_initial, layout_config["original_receiver_centres"], "original_B", "original_R")):
            B, R = quadrature_ports(n, data, receiver_positions)
            max_values["cell_average_initialization_difference"] = max(max_values["cell_average_initialization_difference"], relative(B, full[bname]))
            max_values["receiver_cell_quadrature_difference"] = max(max_values["receiver_cell_quadrature_difference"], relative(R, full[rname]))
        old = np.load(HERE.parent / f"unequal_speed_interface/results/n{n}_full_receiver.npz")
        max_values["original_full_response_regression_difference"] = max(max_values["original_full_response_regression_difference"], relative(full["factorial_receiver_pressure"][0, 0], old["receiver_pressure"]))
        for m in CFG["modes_per_patch"]:
            model = np.load(OUT / f"n{n}_cb_m{m}.npz")
            W, omega = model["eigenvectors"], model["omega"]
            old = np.load(HERE.parent / f"unequal_speed_interface/results/n{n}_craig_bampton_m{m}.npz")
            max_values["original_reduced_response_regression_difference"] = max(max_values["original_reduced_response_regression_difference"], relative(model["factorial_receiver_pressure"][0, 0], old["receiver_pressure"]))
            max_values["original_reduced_frequency_regression_difference"] = max(max_values["original_reduced_frequency_regression_difference"], relative(omega, old["omega"]))
            max_values["reduced_orthonormality"] = max(max_values["reduced_orthonormality"], float(np.linalg.norm(W.T @ W - np.eye(len(omega)))))
            projected = W.T @ (A @ W)
            max_values["projected_eigenproblem_relative_residual"] = max(max_values["projected_eigenproblem_relative_residual"], relative(projected, np.diag(omega ** 2)))
            hi, fi, ri = (model[key] for key in ("matched_full_indices", "matched_continuous_indices", "matched_reduced_indices"))
            Vh, Vf = selected_full_vectors(full, hi), selected_continuous_vectors(n, fine, fi)
            wh = model["matched_full_omega"]
            max_values["full_modal_eigenproblem_relative_residual"] = max(max_values["full_modal_eigenproblem_relative_residual"], relative(A @ Vh, Vh * wh[None] ** 2))
            max_values["reduced_full_mac_absolute_difference"] = max(max_values["reduced_full_mac_absolute_difference"], float(np.max(abs(np.sum(W[:, ri] * Vh, axis=0) ** 2 - model["mac_reduced_full"]))))
            max_values["full_continuous_mac_absolute_difference"] = max(max_values["full_continuous_mac_absolute_difference"], float(np.max(abs(np.sum(Vh * Vf, axis=0) ** 2 - model["mac_full_continuous"]))))
            for a in range(2):
                for b in range(2):
                    br, cr = model["factorial_source_coefficients"][a], model["factorial_receiver_modes"][b]
                    yr = model["factorial_receiver_pressure"][a, b]
                    direct = modal_response(t, omega, br, cr)
                    max_values["receiver_modal_replay_difference"] = max(max_values["receiver_modal_replay_difference"], relative(direct, yr))
                    Hr = model["factorial_reduced_residues"][a, b]
                    wr, wf = model["matched_reduced_omega"], model["matched_continuous_omega"]
                    Aj = (np.cos(np.outer(t, wr)) - np.cos(np.outer(t, wh)))[:, :, None, None] * Hr[None]
                    Bj = (np.cos(np.outer(t, wh)) - np.cos(np.outer(t, wf)))[:, :, None, None] * Hr[None]
                    aa, bb = Aj.sum(axis=1), Bj.sum(axis=1)
                    qq = yr - yf[a, b] - aa - bb
                    max_values["direct_frequency_terms_difference"] = max(max_values["direct_frequency_terms_difference"],
                        float(np.linalg.norm(aa - model["factorial_ritz_frequency"][a, b]) / np.linalg.norm(yf[a, b])),
                        float(np.linalg.norm(bb - model["factorial_grid_frequency"][a, b]) / np.linalg.norm(yf[a, b])))
                    selected = [row for row in budget_rows if int(row["grid"]) == n and int(row["modes_per_patch"]) == m
                        and int(row["source_layout"]) == a and int(row["receiver_layout"]) == b]
                    for row in selected:
                        active = t <= float(row["time_end"]) + 1e-12
                        channels = channel_mask(row, layout_config)
                        ref = yf[a, b][active][:, channels]
                        e = (yr - yf[a, b])[active][:, channels]
                        ra, gb, remainder = (item[active][:, channels] for item in (aa, bb, qq))
                        den = np.sum(ref ** 2)
                        budget = (-np.sum(ra ** 2) - 2 * np.sum(ra * gb) - 2 * np.sum(ra * remainder)) / den
                        diagonal = (-np.sum(Aj[active][:, :, channels] ** 2) - 2 * np.sum(Aj[active][:, :, channels] * Bj[active][:, :, channels])) / den
                        max_values["independent_budget_absolute_difference"] = max(max_values["independent_budget_absolute_difference"], abs(float(budget) - float(row["delta_squared_error"])))
                        max_values["diagonal_budget_absolute_difference"] = max(max_values["diagonal_budget_absolute_difference"], abs(float(diagonal) - float(row["same_mode_term"])))
                        original_error = float(np.sqrt(np.sum(e ** 2) / den))
                        replaced_error = float(np.sqrt(np.sum((e - ra) ** 2) / den))
                        max_values["tabulated_receiver_error_difference"] = max(max_values["tabulated_receiver_error_difference"],
                            abs(original_error - float(row["total_receiver_error"])), abs(replaced_error - float(row["coarse_replacement_error"])))
                        replayed_rows += 1
            selected_times = [0, 40, 160, 640]
            tt = t[selected_times]
            exact = acoustic.exact_mass_weighted_pressure(n, tt)
            full_field = acoustic.separable_pressure(full, tt)
            modal = np.cos(tt[:, None] * omega[None])[:, :, None] * model["coefficients"][None]
            pressure = np.stack([W @ modal[j] for j in range(len(tt))])
            subcell = acoustic.exact_pressure_norm_squared(tt) - np.sum(exact ** 2, axis=1)
            values = np.sum((pressure - exact) ** 2, axis=1) + subcell
            values_h = np.sum((pressure - full_field) ** 2, axis=1)
            max_values["pressure_error_replay_difference"] = max(max_values["pressure_error_replay_difference"],
                float(np.max(abs(values - field[f"m{m}_continuous_error_squared"][selected_times]))),
                float(np.max(abs(values_h - field[f"m{m}_same_grid_error_squared"][selected_times]))))
    n = 16
    mass, _, J, B, R, _, _ = acoustic.assembly(n)
    state0 = np.zeros((J.shape[0], 3))
    state0[:n * n] = B
    state = expm_multiply(J, state0, start=0., stop=.5, num=3, traceA=0.)
    separated = acoustic.separable_setup(n)
    expected = acoustic.separable_pressure(separated, [0., .25, .5])
    checks["independent_sparse_propagation_pressure_difference"] = relative(state[:, :n * n], expected)
    checks["independent_sparse_propagation_receiver_difference"] = relative(np.einsum("ir,tis->trs", R, state[:, :n * n]), acoustic.separable_receiver(separated, [0., .25, .5]))
    checks["independent_sparse_energy_drift"] = float(np.max(abs(np.sum(state ** 2, axis=1) / np.sum(state0 ** 2, axis=0) - 1)))
    checks.update(max_values)
    numeric = {key: value for key, value in checks.items() if not isinstance(value, bool)}
    tolerances = {key: 5e-10 for key in numeric}
    tolerances["independent_budget_absolute_difference"] = 1e-11
    tolerances["diagonal_budget_absolute_difference"] = 1e-11
    tolerances["tabulated_receiver_error_difference"] = 1e-11
    passed = all(value for value in checks.values() if isinstance(value, bool)) and all(value < tolerances[key] for key, value in numeric.items()) and replayed_rows == len(budget_rows) == 1080
    record = {"passed": passed, "npz_archives_crc_checked": len(archives), "budget_rows_replayed": replayed_rows,
              "accuracy_rows_checked": len(accuracy_rows), "checks": checks, "tolerances": tolerances,
              "configuration_sha256": config_digest()}
    dump(OUT / "verification.json", record)
    print(f"Verified {len(archives)} NPZ archives and {replayed_rows} exact budgets: {'passed' if passed else 'failed'}.")
    print(f"Maximum independent budget difference: {max_values['independent_budget_absolute_difference']:.3e}; report: results/verification.json")
    if not passed:
        raise RuntimeError("Second-layout verification failed")


if __name__ == "__main__":
    main()
