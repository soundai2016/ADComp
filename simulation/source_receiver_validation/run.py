#!/usr/bin/env python3
"""Compute both fixed port layouts and their exact modal interventions."""
from common import CFG, HERE, OUT, acoustic, config_digest, dump, residue, response, relative_gaps, save_arrays
import argparse
from datetime import datetime, timezone
import hashlib
import platform
import sys
import time
import numpy as np
import scipy
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_info
from accuracy_cost.run import cb_basis


def catalogue(initial=None, receiver_centres=None):
    initial = acoustic.continuous_data() if initial is None else initial
    receiver_centres = CFG["receiver_centres"] if receiver_centres is None else receiver_centres
    active_max = float(initial["omega"].max())
    halo = active_max + np.pi * np.sqrt(max(np.asarray(CFG["bulk_modulus"]) / np.asarray(CFG["density"])))
    count = int(np.floor(halo / np.pi)) + 2
    entries = []
    for ell in range(count):
        for m, omega in enumerate(acoustic.roots(ell, count + 1)):
            if omega <= halo:
                q2, amplitudes = acoustic.mode_parameters(omega, ell)
                entries.append((omega, m, ell, q2, amplitudes))
    entries.sort(key=lambda row: row[0])
    omega = np.asarray([row[0] for row in entries])
    mm = np.asarray([row[1] for row in entries], dtype=int)
    ll = np.asarray([row[2] for row in entries], dtype=int)
    q2 = np.stack([row[3] for row in entries])
    amplitudes = np.stack([row[4] for row in entries])
    coefficients = np.zeros((len(omega), 3))
    for j, (m, ell) in enumerate(zip(mm, ll)):
        if m <= CFG["max_x_mode"] and ell <= CFG["max_y_mode"]:
            coefficients[j] = initial["coefficients"][ell, m]
    order = CFG["quadrature_order"]
    a = CFG["interface_x"]
    xl, wl = acoustic.gauss_interval(order, 0., a)
    xr, wr = acoustic.gauss_interval(order, a, 1.)
    x, wx = np.r_[xl, xr], np.r_[wl, wr]
    y, wy = acoustic.gauss_interval(order, 0., 1.)
    X = acoustic.x_values(x, q2, amplitudes)
    Y = acoustic.y_values(y, ll.max())[:, ll]
    receivers = np.empty((4, len(omega)))
    for r, (cx, cy) in enumerate(receiver_centres):
        gx = np.exp(-.5 * ((x - cx) / CFG["sigma"]) ** 2)
        gy = np.exp(-.5 * ((y - cy) / CFG["sigma"]) ** 2)
        receivers[r] = (X.T @ (wx * gx)) * (Y.T @ (wy * gy)) / (
            acoustic.gaussian_integrals([0., 1.], cx)[0] * acoustic.gaussian_integrals([0., 1.], cy)[0])
    return {"omega": omega, "m": mm, "ell": ll, "q_squared": q2, "amplitudes": amplitudes,
            "coefficients": coefficients, "receiver_modes": receivers, "active_max": active_max, "halo_max": halo}


def original_continuous_data(layout_config):
    acoustic.CFG["source_centres"] = layout_config["original_source_centres"]
    acoustic.CFG["receiver_centres"] = layout_config["original_receiver_centres"]
    acoustic.continuous_data.cache_clear()
    try:
        data = {key: value.copy() for key, value in acoustic.continuous_data().items()}
    finally:
        acoustic.CFG["source_centres"] = CFG["source_centres"]
        acoustic.CFG["receiver_centres"] = CFG["receiver_centres"]
        acoustic.continuous_data.cache_clear()
    return data


def original_ports(n, mass, initial, layout_config):
    xx, yy = acoustic.averaged_modes(n)
    B = acoustic.synthesize(xx, yy, initial["omega"], initial["coefficients"], [0.])[0] * np.sqrt(mass)[:, None]
    edges = np.linspace(0., 1., n + 1)
    windows = []
    for cx, cy in layout_config["original_receiver_centres"]:
        gx = acoustic.gaussian_integrals(edges, cx)
        gy = acoustic.gaussian_integrals(edges, cy)
        windows.append(np.outer(gx / gx.sum(), gy / gy.sum()).ravel() / np.sqrt(mass))
    return B, np.stack(windows, axis=-1)


def factorial_responses(times, omega, sources, receivers):
    return np.stack([np.stack([response(times, omega, residue(source, receiver))
                              for receiver in receivers]) for source in sources])


def factorial_terms(model, fine_catalogues, times, reference, sources, receivers):
    ri, fi = model["matched_reduced_indices"], model["matched_continuous_indices"]
    sr, sf = model["sign_reduced_full"], model["sign_continuous_full"]
    wr, wh, wf = (model[key] for key in ("matched_reduced_omega", "matched_full_omega", "matched_continuous_omega"))
    terms = {key: np.empty_like(reference) for key in (
        "receiver_pressure", "ritz_frequency", "grid_frequency", "source_coupling", "receiver_coupling", "unmatched")}
    h_reduced = np.empty((2, 2, len(ri), 4, 3))
    h_continuous = np.empty_like(h_reduced)
    for source_layout in range(2):
        for receiver_layout in range(2):
            br, cr = sources[source_layout][ri] * sr[:, None], receivers[receiver_layout][:, ri] * sr[None]
            bf = fine_catalogues[source_layout]["coefficients"][fi] * sf[:, None]
            cf = fine_catalogues[receiver_layout]["receiver_modes"][:, fi] * sf[None]
            Hr, Hf = residue(br, cr), residue(bf, cf)
            h_reduced[source_layout, receiver_layout], h_continuous[source_layout, receiver_layout] = Hr, Hf
            yr = response(times, model["omega"], residue(sources[source_layout], receivers[receiver_layout]))
            ritz = response(times, wr, Hr) - response(times, wh, Hr)
            grid = response(times, wh, Hr) - response(times, wf, Hr)
            source = response(times, wf, residue(br - bf, (cr + cf) / 2))
            receiver = response(times, wf, residue((br + bf) / 2, cr - cf))
            values = (yr, ritz, grid, source, receiver, yr - reference[source_layout, receiver_layout] - ritz - grid - source - receiver)
            for key, value in zip(terms, values):
                terms[key][source_layout, receiver_layout] = value
    return {"factorial_" + key: value for key, value in terms.items()} | {
        "factorial_source_coefficients": np.stack(sources), "factorial_receiver_modes": np.stack(receivers),
        "factorial_reduced_residues": h_reduced, "factorial_continuous_residues": h_continuous}


def project_separated(full, vectors):
    n = len(full["y_eigenvectors"])
    by_y = np.einsum("ijk,jl->lik", vectors.reshape(n, n, -1), full["y_eigenvectors"], optimize=True)
    return np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), by_y).reshape(n * n, -1)


def continuous_overlap(full, fine):
    n = len(full["y_eigenvectors"])
    centres = (np.arange(n) + .5) / n
    ll = fine["ell"]
    xavg = acoustic.x_values(centres, fine["q_squared"], fine["amplitudes"], 1 / n)
    yavg = acoustic.y_values(centres, ll.max())[:, ll] * np.sinc(ll / (2 * n))[None]
    xweighted = xavg * np.sqrt(full["mass"].reshape(n, n)[:, 0])[:, None]
    norms = np.sqrt(np.sum(xweighted ** 2, axis=0) * np.sum(yavg ** 2, axis=0))
    ox = np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), xweighted[None])
    oy = full["y_eigenvectors"].T @ yavg
    overlap = (ox * oy[:, None, :]).reshape(n * n, -1) / norms[None]
    return overlap, norms


def match_overlap(overlap, left_omega, right_omega):
    left, right = linear_sum_assignment(-(overlap ** 2))
    mac = overlap[left, right] ** 2
    accepted = ((mac >= CFG["mac_threshold"])
                & (relative_gaps(left_omega)[left] >= CFG["relative_frequency_gap"])
                & (relative_gaps(right_omega)[right] >= CFG["relative_frequency_gap"])
                & (left_omega[left] > 1e-7) & (right_omega[right] > 1e-7))
    return left, right, mac, np.sign(overlap[left, right]), accepted


def modal_terms(full, reduced, fine, continuum_pairs, times, reference):
    order = np.argsort(full["omega"].ravel())
    coarse_omega = full["omega"].ravel()[order]
    overlap = project_separated(full, reduced["eigenvectors"])[order].T
    ri0, ci0, mac0, signs0, accepted = match_overlap(overlap, reduced["omega"], coarse_omega)
    chain = [(int(i), int(j), continuum_pairs[int(j)][0], float(mac), continuum_pairs[int(j)][1],
              float(sign), continuum_pairs[int(j)][2])
             for i, j, mac, sign, ok in zip(ri0, ci0, mac0, signs0, accepted)
             if ok and int(j) in continuum_pairs]
    ri = np.asarray([row[0] for row in chain], dtype=int)
    ci = np.asarray([row[1] for row in chain], dtype=int)
    fi = np.asarray([row[2] for row in chain], dtype=int)
    sr = np.asarray([row[5] for row in chain])
    sf = np.asarray([row[6] for row in chain])
    br = reduced["coefficients"][ri] * sr[:, None]
    cr = reduced["observation"][:, ri] * sr[None]
    bf = fine["coefficients"][fi] * sf[:, None]
    cf = fine["receiver_modes"][:, fi] * sf[None]
    Hr, Hf = residue(br, cr), residue(bf, cf)
    wr, wh, wf = reduced["omega"][ri], coarse_omega[ci], fine["omega"][fi]
    ritz = response(times, wr, Hr) - response(times, wh, Hr)
    grid = response(times, wh, Hr) - response(times, wf, Hr)
    source = response(times, wf, residue(br - bf, (cr + cf) / 2))
    receiver = response(times, wf, residue((br + bf) / 2, cr - cf))
    yr = reduced["receiver_pressure"]
    unmatched = yr - reference - ritz - grid - source - receiver
    return {"matched_reduced_indices": ri, "matched_full_indices": ci, "matched_continuous_indices": fi,
            "mac_reduced_full": np.asarray([row[3] for row in chain]),
            "mac_full_continuous": np.asarray([row[4] for row in chain]),
            "sign_reduced_full": sr, "sign_continuous_full": sf,
            "matched_reduced_omega": wr, "matched_full_omega": wh, "matched_continuous_omega": wf,
            "matched_reduced_residues": Hr, "matched_continuous_residues": Hf,
            "ritz_frequency": ritz, "grid_frequency": grid, "source_coupling": source,
            "receiver_coupling": receiver, "unmatched": unmatched,
            "no_ritz_frequency_response": yr - ritz,
            "continuous_frequency_response": yr - ritz - grid,
            "matched_continuous_response": response(times, wf, Hf)}


def pressure_terms(n, full, reduced, times):
    data = {"continuous_norm_squared": [], "full_norm_squared": [], "projection_floor_squared": [],
            "full_continuous_error_squared": []}
    for m in reduced:
        data[f"m{m}_continuous_error_squared"] = []
        data[f"m{m}_same_grid_error_squared"] = []
    minimum = np.inf
    for start in range(0, len(times), 16):
        tt = times[start:start + 16]
        exact = acoustic.exact_mass_weighted_pressure(n, tt)
        discrete = acoustic.separable_pressure(full, tt)
        continuous = acoustic.exact_pressure_norm_squared(tt)
        subcell = continuous - np.sum(exact ** 2, axis=1)
        minimum = min(minimum, float(subcell.min()))
        if minimum < -1e-9:
            raise RuntimeError("Negative subcell projection norm")
        subcell = np.maximum(subcell, 0.)
        data["continuous_norm_squared"].append(continuous)
        data["full_norm_squared"].append(np.sum(discrete ** 2, axis=1))
        data["projection_floor_squared"].append(subcell)
        data["full_continuous_error_squared"].append(np.sum((discrete - exact) ** 2, axis=1) + subcell)
        for m, model in reduced.items():
            modal = np.cos(tt[:, None] * model["omega"][None])[:, :, None] * model["coefficients"][None]
            field = (model["eigenvectors"] @ modal.transpose(1, 0, 2).reshape(len(model["omega"]), -1))
            field = field.reshape(n * n, len(tt), 3).transpose(1, 0, 2)
            data[f"m{m}_continuous_error_squared"].append(np.sum((field - exact) ** 2, axis=1) + subcell)
            data[f"m{m}_same_grid_error_squared"].append(np.sum((field - discrete) ** 2, axis=1))
    data = {key: np.concatenate(value) for key, value in data.items()}
    data["minimum_subcell_squared_norm"] = np.asarray(minimum)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", choices=CFG["grids"], default=CFG["grids"])
    args = parser.parse_args()
    import json
    layout_config = json.loads((HERE / "layout_config.json").read_text())
    OUT.mkdir(exist_ok=True)
    dump(OUT / "environment.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
         "config_sha256": config_digest(), "python": sys.version, "platform": platform.platform(),
         "numpy": np.__version__, "scipy": scipy.__version__, "threadpools": threadpool_info(),
         "layout_config_sha256": hashlib.sha256((HERE / "layout_config.json").read_bytes()).hexdigest(),
         "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob("*.py")}})
    times = np.linspace(0., CFG["time_end"], CFG["samples"])
    original_initial = original_continuous_data(layout_config)
    reference = acoustic.exact_receiver(times)
    fine = catalogue()
    original_fine = catalogue(original_initial, layout_config["original_receiver_centres"])
    fine_catalogues = [original_fine, fine]
    factorial_reference = factorial_responses(times, fine["omega"],
        [entry["coefficients"] for entry in fine_catalogues], [entry["receiver_modes"] for entry in fine_catalogues])
    save_arrays(OUT / "continuous_reference.npz", time=times, receiver_pressure=reference, **acoustic.continuous_data())
    save_arrays(OUT / "continuous_catalogue.npz", **fine)
    save_arrays(OUT / "factorial_reference.npz", time=times, receiver_pressure=factorial_reference,
        original_coefficients=original_initial["coefficients"], original_receiver_modes=original_initial["receiver_modes"],
        original_catalogue_coefficients=original_fine["coefficients"], original_catalogue_receiver_modes=original_fine["receiver_modes"])
    for n in args.grids:
        started = time.perf_counter()
        print(f"Grid {n}: separated full spectrum and complete modal matching", flush=True)
        full = acoustic.separable_setup(n)
        mass, A, _, B, R, _, _ = acoustic.assembly(n)
        original_B, original_R = original_ports(n, mass, original_initial, layout_config)
        full["B"], full["R"] = B, R
        full_receiver = acoustic.separable_receiver(full, times)
        factorial_B = [original_B, B]
        factorial_R = [original_R, R]
        full_sources = [project_separated(full, item) for item in factorial_B]
        full_receivers = [project_separated(full, item).T for item in factorial_R]
        factorial_full = factorial_responses(times, full["omega"].ravel(), full_sources, full_receivers)
        order = np.argsort(full["omega"].ravel())
        omega_h = full["omega"].ravel()[order]
        overlap, norms = continuous_overlap(full, fine)
        ii, jj, mac, signs, accepted = match_overlap(overlap[order], omega_h, fine["omega"])
        accepted &= (fine["omega"][jj] <= fine["active_max"] + 1e-9) & (norms[jj] > 1e-8)
        cf = {int(i): (int(j), float(mm), float(sign)) for i, j, mm, sign, ok in zip(ii, jj, mac, signs, accepted) if ok}
        save_arrays(OUT / f"n{n}_full.npz", time=times, receiver_pressure=full_receiver,
                    full_sort_order=order, continuum_full_indices=np.asarray(list(cf), dtype=int),
                    continuum_indices=np.asarray([row[0] for row in cf.values()], dtype=int),
                    continuum_mac=np.asarray([row[1] for row in cf.values()]),
                    original_B=original_B, original_R=original_R, factorial_receiver_pressure=factorial_full, **full)
        del overlap
        models = {}
        for m in CFG["modes_per_patch"]:
            print(f"Grid {n}: Craig--Bampton m={m}", flush=True)
            Q = cb_basis(n, mass, A, m)
            Ar = Q.T @ (A @ Q)
            lam, V = linalg.eigh((Ar + Ar.T) / 2, check_finite=False, driver="evd")
            if lam.min() < -1e-9 * max(lam.max(), 1.):
                raise RuntimeError("Indefinite reduced stiffness")
            lam[lam < 1e-9] = 0.
            W = Q @ V
            coefficients, observation = W.T @ B, R.T @ W
            omega = np.sqrt(lam)
            model = {"omega": omega, "eigenvectors": W, "coefficients": coefficients, "observation": observation,
                     "receiver_pressure": response(times, omega, residue(coefficients, observation))}
            model.update(modal_terms(full, model, fine, cf, times, reference))
            model.update(factorial_terms(model, fine_catalogues, times, factorial_reference,
                [W.T @ item for item in factorial_B], [item.T @ W for item in factorial_R]))
            save_arrays(OUT / f"n{n}_cb_m{m}.npz", time=times, **model)
            models[m] = model
            print(f"Grid {n}, m={m}: {len(model['matched_reduced_indices'])} accepted triples", flush=True)
        print(f"Grid {n}: continuous pressure-field errors", flush=True)
        save_arrays(OUT / f"n{n}_pressure_errors.npz", time=times, **pressure_terms(n, full, models, times))
        print(f"Grid {n} complete in {time.perf_counter() - started:.1f} s (execution log only)", flush=True)


if __name__ == "__main__":
    main()
