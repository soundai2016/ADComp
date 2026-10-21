#!/usr/bin/env python3
"""Pair reduced/FV/continuous modes and retain exact unmatched responses."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from unequal_speed_interface.kernel import (CFG, continuous_data, roots,
    mode_parameters, x_values, y_values, gauss_interval, gaussian_integrals, x_tridiagonal)
from unequal_speed_interface.io import save_arrays
OUT = HERE/"results"


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def residue(b, c):
    return np.einsum("rj,js->jrs", c, b)


def response(times, omega, weight):
    return np.einsum("tj,jrs->trs", np.cos(times[:, None]*omega[None]), weight, optimize=True)


def gap(omega):
    return np.minimum(np.r_[np.inf, np.diff(omega)], np.r_[np.diff(omega), np.inf])/np.maximum(omega, 1e-100)


def match(left, right, omega_left, omega_right, gap_right=None):
    overlap = left.T@right
    il, ir = linear_sum_assignment(-overlap**2)
    mac = overlap[il, ir]**2
    right_gap = gap(omega_right) if gap_right is None else gap_right
    accepted = ((mac >= CFG["mac_threshold"]) &
                (gap(omega_left)[il] >= CFG["relative_frequency_gap"]) &
                (right_gap[ir] >= CFG["relative_frequency_gap"]) &
                (omega_left[il] > 1e-7) & (omega_right[ir] > 1e-7))
    return il, ir, mac, np.sign(overlap[il, ir]), accepted


def continuum_catalogue(n, mass):
    data = continuous_data()
    active_max = float(data["omega"].max())
    halo_max = active_max+np.pi*np.sqrt(max(np.asarray(CFG["bulk_modulus"])/np.asarray(CFG["density"])))
    limit = int(np.floor(halo_max/np.pi))+2
    catalogue = []
    for ell in range(limit):
        for m, omega in enumerate(roots(ell, limit+1)):
            if omega <= halo_max:
                qq, aa = mode_parameters(omega, ell)
                catalogue.append((omega, m, ell, qq, aa))
    catalogue.sort(key=lambda row: row[0])
    omega = np.asarray([r[0] for r in catalogue])
    mm = np.asarray([r[1] for r in catalogue], dtype=int)
    ll = np.asarray([r[2] for r in catalogue], dtype=int)
    q2 = np.stack([r[3] for r in catalogue])
    amp = np.stack([r[4] for r in catalogue])
    xx = (np.arange(n)+.5)/n
    xavg = x_values(xx, q2, amp, 1/n)
    yavg = y_values(xx, ll.max())[:, ll]*np.sinc(ll/(2*n))[None]
    vectors = np.einsum("ik,jk->ijk", xavg, yavg, optimize=True).reshape(n*n, len(omega))*np.sqrt(mass)[:, None]
    norms = np.linalg.norm(vectors, axis=0)
    vectors /= np.maximum(norms, 1e-300)[None]
    b = np.zeros((len(omega), 3))
    for j, (m, ell) in enumerate(zip(mm, ll)):
        if m <= CFG["max_x_mode"] and ell <= CFG["max_y_mode"]:
            b[j] = data["coefficients"][ell, m]
    order = CFG["quadrature_order"]
    xl, wl = gauss_interval(order, 0., .5)
    xr, wr = gauss_interval(order, .5, 1.)
    x, wx = np.r_[xl, xr], np.r_[wl, wr]
    y, wy = gauss_interval(order, 0., 1.)
    X = x_values(x, q2, amp)
    Y = y_values(y, ll.max())[:, ll]
    c = np.empty((4, len(omega)))
    for r, (cx, cy) in enumerate(CFG["receiver_centres"]):
        gx = np.exp(-.5*((x-cx)/CFG["sigma"])**2)
        gy = np.exp(-.5*((y-cy)/CFG["sigma"])**2)
        c[r] = (X.T@(wx*gx))*(Y.T@(wy*gy))/(gaussian_integrals([0., 1.], cx)[0]*gaussian_integrals([0., 1.], cy)[0])
    return {"omega": omega, "m": mm, "ell": ll, "V": vectors, "b": b, "c": c,
            "norms": norms, "active_max": active_max, "halo_max": halo_max,
            "q_squared": q2, "amplitudes": amp}


def norm(x):
    return float(np.linalg.norm(x))


def cosine(a, b):
    return float(np.sum(a*b)/max(norm(a)*norm(b), 1e-300))


def main():
    n = CFG["modal_grid"]
    coarse = np.load(OUT/f"n{n}_full_spectrum.npz")
    t = np.load(OUT/"continuous_reference.npz")["time"]
    yf = np.load(OUT/"continuous_reference.npz")["receiver_pressure"]
    yc = np.load(OUT/f"n{n}_full_receiver.npz")["receiver_pressure"]
    print("Constructing complete continuous frequency catalogue and halo", flush=True)
    fine = continuum_catalogue(n, coarse["mass"])
    # Check every catalogue index against an independently converging 1-D FV
    # spectrum, including modes outside the finite initial-pressure expansion.
    catalogue_root_checks = []
    for nc in (1024, 2048):
        dx, ex, speed2 = x_tridiagonal(nc)
        for ell in np.unique(fine["ell"]):
            selected = np.flatnonzero(fine["ell"] == ell)
            selected = selected[np.argsort(fine["m"][selected])]
            mm = fine["m"][selected]
            if not np.array_equal(mm, np.arange(len(mm))):
                raise RuntimeError("Continuous catalogue omits an internal longitudinal index")
            lv = linalg.eigh_tridiagonal(dx+speed2*(ell*np.pi)**2, ex,
                eigvals_only=True, select="i", select_range=(0, len(mm)-1))
            if ell == 0:
                lv[0] = 0.
            ow = fine["omega"][selected]
            dif = np.abs(np.sqrt(np.maximum(lv, 0.))-ow)
            gaps = np.minimum(np.r_[np.inf,np.diff(ow)],np.r_[np.diff(ow),np.inf])
            catalogue_root_checks.append({"transverse_index":int(ell),"cells":nc,"roots":len(mm),
                "frequency_relative_max_error":float(np.max(dif/np.maximum(ow,1.))),
                "frequency_error_to_longitudinal_gap_max":float(np.max(dif/gaps))})
    candidate = np.flatnonzero(fine["norms"] > 1e-8)
    il, ir, mac, sign, accept = match(coarse["eigenvectors"], fine["V"][:, candidate],
        coarse["omega"], fine["omega"][candidate], gap_right=gap(fine["omega"])[candidate])
    ir = candidate[ir]
    accept &= fine["omega"][ir] <= fine["active_max"]+1e-9
    cf = {int(i): (int(j), float(ma), float(si)) for i, j, ma, si, ok in zip(il, ir, mac, sign, accept) if ok}
    yf_catalogue = response(t, fine["omega"], residue(fine["b"], fine["c"]))
    save_arrays(OUT/"continuous_catalogue.npz", **{k:v for k,v in fine.items() if k != "V"})
    validation = {"catalogue_modes_including_halo": len(fine["omega"]),
        "initial_max_angular_frequency": fine["active_max"], "catalogue_max_angular_frequency": fine["halo_max"],
        "catalogue_receiver_reconstruction_relative_error": norm(yf_catalogue-yf)/norm(yf),
        "accepted_coarse_continuous_pairs": len(cf),
        "catalogue_independent_root_checks": catalogue_root_checks,
        "gap_definition": "min(omega[j]-omega[j-1],omega[j+1]-omega[j])/omega[j], evaluated on the complete frequency catalogue including its upper halo.",
        "matching": "Maximum-sum squared modal overlap; continuous physical modes are averaged to the coarse cells, multiplied by sqrt(M), and normalized only for matching."}
    pair_rows, summary, windows, gram_rows = [], [], [], []
    for path in sorted(OUT.glob(f"n{n}_*m*.npz")):
        raw = np.load(path)
        name = path.stem.split("_", 1)[1]
        if "eigenvectors" not in raw or "receiver_pressure" not in raw:
            continue
        wr = raw["omega"]
        Vr = raw["eigenvectors"]
        br = Vr.T@coarse["B"]
        cr = coarse["R"].T@Vr
        yr = raw["receiver_pressure"]
        ri0, ci0, mac_rc, signs_rc, accepted_rc = match(Vr, coarse["eigenvectors"], wr, coarse["omega"])
        chain = [(int(i), int(j), cf[int(j)][0], float(ma), cf[int(j)][1], float(si), cf[int(j)][2])
                 for i,j,ma,si,ok in zip(ri0, ci0, mac_rc, signs_rc, accepted_rc) if ok and int(j) in cf]
        ri = np.asarray([r[0] for r in chain], dtype=int)
        ci = np.asarray([r[1] for r in chain], dtype=int)
        fi = np.asarray([r[2] for r in chain], dtype=int)
        sr = np.asarray([r[5] for r in chain])
        sf = np.asarray([r[6] for r in chain])
        b_r, c_r = br[ri]*sr[:, None], cr[:, ri]*sr[None]
        b_f, c_f = fine["b"][fi]*sf[:, None], fine["c"][:, fi]*sf[None]
        Hr, Hf = residue(b_r, c_r), residue(b_f, c_f)
        ritz = response(t, wr[ri], Hr)-response(t, coarse["omega"][ci], Hr)
        grid = response(t, coarse["omega"][ci], Hr)-response(t, fine["omega"][fi], Hr)
        source = response(t, fine["omega"][fi], residue(b_r-b_f, (c_r+c_f)/2))
        receiver = response(t, fine["omega"][fi], residue((b_r+b_f)/2, c_r-c_f))
        remainder = yr-yf-ritz-grid-source-receiver
        replacement_coarse = yr-ritz
        replacement_continuous = yr-ritz-grid
        total = yr-yf
        weights = np.linalg.norm(Hf, axis=(1,2))
        wg = weights/max(weights.sum(), 1e-300)
        dg = (coarse["omega"][ci]-fine["omega"][fi])/fine["omega"][fi]
        dr = (wr[ri]-coarse["omega"][ci])/fine["omega"][fi]
        for j, match_row in enumerate(chain):
            pair_rows.append({"model": name, "dimension": len(wr), "rom_mode": int(ri[j]),
                "coarse_mode": int(ci[j]), "continuous_mode": int(fi[j]),
                "continuous_x_index": int(fine["m"][fi[j]]), "continuous_y_index": int(fine["ell"][fi[j]]),
                "mac_rom_coarse": match_row[3], "mac_coarse_continuous": match_row[4],
                "sign_rom_to_coarse": match_row[5], "sign_continuous_to_coarse": match_row[6],
                "omega_rom": float(wr[ri[j]]), "omega_coarse": float(coarse["omega"][ci[j]]),
                "omega_continuous": float(fine["omega"][fi[j]]), "relative_grid_shift": float(dg[j]),
                "relative_ritz_shift": float(dr[j]), "continuous_residue_norm": float(weights[j])})
        components = {"ritz_frequency": ritz, "grid_frequency": grid, "source_coupling": source,
                      "receiver_coupling": receiver, "unmatched": remainder}
        for aname, aa in components.items():
            for bname, bb in components.items():
                gram_rows.append({"model": name, "component_a": aname, "component_b": bname,
                    "normalized_inner_product": float(np.sum(aa*bb)/norm(yf)**2)})
        row = {"model": name, "dimension": len(wr), "accepted_triples": len(chain),
            "same_grid_error": norm(yr-yc)/norm(yc), "total_error": norm(total)/norm(yf),
            "coarse_frequency_replacement_error": norm(replacement_coarse-yf)/norm(yf),
            "continuous_frequency_replacement_error": norm(replacement_continuous-yf)/norm(yf),
            "ritz_grid_cosine": cosine(ritz, grid),
            "weighted_relative_grid_shift": float(wg@dg), "weighted_relative_ritz_shift": float(wg@dr),
            "weighted_shift_compensation_ratio": float(-(wg@dr)/(wg@dg)),
            "matched_continuous_response_omission": norm(yf-response(t, fine["omega"][fi], Hf))/norm(yf),
            "decomposition_relative_defect": norm(total-sum(components.values()))/norm(total),
            "stored_receiver_reconstruction_error": norm(yr-response(t, wr, residue(br, cr)))/norm(yr)}
        row.update({key+"_relative_norm": norm(value)/norm(yf) for key, value in components.items()})
        summary.append(row)
        windows_to_check = [("cumulative", 0., end) for end in (2.,4.,6.,8.)]+[("successive", start, start+2) for start in (0.,2.,4.,6.)]
        for kind, start, end in windows_to_check:
            mask = (t >= start-1e-12)&(t <= end+1e-12)
            for rr in range(4):
                for ss in range(3):
                    ref = yf[mask, rr, ss]
                    original_error = norm(yr[mask, rr, ss]-ref)/norm(ref)
                    replaced_error = norm(replacement_coarse[mask, rr, ss]-ref)/norm(ref)
                    windows.append({"model": name, "window_type": kind, "start": start, "end": end,
                        "source": ss+1, "receiver": rr+1, "original_error": original_error,
                        "coarse_replacement_error": replaced_error,
                        "coarse_replacement_change_percentage_points": 100*(replaced_error-original_error),
                        "reference_squared_norm": norm(ref)**2})
        save_arrays(OUT/f"{name}_decomposition.npz", time=t, rom_response=yr,
            coarse_response=yc, continuous_response=yf, no_ritz_frequency_response=replacement_coarse,
            continuous_frequency_response=replacement_continuous, **components)
        print(json.dumps(row), flush=True)
    csv_write(OUT/"modal_pairs.csv", pair_rows)
    csv_write(OUT/"modal_summary.csv", summary)
    csv_write(OUT/"modal_component_gram.csv", gram_rows)
    csv_write(OUT/"receiver_time_windows.csv", windows)
    validation["passed"] = (validation["catalogue_receiver_reconstruction_relative_error"] < 1e-9
        and all(r["decomposition_relative_defect"] < 1e-11 for r in summary)
        and max(r["frequency_error_to_longitudinal_gap_max"] for r in catalogue_root_checks if r["cells"]==2048) < .1)
    dump(OUT/"modal_validation.json", validation)
    dump(OUT/"modal_summary.json", {"models": summary})
    if not validation["passed"]:
        raise RuntimeError("Modal validation failed")


if __name__ == "__main__":
    main()
