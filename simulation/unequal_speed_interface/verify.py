#!/usr/bin/env python3
"""Independent accuracy and structural checks for the unequal-speed case."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import json
from pathlib import Path
import sys
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from unequal_speed_interface.kernel import (
    CFG, continuous_data, exact_receiver, roots, cosine_q, q_sine, characteristic,
    assembly, separable_setup, separable_pressure, separable_receiver,
    full_eigenvectors, x_tridiagonal, exact_mass_weighted_pressure,
    exact_pressure_norm_squared, gauss_interval, x_values, y_values,
)


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    (HERE/"results").mkdir(exist_ok=True)
    tt = np.linspace(0., CFG["time_end"], CFG["samples"])
    data = continuous_data(192)
    fine = continuous_data(384)
    finer_roots = continuous_data(192, .01)
    rho = np.asarray(CFG["density"])
    kk = np.asarray(CFG["bulk_modulus"])
    q2 = data["q_squared"]
    amp = data["amplitudes"]
    cc = cosine_q(q2, [.5, .5])
    ss = q_sine(q2, [.5, .5])/rho
    pressure_jump = amp[..., 0]*cc[..., 0]-amp[..., 1]*cc[..., 1]
    flux_jump = amp[..., 0]*ss[..., 0]+amp[..., 1]*ss[..., 1]
    flux_scale = np.maximum(np.linalg.norm(amp*ss, axis=-1), 1.)
    pressure_scale = np.maximum(np.linalg.norm(amp*cc, axis=-1), 1.)
    values = {
        "sound_speeds": np.sqrt(kk/rho).tolist(), "impedances": np.sqrt(kk*rho).tolist(),
        "pressure_jump_scaled_max": float(np.max(np.abs(pressure_jump)/pressure_scale)),
        "normal_flux_jump_scaled_max": float(np.max(np.abs(flux_jump)/flux_scale)),
        "x_beta_gram_max_error": float(np.max(np.abs(data["x_gram"]-np.eye(21)[None]))),
        "y_gram_max_error": float(np.max(np.abs(data["y_gram"]-np.eye(17)))),
        "initial_acoustic_energy": (.5*np.sum(data["coefficients"]**2, axis=(0, 1))).tolist(),
        "zero_mode_coefficients": data["coefficients"][0, 0].tolist(),
        "source_quadrature_relative_change_192_384": relative(data["coefficients"], fine["coefficients"]),
        "receiver_quadrature_relative_change_192_384": relative(exact_receiver(tt, 192), exact_receiver(tt, 384)),
        "root_scan_relative_change_002_001": relative(data["omega"], finer_roots["omega"]),
        "receiver_root_scan_relative_change_002_001": relative(exact_receiver(tt), exact_receiver(tt, 192, .01)),
        "number_of_finite_initial_modes": int(data["omega"].size-1),
        "evanescent_segments_in_initial_modes": int(np.sum(q2 < 0)),
        "time_propagation": "Exact modal cosine propagation; no time-stepping error.",
    }
    # The equal-speed spectrum supplies an independent regression target.
    equal_errors = []
    for ell in (0, 1, 4, 8, 16):
        computed = roots(ell, 21, (1., 8.), (1., 8.))
        exact = np.pi*np.sqrt(np.arange(21)**2+ell**2)
        equal_errors.append(float(np.max(np.abs(computed-exact)/np.maximum(exact, 1.))))
    values["equal_speed_spectrum_relative_max_error"] = max(equal_errors)
    # Check the continuous root ordering against converging independent 1-D
    # FV eigenvalues. These FV spectra never define the continuous solution.
    root_checks = []
    for ell in (0, 4, 8, 16):
        omega = data["omega"][ell]
        for n in (512, 1024, 2048):
            d, e, speed2 = x_tridiagonal(n)
            lam = linalg.eigh_tridiagonal(d+speed2*(ell*np.pi)**2, e,
                    eigvals_only=True, select="i", select_range=(0, 20))
            if ell == 0:
                lam[0] = 0.
            estimate = np.sqrt(np.maximum(lam, 0.))
            root_checks.append({"ell": ell, "cells": n,
                "frequency_relative_max_error": float(np.max(np.abs(estimate-omega)/np.maximum(omega, 1.)))})
    values["independent_1d_root_checks"] = root_checks
    values["independent_1d_finest_root_relative_max_error"] = max(r["frequency_relative_max_error"] for r in root_checks if r["cells"] == 2048)
    n = 16
    mass, A, J, B, R, _, _ = assembly(n)
    full = separable_setup(n)
    omega, vectors, _ = full_eigenvectors(full)
    values["separated_eigensystem_relative_residual"] = relative(A@vectors, vectors*omega[None]**2)
    values["separated_eigensystem_orthogonality_error"] = float(np.max(np.abs(vectors.T@vectors-np.eye(n*n))))
    initial = np.zeros((J.shape[0], 3))
    initial[:n*n] = B
    short_times = np.linspace(0., 8., 33)
    state = expm_multiply(J, initial, start=0., stop=8., num=33, traceA=0.)
    values["separated_vs_sparse_pressure_relative_error"] = relative(separable_pressure(full, short_times), state[:, :n*n])
    yy = np.einsum("ir,tis->trs", R, state[:, :n*n], optimize=True)
    values["separated_vs_sparse_receiver_relative_error"] = relative(separable_receiver(full, short_times), yy)
    values["energy_drift"] = float(np.max(np.abs(np.sum(state**2, axis=1)/np.sum(initial**2, axis=0)-1)))
    values["generator_skew_defect"] = float(np.max(np.abs((J+J.T).data), initial=0.))
    # Direct in-cell quadrature verifies the continuous reconstruction identity.
    nq = 10
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    x = ((np.arange(n)[:, None]+.5)/n+nodes[None]/(2*n)).ravel()
    weight = np.tile(weights/(2*n), n)
    beta = np.where(x < .5, 1/kk[0], 1/kk[1])
    y = y_values(x, 16)
    time = .37
    exact = np.zeros((len(x), len(x)))
    for ell in range(17):
        xx = x_values(x, q2[ell], amp[ell])
        value = xx @ (data["coefficients"][ell, :, 0]*np.cos(data["omega"][ell]*time))
        exact += value[:, None]*y[:, ell][None]
    numerical = separable_pressure(full, [time])[0, :, 0]
    p0 = numerical/np.sqrt(mass)
    reconstructed = np.repeat(np.repeat(p0.reshape(n, n), nq, axis=0), nq, axis=1)
    direct = float(np.einsum("ij,i,j->", (reconstructed-exact)**2, weight*beta, weight))
    projected = exact_mass_weighted_pressure(n, [time])[0, :, 0]
    identity = float(np.sum((numerical-projected)**2)+exact_pressure_norm_squared([time])[0, 0]-np.sum(projected**2))
    values["continuous_field_identity_relative_error"] = abs(identity-direct)/direct
    values["norm_definition"] = "Continuous beta-weighted pressure norm, including orthogonal unresolved subcell contribution."
    numerical_keys = [k for k in values if k.endswith(("max_error", "scaled_max", "relative_error", "relative_residual", "orthogonality_error"))]
    # The independent FV-to-continuum convergence has O(h^2), not roundoff, error.
    precise_keys = [k for k in numerical_keys if not k.startswith("independent_1d")]
    precision_checks = precise_keys+[
        "source_quadrature_relative_change_192_384", "receiver_quadrature_relative_change_192_384",
        "root_scan_relative_change_002_001", "receiver_root_scan_relative_change_002_001", "energy_drift"]
    values["passed"] = (all(values[k] < 2e-9 for k in precision_checks)
        and values["independent_1d_finest_root_relative_max_error"] < 2e-4
        and max(abs(v-1.) for v in values["initial_acoustic_energy"]) < 1e-11
        and max(abs(v) for v in values["zero_mode_coefficients"]) < 1e-13)
    (HERE/"results"/"verification.json").write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")
    print(json.dumps(values, indent=2), flush=True)
    if not values["passed"]:
        raise RuntimeError("Unequal-speed verification failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true",
                        help="Verify saved trajectories, modal budgets, window statistics, and output hashes")
    arguments = parser.parse_args()
    if arguments.results:
        from unequal_speed_interface.verify_outputs import main as verify_outputs_main
        verify_outputs_main()
    else:
        main()
