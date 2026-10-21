#!/usr/bin/env python3
"""Check the exact interface modes and the complete separated discrete solver."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from kernel import (CFG, analytic_data, averaged_modes, build_sparse, source_receiver,
                    separable_setup, separable_pressure, separable_receiver, exact_receiver,
                    exact_pressure_norm_squared, cell_projection_norm_squared,
                    exact_mass_weighted_pressure, x_modes, y_modes)

HERE = Path(__file__).resolve().parent


def main():
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available & set(CFG["cpu_affinity"]) or available)
    a = analytic_data(192)
    b = analytic_data(384)
    tt = np.linspace(0., 8., 641)
    result = {
        "impedance_ratio": 8., "normal_incidence_pressure_reflection": 7 / 9,
        "x_weighted_orthogonality_max": float(np.max(np.abs(a["x_gram"] - np.eye(21)))),
        "y_orthogonality_max": float(np.max(np.abs(a["y_gram"] - np.eye(17)))),
        "projection_quadrature_relative_change": float(np.linalg.norm(a["coefficients"] - b["coefficients"]) / np.linalg.norm(b["coefficients"])),
        "receiver_quadrature_relative_change": float(np.linalg.norm(exact_receiver(tt, 192) - exact_receiver(tt, 384)) / np.linalg.norm(exact_receiver(tt, 384))),
        "initial_acoustic_energy": (.5 * np.sum(a["coefficients"] ** 2, axis=(0, 1))).tolist(),
        "zero_mode_coefficients": a["coefficients"][0, 0].tolist(),
    }
    m = np.arange(21)
    A = 1 / np.sqrt(np.where(m == 0, 9 / 16, np.where(m % 2 == 0, 9 / 32, 9 / 4)))
    B = np.where(m % 2 == 0, A, -8 * A)
    pl = A * np.cos(m * np.pi / 2)
    pr = B * np.cos(m * np.pi / 2)
    vl = -A * m * np.pi * np.sin(m * np.pi / 2)
    vr = B * m * np.pi * np.sin(m * np.pi / 2) / 8
    result["analytic_pressure_jump_max"] = float(np.max(np.abs(pl - pr)))
    result["analytic_normal_flux_jump_max"] = float(np.max(np.abs(vl - vr)))
    result["rigid_boundary_normal_derivative_max"] = float(np.max(np.abs(A * m * np.pi * np.sin(0 * m))))
    n = 16
    X, Y, rho, bulk, mass, C, J = build_sparse(n)
    model = separable_setup(n)
    Vx, Vy = model["x_eigenvectors"], model["y_eigenvectors"]
    Ax = (Vx * model["x_eigenvalues"]) @ Vx.T
    Ay = (Vy * model["y_eigenvalues"]) @ Vy.T
    separated_A = sparse.kron(Ax, sparse.eye(n)) + sparse.kron(sparse.eye(n), Ay)
    A_discrete = C @ C.T
    delta = separated_A - A_discrete
    result["separated_stiffness_relative_error"] = float(sparse.linalg.norm(delta) / sparse.linalg.norm(A_discrete))
    Bsource, R = source_receiver(n, mass)
    initial = np.zeros((J.shape[0], 3))
    initial[:n * n] = Bsource
    time_small = np.linspace(0., 8., 33)
    states = expm_multiply(J, initial, start=0., stop=8., num=33, traceA=0.)
    U = separable_pressure(model, time_small)
    yy = np.einsum("ir,tis->trs", R, states[:, :n * n], optimize=True)
    result["separated_vs_sparse_pressure_relative_error"] = float(np.linalg.norm(U - states[:, :n * n]) / np.linalg.norm(states[:, :n * n]))
    result["separated_vs_sparse_receiver_relative_error"] = float(np.linalg.norm(separable_receiver(model, time_small) - yy) / np.linalg.norm(yy))
    result["sparse_energy_relative_drift"] = float(np.max(np.abs(np.sum(states ** 2, axis=1) / np.sum(initial ** 2, axis=0) - 1)))
    result["generator_skew_defect"] = float(np.max(np.abs((J + J.T).data), initial=0.))
    n = 32
    sample_times = np.array([0., .2, 1.3, 8.])
    actual_norm = np.sum(exact_mass_weighted_pressure(n, sample_times) ** 2, axis=1)
    result["cell_projection_parseval_relative_error"] = float(np.linalg.norm(actual_norm - cell_projection_norm_squared(n, sample_times)) / np.linalg.norm(actual_norm))
    # Integrate the piecewise-constant field error directly inside every cell.
    qx, qw = np.polynomial.legendre.leggauss(8)
    xx = ((np.arange(n)[:, None] + .5) / n + qx[None, :] / (2 * n)).ravel()
    ww = np.tile(qw / (2 * n), n)
    beta_x = np.where(xx < .5, 1., 1 / 8)
    t = .37
    modal = a["coefficients"][:, :, 0] * np.cos(a["omega"] * t)
    exact_fine = x_modes(xx) @ modal @ y_modes(xx).T
    full = separable_setup(n)
    U = separable_pressure(full, [t])[0, :, 0]
    p = U / np.sqrt(full["mass"])
    numerical_fine = np.repeat(np.repeat(p.reshape(n, n), 8, axis=0), 8, axis=1)
    direct = float(np.einsum("ij,i,j->", (numerical_fine - exact_fine) ** 2, ww * beta_x, ww))
    average = exact_mass_weighted_pressure(n, [t])[0, :, 0]
    identity = float(np.sum((U - average) ** 2) + exact_pressure_norm_squared([t])[0, 0] - np.sum(average ** 2))
    result["continuous_pressure_error_identity_relative_error"] = abs(direct - identity) / direct
    spectrum_errors = []
    for n in CFG["grids"]:
        full = separable_setup(n)
        exact_discrete_spectrum = 4 * n * n * np.sin(np.pi * np.arange(n) / (2 * n)) ** 2
        spectrum_errors.append(float(np.max(np.abs(full["x_eigenvalues"] - exact_discrete_spectrum)) / (4 * n * n)))
    result["separated_x_spectrum_relative_max_error"] = max(spectrum_errors)
    result["passed"] = all(result[key] < 2e-10 for key in (
        "x_weighted_orthogonality_max", "y_orthogonality_max", "projection_quadrature_relative_change",
        "receiver_quadrature_relative_change", "analytic_pressure_jump_max", "analytic_normal_flux_jump_max",
        "separated_stiffness_relative_error", "separated_vs_sparse_pressure_relative_error",
        "separated_vs_sparse_receiver_relative_error", "sparse_energy_relative_drift",
        "cell_projection_parseval_relative_error", "continuous_pressure_error_identity_relative_error",
        "separated_x_spectrum_relative_max_error"))
    assert result["passed"], result
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
