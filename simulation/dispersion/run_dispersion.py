#!/usr/bin/env python3
"""Controlled acoustic dispersion in a rigid unit square.

Apply the finite-volume operator to sampled continuum modes. At n=16,
independent matrix-exponential propagation checks the modal time dependence.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent


def neumann_laplacian(n: int) -> sparse.csr_matrix:
    """Negative cell-centered Laplacian, with zero normal boundary flux."""
    diagonal = np.full(n, 2.0)
    diagonal[[0, -1]] = 1.0
    one_dimensional = sparse.diags(
        [-np.ones(n - 1), diagonal, -np.ones(n - 1)], [-1, 0, 1],
        format="csr",
    ) * n**2
    identity = sparse.eye(n, format="csr")
    return (sparse.kron(one_dimensional, identity, format="csr") +
            sparse.kron(identity, one_dimensional, format="csr"))


def sampled_mode(n: int, kx: int, ky: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n
    mode = np.cos(np.pi * kx * x)[:, None] * np.cos(np.pi * ky * x)[None, :]
    mode = mode.ravel()
    return mode / np.linalg.norm(mode)


def sinc_integral(frequency: float, duration: float) -> float:
    # np.sinc is sin(pi*x)/(pi*x), including its continuous value at zero.
    return duration * np.sinc(frequency * duration / np.pi)


def cumulative_error(omega_h: float, omega: float, duration: float) -> float:
    norm_h = duration / 2 + sinc_integral(2 * omega_h, duration) / 2
    norm_exact = duration / 2 + sinc_integral(2 * omega, duration) / 2
    inner = (sinc_integral(omega_h - omega, duration) +
             sinc_integral(omega_h + omega, duration)) / 2
    squared_difference = norm_h + norm_exact - 2 * inner
    return float(np.sqrt(max(0.0, squared_difference) / norm_exact))


def main() -> None:
    started = time.perf_counter()
    config_bytes = (HERE / "config.json").read_bytes()
    config = json.loads(config_bytes)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    rows, eigenpair_checks, evolution_checks, core_checks = [], [], [], []
    core_path = HERE.parent / "acoustic_core.py"
    core_spec = importlib.util.spec_from_file_location("acoustic_core", core_path)
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)
    quadrature_errors = []
    actual_waveforms = {}
    for n in config["grids"]:
        operator = neumann_laplacian(n)
        *_, coupling, _generator = core.build_sparse(n, "homogeneous")
        common_operator = (coupling @ coupling.T).tocsr()
        difference = operator - common_operator
        relative_difference = float(np.linalg.norm(difference.data) /
                                    np.linalg.norm(operator.data))
        core_checks.append({"n": n, "relative_frobenius_difference": relative_difference})
        for kx, ky in config["modes"]:
            omega = np.pi * np.hypot(kx, ky)
            omega_h = 2 * n * np.hypot(
                np.sin(np.pi * kx / (2 * n)),
                np.sin(np.pi * ky / (2 * n)),
            )
            mode = sampled_mode(n, kx, ky)
            applied = operator @ mode
            eigenvalue_from_operator = float(np.dot(mode, applied))
            eigen_residual = float(np.linalg.norm(applied - omega_h**2 * mode) /
                                   (omega_h**2 * np.linalg.norm(mode)))
            eigenpair_checks.append({
                "n": n, "kx": kx, "ky": ky,
                "formula_eigenvalue": float(omega_h**2),
                "rayleigh_eigenvalue": eigenvalue_from_operator,
                "relative_residual": eigen_residual,
            })
            for duration in config["observation_times"]:
                error = cumulative_error(omega_h, omega, duration)
                times = np.linspace(0, duration, config["quadrature_samples"])
                exact_wave = np.cos(omega * times)
                discrete_wave = np.cos(omega_h * times)
                quadrature = np.sqrt(np.trapezoid((discrete_wave - exact_wave)**2, times) /
                                     np.trapezoid(exact_wave**2, times))
                quadrature_errors.append(abs(float(quadrature) - error))
                rows.append({
                    "n": n, "h": 1.0 / n, "kx": kx, "ky": ky,
                    "T": duration, "omega": float(omega), "omega_h": float(omega_h),
                    "relative_frequency_lag": float((omega - omega_h) / omega),
                    "phase_lag_radians": float((omega - omega_h) * duration),
                    "phase_delay_time": float((omega - omega_h) * duration / omega),
                    "relative_time_integrated_pressure_error": error,
                    "quadrature_pressure_error": float(quadrature),
                    "points_per_wavelength": float(2 * n / np.hypot(kx, ky)),
                })
            if n == config["matrix_exponential_grid"]:
                size = n * n
                generator = sparse.bmat(
                    [[None, sparse.eye(size)], [-operator, None]], format="csr"
                )
                initial = np.concatenate([mode, np.zeros(size)])
                times = np.linspace(0, max(config["observation_times"]),
                                    config["matrix_exponential_samples"])
                trajectory = expm_multiply(generator, initial, start=0, stop=times[-1],
                                           num=len(times), endpoint=True, traceA=0.0)
                exact_p = np.cos(omega_h * times)[:, None] * mode[None, :]
                exact_pt = -omega_h * np.sin(omega_h * times)[:, None] * mode[None, :]
                pressure_error = float(np.linalg.norm(trajectory[:, :size] - exact_p) /
                                       np.linalg.norm(exact_p))
                derivative_error = float(np.linalg.norm(trajectory[:, size:] - exact_pt) /
                                         np.linalg.norm(exact_pt))
                evolution_checks.append({"n": n, "kx": kx, "ky": ky,
                                         "pressure_relative_error": pressure_error,
                                         "pressure_derivative_relative_error": derivative_error})
                prefix = f"mode_{kx}_{ky}"
                actual_waveforms[prefix + "_modal_pressure"] = trajectory[:, :size] @ mode
                actual_waveforms[prefix + "_modal_derivative"] = trajectory[:, size:] @ mode
                actual_waveforms[prefix + "_continuum_pressure"] = np.cos(omega * times)
                actual_waveforms[prefix + "_discrete_pressure"] = np.cos(omega_h * times)
                actual_waveforms["time"] = times

    with (output / "dispersion_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "matrix_exponential_traces.npz", **actual_waveforms)
    maximum_residual = max(row["relative_residual"] for row in eigenpair_checks)
    maximum_evolution_error = max(max(row["pressure_relative_error"],
                                      row["pressure_derivative_relative_error"])
                                  for row in evolution_checks)
    checks = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acoustic_core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__,
        "operator": "cell-centered Neumann finite-volume negative Laplacian",
        "grid_mode_pairs": len(eigenpair_checks), "result_rows": len(rows),
        "eigenpair_checks": eigenpair_checks,
        "matrix_exponential_checks": evolution_checks,
        "common_acoustic_operator_checks": core_checks,
        "maximum_common_operator_relative_difference": max(
            row["relative_frobenius_difference"] for row in core_checks),
        "maximum_relative_eigenpair_residual": maximum_residual,
        "maximum_matrix_exponential_relative_error": maximum_evolution_error,
        "maximum_quadrature_absolute_error": max(quadrature_errors),
        "wall_seconds": time.perf_counter() - started,
        "passed": (maximum_residual < config["eigenpair_residual_tolerance"] and
                   maximum_evolution_error < config["matrix_exponential_relative_tolerance"] and
                   max(row["relative_frobenius_difference"] for row in core_checks) < 1e-12 and
                   max(quadrature_errors) < 1e-5),
    }
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")

    print(json.dumps({key: checks[key] for key in ["grid_mode_pairs", "result_rows",
                      "maximum_relative_eigenpair_residual", "maximum_matrix_exponential_relative_error",
                      "maximum_quadrature_absolute_error", "wall_seconds", "passed"]}, indent=2))
    if not checks["passed"]:
        raise SystemExit("Dispersion verification failed")


if __name__ == "__main__":
    main()
