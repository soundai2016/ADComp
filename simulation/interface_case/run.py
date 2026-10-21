#!/usr/bin/env python3
"""Accuracy of full and reduced solutions against the interface modal solution."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from accuracy_cost.run import cb_basis, rational_basis
from interface_case.kernel import (CFG, analytic_data, assembly, exact_receiver, exact_pressure_norm_squared,
    exact_mass_weighted_pressure, separable_setup, separable_receiver, separable_pressure)
OUT = HERE / "results"


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def receiver_errors(yy, ref, times):
    answer = {}
    for end in (2., 8.):
        ii = times <= end + 1e-12
        answer[f"T{int(end)}"] = {"relative_error": relative(yy[ii], ref[ii]),
            "per_source": [relative(yy[ii, :, ss], ref[ii, :, ss]) for ss in range(3)]}
    return answer


def field_errors(evaluate, reference, times, cache=None, continuous=False):
    numerators = np.zeros((2, 3))
    denominators = np.zeros((2, 3))
    for start in range(0, len(times), 16):
        stop = min(start + 16, len(times))
        tt = times[start:stop]
        uu = evaluate(tt)
        vv = cache[start:stop] if cache is not None else reference(tt)
        norm2 = exact_pressure_norm_squared(tt) if continuous else None
        # Orthogonality of cell averaging gives the exact unresolved subcell term.
        subcell = norm2 - np.sum(vv ** 2, axis=1) if continuous else None
        if continuous:
            assert np.min(subcell) >= -1e-11
            subcell = np.maximum(subcell, 0.)
        for j, end in enumerate((2., 8.)):
            ii = tt <= end + 1e-12
            numerators[j] += np.sum((uu[ii] - vv[ii]) ** 2, axis=(0, 1))
            if continuous:
                numerators[j] += np.sum(subcell[ii], axis=0)
                denominators[j] += np.sum(norm2[ii], axis=0)
            else:
                denominators[j] += np.sum(vv[ii] ** 2, axis=(0, 1))
    return {f"T{int(end)}": {"relative_error": float(np.sqrt(numerators[j].sum() / denominators[j].sum())),
            "per_source": np.sqrt(numerators[j] / denominators[j]).tolist()}
            for j, end in enumerate((2., 8.))}


def dump(path, answer):
    path.write_text(json.dumps(answer, indent=2, allow_nan=False) + "\n")


def save_arrays(path, **arrays):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=CFG["grids"])
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available & set(CFG["cpu_affinity"]) or available)
    OUT.mkdir(exist_ok=True)
    digest = hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()
    for prior in OUT.glob("aligned_impedance_step_*.json"):
        if json.loads(prior.read_text())["config_sha256"] != digest:
            raise RuntimeError("Existing result configuration differs")
    times = np.linspace(0., CFG["time_end"], CFG["samples"])
    analytic_data()
    yy_exact = exact_receiver(times)
    save_arrays(OUT / "analytic_reference.npz", time=times,
                receiver_pressure=yy_exact, **analytic_data())
    for n in args.grids:
        full = separable_setup(n)
        yy_full = separable_receiver(full, times)
        full_stem = f"aligned_impedance_step_n{n}_full"
        # At n<=128 the two pressure histories occupy at most 504 MB combined.
        exact_cache = exact_mass_weighted_pressure(n, times) if n <= 128 else None
        full_cache = separable_pressure(full, times) if n <= 128 else None
        record = {"grid": n, "method": "full_separable", "dimension": n * n,
                  "config_sha256": digest, "receiver_total_error": receiver_errors(yy_full, yy_exact, times),
                  "pressure_total_error": field_errors(lambda tt: separable_pressure(full, tt),
                    lambda tt: exact_mass_weighted_pressure(n, tt), times, exact_cache, continuous=True)}
        save_arrays(OUT / f"{full_stem}.npz", time=times, receiver_pressure=yy_full, **full)
        dump(OUT / f"{full_stem}.json", record)
        print(full_stem, "receiver", record["receiver_total_error"]["T8"]["relative_error"],
              "pressure", record["pressure_total_error"]["T8"]["relative_error"], flush=True)
        if n not in CFG["reduced_grids"]:
            continue
        mass, A, J, B, R, rho, bulk = assembly(n)
        for m in CFG["modes_per_patch"]:
            for method in ("craig_bampton", "rational_krylov"):
                stem = f"aligned_impedance_step_n{n}_{method}_m{m}"
                if (OUT / f"{stem}.json").exists():
                    continue
                r = n + 2 * m
                if (OUT / f"{stem}.npz").exists():
                    with np.load(OUT / f"{stem}.npz") as raw:
                        Q = raw["orthonormal_basis"]
                        V = raw["reduced_eigenvectors"]
                        lam = raw["reduced_eigenvalues"]
                        coeff = raw["initial_modal_coefficients"]
                    details = {}
                else:
                    if method == "craig_bampton":
                        Q = cb_basis(n, mass, A, m)
                        details = {}
                    else:
                        Q, details = rational_basis(mass, A, B, r)
                    Ar = Q.T @ (A @ Q)
                    lam, V = linalg.eigh((Ar + Ar.T) / 2, check_finite=False, driver="evd")
                    assert lam[0] >= -1e-9 * max(1., lam[-1])
                    lam[lam < 1e-10] = 0.
                    coeff = V.T @ (Q.T @ B)
                W = Q @ V
                observation = R.T @ W
                phase = np.cos(times[:, None] * np.sqrt(lam)[None, :])
                yy = np.einsum("ir,tr,rs->tis", observation, phase, coeff, optimize=True)
                def reduced_pressure(tt):
                    modal = np.cos(tt[:, None] * np.sqrt(lam)[None, :])[:, :, None] * coeff[None, :, :]
                    return np.einsum("ir,trs->tis", W, modal, optimize=True)
                local_error = yy - yy_full
                grid_error = yy_full - yy_exact
                cross_norm = np.linalg.norm(local_error) * np.linalg.norm(grid_error)
                record = {"grid": n, "method": method, "dimension": r, "modes_per_patch": m,
                    "config_sha256": digest, "receiver_total_error": receiver_errors(yy, yy_exact, times),
                    "receiver_same_grid_error": receiver_errors(yy, yy_full, times),
                    "pressure_total_error": field_errors(reduced_pressure, None, times, exact_cache, continuous=True),
                    "pressure_same_grid_error": field_errors(reduced_pressure, None, times, full_cache),
                    "receiver_error_vector_cosine": float(np.sum(local_error * grid_error) / cross_norm) if cross_norm else 0.,
                    "structure": {"orthonormality_frobenius": float(np.linalg.norm(Q.T @ Q - np.eye(r))),
                        "initial_pressure_projection_error": relative(Q @ (Q.T @ B), B),
                        "constant_mode_projection_error": float(np.linalg.norm(np.sqrt(mass) - Q @ (Q.T @ np.sqrt(mass))) / np.linalg.norm(np.sqrt(mass))),
                        **details}}
                if not (OUT / f"{stem}.npz").exists():
                    save_arrays(OUT / f"{stem}.npz", time=times, receiver_pressure=yy,
                        orthonormal_basis=Q, reduced_eigenvectors=V, reduced_eigenvalues=lam,
                        initial_modal_coefficients=coeff, mass=mass)
                dump(OUT / f"{stem}.json", record)
                print(stem, "receiver_total", record["receiver_total_error"]["T8"]["relative_error"],
                      "receiver_same", record["receiver_same_grid_error"]["T8"]["relative_error"], flush=True)
        del exact_cache, full_cache, mass, A, J, B, R, rho, bulk


if __name__ == "__main__":
    main()
