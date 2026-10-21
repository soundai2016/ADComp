#!/usr/bin/env python3
"""Independent modal and reconstruction checks on a small acoustic grid."""
import os
if hasattr(os, "sched_getaffinity"):
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set([cpu for cpu in [6, 7] if cpu in available] or available[-2:]))
from run import assembly, full_run, reduced_run, relative, dump, OUT
import numpy as np
from scipy.linalg import eigh

times = np.linspace(0, 2, 161)
n = 8
mass, A, J, B, R, rho, bulk = assembly(n, "rotated_stripes")
lam, V = eigh(A.toarray())
lam[lam < 1e-10] = 0
exact_U = np.einsum("im,tm,ms->tis", V, np.cos(times[:, None] * np.sqrt(lam)), V.T @ B, optimize=True)
exact_Y = np.einsum("ir,tis->trs", R, exact_U, optimize=True)
full, _, checks = full_run(n, "rotated_stripes", times)
result = {"full_field_relative_difference_from_dense_modes": relative(full["mass_weighted_pressure"], exact_U),
          "full_receiver_relative_difference_from_dense_modes": relative(full["receiver_pressure"], exact_Y),
          "full_structure": checks, "methods": {}}
for method in ["craig_bampton", "rational_krylov"]:
    raw, U, _, structure = reduced_run(n, "rotated_stripes", 8, method, times)
    modes = raw["orthonormal_basis"] @ raw["reduced_eigenvectors"]
    modal = np.cos(times[:, None] * np.sqrt(raw["reduced_eigenvalues"]))[:, :, None] * raw["initial_modal_coefficients"]
    reconstructed = np.einsum("ir,trs->tis", modes, modal, optimize=True)
    YY = np.einsum("ir,tis->trs", R, reconstructed, optimize=True)
    result["methods"][method] = {"stored_representation_field_difference": relative(reconstructed, U),
                                 "stored_representation_receiver_difference": relative(YY, raw["receiver_pressure"]),
                                 "structure": structure}
assert result["full_field_relative_difference_from_dense_modes"] < 1e-10
assert result["full_receiver_relative_difference_from_dense_modes"] < 1e-10
for record in result["methods"].values():
    assert record["stored_representation_field_difference"] < 1e-12
    assert record["stored_representation_receiver_difference"] < 1e-12
    assert record["structure"]["orthonormality_frobenius"] < 1e-10
    assert record["structure"]["constant_mode_projection_error"] < 1e-10
result["status"] = "pass"
OUT.mkdir(exist_ok=True)
dump(OUT / "small_grid_verification.json", result)
print(result)
