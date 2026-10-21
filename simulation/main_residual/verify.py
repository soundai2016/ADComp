#!/usr/bin/env python3
"""Independent small-system and analytic-integration checks."""
import importlib.util
import sys
from pathlib import Path
from run import *


def main():
    if hasattr(os, "sched_setaffinity"):
        available = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(CFG["cpu_affinity"]) & available or available)
    previous = HERE.parent / "studies" / "stage2_2d"
    sys.path.insert(0, str(previous))
    spec = importlib.util.spec_from_file_location("previous_energy_defect_certificate", previous / "energy_defect_certificate.py")
    previous_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous_module)
    omega = np.array([0.0, 0.7, 0.7 + 1e-9, 1.8, 15.0, 45.0])
    analytic = integrated_g_gram(omega, 8.0)[0]
    previous_analytic = previous_module.integrated_g_gram(omega, 8.0)[0]
    nodes, weights = special.roots_legendre(1024)
    values = gfun(omega, 4 * (nodes + 1))
    quadrature = values.T @ ((4 * weights)[:, None] * values)
    gram_relative = np.linalg.norm(analytic - quadrature) / np.linalg.norm(quadrature)
    prior_relative = np.linalg.norm(analytic - previous_analytic) / np.linalg.norm(analytic)
    n = 8
    X, Y, rho, bulk, mass, C, J = build_sparse(n, "double_inclusion")
    B = ports(X, Y, bulk, CORE_CONFIG["source_centres"], True)
    A = (C @ C.T).toarray()
    vals, vecs = linalg.eigh(A)
    vals[vals < 1e-10] = 0
    coefficients = vecs.T @ B
    full_initial = np.zeros((J.shape[0], 3)); full_initial[:n*n] = B
    sparse_full = expm_multiply(J * 2, full_initial, traceA=0.0)
    modal_full = np.concatenate([vecs @ (np.cos(2*np.sqrt(vals))[:, None] * coefficients),
                                 C.T @ vecs @ (gfun(np.sqrt(vals), [2])[0, :, None] * coefficients)])
    full_relative = np.linalg.norm(sparse_full - modal_full) / np.linalg.norm(sparse_full)
    # A non-invariant subspace gives a nonzero residual.
    rng = np.random.default_rng(91805)
    Q, _ = linalg.qr(np.column_stack([np.sqrt(mass), B, rng.normal(size=(n*n, 14))]), mode="economic")
    small = Q.T @ A @ Q
    lam, V = linalg.eigh((small + small.T)/2)
    lam[lam < 1e-10] = 0
    W = Q @ V; a = W.T @ B; CW = C.T @ W
    F = C @ CW - W * lam[None, :]
    integral, _ = guarded_integral(F, np.sqrt(lam), a, 2)
    bound = np.linalg.norm(B - W @ a, axis=0) + np.sqrt(2*integral)
    rom = np.concatenate([W @ (np.cos(2*np.sqrt(lam))[:, None]*a),
                          CW @ (gfun(np.sqrt(lam), [2])[0,:,None]*a)])
    actual = np.linalg.norm(sparse_full - rom, axis=0)
    # Differentiating the modal reconstruction gives (F g a, 0).
    derivative = np.concatenate([-W @ ((np.sqrt(lam)*np.sin(2*np.sqrt(lam)))[:,None]*a),
                                  CW @ (np.cos(2*np.sqrt(lam))[:,None]*a)])
    direct_defect = derivative - J @ rom
    residual_formula = np.concatenate([F @ (gfun(np.sqrt(lam), [2])[0,:,None]*a),
                                       np.zeros((C.shape[1],3))])
    residual_relative = np.linalg.norm(direct_defect-residual_formula)/np.linalg.norm(direct_defect)
    result = {
        "analytic_vs_original_gram_relative_difference": float(prior_relative),
        "analytic_vs_gauss_legendre_gram_relative_difference": float(gram_relative),
        "full_state_sparse_vs_modal_relative_difference": float(full_relative),
        "modal_defect_identity_relative_difference": float(residual_relative),
        "all_small_system_bounds_hold": bool(np.all(actual <= bound)),
        "small_system_bound_over_actual": (bound/actual).tolist(),
    }
    assert prior_relative < 1e-13 and gram_relative < 1e-10
    assert full_relative < 1e-10 and residual_relative < 1e-10
    assert result["all_small_system_bounds_hold"]
    OUT.mkdir(exist_ok=True)
    dump(OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
