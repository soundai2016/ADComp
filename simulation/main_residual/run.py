#!/usr/bin/env python3
"""Residual bounds for the Craig--Bampton and source-conditioned Krylov spaces."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy import linalg, special
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from acoustic_core import build_sparse, ports, CONFIG as CORE_CONFIG
CFG = json.loads((HERE / "config.json").read_text())
OUT = HERE / "results"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gfun(omega, times):
    times = np.asarray(times)
    return times[:, None] * np.sinc(times[:, None] * omega[None, :] / np.pi)


def integrated_g_gram(omega, end):
    """Analytic integral, including the continuous zero-frequency limits."""
    zero = omega == 0
    nonzero = ~zero
    matrix = np.empty((len(omega), len(omega)))
    a = omega[nonzero][:, None]
    b = omega[nonzero][None, :]
    matrix[np.ix_(nonzero, nonzero)] = (
        0.5 * end * (np.sinc((a - b) * end / np.pi) - np.sinc((a + b) * end / np.pi)) / (a * b)
    )
    matrix[np.ix_(zero, zero)] = end ** 3 / 3
    if np.any(zero) and np.any(nonzero):
        x = omega[nonzero] * end
        q = (np.sin(x) - x * np.cos(x)) / omega[nonzero] ** 3
        small = np.abs(x) < 1e-3
        # Integral t*sin(w*t)/w dt; series avoids cancellation near w=0.
        q[small] = end ** 3 * (1/3 - x[small] ** 2/30 + x[small] ** 4/840 - x[small] ** 6/45360)
        matrix[np.ix_(zero, nonzero)] = q[None, :]
        matrix[np.ix_(nonzero, zero)] = q[:, None]
    matrix = (matrix + matrix.T) / 2
    values, vectors = linalg.eigh(matrix, check_finite=False)
    raised = (vectors * np.maximum(values, 0)) @ vectors.T if values[0] < 0 else matrix
    log = {
        "sine_gram_min_eigenvalue": float(values[0]),
        "sine_gram_upward_trace_correction": float(np.maximum(-values, 0).sum()),
        "sine_gram_frobenius_norm": float(np.linalg.norm(matrix)),
    }
    return matrix, raised, log


def guarded_integral(F, omega, coefficients, end):
    gram = F.T @ F
    analytic, raised, record = integrated_g_gram(omega, end)
    matrix = (gram * raised + gram.T * raised.T) / 2
    values, vectors = linalg.eigh(matrix, check_finite=False)
    negative = float(np.maximum(-values, 0).sum())
    matrix = (vectors * np.maximum(values, 0)) @ vectors.T if values[0] < 0 else matrix
    padding = 64 * np.finfo(float).eps * len(omega) * max(float(np.max(values)), np.finfo(float).tiny)
    matrix += padding * np.eye(len(omega))
    integral = np.einsum("is,ij,js->s", coefficients, matrix, coefficients, optimize=True)
    raw = np.einsum("is,ij,js->s", coefficients, gram * analytic, coefficients, optimize=True)
    if np.min(integral) < 0:
        raise RuntimeError("Residual Gram evaluation is negative after guarding")
    record.update({
        "residual_gram_min_eigenvalue": float(values[0]),
        "residual_gram_upward_trace_correction": negative,
        "residual_gram_diagonal_padding": float(padding),
        "min_raw_integrated_defect": float(np.min(raw)),
        "max_integral_guard_increment": float(np.max(integral - raw)),
    })
    return integral, record


def full_endpoints(n, name, C, J, B, prefix=None):
    """Complete pressure/velocity states for same-grid error evaluation."""
    initial = np.zeros((J.shape[0], B.shape[1]))
    initial[:n*n] = B
    start = time.perf_counter()
    # Equally spaced output uses one sparse exponential action; retain T=2,8.
    full = expm_multiply(J, initial, start=0, stop=8, num=5, traceA=0.0)
    selected = full[[1, 4]]
    energy = np.sum(full * full, axis=1)
    record = {
        "material": name, "grid": n,
        "generator_skew_defect": float(np.max(np.abs((J + J.T).data), initial=0.0)),
        "maximum_relative_energy_drift": float(np.max(np.abs(energy / np.sum(B*B, axis=0) - 1))),
        "propagation_seconds": time.perf_counter() - start,
    }
    label = prefix or f"{name}_n{n}"
    np.savez_compressed(OUT / f"{label}_full_endpoints.npz", time=np.array(CFG["times"]),
                        full_state=selected, initial_pressure=B)
    return selected, record


def evaluate_model(path, n, name, method, m, C, B, full):
    raw = np.load(path)
    W = raw["orthonormal_basis"] @ raw["reduced_eigenvectors"]
    lam = raw["reduced_eigenvalues"].copy()
    if np.min(lam) < 0:
        raise RuntimeError("Saved reduced model has negative squared frequencies")
    omega = np.sqrt(lam)
    coeff = raw["initial_modal_coefficients"]
    CW = C.T @ W
    F = C @ CW - W * lam[None, :]
    bnorm = np.linalg.norm(B, axis=0)
    e0 = np.linalg.norm(B - W @ coeff, axis=0)
    audit = {"model": path.name, "sha256": file_hash(path), "dimension": len(lam),
             "initial_coefficient_difference": float(np.linalg.norm(coeff - W.T @ B)),
             "basis_orthonormality_frobenius": float(np.linalg.norm(W.T @ W - np.eye(len(lam)))),
             "ritz_residual_frobenius": float(np.linalg.norm(F)),
             "initial_projection_orthogonality_defect": float(np.max(np.abs(W.T @ (B - W @ coeff))))}
    rows, guards = [], []
    for index, end in enumerate(CFG["times"]):
        integral, guard = guarded_integral(F, omega, coeff, end)
        bound = e0 + np.sqrt(end * integral)
        guarded = bound + 1e-11 * bnorm
        p = W @ (np.cos(omega * end)[:, None] * coeff)
        u = CW @ (gfun(omega, [end])[0, :, None] * coeff)
        reference = full[index]
        actual = np.sqrt(np.sum((reference[:n*n] - p)**2, axis=0)
                         + np.sum((reference[n*n:] - u)**2, axis=0))
        triangle = bnorm + np.sqrt(np.sum(p*p, axis=0) + np.sum(u*u, axis=0)) + 1e-11*bnorm
        capped = np.minimum(guarded, triangle)
        for source in CFG["source_indices"]:
            rows.append({
                "material": name, "grid": n, "method": method,
                "modes_per_patch": m, "dimension": len(lam), "source": source, "time": end,
                "actual_relative_state_error": float(actual[source] / bnorm[source]),
                "initial_relative_state_error": float(e0[source] / bnorm[source]),
                "integrated_defect_squared": float(integral[source]),
                "relative_bound": float(bound[source] / bnorm[source]),
                "relative_guarded_bound": float(guarded[source] / bnorm[source]),
                "relative_triangle_bound": float(triangle[source] / bnorm[source]),
                "relative_bound_with_triangle_cap": float(capped[source] / bnorm[source]),
                "bound_effectivity": float(guarded[source] / max(actual[source], 1e-300)),
                "capped_bound_effectivity": float(capped[source] / max(actual[source], 1e-300)),
                "bound_minus_error": float((guarded[source] - actual[source]) / bnorm[source]),
                "bound_holds": bool(actual[source] <= guarded[source]),
                "bound_accepts_5percent": bool(guarded[source] <= .05 * bnorm[source]),
                "bound_accepts_10percent": bool(guarded[source] <= .1 * bnorm[source]),
                "actual_within_5percent": bool(actual[source] <= .05 * bnorm[source]),
                "actual_within_10percent": bool(actual[source] <= .1 * bnorm[source]),
            })
        guards.append({"model": path.name, "time": end, **guard})
    return rows, guards, audit


def summarize(rows):
    selections = []
    for key in sorted(set((r["material"], r["grid"], r["method"], r["source"], r["time"]) for r in rows)):
        rr = [r for r in rows if (r["material"], r["grid"], r["method"], r["source"], r["time"]) == key]
        for threshold in CFG["acceptance_thresholds"]:
            accepted = [r for r in rr if r["relative_guarded_bound"] <= threshold]
            chosen = min(accepted, key=lambda r: r["dimension"]) if accepted else None
            selections.append(dict(zip(("material", "grid", "method", "source", "time"), key)) | {
                "threshold": threshold, "selected_dimension": chosen["dimension"] if chosen else None,
                "accepted": chosen is not None,
                "selected_bound": chosen["relative_guarded_bound"] if chosen else None,
                "selected_actual_error_audit": chosen["actual_relative_state_error"] if chosen else None,
                "any_actual_model_meets_threshold": any(r["actual_relative_state_error"] <= threshold for r in rr),
            })
    groups = []
    for key in sorted(set((r["method"], r["grid"], r["dimension"], r["time"]) for r in rows)):
        rr = [r for r in rows if (r["method"], r["grid"], r["dimension"], r["time"]) == key]
        groups.append(dict(zip(("method", "grid", "dimension", "time"), key)) | {
            "cases": len(rr),
            **{f"{stat}_{metric}": float(fn([r[metric] for r in rr]))
               for stat, fn in [("min", np.min), ("median", np.median), ("max", np.max), ("mean", np.mean)]
               for metric in ["actual_relative_state_error", "relative_guarded_bound", "bound_effectivity"]},
        })
    counts = []
    for key in sorted(set((r["method"], r["grid"], r["time"], r["threshold"]) for r in selections)):
        ss = [r for r in selections if (r["method"], r["grid"], r["time"], r["threshold"]) == key]
        counts.append(dict(zip(("method", "grid", "time", "threshold"), key)) | {
            "sources": len(ss), "bound_accepted": sum(r["accepted"] for r in ss),
            "actual_any_rank_accepted": sum(r["any_actual_model_meets_threshold"] for r in ss),
        })
    summary = {
        "model_count": len(rows) // (len(CFG["times"]) * len(CFG["source_indices"])),
        "rows": len(rows), "all_bounds_hold": all(r["bound_holds"] for r in rows),
        "minimum_bound_minus_error": min(r["bound_minus_error"] for r in rows),
        "groups": groups, "selection_counts": counts,
    }
    return summary, selections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, nargs="+", default=CFG["cpu_affinity"])
    parser.add_argument("--reuse-references", action="store_true", help="Read previously computed complete endpoint states")
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(args.cpus) & available or set(sorted(available)[:2]))
    OUT.mkdir(exist_ok=True)
    rows, guards, models, references = [], [], [], []
    start = time.perf_counter()
    for name in CFG["materials"]:
        for n in CFG["grids"]:
            print(f"full state {name} n={n}", flush=True)
            X, Y, rho, bulk, mass, C, J = build_sparse(n, name)
            B = ports(X, Y, bulk, CORE_CONFIG["source_centres"], True)
            if args.reuse_references:
                saved = np.load(OUT / f"{name}_n{n}_full_endpoints.npz")
                if not np.array_equal(saved["initial_pressure"], B):
                    raise RuntimeError("Saved full-reference initial condition differs")
                full = saved["full_state"]
                previous_records = json.loads((OUT / "full_reference_checks.json").read_text())
                record = next(r for r in previous_records if r["material"] == name and r["grid"] == n)
            else:
                full, record = full_endpoints(n, name, C, J, B)
            references.append(record)
            for method in CFG["methods"]:
                for m in CFG["modes_per_patch"]:
                    path = HERE.parent / "accuracy_cost" / "results" / f"{name}_n{n}_{method}_m{m}.npz"
                    rr, gg, audit = evaluate_model(path, n, name, method, m, C, B, full)
                    rows.extend(rr); guards.extend(gg); models.append(audit)
                    print(f"  {method} m={m}: worst T=8 error {max(r['actual_relative_state_error'] for r in rr if r['time']==8):.5g}", flush=True)
    summary, selections = summarize(rows)
    summary["wall_seconds"] = time.perf_counter() - start
    summary["full_reference_reused"] = args.reuse_references
    summary["config_sha256"] = file_hash(HERE / "config.json")
    csv_write(OUT / "bounds.csv", rows)
    csv_write(OUT / "selection.csv", selections)
    dump(OUT / "summary.json", summary)
    dump(OUT / "floating_point_guards.json", guards)
    dump(OUT / "model_integrity.json", models)
    dump(OUT / "full_reference_checks.json", references)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["groups", "selection_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
