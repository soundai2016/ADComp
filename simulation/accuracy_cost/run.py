#!/usr/bin/env python3
"""Same-grid accuracy and cost of local and source-conditioned acoustic reduction."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh, expm_multiply, splu, LinearOperator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from acoustic_core import build_sparse, ports, CONFIG as CORE_CONFIG
CFG = json.loads((HERE / "config.json").read_text())
assert CORE_CONFIG["sigma"] == CFG["sigma"], "Port widths differ between experiment and acoustic kernel"
OUT = HERE / "results"
STAGES = CFG["timing_stages"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return time.perf_counter()


def assembly(n, name, need_stiffness=True):
    X, Y, rho, bulk, mass, C, J = build_sparse(n, name)
    B = ports(X, Y, bulk, CFG["source_centres"], True)
    R = ports(X, Y, bulk, CFG["receiver_centres"], False)
    A = (C @ C.T).tocsr() if need_stiffness else None
    return mass, A, J, B, R, rho, bulk


def cb_basis(n, mass, A, m):
    middle = n // 2 - 1
    interface = np.arange(middle * n, (middle + 1) * n)
    interiors = [np.arange(middle * n), np.arange((middle + 1) * n, n * n)]
    lift = np.zeros((n * n, n))
    lift[interface] = np.eye(n)
    modes = []
    for ii in interiors:
        aii = A[ii][:, ii].tocsc()
        aib = A[ii][:, interface].toarray()
        lu = splu(aii)
        lift[ii] = -lu.solve(aib)
        inverse = LinearOperator(aii.shape, matvec=lu.solve, matmat=lu.solve)
        values, vectors = eigsh(aii, k=m, sigma=0, which="LM", OPinv=inverse,
                               tol=CFG["cb_eigen_tolerance"],
                               v0=np.random.default_rng(123).normal(size=len(ii)))
        order = np.argsort(values)
        block = np.zeros((n * n, m))
        block[ii] = vectors[:, order]
        modes.append(block)
    Q, _ = linalg.qr(np.column_stack([lift] + modes), mode="economic", check_finite=False)
    return Q


def append_orthogonal(Q, W, remaining):
    # Two passes protect the rational basis against repeated low-frequency directions.
    scale = np.max(np.linalg.norm(W, axis=0), initial=0.0)
    if scale == 0:
        return Q
    for _ in range(2):
        W = W - Q @ (Q.T @ W)
    U, RR, _ = linalg.qr(W, mode="economic", pivoting=True, check_finite=False)
    rank = int(np.sum(np.abs(np.diag(RR)) > 1e-12 * scale))
    if rank:
        Q = np.column_stack([Q, U[:, :min(rank, remaining)]])
    return Q


def rational_basis(mass, A, B, r):
    constant = np.sqrt(mass)
    constant /= np.linalg.norm(constant)
    Q = constant[:, None]
    Q = append_orthogonal(Q, B.copy(), r - Q.shape[1])
    factors = {}
    frequencies = CFG["krylov_frequencies"]
    iteration = 0
    while Q.shape[1] < r:
        shift_index = iteration % len(frequencies)
        if shift_index not in factors:
            s = CFG["krylov_shift_damping"] + 2j * np.pi * frequencies[shift_index]
            factors[shift_index] = splu((A + (s * s) * sparse.eye(A.shape[0], format="csr")).tocsc())
        W = factors[shift_index].solve(np.asarray(Q[:, -3:], dtype=np.complex128))
        Q = append_orthogonal(Q, np.column_stack([W.real, W.imag]), r - Q.shape[1])
        iteration += 1
        if iteration > 4 * r:
            raise RuntimeError("Rational Krylov basis broke down before the prescribed dimension")
    Q, _ = linalg.qr(Q, mode="economic", check_finite=False)
    return Q, {"iterations": iteration, "distinct_shift_factorizations": len(factors)}


def full_run(n, name, times):
    timing = {key: 0.0 for key in STAGES}
    start = now()
    mass, A, J, B, R, rho, bulk = assembly(n, name, need_stiffness=False)
    timing["assembly"] = now() - start
    start = now()
    initial = np.zeros((J.shape[0], 3))
    initial[:n * n] = B
    state = initial
    U = np.empty((len(times), n * n, 3))
    U[0] = B
    timing["propagation"] = now() - start
    energy0 = np.sum(initial ** 2, axis=0)
    max_drift = 0.0
    chunk_steps = int(round(CFG["chunk_duration"] / (times[1] - times[0])))
    for index in range(0, len(times) - 1, chunk_steps):
        steps = min(chunk_steps, len(times) - 1 - index)
        start = now()
        block = expm_multiply(J, state, start=0.0, stop=steps * (times[1] - times[0]),
                              num=steps + 1, traceA=0.0)
        U[index + 1:index + steps + 1] = block[1:, :n * n]
        state = block[-1].copy()
        timing["propagation"] += now() - start
        energies = np.sum(block * block, axis=1)
        max_drift = max(max_drift, float(np.max(np.abs(energies / energy0 - 1))))
    start = now()
    Y = np.einsum("ir,tis->trs", R, U, optimize=True)
    timing["receiver_output"] = now() - start
    start = now()
    pressure = U / np.sqrt(mass)[None, :, None]
    timing["field_reconstruction"] = now() - start
    return {"time": times, "mass_weighted_pressure": U, "receiver_pressure": Y,
            "mass": mass, "rho": rho, "bulk_modulus": bulk}, timing, {
            "energy_relative_drift": max_drift,
            "generator_skew_defect": float(np.max(np.abs((J + J.T).data), initial=0.0))}


def reduced_run(n, name, m, method, times):
    timing = {key: 0.0 for key in STAGES}
    start = now()
    mass, A, J, B, R, rho, bulk = assembly(n, name)
    timing["assembly"] = now() - start
    r = n + 2 * m
    start = now()
    details = {}
    if method == "craig_bampton":
        Q = cb_basis(n, mass, A, m)
    elif method == "rational_krylov":
        Q, details = rational_basis(mass, A, B, r)
    else:
        raise ValueError(method)
    timing["basis"] = now() - start
    start = now()
    Ar = Q.T @ (A @ Q)
    Ar = (Ar + Ar.T) / 2
    lam, V = linalg.eigh(Ar, check_finite=False, driver="evd")
    negative_min = float(lam[0])
    if negative_min < -1e-9 * max(lam[-1], 1.0):
        raise RuntimeError("Projected stiffness is indefinite")
    lam[lam < 1e-10] = 0.0
    coefficients = V.T @ (Q.T @ B)
    reduced_receivers = R.T @ Q
    timing["projection_eigensolve"] = now() - start
    start = now()
    omega = np.sqrt(lam)
    modal = np.cos(times[:, None] * omega[None, :])[:, :, None] * coefficients[None, :, :]
    timing["propagation"] = now() - start
    start = now()
    receiver_modes = reduced_receivers @ V
    Y = np.einsum("ir,trs->tis", receiver_modes, modal, optimize=True)
    timing["receiver_output"] = now() - start
    start = now()
    physical_modes = (Q @ V) / np.sqrt(mass[:, None])
    pressure = np.einsum("ir,trs->tis", physical_modes, modal, optimize=True)
    timing["field_reconstruction"] = now() - start
    U = pressure * np.sqrt(mass)[None, :, None]
    constant = np.sqrt(mass)
    constant /= np.linalg.norm(constant)
    AQ = A @ Q
    structure = {
        "orthonormality_frobenius": float(np.linalg.norm(Q.T @ Q - np.eye(r))),
        "projected_stiffness_min_eigenvalue_before_roundoff_clip": negative_min,
        "constant_mode_projection_error": float(np.linalg.norm(constant - Q @ (Q.T @ constant))),
        "projected_stiffness_symmetry_relative": float(np.linalg.norm(Q.T @ AQ - AQ.T @ Q) / max(np.linalg.norm(Ar), 1e-300)),
        "initial_pressure_projection_error": float(np.linalg.norm(B - Q @ (Q.T @ B)) / np.linalg.norm(B)),
        **details,
    }
    raw = {"time": times, "receiver_pressure": Y, "orthonormal_basis": Q, "reduced_eigenvectors": V,
           "reduced_eigenvalues": lam, "initial_modal_coefficients": coefficients, "mass": mass}
    return raw, U, timing, structure


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def summarize_timing(repetitions):
    median = {key: float(np.median([rep[key] for rep in repetitions])) for key in STAGES}
    # Total is the median of each complete repetition, not a sum of component medians.
    for name, keys in {
        "setup_seconds": ["assembly", "basis", "projection_eigensolve"],
        "receiver_online_seconds": ["propagation", "receiver_output"],
        "field_online_seconds": ["propagation", "field_reconstruction"],
        "both_online_seconds": ["propagation", "receiver_output", "field_reconstruction"],
        "receiver_total_seconds": ["assembly", "basis", "projection_eigensolve", "propagation", "receiver_output"],
        "field_total_seconds": ["assembly", "basis", "projection_eigensolve", "propagation", "field_reconstruction"],
        "both_total_seconds": STAGES,
    }.items():
        median[name] = float(np.median([sum(rep[key] for key in keys) for rep in repetitions]))
    return median


def warmup():
    times = np.linspace(0, .2, 17)
    full_run(8, "rotated_stripes", times)
    reduced_run(8, "rotated_stripes", 4, "craig_bampton", times)
    reduced_run(8, "rotated_stripes", 4, "rational_krylov", times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=CFG["grids"])
    parser.add_argument("--materials", nargs="+", default=CFG["materials"])
    parser.add_argument("--cpus", type=int, nargs="+", help="Override requested CPU affinity; default from config.json")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    requested_affinity = args.cpus if args.cpus is not None else CFG["cpu_affinity"]
    used_affinity = None
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        used = set(requested_affinity) & available
        if not used:
            used = set(sorted(available)[:CFG["threads"]])
        os.sched_setaffinity(0, used)
        used_affinity = sorted(os.sched_getaffinity(0))
        if used_affinity != sorted(requested_affinity):
            print(f"Requested CPU affinity {requested_affinity}; available affinity used: {used_affinity}", flush=True)
    digest = hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()
    for prior in OUT.glob("*_timing.json"):
        prior_record = json.loads(prior.read_text())
        if prior_record["config_sha256"] != digest:
            raise RuntimeError("Configuration differs from existing measurements")
        if prior_record.get("cpu_affinity") != used_affinity:
            raise RuntimeError("CPU affinity differs from existing measurements; use a fresh results directory")
    dump(OUT / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
         "scipy": scipy.__version__, "platform": platform.platform(),
         "requested_cpu_affinity": requested_affinity, "cpu_affinity": used_affinity,
         "cpu_information": Path("/proc/cpuinfo").read_text().split("\n\n")[0] if Path("/proc/cpuinfo").exists() else platform.processor(),
         "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
         "config_sha256": digest,
         "code_sha256": {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [HERE / "run.py", HERE.parent / "acoustic_core.py"]}})
    warmup()
    times = np.linspace(0, CFG["time_end"], CFG["samples"])
    execution_start = time.time()
    for n in args.grids:
        for name in args.materials:
            stem = f"{name}_n{n}_full"
            path = OUT / f"{stem}_timing.json"
            if not path.exists():
                repetitions = []
                interval_start = time.time()
                for repetition in range(CFG["repetitions"]):
                    answer, timing, structure = full_run(n, name, times)
                    repetitions.append(timing)
                np.savez_compressed(OUT / f"{stem}.npz", **answer)
                record = {"config_sha256": digest, "material": name, "grid": n, "method": "full_sparse_exponential",
                          "cpu_affinity": used_affinity,
                          "dimension": n * n, "repetitions": repetitions, "median": summarize_timing(repetitions),
                          "structure": structure, "timed_interval_unix": [interval_start, time.time()]}
                dump(path, record)
                print(stem, record["median"], flush=True)
            with np.load(OUT / f"{stem}.npz") as reference:
                full_U = reference["mass_weighted_pressure"]
                full_Y = reference["receiver_pressure"]
            for m in CFG["modes_per_patch"]:
                for method in ("craig_bampton", "rational_krylov"):
                    reduced_stem = f"{name}_n{n}_{method}_m{m}"
                    target = OUT / f"{reduced_stem}_timing.json"
                    if target.exists():
                        continue
                    repetitions = []
                    interval_start = time.time()
                    for repetition in range(CFG["repetitions"]):
                        answer, U, timing, structure = reduced_run(n, name, m, method, times)
                        repetitions.append(timing)
                    errors = {"receiver_relative_error": relative(answer["receiver_pressure"], full_Y),
                              "mass_weighted_pressure_relative_error": relative(U, full_U),
                              "receiver_per_source": [relative(answer["receiver_pressure"][:, :, s], full_Y[:, :, s]) for s in range(3)],
                              "mass_weighted_pressure_per_source": [relative(U[:, :, s], full_U[:, :, s]) for s in range(3)]}
                    record = {"config_sha256": digest, "material": name, "grid": n, "method": method,
                              "cpu_affinity": used_affinity,
                              "modes_per_patch": m, "dimension": n + 2 * m,
                              "repetitions": repetitions, "median": summarize_timing(repetitions), "errors": errors,
                              "threshold_pass": all(errors[key] <= value for key, value in CFG["accuracy_thresholds"].items()),
                              "structure": structure, "timed_interval_unix": [interval_start, time.time()]}
                    np.savez_compressed(OUT / f"{reduced_stem}.npz", **answer)
                    dump(target, record)
                    print(reduced_stem, errors["receiver_relative_error"], errors["mass_weighted_pressure_relative_error"],
                          "total", record["median"]["both_total_seconds"], flush=True)
    dump(OUT / "execution.json", {"start_unix": execution_start, "end_unix": time.time(),
         "grids": args.grids, "materials": args.materials, "affinity": used_affinity,
         "config_sha256": digest, "timing_run": "Sequential cases, methods and repetitions within this process."})


if __name__ == "__main__":
    main()
