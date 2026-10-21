#!/usr/bin/env python3
"""Three-grid full/CB accuracy study against continuous interface modes."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from accuracy_cost.run import cb_basis, rational_basis
from unequal_speed_interface.io import save_arrays
from unequal_speed_interface.kernel import (CFG, continuous_data, assembly,
    exact_receiver, exact_pressure_norm_squared, exact_mass_weighted_pressure,
    separable_setup, separable_pressure, separable_receiver, full_eigenvectors)
OUT = HERE/"results"


def dump(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def receiver_errors(actual, reference, times):
    return {f"T{end}": {"relative_error": relative(actual[times <= end+1e-12], reference[times <= end+1e-12]),
            "per_source": [relative(actual[times <= end+1e-12, :, s], reference[times <= end+1e-12, :, s]) for s in range(3)]}
            for end in (2, 8)}


def reduced_pressure(model, times):
    modal = np.cos(np.asarray(times)[:, None]*model["omega"][None])[:, :, None]*model["coefficients"][None]
    return (model["vectors"] @ modal.transpose(1, 0, 2).reshape(len(model["omega"]), -1)).reshape(-1, len(times), 3).transpose(1, 0, 2)


def field_summary(numerator, denominator):
    return {f"T{end}": {"relative_error": float(np.sqrt(numerator[j].sum()/denominator[j].sum())),
            "per_source": np.sqrt(numerator[j]/denominator[j]).tolist()}
            for j, end in enumerate((2, 8))}


def field_errors(n, full, models, times):
    arrays = {"denominator_continuous": np.zeros((2, 3)),
              "denominator_discrete": np.zeros((2, 3)), "projection_floor": np.zeros((2, 3)),
              "full": np.zeros((2, 3))}
    for name in models:
        arrays[name+"_continuous"] = np.zeros((2, 3))
        arrays[name+"_discrete"] = np.zeros((2, 3))
    subcell_min = np.inf
    for start in range(0, len(times), 16):
        tt = times[start:start+16]
        exact = exact_mass_weighted_pressure(n, tt)
        discrete = separable_pressure(full, tt)
        continuous_norm = exact_pressure_norm_squared(tt)
        unresolved = continuous_norm-np.sum(exact**2, axis=1)
        subcell_min = min(subcell_min, float(unresolved.min()))
        if unresolved.min() < -1e-9:
            raise RuntimeError("Cell averages are not an orthogonal projection")
        unresolved = np.maximum(unresolved, 0.)
        contributions = {
            "denominator_continuous": continuous_norm,
            "denominator_discrete": np.sum(discrete**2, axis=1),
            "projection_floor": unresolved,
            "full": np.sum((discrete-exact)**2, axis=1)+unresolved}
        for name, model in models.items():
            reduced = reduced_pressure(model, tt)
            contributions[name+"_continuous"] = np.sum((reduced-exact)**2, axis=1)+unresolved
            contributions[name+"_discrete"] = np.sum((reduced-discrete)**2, axis=1)
        for j, end in enumerate((2, 8)):
            active = tt <= end+1e-12
            for key, value in contributions.items():
                arrays[key][j] += value[active].sum(axis=0)
    answer = {"full": field_summary(arrays["full"], arrays["denominator_continuous"]),
              "projection_floor": field_summary(arrays["projection_floor"], arrays["denominator_continuous"]),
              "minimum_subcell_squared_norm": subcell_min}
    for name in models:
        answer[name] = {
            "total": field_summary(arrays[name+"_continuous"], arrays["denominator_continuous"]),
            "same_grid": field_summary(arrays[name+"_discrete"], arrays["denominator_discrete"])}
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=CFG["grids"])
    parser.add_argument("--no-rk", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    digest = hashlib.sha256((HERE/"config.json").read_bytes()).hexdigest()
    verification = json.loads((OUT/"verification.json").read_text())
    if not verification["passed"]:
        raise RuntimeError("Run verify.py successfully before the accuracy experiment")
    times = np.linspace(0., CFG["time_end"], CFG["samples"])
    reference = exact_receiver(times)
    save_arrays(OUT/"continuous_reference.npz", time=times,
                       receiver_pressure=reference, **continuous_data())
    dump(OUT/"environment.json", {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "threadpools": threadpool_info(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "timing_scope": CFG["timing_scope"], "config_sha256": digest,
        "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob("*.py")}})
    for n in args.grids:
        started = time.perf_counter()
        print(f"Building full discrete spectrum n={n}", flush=True)
        full = separable_setup(n)
        full_receiver = separable_receiver(full, times)
        mass, A, J, B, R, rho, bulk = assembly(n)
        models = {}
        choices = [("craig_bampton", m) for m in CFG["modes_per_patch"]]
        if n == CFG["modal_grid"] and not args.no_rk:
            choices.extend(("rational_krylov", m) for m in CFG["rational_krylov_modes_per_patch"])
        for method, m in choices:
            name = f"{method}_m{m}"
            print(f"Building n={n}, {name}", flush=True)
            details = {}
            if method == "craig_bampton":
                Q = cb_basis(n, mass, A, m)
            else:
                Q, details = rational_basis(mass, A, B, n+2*m)
            Ar = Q.T @ (A@Q)
            lam, V = linalg.eigh((Ar+Ar.T)/2, check_finite=False, driver="evd")
            if lam[0] < -1e-9*max(lam[-1], 1.):
                raise RuntimeError("Reduced stiffness is indefinite")
            lam[lam < 1e-9] = 0.
            W = Q@V
            coeff = W.T@B
            observation = R.T@W
            omega = np.sqrt(lam)
            yy = np.einsum("ir,tr,rs->tis", observation, np.cos(times[:, None]*omega[None]), coeff, optimize=True)
            models[name] = {"omega": omega, "vectors": W, "coefficients": coeff,
                "observation": observation, "receiver": yy, "method": method, "m": m,
                "structure": {"orthonormality_frobenius": float(np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]))),
                    "initial_pressure_projection_error": relative(Q@(Q.T@B), B),
                    "constant_mode_projection_error": relative(Q@(Q.T@np.sqrt(mass)), np.sqrt(mass)), **details}}
            arrays = {"time": times, "omega": omega, "coefficients": coeff,
                      "observation": observation, "receiver_pressure": yy}
            if n == CFG["modal_grid"]:
                arrays["eigenvectors"] = W
            save_arrays(OUT/f"n{n}_{name}.npz", **arrays)
        print(f"Evaluating continuous pressure errors n={n}", flush=True)
        field = field_errors(n, full, models, times)
        full_record = {"grid": n, "method": "full_separable", "dimension": n*n,
            "config_sha256": digest, "receiver_total_error": receiver_errors(full_receiver, reference, times),
            "pressure_total_error": field["full"], "pressure_cell_projection_floor": field["projection_floor"],
            "minimum_subcell_squared_norm": field["minimum_subcell_squared_norm"]}
        dump(OUT/f"n{n}_full.json", full_record)
        save_arrays(OUT/f"n{n}_full_receiver.npz", time=times, receiver_pressure=full_receiver)
        if n == CFG["modal_grid"]:
            omega, vectors, order = full_eigenvectors(full)
            save_arrays(OUT/f"n{n}_full_spectrum.npz", omega=omega, eigenvectors=vectors,
                mass=mass, source_coupling=vectors.T@B, receiver_coupling=R.T@vectors, B=B, R=R)
        print(f"n={n} full: receiver {100*full_record['receiver_total_error']['T8']['relative_error']:.6f}%, field {100*field['full']['T8']['relative_error']:.6f}%", flush=True)
        for name, model in models.items():
            reduction_error = model["receiver"]-full_receiver
            grid_error = full_receiver-reference
            cosine = {}
            for end in (2, 8):
                active = times <= end+1e-12
                er, eg = reduction_error[active], grid_error[active]
                cosine[f"T{end}"] = float(np.sum(er*eg)/max(np.linalg.norm(er)*np.linalg.norm(eg), 1e-300))
            record = {"grid": n, "method": model["method"], "modes_per_patch": model["m"],
                "dimension": len(model["omega"]), "config_sha256": digest,
                "receiver_total_error": receiver_errors(model["receiver"], reference, times),
                "receiver_same_grid_error": receiver_errors(model["receiver"], full_receiver, times),
                "pressure_total_error": field[name]["total"],
                "pressure_same_grid_error": field[name]["same_grid"],
                "pressure_cell_projection_floor": field["projection_floor"],
                "receiver_reduction_grid_error_cosine": cosine, "structure": model["structure"]}
            dump(OUT/f"n{n}_{name}.json", record)
            print(f"n={n} {name}: receiver total {100*record['receiver_total_error']['T8']['relative_error']:.6f}%, same grid {100*record['receiver_same_grid_error']['T8']['relative_error']:.6f}%", flush=True)
        print(f"Completed accuracy grid n={n} in {time.perf_counter()-started:.2f} s (execution log, not a timing benchmark).", flush=True)
    rows = []
    for file in sorted(OUT.glob("n*_*.json")):
        rec = json.loads(file.read_text())
        for end in (2, 8):
            window = f"T{end}"
            rows.append({"grid": rec["grid"], "method": rec["method"], "modes_per_patch": rec.get("modes_per_patch", ""),
                "dimension": rec["dimension"], "time_end": end,
                "receiver_total_error": rec["receiver_total_error"][window]["relative_error"],
                "continuous_pressure_total_error": rec["pressure_total_error"][window]["relative_error"],
                "receiver_same_grid_error": rec.get("receiver_same_grid_error", {}).get(window, {}).get("relative_error", ""),
                "pressure_same_grid_error": rec.get("pressure_same_grid_error", {}).get(window, {}).get("relative_error", ""),
                "pressure_cell_projection_floor": rec["pressure_cell_projection_floor"][window]["relative_error"],
                "receiver_error_vector_cosine": rec.get("receiver_reduction_grid_error_cosine", {}).get(window, "")})
    with (OUT/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(OUT/"summary.json", {"configuration": CFG, "rows": rows})


if __name__ == "__main__":
    main()
