#!/usr/bin/env python3
"""Extract compact modal inputs and recompute assignments from full eigensystems."""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from core import GRIDS, RANKS, assignment, chain, modal_response, relative, sha256, store_assignment

HERE = Path(__file__).resolve().parent


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in (z.files if keys is None else keys)}


def average_x_modes(x, fine, width, interface=0.5):
    """Exact cell averages of the supplied analytic piecewise cosine modes."""
    result = np.empty((len(x), len(fine["omega"])))
    for side, mask, distance in ((0, x < interface, x[x < interface]),
                                 (1, x >= interface, 1 - x[x >= interface])):
        q2 = fine["q_squared"][:, side]
        positive = q2 >= 0
        values = np.empty((len(distance), len(q2)))
        q = np.sqrt(q2[positive])
        values[:, positive] = np.cos(distance[:, None] * q[None]) * np.sinc(q * width / (2 * np.pi))[None]
        q = np.sqrt(-q2[~positive])
        z = q * width / 2
        values[:, ~positive] = np.cosh(distance[:, None] * q[None]) * (np.sinh(z) / z)[None]
        result[mask] = values * fine["amplitudes"][None, :, side]
    return result


def continuous_overlap(full, fine):
    n = len(full["y_eigenvectors"])
    x = (np.arange(n) + 0.5) / n
    ell = fine["ell"]
    xavg = average_x_modes(x, fine, 1 / n)
    # Cell averages preserve the continuous modes' mass-weighted normalization.
    all_ell = np.arange(ell.max() + 1)
    yvalues = np.cos(np.pi * x[:, None] * all_ell) * np.where(all_ell == 0, 1., np.sqrt(2.))
    yavg = yvalues[:, ell] * np.sinc(ell / (2 * n))[None]
    xweighted = xavg * np.sqrt(full["mass"].reshape(n, n)[:, 0])[:, None]
    norms = np.sqrt(np.sum(xweighted ** 2, axis=0) * np.sum(yavg ** 2, axis=0))
    ox = np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), xweighted[None])
    oy = full["y_eigenvectors"].T @ yavg
    return (ox * oy[:, None, :]).reshape(n * n, -1) / norms[None], norms


def project_separated(full, vectors):
    n = len(full["y_eigenvectors"])
    by_y = np.einsum("ijk,jl->lik", vectors.reshape(n, n, -1), full["y_eigenvectors"], optimize=True)
    return np.matmul(full["x_eigenvectors"].transpose(0, 2, 1), by_y).reshape(n * n, -1)


def extract(root, output_dir):
    root = root.resolve()
    base = root / "source_receiver_validation"
    src = base / "results"
    if not (src / "factorial_reference.npz").is_file():
        raise FileNotFoundError("--simulation-root must contain source_receiver_validation/results; run the full simulation first")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = []

    def read(name, keys=None):
        path = src / name
        input_paths.append(path)
        return load(path, keys)

    fine = read("continuous_catalogue.npz")
    reference = read("factorial_reference.npz")
    arrays = dict(time=reference["time"], omega_f=fine["omega"], active_max=fine["active_max"],
        bf=np.stack([reference["original_catalogue_coefficients"], fine["coefficients"]]),
        cf=np.stack([reference["original_catalogue_receiver_modes"], fine["receiver_modes"]]))
    checks = dict(reference_modal_replay_relative_difference=0., rom_modal_replay_relative_difference=0.,
                  baseline_assignment_indices_identical=True, baseline_assignment_mac_max_abs_difference=0.)
    for a in range(2):
        for b in range(2):
            replay = modal_response(arrays["time"], arrays["omega_f"], arrays["bf"][a], arrays["cf"][b])
            checks["reference_modal_replay_relative_difference"] = max(checks["reference_modal_replay_relative_difference"],
                relative(replay, reference["receiver_pressure"][a, b]))
    for n in GRIDS:
        print(f"Recomputing global full/continuous assignment for n={n}", flush=True)
        full = read(f"n{n}_full.npz", ["omega", "x_eigenvectors", "y_eigenvectors", "mass"])
        order = np.argsort(full["omega"].ravel())
        omega_h = full["omega"].ravel()[order]
        overlap, norms = continuous_overlap(full, fine)
        af = assignment(overlap[order], omega_h, arrays["omega_f"])
        del overlap
        arrays[f"n{n}_omega_h"] = omega_h
        arrays[f"n{n}_continuous_mode_norms"] = norms
        store_assignment(arrays, f"n{n}_af", af)
        for m in RANKS:
            prefix = f"n{n}_m{m}"
            model = read(f"n{n}_cb_m{m}.npz", ["omega", "eigenvectors", "factorial_source_coefficients",
                "factorial_receiver_modes", "factorial_receiver_pressure", "matched_reduced_indices",
                "matched_full_indices", "matched_continuous_indices", "mac_reduced_full", "mac_full_continuous"])
            overlap = project_separated(full, model["eigenvectors"])[order].T
            ar = assignment(overlap, model["omega"], omega_h)
            del overlap
            store_assignment(arrays, prefix + "_ar", ar)
            arrays[prefix + "_omega_r"] = model["omega"]
            arrays[prefix + "_br"] = model["factorial_source_coefficients"]
            arrays[prefix + "_cr"] = model["factorial_receiver_modes"]
            triples = np.asarray(chain(ar, af, arrays["omega_f"], float(arrays["active_max"]), norms, 0.9, 0.005))
            arrays[prefix + "_default_triples"] = triples[:, :3]
            for j, key in enumerate(("matched_reduced_indices", "matched_full_indices", "matched_continuous_indices")):
                checks["baseline_assignment_indices_identical"] &= bool(np.array_equal(triples[:, j], model[key]))
            checks["baseline_assignment_mac_max_abs_difference"] = max(checks["baseline_assignment_mac_max_abs_difference"],
                float(np.max(abs(ar["mac"][triples[:, 3]] - model["mac_reduced_full"]))),
                float(np.max(abs(af["mac"][triples[:, 4]] - model["mac_full_continuous"]))))
            for a in range(2):
                for b in range(2):
                    replay = modal_response(arrays["time"], model["omega"], model["factorial_source_coefficients"][a], model["factorial_receiver_modes"][b])
                    checks["rom_modal_replay_relative_difference"] = max(checks["rom_modal_replay_relative_difference"],
                        relative(replay, model["factorial_receiver_pressure"][a, b]))
            print(f"  m={m}: {len(triples)} default triples; all modal port factors retained", flush=True)
    fields = ["grid", "modes_per_patch", "source_layout", "receiver_layout", "time_end", "group_code", "source", "receiver",
        "total_receiver_error", "coarse_replacement_error", "change_percentage_points", "delta_squared_error",
        "same_mode_term", "cross_mode_term", "remainder_cross_term", "source_cross_term", "receiver_cross_term",
        "unmatched_cross_term", "reference_squared_norm"]
    path = src / "finite_time_budgets.csv"
    input_paths.append(path)
    with path.open(newline="") as stream:
        baseline = [row for row in csv.DictReader(stream) if row["group"] in ("aggregate", "channel")]
    for row in baseline:
        row["group_code"] = 0 if row["group"] == "aggregate" else 1
    arrays["baseline_expected_fields"] = np.asarray(fields)
    arrays["baseline_expected"] = np.asarray([[float(row[key]) for key in fields] for row in baseline])
    accuracy_path = src / "accuracy.csv"
    input_paths.append(accuracy_path)
    (output_dir / "accuracy.csv").write_bytes(accuracy_path.read_bytes())
    assert len(baseline) == 936, f"Unexpected aggregate/channel budget count: {len(baseline)}"
    assert checks["baseline_assignment_indices_identical"]
    assert max(float(v) for k, v in checks.items() if k != "baseline_assignment_indices_identical") < 1e-10, checks
    compact = output_dir / "compact_inputs.npz"
    np.savez_compressed(compact, **arrays)
    metadata = dict(description="Modal frequencies, port factors and port-independent matching assignments.",
        compact_sha256=sha256(compact), accuracy_sha256=sha256(accuracy_path),
        source_sha256={str(path.relative_to(root)): sha256(path) for path in input_paths},
        extraction_checks=checks,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        source_layouts=[[[0.19, 0.76], [0.54, 0.47], [0.81, 0.22]], [[0.16, 0.27], [0.39, 0.83], [0.84, 0.61]]],
        receiver_layouts=[[[0.22, 0.21], [0.79, 0.77], [0.28, 0.69], [0.68, 0.33]], [[0.12, 0.76], [0.35, 0.41], [0.65, 0.18], [0.88, 0.87]]])
    (output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {compact.stat().st_size} bytes of modal inputs; assignment and response checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-root", type=Path, default=HERE.parent / "simulation", help="Directory containing the full simulation modules and results")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "data" / "analysis")
    args = parser.parse_args()
    extract(args.simulation_root, args.output_dir)


if __name__ == "__main__":
    main()
