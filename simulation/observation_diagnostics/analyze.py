#!/usr/bin/env python3
"""Observation and time-window dependence of fixed modal interventions."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
OUT = HERE / "results"
CFG = json.loads((HERE / "config.json").read_text())
KEYS = {"remove_ritz": "no_ritz_frequency_response", "fine_frequency": "fine_frequency_response"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def windows():
    for end in CFG["cumulative_end_times"]:
        yield "cumulative", 0.0, end
    edges = CFG["successive_edges"]
    for start, end in zip(edges[:-1], edges[1:]):
        yield "disjoint", start, end


def main():
    OUT.mkdir(exist_ok=True)
    channel_rows, group_rows, inputs = [], [], {}
    max_identity = 0.0
    max_ritz_identity = 0.0
    tol = CFG["squared_relative_change_zero_tolerance"]
    for material in CFG["materials"]:
        references = {}
        for grid in CFG["reference_grids"]:
            path = SOURCE / "propagation" / "results" / f"{material}_n{grid}.npz"
            with np.load(path) as data:
                references[grid] = (data["time"], data["receiver_pressure"])
            inputs[str(path.relative_to(SOURCE))] = digest(path)
        for method, modes in CFG["models"]:
            path = SOURCE / "modal_diagnostics" / "results" / f"{material}_{method}_m{modes}_decomposition.npz"
            with np.load(path) as data:
                time = data["time"]
                original = data["rom_response"]
                changed = {name: data[key] for name, key in KEYS.items()}
                ritz = data["ritz_frequency"]
            inputs[str(path.relative_to(SOURCE))] = digest(path)
            max_ritz_identity = max(max_ritz_identity,
                float(np.linalg.norm(changed["remove_ritz"] - original + ritz)
                      / np.linalg.norm(original)))
            for grid, (reference_time, reference) in references.items():
                np.testing.assert_array_equal(reference_time, time)
                for kind, start, end in windows():
                    use = (time <= end) & ((time >= start) if start == 0 else (time > start))
                    yref, y = reference[use], original[use]
                    energy = np.sum(yref * yref, axis=0)
                    assert np.all(energy > 0)
                    total_energy = float(energy.sum())
                    error = y - yref
                    sse = np.sum(error * error, axis=0)
                    common = dict(material=material, method=method, modes_per_patch=modes,
                                  reference_grid=grid, window_kind=kind, start=start, end=end,
                                  samples=int(use.sum()))
                    intervention = {}
                    for name, trajectory in changed.items():
                        delta = trajectory[use] - y
                        new_sse = np.sum((trajectory[use] - yref) ** 2, axis=0)
                        square = np.sum(delta * delta, axis=0)
                        cross = 2 * np.sum(error * delta, axis=0)
                        increment = new_sse - sse
                        residual = np.abs(increment - square - cross) / total_energy
                        max_identity = max(max_identity, float(residual.max()))
                        intervention[name] = (new_sse, square, cross, increment)
                    for source in range(3):
                        for receiver in range(4):
                            k = (receiver, source)
                            row = dict(common, source=source + 1, receiver=receiver + 1,
                                reference_energy=float(energy[k]),
                                reference_energy_share=float(energy[k] / total_energy),
                                original_relative_error=float(np.sqrt(sse[k] / energy[k])),
                                original_squared_error_contribution=float(sse[k] / total_energy))
                            for name, (new_sse, square, cross, increment) in intervention.items():
                                row.update({
                                    f"{name}_relative_error": float(np.sqrt(new_sse[k] / energy[k])),
                                    f"{name}_error_change": float((np.sqrt(new_sse[k]) - np.sqrt(sse[k])) / np.sqrt(energy[k])),
                                    f"{name}_squared_error_change_contribution": float(increment[k] / total_energy),
                                    f"{name}_intervention_square_contribution": float(square[k] / total_energy),
                                    f"{name}_intervention_cross_contribution": float(cross[k] / total_energy),
                                    f"{name}_change_sign": int(np.sign(increment[k])) if abs(increment[k]) > tol * energy[k] else 0,
                                })
                            channel_rows.append(row)
                    groups = [("all", 0, np.ones((4, 3), dtype=bool))]
                    for source in range(3):
                        select = np.zeros((4, 3), dtype=bool); select[:, source] = True
                        groups.append(("source", source + 1, select))
                    for receiver in range(4):
                        select = np.zeros((4, 3), dtype=bool); select[receiver, :] = True
                        groups.append(("receiver", receiver + 1, select))
                    for group, index, select in groups:
                        den = float(energy[select].sum())
                        old = float(sse[select].sum())
                        row = dict(common, group=group, index=index,
                            reference_energy_share=den / total_energy,
                            original_relative_error=float(np.sqrt(old / den)))
                        for name, (new_sse, square, cross, increment) in intervention.items():
                            ratio = increment / energy
                            positive, negative = ratio > tol, ratio < -tol
                            row.update({
                                f"{name}_relative_error": float(np.sqrt(new_sse[select].sum() / den)),
                                f"{name}_error_change": float((np.sqrt(new_sse[select].sum()) - np.sqrt(old)) / np.sqrt(den)),
                                f"{name}_squared_error_change_contribution": float(increment[select].sum() / total_energy),
                                f"{name}_increased_error_channels": int((positive & select).sum()),
                                f"{name}_decreased_error_channels": int((negative & select).sum()),
                                f"{name}_neutral_channels": int((~(positive | negative) & select).sum()),
                                f"{name}_increased_error_energy_share": float(energy[positive & select].sum() / den),
                                f"{name}_decreased_error_energy_share": float(energy[negative & select].sum() / den),
                            })
                        group_rows.append(row)
    write_csv("channel_errors.csv", channel_rows)
    write_csv("group_errors.csv", group_rows)
    primary = [r for r in channel_rows if r["reference_grid"] == 512]
    alternative = {(r["material"], r["method"], r["modes_per_patch"], r["window_kind"], r["start"], r["end"], r["source"], r["receiver"]): r
                   for r in channel_rows if r["reference_grid"] == 384}
    sensitivity = []
    for row in primary:
        key = tuple(row[k] for k in ("material", "method", "modes_per_patch", "window_kind", "start", "end", "source", "receiver"))
        other = alternative[key]
        record = {k: row[k] for k in ("material", "method", "modes_per_patch", "window_kind", "start", "end", "source", "receiver")}
        for name in KEYS:
            record[f"{name}_change_difference"] = row[f"{name}_error_change"] - other[f"{name}_error_change"]
            record[f"{name}_sign_changed"] = row[f"{name}_change_sign"] != other[f"{name}_change_sign"]
        sensitivity.append(record)
    write_csv("reference_sensitivity.csv", sensitivity)
    focal = [r for r in group_rows if r["group"] == "all" and r["reference_grid"] == 512
             and r["method"] == "craig_bampton" and r["modes_per_patch"] == 32]
    focal_sensitivity = [r for r in sensitivity if r["method"] == "craig_bampton" and r["modes_per_patch"] == 32]
    summary = dict(channel_rows=len(channel_rows), group_rows=len(group_rows),
        max_normalized_squared_error_identity_residual=max_identity,
        max_relative_remove_ritz_trajectory_identity=max_ritz_identity,
        focal_reference_sign_changes={name: sum(r[f"{name}_sign_changed"] for r in focal_sensitivity) for name in KEYS},
        focal_max_absolute_reference_change_difference={name: max(abs(r[f"{name}_change_difference"]) for r in focal_sensitivity) for name in KEYS},
        focal_aggregate_results=focal,
        input_sha256=inputs, config_sha256=digest(HERE / "config.json"),
        analysis_code_sha256=digest(Path(__file__)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("focal_aggregate_results", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
