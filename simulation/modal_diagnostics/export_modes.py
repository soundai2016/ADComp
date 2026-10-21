#!/usr/bin/env python3
"""Export pressure modes selected by the fixed coupling-frequency criterion."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def main():
    config = json.loads((HERE / "config.json").read_text())
    selected, fields = [], {}
    for material in config["materials"]:
        with (HERE / "results" / f"{material}_modal_pairs.csv").open() as stream:
            pairs = [row for row in csv.DictReader(stream) if row["method"] == "craig_bampton" and row["m"] == "32"]
        def score(row):
            return float(row["fine_residue_norm"]) * abs(float(row["omega_rom"]) - float(row["omega_coarse"]))
        chosen = max(pairs, key=score)
        coarse_index, reduced_index, fine_index = (int(chosen[key]) for key in ("coarse_mode", "rom_mode", "fine_mode"))
        coarse = load(HERE / "results" / f"{material}_n32_spectrum.npz")
        fine = load(HERE / "results" / f"{material}_n128_spectrum.npz")
        reduced = load(HERE.parent / "accuracy_cost/results" / f"{material}_n32_craig_bampton_m32.npz")
        coarse_match = load(HERE / "results" / f"{material}_coarse_fine_matching.npz")
        reduced_match = load(HERE / "results" / f"{material}_craig_bampton_m32_matching.npz")
        cf = (coarse_match["coarse_indices"] == coarse_index) & (coarse_match["fine_indices"] == fine_index)
        rc = (reduced_match["rom_indices"] == reduced_index) & (reduced_match["coarse_indices"] == coarse_index)
        if cf.sum() != 1 or rc.sum() != 1:
            raise ValueError("The selected mode must have unique coarse/fine and reduced/coarse matches.")
        fine_sign = float(coarse_match["sign"][cf][0])
        reduced_sign = float(reduced_match["sign"][rc][0])
        fields[material + "_fine_pressure_mode"] = (fine_sign * fine["eigenvectors"][:, fine_index] / np.sqrt(fine["mass"])).reshape(128, 128)
        fields[material + "_coarse_pressure_mode"] = (coarse["eigenvectors"][:, coarse_index] / np.sqrt(coarse["mass"])).reshape(32, 32)
        fields[material + "_rom_pressure_mode"] = (reduced_sign * (reduced["orthonormal_basis"] @ reduced["reduced_eigenvectors"][:, reduced_index]) / np.sqrt(reduced["mass"])).reshape(32, 32)
        frequencies = [float(chosen[key]) / (2 * np.pi) for key in ("omega_fine", "omega_coarse", "omega_rom")]
        selected.append({"material": material, "criterion_value": score(chosen), "coarse_mode": coarse_index,
            "fine_mode": fine_index, "rom_mode": reduced_index, "fine_frequency": frequencies[0],
            "coarse_frequency": frequencies[1], "rom_frequency": frequencies[2],
            "mac_coarse_fine": float(chosen["mac_coarse_fine"]), "mac_rom_coarse": float(chosen["mac_rom_coarse"]),
            "fine_residue_norm": float(chosen["fine_residue_norm"]), "fine_sign_to_coarse": fine_sign,
            "rom_sign_to_coarse": reduced_sign})
    np.savez_compressed(HERE / "results/illustrated_pressure_modes.npz", **fields)
    metadata = {
        "selection_rule": "For accepted Craig-Bampton m32 triples, maximize fine_residue_norm * abs(omega_rom - omega_coarse) separately for each material.",
        "normalization": "Unit mass-weighted pressure norm; divide the mass-normalized vector by sqrt(mass). Fine and reduced signs align to the coarse mode using the MAC assignment.",
        "interpretation": "Illustrative modes. Quantitative diagnostics use all accepted modes and retain unmatched contributions.",
        "selected_modes": selected,
    }
    (HERE / "results/illustrated_mode_selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
