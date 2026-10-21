"""Independently validate the saved numerical outputs without CB solves."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"


def norm(array):
    return float(np.linalg.norm(array))


def read_csv(name):
    return list(csv.DictReader((OUT/name).open()))


def main():
    config = json.loads((HERE/"config.json").read_text())
    manifest = json.loads((OUT/"output_manifest.json").read_text())
    mismatches = []
    for family in ("source_hashes", "artifact_hashes"):
        for name, expected in manifest[family].items():
            path = HERE/name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                mismatches.append(name)
    reference = np.load(OUT/"continuous_reference.npz")
    times, yf = reference["time"], reference["receiver_pressure"]
    config_hash = hashlib.sha256((HERE/"config.json").read_bytes()).hexdigest()
    rows = read_csv("summary.csv")
    records = json.loads((OUT/"summary.json").read_text())["rows"]
    expected_models = len(config["grids"])*(1+len(config["modes_per_patch"]))+len(config["rational_krylov_modes_per_patch"])
    total_error_defects, local_error_defects, cosine_defects, per_source_defects = [], [], [], []
    scalar_checks = []
    for row, json_row in zip(rows, records):
        n, end = int(row["grid"]), int(row["time_end"])
        active = times <= end+1e-12
        full = np.load(OUT/f"n{n}_full_receiver.npz")["receiver_pressure"]
        if row["method"] == "full_separable":
            filename = OUT/f"n{n}_full.json"
            observed = full
        else:
            stem = f"n{n}_{row['method']}_m{int(row['modes_per_patch'])}"
            filename = OUT/f"{stem}.json"
            observed = np.load(OUT/f"{stem}.npz")["receiver_pressure"]
        stored = json.loads(filename.read_text())
        scalar_checks.append(stored["config_sha256"] == config_hash)
        scalar_checks.append(int(json_row["grid"])==n and int(json_row["time_end"])==end and json_row["method"]==row["method"])
        error = norm(observed[active]-yf[active])/norm(yf[active])
        total_error_defects.extend((abs(error-float(row["receiver_total_error"])),
            abs(error-stored["receiver_total_error"][f"T{end}"]["relative_error"]),
            abs(error-json_row["receiver_total_error"])))
        for s in range(3):
            per_source = norm(observed[active,:,s]-yf[active,:,s])/norm(yf[active,:,s])
            per_source_defects.append(abs(per_source-stored["receiver_total_error"][f"T{end}"]["per_source"][s]))
        if row["method"] != "full_separable":
            er, eg = observed[active]-full[active], full[active]-yf[active]
            local_error = norm(er)/norm(full[active])
            cosine = float(np.sum(er*eg)/max(norm(er)*norm(eg),1e-300))
            local_error_defects.extend((abs(local_error-float(row["receiver_same_grid_error"])),
                abs(local_error-stored["receiver_same_grid_error"][f"T{end}"]["relative_error"])))
            cosine_defects.extend((abs(cosine-float(row["receiver_error_vector_cosine"])),
                abs(cosine-stored["receiver_reduction_grid_error_cosine"][f"T{end}"]),
                abs(cosine-json_row["receiver_error_vector_cosine"])))
    pairs = read_csv("modal_pairs.csv")
    modal_rows = json.loads((OUT/"modal_summary.json").read_text())["models"]
    fine = np.load(OUT/"continuous_catalogue.npz")
    coarse = np.load(OUT/f"n{config['modal_grid']}_full_spectrum.npz")
    decompositions = {}
    budget_defects, frequency_term_defects, counterfactual_defects, modal_stat_defects = [], [], [], []
    match_counts = {}
    component_names = ("ritz_frequency", "grid_frequency", "source_coupling", "receiver_coupling", "unmatched")
    for row in modal_rows:
        name = row["model"]
        paired = [p for p in pairs if p["model"]==name]
        match_counts[name] = len(paired)
        scalar_checks.append(len(paired)==row["accepted_triples"])
        ri = np.asarray([int(p["rom_mode"]) for p in paired])
        ci = np.asarray([int(p["coarse_mode"]) for p in paired])
        fi = np.asarray([int(p["continuous_mode"]) for p in paired])
        raw = np.load(OUT/f"n{config['modal_grid']}_{name}.npz")
        data = dict(np.load(OUT/f"{name}_decomposition.npz"))
        decompositions[name] = data
        Hr = np.einsum("rj,js->jrs", raw["observation"][:,ri], raw["coefficients"][ri])
        Hf = np.einsum("rj,js->jrs", fine["c"][:,fi], fine["b"][fi])
        cos_r = np.cos(times[:,None]*raw["omega"][ri][None])
        cos_h = np.cos(times[:,None]*coarse["omega"][ci][None])
        cos_f = np.cos(times[:,None]*fine["omega"][fi][None])
        ritz = np.einsum("tj,jrs->trs",cos_r-cos_h,Hr,optimize=True)
        grid = np.einsum("tj,jrs->trs",cos_h-cos_f,Hr,optimize=True)
        coupling = np.einsum("tj,jrs->trs",cos_f,Hr-Hf,optimize=True)
        frequency_term_defects.extend((norm(ritz-data["ritz_frequency"])/norm(yf),
            norm(grid-data["grid_frequency"])/norm(yf),
            norm(coupling-data["source_coupling"]-data["receiver_coupling"])/norm(yf)))
        error = data["rom_response"]-yf
        budget_defects.append(norm(error-sum(data[key] for key in component_names))/norm(error))
        counterfactual_defects.extend((norm(data["no_ritz_frequency_response"]-(raw["receiver_pressure"]-ritz))/norm(yf),
            norm(data["continuous_frequency_response"]-(raw["receiver_pressure"]-ritz-grid))/norm(yf)))
        for key,array in (("total_error",error),
                ("coarse_frequency_replacement_error",data["no_ritz_frequency_response"]-yf),
                ("continuous_frequency_replacement_error",data["continuous_frequency_response"]-yf)):
            modal_stat_defects.append(abs(norm(array)/norm(yf)-row[key]))
        for key in component_names:
            modal_stat_defects.append(abs(norm(data[key])/norm(yf)-row[key+"_relative_norm"]))
        wg = np.linalg.norm(Hf,axis=(1,2))
        wg /= wg.sum()
        dg = (coarse["omega"][ci]-fine["omega"][fi])/fine["omega"][fi]
        dr = (raw["omega"][ri]-coarse["omega"][ci])/fine["omega"][fi]
        modal_stat_defects.extend((abs(float(wg@dg)-row["weighted_relative_grid_shift"]),
            abs(float(wg@dr)-row["weighted_relative_ritz_shift"])))
    gram_defects = []
    for row in read_csv("modal_component_gram.csv"):
        d = decompositions[row["model"]]
        val = float(np.sum(d[row["component_a"]]*d[row["component_b"]])/norm(yf)**2)
        gram_defects.append(abs(val-float(row["normalized_inner_product"])))
    windows = read_csv("receiver_time_windows.csv")
    window_defects = []
    for row in windows:
        active = (times >= float(row["start"])-1e-12)&(times <= float(row["end"])+1e-12)
        r, s = int(row["receiver"])-1, int(row["source"])-1
        data = decompositions[row["model"]]
        ref = yf[active,r,s]
        original = norm(data["rom_response"][active,r,s]-ref)/norm(ref)
        replaced = norm(data["no_ritz_frequency_response"][active,r,s]-ref)/norm(ref)
        window_defects.extend((abs(original-float(row["original_error"])),
            abs(replaced-float(row["coarse_replacement_error"])),
            abs(100*(replaced-original)-float(row["coarse_replacement_change_percentage_points"]))))
    report = {
        "scope": "Independent checks from receiver traces and modal factors; no CB basis rebuild and no rerun of large-grid pressure histories.",
        "expected_model_count":expected_models,"model_count":len(rows)//2,
        "summary_csv_rows":len(rows),"summary_json_rows":len(records),
        "modal_model_count":len(modal_rows),"matched_triples":match_counts,
        "receiver_window_rows":len(windows),"expected_receiver_window_rows":len(modal_rows)*8*12,
        "source_hashes_checked":len(manifest["source_hashes"]),
        "artifact_hashes_checked":len(manifest["artifact_hashes"]),"hash_mismatches":mismatches,
        "receiver_total_statistic_max_absolute_defect":max(total_error_defects,default=0.),
        "receiver_same_grid_statistic_max_absolute_defect":max(local_error_defects,default=0.),
        "receiver_per_source_statistic_max_absolute_defect":max(per_source_defects,default=0.),
        "window_error_cosine_max_absolute_defect":max(cosine_defects,default=0.),
        "modal_budget_max_relative_defect":max(budget_defects,default=0.),
        "frequency_terms_reconstructed_max_normalized_defect":max(frequency_term_defects,default=0.),
        "counterfactual_reconstructed_max_normalized_defect":max(counterfactual_defects,default=0.),
        "modal_statistics_max_absolute_defect":max(modal_stat_defects,default=0.),
        "component_gram_max_absolute_defect":max(gram_defects,default=0.),
        "receiver_window_statistics_max_absolute_defect":max(window_defects,default=0.),
    }
    defect_keys = [k for k in report if "defect" in k]
    report["passed"] = (not mismatches and all(scalar_checks)
        and len(rows)==len(records)==2*expected_models
        and len(windows)==report["expected_receiver_window_rows"]
        and all(report[k] < 2e-10 for k in defect_keys))
    (OUT/"output_verification.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise RuntimeError("Unequal-speed output verification failed")


if __name__ == "__main__":
    main()
