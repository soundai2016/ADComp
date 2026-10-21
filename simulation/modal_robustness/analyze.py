#!/usr/bin/env python3
"""Summarize the modal sweep and verify numerical identities."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
CFG=json.loads((HERE/"config.json").read_text())


def read(kind):
    data=[]
    for name in CFG["materials"]:
        with (OUT/f"{name}_{kind}.csv").open() as stream:
            for raw in csv.DictReader(stream):
                row={}
                for key,value in raw.items():
                    if value in ("True","False"):
                        row[key]=value=="True"
                    else:
                        try:row[key]=float(value)
                        except ValueError:row[key]=value
                data.append(row)
    return data


def span(rows,key):
    return [min(r[key] for r in rows),max(r[key] for r in rows)]


def main():
    sensitivity=read("sensitivity");channels=read("channel_sensitivity")
    finite=read("finite_time");baseline=read("baseline_check")
    clusters=read("clusters");reference=read("reference_matching")
    base=lambda r:r["mac_threshold"]==CFG["baseline_mac"] and r["relative_gap"]==CFG["baseline_gap"]
    primary=lambda r:r["method"]=="craig_bampton" and r["m"]==32
    summary=[]
    for name in CFG["materials"]:
        for grid in CFG["fine_grids"]:
            own=[r for r in sensitivity if r["material"]==name and r["fine_grid"]==grid and r["selection"]=="own"]
            fixed=[r for r in own if r["path"]=="rom"]
            fixedbase=next(r for r in fixed if base(r))
            paths=[r for r in own if base(r)]
            c=[r for r in channels if r["material"]==name and r["fine_grid"]==grid and r["selection"]=="own"
                and r["path"]=="rom" and r["group"]=="channel" and r["window_kind"]=="cumulative" and r["end"]==8]
            bch={(r["source"],r["receiver"]):r for r in c if base(r)}
            changed=sum(r["sign"]!=bch[(r["source"],r["receiver"])]["sign"] for r in c)
            f=[r for r in finite if r["material"]==name and r["fine_grid"]==grid and primary(r)
                and r["group"]=="channel" and r["window_kind"]!="short_cumulative"]
            nonneutral=[r for r in f if r["exact_sign"]!=0]
            all8=next(r for r in finite if r["material"]==name and r["fine_grid"]==grid and primary(r)
                and r["group"]=="all" and r["window_kind"]=="cumulative" and r["end"]==8)
            summary.append(dict(material=name,fine_grid=grid,default_matched_count=fixedbase["matched_count"],
                matched_count_range=span(fixed,"matched_count"),common_count=fixedbase["common_count"],
                fixed_rom_error_change_pp_range=[100*v for v in span(fixed,"error_change")],
                all_paths_all_thresholds_error_change_pp_range=[100*v for v in span(own,"error_change")],
                default_path_error_changes_pp={r["path"]:100*r["error_change"] for r in paths},
                frequency_compensation_ratio_default=fixedbase["frequency_compensation_ratio"],
                frequency_compensation_ratio_range=span(fixed,"frequency_compensation_ratio"),
                modal_coverage_difference=fixedbase["eigenmode_coverage_norm"],
                path_interaction_norm=fixedbase["path_interaction_norm"],
                channel_full_window_sign_changes_across_thresholds=changed,
                full_window_threshold_channel_comparisons=len(c),
                linear_sign_agreements=sum(r["linear_sign"]==r["exact_sign"] for r in nonneutral),
                linear_sign_comparisons=len(nonneutral),
                reference_margin_gt_one=sum(r["reference_sign_margin"]>1 for r in f),
                reference_margin_comparisons=len(f),minimum_reference_sign_margin=min(r["reference_sign_margin"] for r in f),
                aggregate_T8_exact_squared_change=all8["exact_squared_error_change"],
                aggregate_T8_linear_squared_change=all8["linear_full"],
                aggregate_T8_ritz_linearization_error=all8["ritz_linearization_error"],
                aggregate_T8_grid_linearization_error=all8["grid_linearization_error"],
                aggregate_T8_maximum_phase_offset=all8["maximum_phase_offset"]))
    checks={}
    verifications=[json.loads((OUT/f"{n}_verification.json").read_text()) for n in CFG["materials"]]
    checks["all_declared_sweep_rows_present"]=len(sensitivity)==len(CFG["materials"])*len(CFG["fine_grids"])*9*2*3
    checks["all_declared_channel_rows_present"]=len(channels)==len(sensitivity)*8*13
    checks["all_current_configuration_hashes_match"]=all(v["config_sha256"]==hashlib.sha256((HERE/"config.json").read_bytes()).hexdigest() for v in verifications)
    checks["all_current_analysis_hashes_match"]=all(v["analysis_sha256"]==hashlib.sha256((HERE/"run.py").read_bytes()).hexdigest() for v in verifications)
    checks["baseline_stable_quantities_reproduced"]=max(r["absolute_difference"] for r in baseline if not r["angle_near_zero_component"])<1e-10
    checks["original_order_baseline_reproduced"]=max(r["original_order_absolute_difference"] for r in baseline)<1e-10
    checks["exact_paths_close"]=max(r["decomposition_defect"] for r in sensitivity)<1e-12
    checks["exact_budgets_close"]=max(r["exact_budget_defect"] for r in finite)<1e-12
    checks["gram_identity_closes"]=max(r["linear_gram_identity_defect"] for r in finite)<1e-12
    checks["gram_positive_semidefinite_to_roundoff"]=min(r["gram_relative_minimum_eigenvalue"] for r in finite)>-1e-12
    checks["reference_change_identity_closes"]=max(r["reference_change_identity_defect"] for r in finite)<1e-12
    checks["reference_change_within_cauchy_bound"]=all(abs(r["reference_change_exact_cross"])<=r["reference_change_bound"]+1e-12 for r in finite)
    checks["cluster_basis_rotation_preserves_response"]=max(r["block_response_basis_rotation_relative_defect"] for r in clusters)<1e-10
    checks["common_sets_give_identical_results_across_thresholds"]=True
    for name in CFG["materials"]:
        for grid in CFG["fine_grids"]:
            for path in ("rom","fine","symmetric"):
                r=[r for r in sensitivity if r["material"]==name and r["fine_grid"]==grid and r["path"]==path and r["selection"]=="common"]
                checks["common_sets_give_identical_results_across_thresholds"] &= max(x["error_change"] for x in r)-min(x["error_change"] for x in r)<1e-14
    result=dict(material_summary=summary,reference_matching=reference,verification=checks,
        counts=dict(sensitivity=len(sensitivity),channel_sensitivity=len(channels),finite_time=len(finite),cluster=len(clusters)),
        maximum_baseline_difference_any_quantity=max(r["absolute_difference"] for r in baseline),
        maximum_baseline_difference_stable_quantities=max(r["absolute_difference"] for r in baseline if not r["angle_near_zero_component"]),
        maximum_original_order_baseline_difference=max(r["original_order_absolute_difference"] for r in baseline),
        ill_conditioned_angle_comparisons=[r for r in baseline if r["angle_near_zero_component"]],
        reverse_checker_channel=[r for r in finite if r["material"]=="smooth_checker" and r["fine_grid"]==128 and primary(r)
            and r["group"]=="channel" and r["source"]==2 and r["receiver"]==2 and r["window_kind"]=="cumulative" and r["end"]==8],
        max_exact_budget_defect=max(r["exact_budget_defect"] for r in finite),
        maximum_reference_change_identity_defect=max(r["reference_change_identity_defect"] for r in finite),
        maximum_cluster_rotation_defect=max(r["block_response_basis_rotation_relative_defect"] for r in clusters),
        no_success_condition_on_scientific_signs=True,
        scope=CFG["prediction_scope"],
        linearization_caution="The exact q is retained in the linear approximation. A correct sign is not a reference-free forecast; vector linearization errors are reported separately.")
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(verification=checks,counts=result["counts"],summary=summary),indent=2))
    if not all(checks.values()):
        raise RuntimeError("One or more reproducibility or mathematical verification gates failed")


if __name__=="__main__":main()
