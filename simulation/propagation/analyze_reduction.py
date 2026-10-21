#!/usr/bin/env python3
"""Evaluate stored Craig-Bampton receiver traces against a refined propagation grid."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
NAMES=['rotated_stripes','double_inclusion','smooth_checker']
RANKS=[0,4,8,16,32,64,96,128]


def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def main():
    parser=argparse.ArgumentParser()
    configured_grids=json.loads((ROOT/'experiment_config.json').read_text())['propagation_grids']
    parser.add_argument('--comparison-grid',type=int,default=max(configured_grids))
    args=parser.parse_args()
    results=ROOT/'results'
    archive=results/'reduced_responses.npz'
    if not archive.exists():
        raise SystemExit('Run propagation/assemble_reduced_responses.py first.')
    old=np.load(archive)
    rows=[]
    for name in NAMES:
        fine=np.load(results/f'{name}_n{args.comparison_grid}.npz')
        if not np.array_equal(old['time'],fine['time']):
            raise ValueError('Time grids differ; no implicit interpolation is permitted.')
        for T in [2.,8.]:
            mask=(old['time']>0)&(old['time']<=T)
            full=old[f'{name}_full32'][mask]
            ref=fine['receiver_pressure'][mask]
            for r in RANKS:
                y=old[f'{name}_r{r}'][mask]
                reduction=y-full
                grid=full-ref
                cross=float(np.vdot(reduction,grid).real)
                reference_squared=float(np.vdot(ref,ref).real)
                cross_normalized=2*cross/reference_squared
                cosine=cross/max(float(np.linalg.norm(reduction)*np.linalg.norm(grid)),1e-30)
                identity_residual=abs(rel(y,ref)**2-(float(np.vdot(reduction,reduction).real)/reference_squared+rel(full,ref)**2+cross_normalized))
                rows.append({'material':name,'modes_per_patch':r,'pressure_dof':32+2*r,'time_end':T,
                    'comparison_grid':args.comparison_grid,'same_grid_reduction_difference':rel(y,full),
                    'difference_from_fine_comparator':rel(y,ref),'full32_difference_from_fine_comparator':rel(full,ref),
                    'reduction_grid_error_cosine':cosine,'cross_term_normalized':cross_normalized,
                    'error_decomposition_identity_residual':identity_residual,
                    'worst_source_difference_from_fine_comparator':max(rel(y[:,:,s],ref[:,:,s]) for s in range(3))})
    with (results/'reduction_comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for r in RANKS:
        ss=[x for x in rows if x['time_end']==8. and x['modes_per_patch']==r]
        summary.append({'modes_per_patch':r,'pressure_dof':32+2*r,
            **{k:float(np.mean([x[k] for x in ss])) for k in ['same_grid_reduction_difference','difference_from_fine_comparator','full32_difference_from_fine_comparator','reduction_grid_error_cosine','cross_term_normalized']}})
    (results/'reduction_comparison_summary.json').write_text(json.dumps({
        'analysis':'Craig-Bampton reduced responses evaluated against the refined propagation grid',
        'comparison_grid':args.comparison_grid,'comparator_is_not_continuum_truth':True,
        'same_materials_and_sources_as_development_cases':True,'time_window':'0<t<=8',
        'error_decomposition_max_residual':max(x['error_decomposition_identity_residual'] for x in rows),
        'mean_over_materials':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
