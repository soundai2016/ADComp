#!/usr/bin/env python3
"""Summarize receiver-waveform differences between propagation grids."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
NAMES=['rotated_stripes','double_inclusion','smooth_checker']
GRIDS=json.loads((ROOT/'experiment_config.json').read_text())['propagation_grids']
LABELS={'rotated_stripes':'Rotated stripes',
        'double_inclusion':'Double inclusion',
        'smooth_checker':'Smooth checker'}


def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    data={}
    for name in NAMES:
        for n in GRIDS:
            path=RESULTS/f'{name}_n{n}.npz'
            if path.exists():
                with np.load(path) as z:
                    data[name,n]={k:z[k].copy() for k in ('time','receiver_pressure','wall_seconds','energy_relative_drift')}
                    data[name,n]['energy_check_count']=len(z['energy_check_times']) if 'energy_check_times' in z else len(z['time'])
    missing=[f'{name}_n{n}.npz' for name in NAMES for n in GRIDS if (name,n) not in data]
    if missing:
        raise FileNotFoundError('Complete the configured grid ladder before analysis: '+', '.join(missing))
    for name in NAMES:
        times=data[name,GRIDS[0]]['time']
        if any(not np.array_equal(times,data[name,n]['time']) for n in GRIDS[1:]):
            raise ValueError('Propagation output times must coincide exactly.')
    rows=[]
    for name in NAMES:
        ns=[n for n in GRIDS if (name,n) in data]
        if len(ns)<2:
            continue
        pairs=[('adjacent',a,b) for a,b in zip(ns[:-1],ns[1:])]
        pairs += [('finest_computed_comparator',n,ns[-1]) for n in ns[:-1]]
        for kind,n,m in pairs:
            times=data[name,n]['time']
            for horizon in [2.,8.]:
                mask=(times>0)&(times<=horizon)
                a=data[name,n]['receiver_pressure'][mask]
                b=data[name,m]['receiver_pressure'][mask]
                each=[rel(a[:,:,s],b[:,:,s]) for s in range(3)]
                rows.append({'material':name,'comparison':kind,'coarse_grid':n,'fine_grid':m,
                    'time_end':horizon,'relative_difference_fine_denominator':rel(a,b),
                    'relative_difference_coarse_denominator':rel(b,a),
                    'source1_relative_difference':each[0],'source2_relative_difference':each[1],
                    'source3_relative_difference':each[2],'worst_source_relative_difference':max(each),
                    'denominator':'fine-grid receiver waveform norm; sums over output times t>0, receivers and sources'})
    with (RESULTS/'grid_comparisons.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for a,b in zip(GRIDS[:-1],GRIDS[1:]):
        for horizon in [2.,8.]:
            ss=[r for r in rows if r['comparison']=='adjacent' and r['coarse_grid']==a and r['fine_grid']==b and r['time_end']==horizon]
            if ss:
                summary.append({'coarse_grid':a,'fine_grid':b,'time_end':horizon,'materials_completed':len(ss),
                    'mean_relative_difference':float(np.mean([r['relative_difference_fine_denominator'] for r in ss])),
                    'max_material_relative_difference':max(r['relative_difference_fine_denominator'] for r in ss),
                    'max_source_relative_difference':max(r['worst_source_relative_difference'] for r in ss)})
    result={'analysis':'receiver-waveform differences between finite grids',
        'finest_grid_is_not_truth':True,'complete_requested_ladder':len(data)==len(NAMES)*len(GRIDS),
        'adjacent_grid_summary':summary,
        'max_energy_relative_drift':max(float(d['energy_relative_drift']) for d in data.values()),
        'energy_sampling':'All 641 output times for sparse exponentials; the five snapshot times for Chebyshev propagation.',
        'sum_simulation_wall_seconds':sum(float(d['wall_seconds']) for d in data.values()),
        'definition':'relative waveform difference uses fine-grid denominator; each material has 3 sources and 4 receivers',
        'scope':'Three smooth development media, rigid walls, fixed sigma=0.07, nondimensional T=8; 3 sources and 4 receivers.'}
    clock_path=RESULTS/'clock_check_rotated_stripes_n64.npz'
    if clock_path.exists() and ('rotated_stripes',64) in data:
        with np.load(clock_path) as z:
            yy=z['receiver_pressure'];tt=z['time']
            integral_fine=float(np.trapezoid(np.sum(yy**2,axis=(1,2)),tt))
            integral_coarse=float(np.trapezoid(np.sum(yy[::2]**2,axis=(1,2)),tt[::2]))
            result['output_time_quadrature_audit']={
                'grid':64,'material':'rotated_stripes','fine_samples':len(tt),'coarse_samples':len(tt[::2]),
                'waveform_norm_relative_change':abs(np.sqrt(integral_fine/integral_coarse)-1),
                'interpretation':'checks the waveform norm quadrature only, not every cross-grid difference numerator'}
    dump(RESULTS/'audit_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
