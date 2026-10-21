#!/usr/bin/env python3
"""Complete cost envelopes for the fixed SV/CB family on a uniform grid sweep."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def encode(row):
    if row is None:return None
    keys=('grid','method','modes_per_patch','requested_cfl','dimension','total_receiver_error',
          'alternative_reference_total_receiver_error','total_seconds','minimum_total_seconds','maximum_total_seconds')
    return {key:row[key] for key in keys}


def identify(row):
    return None if row is None else (row['grid'],row['method'],row['modes_per_patch'],row['requested_cfl'])


def select(rows,epsilon):
    selected={}
    for ref,key in [('primary','total_receiver_error'),('alternative','alternative_reference_total_receiver_error')]:
        for group in ('full','reduced'):
            available=[row for row in rows if row[key]<=epsilon and row['method'].startswith('full_')==(group=='full')]
            selected[ref+'_'+group]=min(available,key=lambda r:r['total_seconds']) if available else None
    result={'tolerance':epsilon,**{key:encode(value) for key,value in selected.items()}}
    full,reduced=selected['primary_full'],selected['primary_reduced']
    result['median_full_over_reduced']=None if full is None or reduced is None else full['total_seconds']/reduced['total_seconds']
    result['fastest_full_over_slowest_reduced']=None if full is None or reduced is None else full['minimum_total_seconds']/reduced['maximum_total_seconds']
    result['slowest_full_over_fastest_reduced']=None if full is None or reduced is None else full['maximum_total_seconds']/reduced['minimum_total_seconds']
    alternate_full,alternate_reduced=selected['alternative_full'],selected['alternative_reduced']
    result['alternative_median_full_over_reduced']=None if alternate_full is None or alternate_reduced is None else alternate_full['total_seconds']/alternate_reduced['total_seconds']
    result['alternative_fastest_full_over_slowest_reduced']=None if alternate_full is None or alternate_reduced is None else alternate_full['minimum_total_seconds']/alternate_reduced['maximum_total_seconds']
    result['selection_unchanged_by_reference']=all(identify(selected['primary_'+group])==identify(selected['alternative_'+group]) for group in ('full','reduced'))
    result['candidates_changing_threshold_classification']=sum((r['total_receiver_error']<=epsilon)!=(r['alternative_reference_total_receiver_error']<=epsilon) for r in rows)
    return result


def envelope(rows,lower,upper):
    breaks=sorted({lower,upper}|{r[key] for r in rows for key in ('total_receiver_error','alternative_reference_total_receiver_error') if lower<r[key]<upper})
    points=[select(rows,epsilon) for epsilon in breaks]
    intervals=[{'lower_tolerance':left['tolerance'],'upper_tolerance':right['tolerance'],
        'lower_closed':True,'upper_closed':False,**{k:v for k,v in left.items() if k!='tolerance'}}
        for left,right in zip(points[:-1],points[1:])]
    ratios=[p['median_full_over_reduced'] for p in points if p['median_full_over_reduced'] is not None]
    conservative=[p['fastest_full_over_slowest_reduced'] for p in points if p['fastest_full_over_slowest_reduced'] is not None]
    alternate_ratios=[p['alternative_median_full_over_reduced'] for p in points if p['alternative_median_full_over_reduced'] is not None]
    merged=[]
    for interval in intervals:
        pair=(identify(interval['primary_full']),identify(interval['primary_reduced']))
        if merged and pair==merged[-1]['candidate_pair']:
            merged[-1]['upper_tolerance']=interval['upper_tolerance']
            merged[-1]['selection_unchanged_throughout_interval'] &= interval['selection_unchanged_by_reference']
        else:
            merged.append({**interval,'candidate_pair':pair,
                'selection_unchanged_throughout_interval':interval['selection_unchanged_by_reference']})
    return {'breakpoints':breaks,'points':points,'intervals':intervals,'endpoint':points[-1],
            'merged_primary_selection_intervals':merged,
            'median_reduction_advantage_intervals':[p for p in merged if p['median_full_over_reduced'] is not None and p['median_full_over_reduced']>1],
            'separated_timing_range_advantage_intervals':[p for p in merged if p['fastest_full_over_slowest_reduced'] is not None and p['fastest_full_over_slowest_reduced']>1],
            'maximum_median_full_over_reduced':max(ratios,default=None),
            'maximum_fastest_full_over_slowest_reduced':max(conservative,default=None),
            'maximum_alternative_median_full_over_reduced':max(alternate_ratios,default=None),
            'minimum_median_full_over_reduced':min(ratios,default=None)}


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    expected=len(cfg['materials'])*len(cfg['grids'])*(len(cfg['explicit_cfl_values'])+1)
    paths=sorted((HERE/'results'/'candidates').glob('*_timing.json'))
    if len(paths)!=expected:
        raise RuntimeError(f'Complete batch required: expected {expected}, found {len(paths)}')
    references={}
    ref_hashes={}
    for material in cfg['materials']:
        references[material]={}
        for grid in (cfg['reference_grid'],cfg['alternative_reference_grid']):
            path=ROOT/'propagation'/'results'/f'{material}_n{grid}.npz'
            with np.load(path) as data:
                references[material][grid]=data['receiver_pressure'].copy()
                np.testing.assert_array_equal(data['time'],np.linspace(0,cfg['time_end'],cfg['samples']))
            ref_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    for path in paths:
        record=json.loads(path.read_text())
        material=record['material']
        with np.load(path.with_name(path.name.replace('_timing.json','.npz'))) as data:
            y=data['receiver_pressure'].copy()
            np.testing.assert_array_equal(data['time'],np.linspace(0,cfg['time_end'],cfg['samples']))
        reference=references[material][cfg['reference_grid']]
        alternative=references[material][cfg['alternative_reference_grid']]
        rows.append({key:record[key] for key in ('material','grid','method','modes_per_patch','requested_cfl','dimension')})
        rows[-1].update(total_receiver_error=relative(y,reference),
            alternative_reference_total_receiver_error=relative(y,alternative),
            reference_grid_difference=relative(alternative,reference),
            per_source_total_receiver_error=[relative(y[:,:,s],reference[:,:,s]) for s in range(3)],
            total_seconds=record['median']['total_seconds'],minimum_total_seconds=record['median']['minimum_total_seconds'],
            maximum_total_seconds=record['median']['maximum_total_seconds'],
            time_repetitions=[sum(stages.values()) for stages in record['repetitions']])
    seen={(r['material'],r['grid'],r['method'],r['requested_cfl']) for r in rows}
    prescribed={(material,n,method,cfl) for material in cfg['materials'] for n in cfg['grids']
                for method,cfl in [('full_stormer_verlet',c) for c in cfg['explicit_cfl_values']]+[('craig_bampton',None)]}
    assert seen==prescribed and len(rows)==len(seen)
    write_csv(HERE/'results'/'receiver_errors.csv',rows)
    lower,upper=min(cfg['receiver_tolerances']),max(cfg['receiver_tolerances'])
    envelopes={}
    sparse_envelopes={}
    selections=[]
    flat_intervals=[]
    for material in cfg['materials']:
        candidates=[r for r in rows if r['material']==material]
        envelopes[material]=envelope(candidates,lower,upper)
        sparse_envelopes[material]=envelope([r for r in candidates if r['grid'] in (64,96,128)],lower,upper)
        for epsilon in cfg['receiver_tolerances']:
            selections.append({'material':material,**select(candidates,epsilon)})
        for interval in envelopes[material]['intervals']:
            flat={'material':material,**{k:v for k,v in interval.items() if not isinstance(v,dict) and k not in ('primary_full','primary_reduced','alternative_full','alternative_reduced')}}
            for group in ('primary_full','primary_reduced','alternative_full','alternative_reduced'):
                if interval[group] is not None:
                    flat.update({group+'_'+k:v for k,v in interval[group].items()})
            flat_intervals.append(flat)
    write_csv(HERE/'results'/'tolerance_cost_intervals.csv',flat_intervals)
    dump(HERE/'results'/'tolerance_cost_envelope.json',{'range':[lower,upper],
        'materials':envelopes,'sparse_grid_subset_same_batch':sparse_envelopes,
        'scope':'Every candidate error from both references supplies a breakpoint. Intervals are left closed and right open; the 10% endpoint is retained. Both solver families are compared only when both have a passing candidate.'})
    summary={'complete':True,'settings':len(rows),'repetitions':cfg['repetitions'],
        'reference_grid':cfg['reference_grid'],'alternative_reference_grid':cfg['alternative_reference_grid'],
        'reference_sha256':ref_hashes,'receiver_tolerances':cfg['receiver_tolerances'],'selections':selections,
        'envelope_extrema':{material:{k:v for k,v in envelopes[material].items() if k.startswith(('maximum','minimum'))} for material in cfg['materials']},
        'sparse_subset_envelope_extrema':{material:{k:v for k,v in sparse_envelopes[material].items() if k.startswith(('maximum','minimum'))} for material in cfg['materials']},
        'candidate_family':'Fixed Craig-Bampton m128, Stormer-Verlet CFL0.25/0.125, n=64:8:128.',
        'cost':'Setup-inclusive three-source receiver production. Each median uses three complete stage sums.'}
    dump(HERE/'results'/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
