#!/usr/bin/env python3
"""Descriptive least-cost envelopes over the prespecified 5–10% receiver-error range."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def identify(row):
    return None if row is None else (row['method'],row['grid'],row['modes_per_patch'],row.get('requested_cfl'))


def selected(row):
    if row is None:return None
    keys=('grid','method','dimension','modes_per_patch','requested_cfl','total_receiver_error',
          'alternative_reference_total_receiver_error','total_seconds','minimum_total_seconds','maximum_total_seconds')
    return {key:row.get(key) for key in keys}


def calculate(cfg,rows):
    lower,upper=min(cfg['smooth_receiver_tolerances']),max(cfg['smooth_receiver_tolerances'])
    materials={}
    interval_rows=[]
    for material in cfg['smooth_materials']:
        candidates=[row for row in rows if row['material']==material and 'total_seconds' in row]
        breaks=sorted({lower,upper}|{row[key] for row in candidates for key in ('total_receiver_error','alternative_reference_total_receiver_error') if lower<row[key]<upper})
        points=[]
        for epsilon in breaks:
            result={'tolerance':epsilon}
            bests={}
            for reference,error_key in [('primary','total_receiver_error'),('alternative','alternative_reference_total_receiver_error')]:
                for kind in ('full','reduced'):
                    passing=[row for row in candidates if row[error_key]<=epsilon and row['method'].startswith('full_')==(kind=='full')]
                    best=min(passing,key=lambda row:row['total_seconds']) if passing else None
                    bests[(reference,kind)]=best
                    result[f'{reference}_{kind}']=selected(best)
            full,reduced=bests[('primary','full')],bests[('primary','reduced')]
            result['median_full_over_reduced']=None if full is None or reduced is None else full['total_seconds']/reduced['total_seconds']
            result['fastest_full_over_slowest_reduced']=None if full is None or reduced is None else full['minimum_total_seconds']/reduced['maximum_total_seconds']
            result['selection_unchanged_by_reference']=all(identify(bests[('primary',kind)])==identify(bests[('alternative',kind)]) for kind in ('full','reduced'))
            points.append(result)
        intervals=[]
        for left,right in zip(points[:-1],points[1:]):
            intervals.append({'lower_tolerance':left['tolerance'],'upper_tolerance':right['tolerance'],
                              'lower_closed':True,'upper_closed':False,**{key:value for key,value in left.items() if key!='tolerance'}})
        materials[material]={'breakpoints':breaks,'intervals':intervals,'endpoint':points[-1],'points':points}
        for interval in intervals:
            flat={'material':material,**{key:value for key,value in interval.items() if key not in ('primary_full','primary_reduced','alternative_full','alternative_reduced')}}
            for group in ('primary_full','primary_reduced','alternative_full','alternative_reduced'):
                model=interval[group]
                if model is not None:
                    flat.update({f'{group}_{key}':value for key,value in model.items()})
            interval_rows.append(flat)
    result={'range':[lower,upper],
            'interpretation':'Descriptive lower envelopes of the complete tested candidate set. Breakpoints are all candidate errors within the prespecified range, using both propagation references. These intervals do not constitute additional independently tested tolerances.',
            'cost':'Median complete receiver-output cost; uncertainty columns give the observed three-run minima and maxima, not confidence intervals.',
            'materials':materials}
    (HERE/'results'/'tolerance_cost_envelope.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keys=list(dict.fromkeys(key for row in interval_rows for key in row))
    with (HERE/'results'/'tolerance_cost_intervals.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(interval_rows)
    return result


