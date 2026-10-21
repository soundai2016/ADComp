#!/usr/bin/env python3
"""Select the least expensive interface-case candidate at fixed physical tolerances."""
import argparse
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    rows=[]
    for path in sorted((HERE/'results'/'interface').glob('*_timing.json')):
        record=json.loads(path.read_text())
        for error in record['errors']:
            row={key:record[key] for key in ('material','grid','method','dimension','modes_per_patch')}
            row.update(error)
            if error['time_end']==cfg['timing_time_end']:
                row.update(record['median'])
            rows.append(row)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (HERE/'results'/'interface_errors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value) if isinstance(value,list) else value for key,value in row.items()})
    selections=[]
    candidates=[row for row in rows if 'both_total_seconds' in row]
    expected=len(cfg['interface_full_grids'])+2*len(cfg['interface_reduced_grids'])*len(cfg['interface_modes_per_patch'])
    if len(candidates)!=expected and not args.allow_partial:
        raise RuntimeError(f'Incomplete timed candidate set: expected {expected}, found {len(candidates)}')
    for tolerance in cfg['interface_receiver_tolerances']:
        for output in ('receiver_only','receiver_and_field'):
            column='receiver_total_seconds' if output=='receiver_only' else 'both_total_seconds'
            min_column=column.replace('_seconds','_minimum_seconds');max_column=column.replace('_seconds','_maximum_seconds')
            passing=[row for row in candidates if row['receiver_error']<=tolerance
                     and (output=='receiver_only' or row['continuous_field_error']<=cfg['interface_field_tolerance'])]
            full=[row for row in passing if row['method'].startswith('full_')]
            reduced=[row for row in passing if not row['method'].startswith('full_')]
            best_full=min(full,key=lambda row:row[column]) if full else None
            best_reduced=min(reduced,key=lambda row:row[column]) if reduced else None
            best=min(passing,key=lambda row:row[column]) if passing else None
            def encode(row):
                if row is None:return None
                answer={key:row[key] for key in ('grid','method','modes_per_patch','dimension','receiver_error','continuous_field_error','projection_floor')}
                answer['total_seconds']=row[column]
                answer['minimum_total_seconds']=row[min_column]
                answer['maximum_total_seconds']=row[max_column]
                answer['receiver_margin']=tolerance-row['receiver_error']
                answer['field_margin']=cfg['interface_field_tolerance']-row['continuous_field_error']
                return answer
            selections.append({'receiver_tolerance':tolerance,'field_tolerance':cfg['interface_field_tolerance'] if output=='receiver_and_field' else None,
                'output':output,'tested_candidates':len(candidates),'passing_candidates':len(passing),
                'least_cost_full':encode(best_full),'least_cost_reduced':encode(best_reduced),'least_cost_any':encode(best),
                'speedup_over_least_cost_passing_full':None if best_full is None or best_reduced is None else best_full[column]/best_reduced[column],
                'fastest_full_over_slowest_reduced':None if best_full is None or best_reduced is None else best_full[min_column]/best_reduced[max_column],
                'slowest_full_over_fastest_reduced':None if best_full is None or best_reduced is None else best_full[max_column]/best_reduced[min_column],
                'timing_ranges_separated_in_favor_of_reduced':None if best_full is None or best_reduced is None else best_reduced[max_column]<best_full[min_column]})
    summary={'reference':'Exact finite-band continuum solution, with coefficient/receiver quadrature convergence independently checked.',
             'field_norm':'Compressibility-weighted L2 of piecewise-constant pressure against the continuum solution; includes within-cell projection loss.',
             'time_end':cfg['timing_time_end'],'complete':len(candidates)==expected,'expected_settings':expected,'number_settings':len(candidates),'selections':selections}
    (HERE/'results'/'interface_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
