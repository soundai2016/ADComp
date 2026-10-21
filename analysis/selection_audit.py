#!/usr/bin/env python3
"""Compare same-grid and continuous-reference candidate selections."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
from statistics import mean

HERE = Path(__file__).resolve().parent


def audit(data_dir, output_dir, simulation_root=None):
    source = data_dir / 'accuracy.csv'
    provenance = json.loads((data_dir / 'provenance.json').read_text())
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != provenance['accuracy_sha256']:
        raise RuntimeError('accuracy.csv differs from its recorded SHA-256')
    source_verified = False
    if simulation_root is not None:
        simulation_input = simulation_root / 'source_receiver_validation/results/accuracy.csv'
        if hashlib.sha256(simulation_input.read_bytes()).hexdigest() != actual_sha:
            raise RuntimeError('Simulation accuracy.csv differs from the bundled input')
        source_verified = True
    output_dir.mkdir(parents=True, exist_ok=True)
    with source.open(newline='') as f:
        raw = list(csv.DictReader(f))
    rows = [r for r in raw if r['group'] == 'aggregate' and r['method'] == 'craig_bampton']
    assert len(rows) == 72, len(rows)
    keycols = ('grid', 'source_layout', 'receiver_layout', 'time_end')
    groups = {}
    for r in rows:
        key = (int(r['grid']), int(r['source_layout']), int(r['receiver_layout']), float(r['time_end']))
        groups.setdefault(key, []).append(r)
    assert set(groups) == set(itertools.product((64, 128), (0,1), (0,1), (.5,2.,8.)))
    results = []
    for key in sorted(groups):
        candidates = groups[key]
        assert sorted(int(c['modes_per_patch']) for c in candidates) == [32,64,128]
        assert len(set(c['reference_squared_norm'] for c in candidates)) == 1
        assert float(candidates[0]['reference_squared_norm']) > 0
        # Prefer the smaller dimension when errors tie.
        same = min(candidates, key=lambda c: (float(c['same_grid_receiver_error']), int(c['modes_per_patch'])))
        continuous = min(candidates, key=lambda c: (float(c['total_receiver_error']), int(c['modes_per_patch'])))
        for metric in ('same_grid_receiver_error', 'total_receiver_error'):
            assert len(set(float(c[metric]) for c in candidates)) == 3, ('tie',key,metric)
        out = dict(zip(keycols,key))
        out.update({
            'same_grid_selected_m':int(same['modes_per_patch']),
            'continuous_selected_m':int(continuous['modes_per_patch']),
            'same_grid_selected_dimension':int(same['dimension']),
            'continuous_selected_dimension':int(continuous['dimension']),
            'same_grid_error_at_same_grid_choice_percent':100*float(same['same_grid_receiver_error']),
            'same_grid_error_at_continuous_choice_percent':100*float(continuous['same_grid_receiver_error']),
            'continuous_error_at_same_grid_choice_percent':100*float(same['total_receiver_error']),
            'continuous_error_at_continuous_choice_percent':100*float(continuous['total_receiver_error']),
            'selection_loss_percentage_points':100*(float(same['total_receiver_error'])-float(continuous['total_receiver_error'])),
            'different_choice':int(same['modes_per_patch'] != continuous['modes_per_patch']),
            'reference_squared_norm':float(candidates[0]['reference_squared_norm']),
        })
        for c in sorted(candidates,key=lambda r:int(r['modes_per_patch'])):
            m = int(c['modes_per_patch'])
            out[f'm{m}_same_grid_error_percent'] = 100*float(c['same_grid_receiver_error'])
            out[f'm{m}_continuous_error_percent'] = 100*float(c['total_receiver_error'])
        results.append(out)
    with (output_dir / 'selection_regret.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    positive = [r for r in results if r['different_choice']]
    summary = {
        'source_path':source.name,
        'source_sha256':actual_sha,
        'source_rows':len(raw),
        'aggregate_cb_rows':len(rows),
        'comparison_cases':len(results),
        'different_choice_cases':len(positive),
        'same_grid_selected_m_counts':{str(m):sum(r['same_grid_selected_m']==m for r in results) for m in (32,64,128)},
        'continuous_selected_m_counts':{str(m):sum(r['continuous_selected_m']==m for r in results) for m in (32,64,128)},
        'mean_selection_loss_pp_all_cases':mean(r['selection_loss_percentage_points'] for r in results),
        'mean_selection_loss_pp_different_choice_cases':mean(r['selection_loss_percentage_points'] for r in positive),
        'maximum_selection_loss_case':max(results,key=lambda r:r['selection_loss_percentage_points']),
        'by_time':{str(T):{
            'cases':sum(r['time_end']==T for r in results),
            'different_choice_cases':sum(r['time_end']==T and r['different_choice'] for r in results),
            'mean_loss_pp':mean(r['selection_loss_percentage_points'] for r in results if r['time_end']==T)
        } for T in (.5,2.,8.)},
        'by_grid':{str(n):{
            'cases':sum(r['grid']==n for r in results),
            'different_choice_cases':sum(r['grid']==n and r['different_choice'] for r in results),
            'mean_loss_pp':mean(r['selection_loss_percentage_points'] for r in results if r['grid']==n)
        } for n in (64,128)},
        'tie_cases':0,
        'definition':'100 * (total_receiver_error at same-grid minimum - total_receiver_error at continuous-reference minimum)',
        'candidate_set':[32,64,128],
        'coverage':'2 grids x 4 source/receiver layouts x 3 cumulative windows; 12 channels jointly; finite 356-mode continuous initial field.',
        'normalization':'same_grid_receiver_error uses norm(Y_h); total_receiver_error uses norm(Y_cont). Both denominators are constant over m within a case. Selection loss compares total_receiver_error values with the same continuous denominator.',
        'scope':'Retrospective comparison of three candidates requiring a continuous reference.',
    }
    (output_dir/'selection_regret_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    maximum_loss = summary['maximum_selection_loss_case']['selection_loss_percentage_points']
    print(f"Verified {len(results)} selection cases: {len(positive)} different choices, "
          f"maximum loss {maximum_loss:.6f} percentage points.")
    verification = dict(passed=True, source_sha256=actual_sha,
        simulation_input_hash_verified=source_verified, source_rows=len(raw),
        aggregate_cb_rows=len(rows), comparison_cases=len(results),
        no_metric_ties=True, all_candidate_sets_complete=True,
        all_within_case_reference_denominators_identical=True)
    (output_dir / 'selection_verification.json').write_text(json.dumps(verification, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'data' / 'analysis')
    parser.add_argument('--output-dir', type=Path, default=HERE.parent / 'results' / 'analysis')
    parser.add_argument('--simulation-root', type=Path, help='Verify accuracy.csv against this simulation directory')
    args = parser.parse_args()
    audit(args.data_dir, args.output_dir, args.simulation_root)


if __name__ == '__main__':
    main()
