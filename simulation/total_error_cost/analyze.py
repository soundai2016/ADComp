#!/usr/bin/env python3
"""Total receiver error and cost against independently refined propagation grids."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def read_traces(path):
    with np.load(path) as data:
        return data['time'].copy(), data['receiver_pressure'].copy()


def assess(times, y, full, ref, alternative, end):
    keep = times <= end + 1e-12
    y, full, ref, alternative = (array[keep] for array in (y, full, ref, alternative))
    grid = full-ref
    reduced = y-full
    total = y-ref
    denominator = np.sum(ref*ref)
    product = float(np.sum(grid*reduced)/denominator)
    grid_norm = np.linalg.norm(grid)
    reduced_norm = np.linalg.norm(reduced)
    return {
        'time_end': end,
        'total_receiver_error': relative(y, ref),
        'alternative_reference_total_receiver_error': relative(y, alternative),
        'grid_receiver_error': relative(full, ref),
        'same_grid_receiver_error': relative(y, full),
        'same_denominator_non_grid_error': float(reduced_norm/np.sqrt(denominator)),
        'cross_term': 2*product,
        'error_vector_cosine': float(np.sum(grid*reduced)/(grid_norm*reduced_norm)) if grid_norm*reduced_norm else 0.0,
        'error_identity_absolute_residual': float(abs(np.sum(total*total)/denominator - (grid_norm**2+reduced_norm**2)/denominator - 2*product)),
        'reference_grid_difference': relative(alternative, ref),
        'worst_source_total_receiver_error': max(relative(y[:,:,s], ref[:,:,s]) for s in range(y.shape[2])),
        'per_source_total_receiver_error': [relative(y[:,:,s], ref[:,:,s]) for s in range(y.shape[2])],
    }


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true', help='Analyze only completed materials; no missing-reference substitution.')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    out = HERE/'results'
    out.mkdir(exist_ok=True)
    rows = []
    for material in cfg['smooth_materials']:
        fine_path = ROOT/'propagation'/'results'/f'{material}_n{cfg["reference_grid"]}.npz'
        alternative_path = ROOT/'propagation'/'results'/f'{material}_n{cfg["alternative_reference_grid"]}.npz'
        if not fine_path.exists() or not alternative_path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(fine_path if not fine_path.exists() else alternative_path)
        times, reference = read_traces(fine_path)
        other_times, alternative = read_traces(alternative_path)
        np.testing.assert_array_equal(times, other_times)
        for n in cfg['rescoring_full_grids']:
            time_full, full = read_traces(ROOT/'propagation'/'results'/f'{material}_n{n}.npz')
            np.testing.assert_array_equal(times, time_full)
            for end in cfg['error_windows']:
                rows.append({'material': material, 'grid': n, 'method': 'full_sparse_exponential',
                             'modes_per_patch': None, 'dimension': n*n,
                             'source': 'propagation', **assess(times, full, full, reference, alternative, end)})
        for data_dir, source in ((ROOT/'accuracy_cost'/'results', 'same_grid_study'), (out/'smooth', 'total_error_study')):
            for path in sorted(data_dir.glob(f'{material}_n*_timing.json')):
                record = json.loads(path.read_text())
                raw_path = path.with_name(path.name.replace('_timing.json', '.npz'))
                time_model, y = read_traces(raw_path)
                np.testing.assert_array_equal(times, time_model)
                n = record['grid']
                _, full = read_traces(ROOT/'propagation'/'results'/f'{material}_n{n}.npz')
                for end in cfg['error_windows']:
                    row = {'material': material, 'grid': n, 'method': record['method'],
                           'modes_per_patch': record.get('modes_per_patch'), 'requested_cfl': record.get('requested_cfl'), 'dimension': record['dimension'],
                           'source': source, **assess(times, y, full, reference, alternative, end)}
    # Cost selection uses the uniformly timed receiver-output experiment.
                    if source == 'total_error_study' and end == cfg['timing_time_end']:
                        row['total_seconds'] = record['median']['total_seconds']
                        row['minimum_total_seconds'] = record['median']['minimum_total_seconds']
                        row['maximum_total_seconds'] = record['median']['maximum_total_seconds']
                        row['time_repetitions'] = [sum(rep.values()) for rep in record['repetitions']]
                    rows.append(row)
    write_csv(out/'total_receiver_errors.csv', rows)
    expected_per_material = len(cfg['smooth_full_grids'])+len(cfg['chebyshev_full_grids'])+len(cfg['explicit_full_grids'])*len(cfg['explicit_cfl_values'])+2*sum(len(ms) for ms in cfg['smooth_reduced_modes'].values())
    completed = {m:sum(row['material']==m and 'total_seconds' in row for row in rows) for m in cfg['smooth_materials']}
    complete = all(count==expected_per_material for count in completed.values())
    if not complete and not args.allow_partial:
        raise RuntimeError(f'Incomplete timed candidate set: expected {expected_per_material} per material, found {completed}')
    selections = []
    for material in cfg['smooth_materials']:
        candidates = [row for row in rows if row['material']==material and 'total_seconds' in row]
        for threshold in cfg['smooth_receiver_tolerances']:
            passing = [row for row in candidates if row['total_receiver_error'] <= threshold]
            full = [row for row in passing if row['method'].startswith('full_')]
            rom = [row for row in passing if not row['method'].startswith('full_')]
            best_full = min(full, key=lambda row: row['total_seconds']) if full else None
            best_rom = min(rom, key=lambda row: row['total_seconds']) if rom else None
            best_any = min(passing, key=lambda row: row['total_seconds']) if passing else None
            def encode(row):
                if row is None:
                    return None
                result = {key: row[key] for key in ('grid','method','modes_per_patch','requested_cfl','dimension','total_receiver_error','alternative_reference_total_receiver_error','reference_grid_difference','total_seconds','minimum_total_seconds','maximum_total_seconds')}
                result['primary_margin'] = threshold-row['total_receiver_error']
                result['alternative_margin'] = threshold-row['alternative_reference_total_receiver_error']
                return result
            alternative_passing = [row for row in candidates if row['alternative_reference_total_receiver_error'] <= threshold]
            alternative_best = min(alternative_passing, key=lambda row: row['total_seconds']) if alternative_passing else None
            identify = lambda row: None if row is None else (row['method'],row['grid'],row['modes_per_patch'],row['requested_cfl'])
            selections.append({'material': material, 'receiver_tolerance': threshold,
                               'tested_candidates': len(candidates), 'passing_candidates': len(passing),
                               'least_cost_full': encode(best_full), 'least_cost_reduced': encode(best_rom),
                               'least_cost_any': encode(best_any),
                               'least_cost_any_alternative_reference': encode(alternative_best),
                               'selection_unchanged_by_reference': identify(best_any)==identify(alternative_best),
                               'candidates_changing_threshold_classification': sum((row['total_receiver_error']<=threshold)!=(row['alternative_reference_total_receiver_error']<=threshold) for row in candidates),
                               'speedup_over_least_cost_passing_full': None if best_full is None or best_rom is None else best_full['total_seconds']/best_rom['total_seconds'],
                               'fastest_full_over_slowest_reduced': None if best_full is None or best_rom is None else best_full['minimum_total_seconds']/best_rom['maximum_total_seconds'],
                               'slowest_full_over_fastest_reduced': None if best_full is None or best_rom is None else best_full['maximum_total_seconds']/best_rom['minimum_total_seconds'],
                               'timing_ranges_separated_in_favor_of_reduced': None if best_full is None or best_rom is None else best_rom['maximum_total_seconds']<best_full['minimum_total_seconds']})
    summary = {'reference_grid': cfg['reference_grid'], 'alternative_reference_grid': cfg['alternative_reference_grid'],
               'error_windows': cfg['error_windows'], 'timing_time_end': cfg['timing_time_end'],
               'norm': 'Frobenius norm of three-source, four-receiver traces at all output times, including t=0.',
               'reference_scope': 'Finite-grid comparison; neither the finest grid nor successive-grid differences certify continuum accuracy.',
               'timing_scope': 'Fresh receiver-output experiment; independent assembly and reduced basis for each three-source query. No field-output timing is included.',
               'complete': complete, 'expected_timed_candidates_per_material': expected_per_material, 'completed_timed_candidates': completed, 'number_error_rows': len(rows), 'material_rows': {m:sum(row['material']==m for row in rows) for m in cfg['smooth_materials']},
               'maximum_error_identity_residual': max((row['error_identity_absolute_residual'] for row in rows), default=0),
               'selections': selections}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if complete:
        from envelope import calculate
        calculate(cfg, rows)
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    main()
