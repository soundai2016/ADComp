#!/usr/bin/env python3
"""Recompute modal, segmentation, sampling, and polynomial reference checks."""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from run_propagation import CONFIG, MATERIALS, build_sparse, ports, propagate, relative
from chebyshev_reference import propagate as polynomial_propagate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODAL_CASES = [(name, n) for n in (32, 48) for name in MATERIALS] + [('rotated_stripes', 64)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_case(name, n):
    path = RESULTS / f'{name}_n{n}.npz'
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def modal_check(name, n):
    reference = load_case(name, n)
    started = time.perf_counter()
    x, y, _, bulk, _, coupling, _ = build_sparse(n, name)
    sources = ports(x, y, bulk, CONFIG['source_centres'], True)
    receivers = ports(x, y, bulk, CONFIG['receiver_centres'], False)
    values, vectors = eigh((coupling @ coupling.T).toarray())
    values = np.maximum(values, 0.)
    modal = np.einsum('rm,tm,ms->trs', receivers.T @ vectors,
        np.cos(reference['time'][:, None] * np.sqrt(values)[None, :]),
        vectors.T @ sources, optimize=True)
    difference = relative(reference['receiver_pressure'], modal)
    result = dict(material=name, grid=n, samples=len(reference['time']),
                  modal_relative_difference=difference, seconds=time.perf_counter() - started,
                  passed=difference < 1e-10)
    print(json.dumps({'check': 'dense_modal', **result}), flush=True)
    return result


def clock_checks():
    n, name = 64, 'rotated_stripes'
    baseline = load_case(name, n)
    if int(baseline['samples']) != 641 or float(baseline['chunk_duration']) != 1.:
        baseline = propagate(n, name, samples=641, chunk_duration=1.)
    if not np.array_equal(baseline['time'], np.linspace(0., 8., 641)):
        raise ValueError('The segmentation comparison requires 641 samples on [0, 8].')

    fine = propagate(n, name, samples=1281, chunk_duration=.5)
    np.savez_compressed(RESULTS / 'clock_check_rotated_stripes_n64.npz', **fine)
    if not np.array_equal(fine['time'][::2], baseline['time']):
        raise ValueError('Doubled-sampling output times do not contain the baseline times.')
    common = relative(fine['receiver_pressure'][::2], baseline['receiver_pressure'])
    power = np.sum(fine['receiver_pressure'] ** 2, axis=(1, 2))
    integral_fine = float(np.trapezoid(power, fine['time']))
    integral_coarse = float(np.trapezoid(power[::2], fine['time'][::2]))
    sampling = dict(grid=n, material=name, fine_samples=1281, coarse_samples=641,
        chunk_durations=[1., .5], relative_trace_difference=common,
        max_absolute_trace_difference=float(np.max(abs(fine['receiver_pressure'][::2] - baseline['receiver_pressure']))),
        energy_relative_drift=float(fine['energy_relative_drift']),
        waveform_norm_relative_change=abs(np.sqrt(integral_fine / integral_coarse) - 1),
        quantity='Trapezoidal norm of one waveform, not a cross-grid error numerator.',
        passed=common < 1e-10 and np.isfinite(integral_fine) and integral_coarse > 0)
    sampling['passed'] = bool(sampling['passed'])
    write_json(RESULTS / 'clock_check.json', sampling)
    print(json.dumps({'check': 'output_sampling', **sampling}), flush=True)

    alternate = propagate(n, name, samples=641, chunk_duration=.6625)
    np.savez_compressed(RESULTS / 'clock_check_alternate_chunk_rotated_stripes_n64.npz', **alternate)
    difference = relative(alternate['receiver_pressure'], baseline['receiver_pressure'])
    segmentation = dict(grid=n, material=name, samples=641,
        chunk_durations=[1., .6625], chunk_steps=[80, 53],
        relative_trace_difference=difference,
        max_absolute_trace_difference=float(np.max(abs(alternate['receiver_pressure'] - baseline['receiver_pressure']))),
        energy_relative_drift=float(alternate['energy_relative_drift']), passed=difference < 1e-10)
    write_json(RESULTS / 'clock_check_alternate_chunk.json', segmentation)
    print(json.dumps({'check': 'segmentation', **segmentation}), flush=True)
    return sampling, segmentation


def polynomial_check(n):
    name = 'rotated_stripes'
    reference = load_case(name, n)
    answer = polynomial_propagate(n, name)
    if not np.array_equal(answer['time'], reference['time']):
        raise ValueError('Chebyshev and sparse-exponential observation times differ.')
    record = {key: float(answer[key]) for key in ('energy_relative_drift', 'wall_seconds',
        'uniform_cosine_tail_bound', 'squared_frequency_upper_bound')}
    record.update(material=name, grid=n, polynomial_degree=int(answer['polynomial_degree']),
                  energy_check_times=answer['energy_check_times'].tolist())
    for key in ('receiver_pressure', 'pressure_snapshots'):
        record[key + '_relative_difference'] = relative(answer[key], reference[key])
    record['passed'] = max(record[key + '_relative_difference']
                           for key in ('receiver_pressure', 'pressure_snapshots')) < 2e-10
    print(json.dumps({'check': 'chebyshev', **record}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Check only the prescribed striped 32 x 32 modal and polynomial case.')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = [('rotated_stripes', 32)] if args.quick else MODAL_CASES
    grids = [32] if args.quick else [32, 64, 256]
    required = {f'{name}_n{n}.npz' for name, n in cases}
    required.update(f'rotated_stripes_n{n}.npz' for n in grids)
    if not args.quick:
        required.add('rotated_stripes_n64.npz')
    missing = [name for name in sorted(required) if not (RESULTS / name).is_file()]
    if missing:
        raise FileNotFoundError('Run the propagation grid ladder first: ' + ', '.join(missing))
    modal = [modal_check(name, n) for name, n in cases]
    sampling, segmentation = (None, None) if args.quick else clock_checks()
    polynomial = [polynomial_check(n) for n in grids]
    suffix = '_quick' if args.quick else ''
    write_json(RESULTS / f'modal_comparison{suffix}.json', modal)
    polynomial_name = 'chebyshev_verification_quick.json' if args.quick else 'chebyshev_verification_n32_64_256.json'
    write_json(RESULTS / polynomial_name, polynomial)
    checks = modal + polynomial + ([sampling, segmentation] if sampling is not None else [])
    summary = dict(mode='quick' if args.quick else 'full', passed=all(row['passed'] for row in checks),
        modal_cases=len(modal), polynomial_cases=len(polynomial),
        maximum_modal_relative_difference=max(row['modal_relative_difference'] for row in modal),
        maximum_polynomial_relative_difference=max(row[key + '_relative_difference']
            for row in polynomial for key in ('receiver_pressure', 'pressure_snapshots')),
        output_sampling=sampling, segmentation=segmentation,
        input_sha256={name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in sorted(required)})
    write_json(RESULTS / f'reference_checks{suffix}.json', summary)
    if not summary['passed']:
        raise SystemExit('A propagation reference check failed.')


if __name__ == '__main__':
    main()
