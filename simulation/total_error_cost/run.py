#!/usr/bin/env python3
"""Receiver-output cost, including independent construction of each reduced model."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
STAGES = ['assembly', 'basis', 'projection_eigensolve', 'propagation', 'receiver_output']


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module('same_grid_cost', ROOT / 'accuracy_cost' / 'run.py')


def dependency_hashes():
    paths = [ROOT/'acoustic_core.py', ROOT/'accuracy_cost'/'run.py', ROOT/'accuracy_cost'/'config.json',
             ROOT/'propagation'/'chebyshev_reference.py']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def assemble(n, name, stiffness=True):
    if name == 'impedance_interface':
        interface = load_module('impedance_kernel_cost', ROOT / 'interface_case' / 'kernel.py')
        return interface.assembly(n, need_stiffness=stiffness)
    return BASE.assembly(n, name, need_stiffness=stiffness)


def full_receiver(n, name, times, chunk_duration):
    stages = {key: 0.0 for key in STAGES}
    start = time.perf_counter()
    mass, A, J, B, R, rho, bulk = assemble(n, name, stiffness=False)
    stages['assembly'] = time.perf_counter() - start
    start = time.perf_counter()
    state = np.zeros((J.shape[0], B.shape[1]))
    state[:n*n] = B
    Y = np.empty((len(times), R.shape[1], B.shape[1]))
    stages['propagation'] = time.perf_counter() - start
    start = time.perf_counter()
    Y[0] = R.T @ B
    stages['receiver_output'] = time.perf_counter() - start
    steps_per_chunk = max(1, int(round(chunk_duration / (times[1] - times[0]))))
    energy0 = np.sum(state * state, axis=0)
    max_energy_drift = 0.0
    for index in range(0, len(times)-1, steps_per_chunk):
        steps = min(steps_per_chunk, len(times)-1-index)
        start = time.perf_counter()
        block = expm_multiply(J, state, start=0.0, stop=steps*(times[1]-times[0]),
                              num=steps+1, traceA=0.0)
        state = block[-1].copy()
        stages['propagation'] += time.perf_counter() - start
        start = time.perf_counter()
        Y[index+1:index+steps+1] = np.einsum('ir,tis->trs', R, block[1:, :n*n], optimize=True)
        stages['receiver_output'] += time.perf_counter() - start
        # Diagnostic calculations do not contribute to output-production cost.
        max_energy_drift = max(max_energy_drift, float(np.max(np.abs(np.sum(state*state, axis=0)/energy0-1))))
    return {'time': times, 'receiver_pressure': Y}, stages, {'energy_relative_drift': max_energy_drift}



def chebyshev_receiver(n, name, times, tolerance):
    stages = {key: 0.0 for key in STAGES}
    started = time.perf_counter()
    mass, A, J, B, R, rho, bulk = assemble(n, name, stiffness=True)
    stages['assembly'] = time.perf_counter()-started
    chebyshev = load_module('chebyshev_cost', ROOT/'propagation'/'chebyshev_reference.py')
    started = time.perf_counter()
    Y, diagnostics = chebyshev.receiver_action(A, B, R, times, tail_tolerance=tolerance)
    stages['propagation'] = time.perf_counter()-started
    return {'time': times, 'receiver_pressure': Y}, stages, diagnostics


def explicit_receiver(n, name, cfl, times):
    stages = {key: 0.0 for key in STAGES}
    start = time.perf_counter()
    mass, A, J, B, R, rho, bulk = assemble(n, name, stiffness=True)
    speed = float(np.sqrt(bulk / rho).max())
    output_dt = float(times[1] - times[0])
    substeps = max(1, int(np.ceil(output_dt * speed * n / cfl)))
    dt = output_dt / substeps
    # Gershgorin gives a sufficient central-difference stability bound.
    maximum_row_sum = float(np.max(np.asarray(np.abs(A).sum(axis=1))))
    if dt * np.sqrt(maximum_row_sum) >= 2:
        raise RuntimeError('Explicit time step violates the sufficient spectral stability bound')
    stages['assembly'] = time.perf_counter() - start
    start = time.perf_counter()
    previous = B.copy()
    current = B - .5 * dt * dt * (A @ B)
    Y = np.empty((len(times), R.shape[1], B.shape[1]))
    stages['propagation'] = time.perf_counter() - start
    start = time.perf_counter()
    Y[0] = R.T @ B
    stages['receiver_output'] = time.perf_counter() - start
    step = 1
    for output_index in range(1, len(times)):
        target_step = output_index * substeps
        start = time.perf_counter()
        while step < target_step:
            following = 2 * current - previous - dt * dt * (A @ current)
            previous, current = current, following
            step += 1
        stages['propagation'] += time.perf_counter() - start
        start = time.perf_counter()
        Y[output_index] = R.T @ current
        stages['receiver_output'] += time.perf_counter() - start
    return {'time': times, 'receiver_pressure': Y}, stages, {
        'requested_cfl': cfl, 'actual_cfl': dt * speed * n,
        'time_step': dt, 'substeps_per_output': substeps, 'time_steps': step,
        'sufficient_stability_product': dt * np.sqrt(maximum_row_sum)}


def reduced_receiver(n, name, m, method, times, return_spatial=False):
    stages = {key: 0.0 for key in STAGES}
    start = time.perf_counter()
    mass, A, J, B, R, rho, bulk = assemble(n, name)
    stages['assembly'] = time.perf_counter() - start
    start = time.perf_counter()
    details = {}
    if method == 'craig_bampton':
        Q = BASE.cb_basis(n, mass, A, m)
    elif method == 'rational_krylov':
        Q, details = BASE.rational_basis(mass, A, B, n+2*m)
    else:
        raise ValueError(method)
    stages['basis'] = time.perf_counter() - start
    start = time.perf_counter()
    Ar = Q.T @ (A @ Q)
    Ar = (Ar + Ar.T) / 2
    lam, V = linalg.eigh(Ar, check_finite=False, driver='evd')
    minimum = float(lam[0])
    if minimum < -1e-9*max(lam[-1], 1.0):
        raise RuntimeError('Projected stiffness is indefinite')
    lam[lam < 1e-10] = 0.0
    coefficients = V.T @ (Q.T @ B)
    receiver_modes = (R.T @ Q) @ V
    stages['projection_eigensolve'] = time.perf_counter() - start
    start = time.perf_counter()
    modal = np.cos(times[:, None] * np.sqrt(lam)[None, :])[:, :, None] * coefficients[None, :, :]
    stages['propagation'] = time.perf_counter() - start
    start = time.perf_counter()
    Y = np.einsum('ir,trs->tis', receiver_modes, modal, optimize=True)
    stages['receiver_output'] = time.perf_counter() - start
    # The receiver representation reconstructs all saved traces without storing an N by r basis.
    raw = {'time': times, 'receiver_pressure': Y, 'reduced_eigenvalues': lam,
           'receiver_modal_matrix': receiver_modes, 'initial_modal_coefficients': coefficients}
    constant = np.sqrt(mass)
    constant /= np.linalg.norm(constant)
    details.update({'orthonormality_frobenius': float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))),
                    'constant_mode_projection_error': float(np.linalg.norm(constant - Q @ (Q.T @ constant))),
                    'initial_projection_relative_error': float(np.linalg.norm(B-Q@(Q.T@B))/np.linalg.norm(B)),
                    'minimum_eigenvalue_before_clipping': minimum})
    if return_spatial:
        raw['_orthonormal_basis'] = Q
        raw['_reduced_eigenvectors'] = V
        raw['_mass'] = mass
    return raw, stages, details


def summarize(repetitions):
    result = {key: float(np.median([row[key] for row in repetitions])) for key in STAGES}
    totals = [sum(row.values()) for row in repetitions]
    result.update(total_seconds=float(np.median(totals)),
                  minimum_total_seconds=float(min(totals)), maximum_total_seconds=float(max(totals)))
    return result


def execute(cfg, materials, cpus, dry_run=False):
    out = HERE / 'results' / 'smooth'
    out.mkdir(exist_ok=True, parents=True)
    used = None
    if hasattr(os, 'sched_getaffinity'):
        available = set(os.sched_getaffinity(0))
        chosen = set(cpus) & available
        if not chosen:
            chosen = set(sorted(available)[:cfg['threads']])
        os.sched_setaffinity(0, chosen)
        used = sorted(os.sched_getaffinity(0))
    config_digest = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependencies = dependency_hashes()
    times = np.linspace(0, cfg['timing_time_end'], cfg['samples'])
    cases = []
    for name in materials:
        for n in cfg['smooth_full_grids']:
            cases.append((name, n, None, 'full_sparse_exponential'))
        for n in cfg['chebyshev_full_grids']:
            cases.append((name, n, None, 'full_chebyshev_bessel'))
        for n in cfg['explicit_full_grids']:
            for cfl in cfg['explicit_cfl_values']:
                cases.append((name, n, cfl, 'full_stormer_verlet'))
        for n, ranks in cfg['smooth_reduced_modes'].items():
            for m in ranks:
                for method in ('craig_bampton', 'rational_krylov'):
                    cases.append((name, int(n), m, method))
    if dry_run:
        print(json.dumps({'settings': len(cases), 'repetitions': cfg['repetitions'], 'cases': cases}, indent=2))
        return
    for previous in out.glob('*_timing.json'):
        record = json.loads(previous.read_text())
        if record['config_sha256'] != config_digest or record['code_sha256'] != code_digest or record.get('dependency_sha256') != dependencies:
            raise RuntimeError('Existing timings use a different configuration or implementation')
        if record['cpu_affinity'] != used:
            raise RuntimeError('Existing timings use different CPU affinity')
    dump(out/'environment.json', {'platform': platform.platform(), 'python': platform.python_version(),
         'numpy': np.__version__, 'scipy': scipy.__version__, 'cpu_affinity': used,
         'cpu_information': Path('/proc/cpuinfo').read_text().split('\n\n')[0] if Path('/proc/cpuinfo').exists() else platform.processor(),
         'thread_environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
         'threads': cfg['threads'], 'config_sha256': config_digest, 'code_sha256': code_digest, 'dependency_sha256': dependencies,
         'shared_basis_code_sha256': hashlib.sha256((ROOT/'accuracy_cost'/'run.py').read_bytes()).hexdigest(),
         'kernel_sha256': hashlib.sha256((ROOT/'acoustic_core.py').read_bytes()).hexdigest()})
    full_receiver(8, 'rotated_stripes', np.linspace(0, .2, 17), .1)
    for method in ('craig_bampton', 'rational_krylov'):
        reduced_receiver(8, 'rotated_stripes', 4, method, np.linspace(0, .2, 17))
    explicit_receiver(8, 'rotated_stripes', .25, np.linspace(0, .2, 17))
    chebyshev_receiver(8, 'rotated_stripes', np.linspace(0,.2,17), cfg['chebyshev_tail_tolerance'])
    for name, n, m, method in cases:
        parameter_suffix = f'_cfl{str(m).replace(".", "p")}' if method == 'full_stormer_verlet' else (f'_m{m}' if m is not None else '')
        stem = f'{name}_n{n}_{method}' + parameter_suffix
        target = out/f'{stem}_timing.json'
        if target.exists():
            continue
        interval_start = time.time()
        repetitions = []
        for _ in range(cfg['repetitions']):
            if method == 'full_chebyshev_bessel':
                raw, stages, diagnostics = chebyshev_receiver(n, name, times, cfg['chebyshev_tail_tolerance'])
            elif method == 'full_stormer_verlet':
                raw, stages, diagnostics = explicit_receiver(n, name, m, times)
            elif m is None:
                raw, stages, diagnostics = full_receiver(n, name, times, cfg['chunk_duration'])
            else:
                raw, stages, diagnostics = reduced_receiver(n, name, m, method, times)
            repetitions.append(stages)
        interval_end = time.time()
        np.savez_compressed(out/f'{stem}.npz', **raw)
        record = {'material': name, 'grid': n, 'method': method, 'modes_per_patch': m if not method.startswith('full_') else None,
                  'requested_cfl': m if method == 'full_stormer_verlet' else None,
                  'dimension': n*n if method.startswith('full_') else n+2*m, 'repetitions': repetitions,
                  'median': summarize(repetitions), 'diagnostics': diagnostics,
                  'cpu_affinity': used, 'config_sha256': config_digest, 'code_sha256': code_digest, 'dependency_sha256': dependencies,
                  'timed_interval_unix': [interval_start, interval_end]}
        dump(target, record)
        print(json.dumps({'case': stem, **record['median']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', nargs='+')
    parser.add_argument('--cpus', type=int, nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    execute(cfg, args.materials or cfg['smooth_materials'], args.cpus or cfg['cpu_affinity'], args.dry_run)


if __name__ == '__main__':
    main()
