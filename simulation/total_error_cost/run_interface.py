#!/usr/bin/env python3
"""Cost and continuum error in the interface-aligned impedance benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import run
import numpy as np

HERE = Path(__file__).resolve().parent
KERNEL = run.load_module('impedance_kernel_cost', HERE.parent/'interface_case'/'kernel.py')


def field_metrics(n, times, pressure_blocks, mass):
    """Compute continuum L2 errors of the piecewise-constant physical pressure."""
    sums = {end: {'difference': np.zeros(3), 'continuum_norm': np.zeros(3),
                  'cell_average_norm': np.zeros(3), 'projection_loss': np.zeros(3)} for end in (2.,8.)}
    minimum_projection_loss = float("inf")
    for indices, physical in pressure_blocks:
        tt = times[indices]
        exact = KERNEL.exact_mass_weighted_pressure(n, tt)
        discrete = physical * np.sqrt(mass)[None, :, None]
        difference = np.sum((discrete-exact)**2, axis=1)
        cell_average_norm = np.sum(exact**2, axis=1)
        continuum_norm = KERNEL.exact_pressure_norm_squared(tt)
        projection_loss = continuum_norm-cell_average_norm
        minimum_projection_loss = min(minimum_projection_loss, float(projection_loss.min()))
        if minimum_projection_loss < -1e-10:
            raise RuntimeError('Cell averaging increased the exact L2 norm')
        projection_loss = np.maximum(projection_loss, 0)
        for end, accumulator in sums.items():
            keep = tt <= end+1e-12
            for key, values in [('difference',difference),('continuum_norm',continuum_norm),
                                ('cell_average_norm',cell_average_norm),('projection_loss',projection_loss)]:
                accumulator[key] += values[keep].sum(axis=0)
    rows = []
    for end, value in sums.items():
        numerator = value['difference']+value['projection_loss']
        rows.append({'time_end': end,
            'continuous_field_error': float(np.sqrt(numerator.sum()/value['continuum_norm'].sum())),
            'cell_average_field_error': float(np.sqrt(value['difference'].sum()/value['cell_average_norm'].sum())),
            'projection_floor': float(np.sqrt(value['projection_loss'].sum()/value['continuum_norm'].sum())),
            'continuous_field_error_per_source': np.sqrt(numerator/value['continuum_norm']).tolist()})
    return rows, minimum_projection_loss


def full_run(n, times, chunk, evaluate_errors, keep_model=False):
    stages = {key: 0.0 for key in run.STAGES+['field_reconstruction']}
    started = time.perf_counter()
    model = KERNEL.separable_setup(n)
    stages['assembly'] = time.perf_counter()-started
    started = time.perf_counter()
    y = KERNEL.separable_receiver(model, times)
    stages['receiver_output'] = time.perf_counter()-started
    mass = model['mass']
    def blocks():
        for index in range(0, len(times), chunk):
            ids = np.arange(index, min(index+chunk, len(times)))
            started = time.perf_counter()
            physical = KERNEL.separable_pressure(model, times[ids])/np.sqrt(mass)[None,:,None]
            stages['field_reconstruction'] += time.perf_counter()-started
            yield ids, physical
    if evaluate_errors:
        errors, minimum_projection_loss = field_metrics(n, times, blocks(), mass)
    else:
        for ids, physical in blocks():
            pass
        errors, minimum_projection_loss = [], 0.0
    raw = {'time': times, 'receiver_pressure': y}
    if keep_model:
        raw['_separable_model'] = model
    return raw, stages, errors, {'minimum_projection_loss': minimum_projection_loss}


def reduced_run(n, m, method, times, chunk, evaluate_errors, keep_model=False):
    raw, stages, diagnostics = run.reduced_receiver(n, 'impedance_interface', m, method, times, return_spatial=True)
    Q = raw.pop('_orthonormal_basis')
    V = raw.pop('_reduced_eigenvectors')
    mass = raw.pop('_mass')
    started = time.perf_counter()
    spatial_modes = (Q@V)/np.sqrt(mass[:,None])
    stages['field_reconstruction'] = time.perf_counter()-started
    def blocks():
        for index in range(0, len(times), chunk):
            ids = np.arange(index, min(index+chunk, len(times)))
            started = time.perf_counter()
            modal = np.cos(times[ids,None]*np.sqrt(raw['reduced_eigenvalues'])[None,:])[:,:,None]*raw['initial_modal_coefficients'][None,:,:]
            physical = np.einsum('ir,trs->tis', spatial_modes, modal, optimize=True)
            stages['field_reconstruction'] += time.perf_counter()-started
            yield ids, physical
    if evaluate_errors:
        errors, minimum_projection_loss = field_metrics(n, times, blocks(), mass)
    else:
        for ids, physical in blocks():
            pass
        errors, minimum_projection_loss = [], 0.0
    diagnostics['minimum_projection_loss'] = minimum_projection_loss
    if keep_model:
        raw['_physical_spatial_modes'] = spatial_modes
        raw['_mass'] = mass
    return raw, stages, errors, diagnostics


def evaluate_fields(n, times, raw, chunk):
    model = raw.pop('_separable_model', None)
    spatial = raw.pop('_physical_spatial_modes', None)
    mass = model['mass'] if model is not None else raw.pop('_mass')
    def blocks():
        for index in range(0,len(times),chunk):
            ids = np.arange(index,min(index+chunk,len(times)))
            if model is not None:
                pressure = KERNEL.separable_pressure(model,times[ids])/np.sqrt(mass)[None,:,None]
            else:
                modal = np.cos(times[ids,None]*np.sqrt(raw['reduced_eigenvalues'])[None,:])[:,:,None]*raw['initial_modal_coefficients'][None,:,:]
                pressure = np.einsum('ir,trs->tis',spatial,modal,optimize=True)
            yield ids,pressure
    return field_metrics(n,times,blocks(),mass)


def summary(repetitions):
    result = {key: float(np.median([row[key] for row in repetitions])) for key in run.STAGES+['field_reconstruction']}
    for name, keys in [('receiver_total_seconds', run.STAGES), ('both_total_seconds', run.STAGES+['field_reconstruction'])]:
        totals = [sum(row[key] for key in keys) for row in repetitions]
        result[name] = float(np.median(totals))
        result[name.replace('_seconds','_minimum_seconds')] = float(min(totals))
        result[name.replace('_seconds','_maximum_seconds')] = float(max(totals))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', type=int, nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    requested = args.cpus or cfg['cpu_affinity']
    if hasattr(os, 'sched_getaffinity'):
        available = os.sched_getaffinity(0)
        chosen = set(requested)&available or set(sorted(available)[:cfg['threads']])
        os.sched_setaffinity(0, chosen)
        affinity = sorted(os.sched_getaffinity(0))
    else:
        affinity = None
    cases = [(n,None,'full_separable') for n in cfg['interface_full_grids']]
    cases += [(n,m,method) for n in cfg['interface_reduced_grids'] for m in cfg['interface_modes_per_patch']
              for method in ('craig_bampton','rational_krylov')]
    if args.dry_run:
        print(json.dumps({'settings':len(cases),'cases':cases},indent=2))
        return
    out = HERE/'results'/'interface'
    out.mkdir(exist_ok=True, parents=True)
    digest = hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run.py',Path(__file__),HERE.parent/'interface_case'/'kernel.py']}
    dependencies = run.dependency_hashes()
    dependencies['interface_case/config.json'] = hashlib.sha256((HERE.parent/'interface_case'/'config.json').read_bytes()).hexdigest()
    for path in out.glob('*_timing.json'):
        record = json.loads(path.read_text())
        if record['config_sha256'] != digest or record['code_sha256'] != code or record.get('dependency_sha256') != dependencies or record['cpu_affinity'] != affinity:
            raise RuntimeError('Existing measurements use a different implementation or configuration')
    times = np.linspace(0,cfg['timing_time_end'],cfg['samples'])
    # All methods use the same prescribed initial modal coefficients.
    KERNEL.analytic_data()
    exact_receiver = KERNEL.exact_receiver(times)
    warm_times = np.linspace(0,.2,17)
    full_run(8,warm_times,8,False)
    for method in ('craig_bampton','rational_krylov'):
        reduced_run(8,4,method,warm_times,8,False)
    for n,m,method in cases:
        stem = f'impedance_interface_n{n}_{method}'+(f'_m{m}' if m is not None else '')
        target = out/f'{stem}_timing.json'
        if target.exists():
            continue
        repetitions=[]
        interval_start=time.time()
        for repeat in range(cfg['repetitions']):
            keep_model = repeat == cfg['repetitions']-1
            if m is None:
                raw,stages,errors,diagnostics=full_run(n,times,cfg['field_chunk_samples'],False,keep_model=keep_model)
            else:
                raw,stages,errors,diagnostics=reduced_run(n,m,method,times,cfg['field_chunk_samples'],False,keep_model=keep_model)
            repetitions.append(stages)
        interval_end=time.time()
        # Evaluate reference errors outside the timed repetitions.
        errors, minimum_projection_loss = evaluate_fields(n,times,raw,cfg['field_chunk_samples'])
        diagnostics['minimum_projection_loss'] = minimum_projection_loss
        for row in errors:
            keep=times<=row['time_end']+1e-12
            y,ref=raw['receiver_pressure'][keep],exact_receiver[keep]
            row['receiver_error']=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
            row['receiver_error_per_source']=[float(np.linalg.norm(y[:,:,s]-ref[:,:,s])/np.linalg.norm(ref[:,:,s])) for s in range(3)]
        np.savez_compressed(out/f'{stem}.npz',**raw)
        record={'material':'impedance_interface','grid':n,'modes_per_patch':m,'method':method,
                'dimension':n*n if m is None else n+2*m,'repetitions':repetitions,'median':summary(repetitions),
                'errors':errors,'diagnostics':diagnostics,'config_sha256':digest,'code_sha256':code,'dependency_sha256':dependencies,
                'cpu_affinity':affinity,'timed_interval_unix':[interval_start,interval_end]}
        run.dump(target,record)
        print(json.dumps({'case':stem,'cost':record['median'],'errors':errors}),flush=True)


if __name__=='__main__':
    main()
