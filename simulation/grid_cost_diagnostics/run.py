#!/usr/bin/env python3
"""Uniform-grid cost diagnostic for the common receiver solvers."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('receiver_cost_solver', ROOT/'total_error_cost'/'run.py')
SOLVER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = SOLVER
spec.loader.exec_module(SOLVER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes():
    paths = [HERE/'config.json', HERE/'run.py', ROOT/'total_error_cost'/'run.py',
             ROOT/'total_error_cost'/'config.json', ROOT/'accuracy_cost'/'run.py',
             ROOT/'accuracy_cost'/'config.json', ROOT/'acoustic_core.py',
             ROOT/'propagation'/'chebyshev_reference.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def candidates(cfg):
    for material in cfg['materials']:
        for n in cfg['grids']:
            for cfl in cfg['explicit_cfl_values']:
                yield {'material': material, 'grid': n, 'method': 'full_stormer_verlet',
                       'requested_cfl': cfl, 'modes_per_patch': None, 'dimension': n*n}
            yield {'material': material, 'grid': n, 'method': 'craig_bampton',
                   'requested_cfl': None, 'modes_per_patch': cfg['modes_per_patch'],
                   'dimension': n+2*cfg['modes_per_patch']}


def stem(candidate):
    suffix = ('_cfl'+str(candidate['requested_cfl']).replace('.', 'p')
              if candidate['method']=='full_stormer_verlet' else '_m'+str(candidate['modes_per_patch']))
    return f"{candidate['material']}_n{candidate['grid']}_{candidate['method']}"+suffix


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    cases = list(candidates(cfg))
    dependencies = hashes()
    if args.dry_run:
        print(json.dumps({'settings':len(cases), 'repetitions':cfg['repetitions'],
                          'dependency_sha256':dependencies, 'candidates':cases}, indent=2))
        return
    requested = set(cfg['cpu_affinity'])
    if not requested.issubset(os.sched_getaffinity(0)):
        raise RuntimeError('Requested CPUs unavailable; do not substitute timing hardware')
    os.sched_setaffinity(0, requested)
    out = HERE/'results'/'candidates'
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob('*_timing.json'):
        previous = json.loads(path.read_text())
        if previous['dependency_sha256'] != dependencies or previous['cpu_affinity'] != cfg['cpu_affinity']:
            raise RuntimeError('Stored timings have different dependencies or affinity')
    started = time.time()
    environment = {'platform':platform.platform(), 'python':platform.python_version(),
        'numpy':np.__version__, 'scipy':scipy.__version__, 'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'threads':cfg['threads'], 'dependency_sha256':dependencies, 'started_unix':started,
        'started_utc':utc(started), 'settings':len(cases), 'repetitions':cfg['repetitions'],
        'cpu_information':Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'thread_environment':{key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    times = np.linspace(0, cfg['time_end'], cfg['samples'])
    with threadpool_limits(limits=cfg['threads']):
        environment['threadpools'] = threadpool_info()
        dump(HERE/'results'/'environment.json', environment)
        warm_times = np.linspace(0, .2, 17)
        SOLVER.explicit_receiver(8, 'rotated_stripes', .25, warm_times)
        SOLVER.explicit_receiver(8, 'rotated_stripes', .125, warm_times)
        SOLVER.reduced_receiver(8, 'rotated_stripes', 4, 'craig_bampton', warm_times)
        for index, candidate in enumerate(cases):
            name = stem(candidate)
            target = out/f'{name}_timing.json'
            if target.exists():
                continue
            repetitions, intervals, diagnostics, traces = [], [], [], []
            case_start = time.time()
            for repetition in range(cfg['repetitions']):
                rep_start = time.time()
                if candidate['method']=='full_stormer_verlet':
                    raw, stages, diagnostic = SOLVER.explicit_receiver(candidate['grid'], candidate['material'], candidate['requested_cfl'], times)
                else:
                    raw, stages, diagnostic = SOLVER.reduced_receiver(candidate['grid'], candidate['material'], candidate['modes_per_patch'], candidate['method'], times)
                rep_end = time.time()
                repetitions.append(stages)
                intervals.append({'start_unix':rep_start, 'end_unix':rep_end,
                                  'start_utc':utc(rep_start), 'end_utc':utc(rep_end)})
                diagnostics.append(diagnostic)
                traces.append(raw['receiver_pressure'].copy())
            case_end = time.time()
            raw['receiver_pressure_repetitions'] = np.stack(traces)
            temporary = out/f'{name}.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **raw)
            raw_path = out/f'{name}.npz'
            temporary.replace(raw_path)
            record = {**candidate, 'repetitions':repetitions, 'repetition_intervals':intervals,
                'median':SOLVER.summarize(repetitions), 'diagnostics':diagnostics,
                'cpu_affinity':cfg['cpu_affinity'], 'dependency_sha256':dependencies,
                'timed_interval_unix':[case_start,case_end],
                'maximum_repeated_trace_difference':max(float(np.linalg.norm(y-traces[-1])/np.linalg.norm(traces[-1])) for y in traces),
                'raw_sha256':digest(raw_path), 'receiver_shape':list(traces[-1].shape),
                'full_pressure_dimension':candidate['grid']**2}
            dump(target, record)
            print(json.dumps({'completed':index+1, 'settings':len(cases), 'candidate':name,
                              'median_seconds':record['median']['total_seconds']}), flush=True)
    if hashes()!=dependencies:
        raise RuntimeError('A runtime dependency changed during timing')
    environment.update(finished_unix=time.time(), finished_utc=utc(time.time()),
                       completed_settings=len(list(out.glob('*_timing.json'))))
    dump(HERE/'results'/'environment.json', environment)


if __name__=='__main__':
    main()
