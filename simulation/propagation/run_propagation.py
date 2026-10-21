#!/usr/bin/env python3
"""Finite-grid acoustic propagation with full-order matrix functions."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from acoustic_core import material, build_sparse, ports, CONFIG
EXPERIMENT = json.loads((ROOT / 'experiment_config.json').read_text())
MATERIALS = EXPERIMENT['materials']


def propagate(n, name, samples=641, chunk_duration=.5, progress=False):
    started = time.perf_counter()
    X,Y,rho,bulk,mass,C,J = build_sparse(n,name)
    B = ports(X,Y,bulk,CONFIG['source_centres'],True)
    R = ports(X,Y,bulk,CONFIG['receiver_centres'],False)
    state = np.zeros((J.shape[0],len(CONFIG['source_centres'])))
    state[:n*n] = B
    energy0 = np.sum(state*state,axis=0)
    times = np.linspace(0,CONFIG['time_end'],samples)
    dt = times[1]-times[0]
    chunk_steps = max(1,int(round(chunk_duration/dt)))
    traces = np.empty((samples,len(CONFIG['receiver_centres']),len(CONFIG['source_centres'])))
    traces[0] = R.T@B
    snap_ids = np.linspace(0,samples-1,5,dtype=int)
    snapshots = {0: (state[:n*n]/np.sqrt(mass[:,None])).copy()}
    max_drift = 0.
    for start in range(0,samples-1,chunk_steps):
        steps = min(chunk_steps,samples-1-start)
        block = expm_multiply(J,state,start=0.,stop=steps*dt,num=steps+1,traceA=0.)
        traces[start+1:start+steps+1] = np.einsum('ir,tis->trs',R,block[1:,:n*n],optimize=True)
        energies = np.sum(block*block,axis=1)
        max_drift = max(max_drift,float(np.max(np.abs(energies/energy0[None,:]-1))))
        for index in snap_ids:
            if start < index <= start+steps:
                snapshots[int(index)] = (block[index-start,:n*n]/np.sqrt(mass[:,None])).copy()
        state = block[-1].copy()
        if progress:
            print(json.dumps({'state':'progress','material':name,'grid':n,
                'time':float(times[start+steps]),'wall_seconds':time.perf_counter()-started,
                'energy_relative_drift':max_drift}),flush=True)
    return {'time':times,'receiver_pressure':traces,
        'snapshot_times':times[snap_ids],
        'pressure_snapshots':np.stack([snapshots[int(i)] for i in snap_ids]),
        'rho':rho,'bulk_modulus':bulk,
        'energy_relative_drift':np.array(max_drift),
        'wall_seconds':np.array(time.perf_counter()-started),
        'grid':np.array(n),'samples':np.array(samples),
        'chunk_duration':np.array(chunk_duration)}


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify():
    from scipy.linalg import eigh
    n,name=8,'rotated_stripes'
    answer=propagate(n,name,samples=81,chunk_duration=.5)
    X,Y,rho,bulk,mass,C,J=build_sparse(n,name)
    B=ports(X,Y,bulk,CONFIG['source_centres'],True)
    R=ports(X,Y,bulk,CONFIG['receiver_centres'],False)
    lam,V=eigh((C@C.T).toarray())
    lam=np.maximum(lam,0.)
    modal=np.einsum('rm,tm,ms->trs',R.T@V,
        np.cos(answer['time'][:,None]*np.sqrt(lam)[None,:]),V.T@B,optimize=True)
    residual=relative(answer['receiver_pressure'],modal)
    skew=J+J.T
    skew_max=float(np.max(np.abs(skew.data),initial=0.))
    result={'status':'pass' if residual<1e-10 and skew_max<1e-13 else 'fail',
        'grid':n,'modal_relative_difference':residual,'generator_skew_defect':skew_max,
        'energy_relative_drift':float(answer['energy_relative_drift']),
        'chunk_duration':.5}
    if (ROOT/'results'/'reduced_responses.npz').exists():
        with np.load(ROOT/'results'/'reduced_responses.npz') as responses:
            matches={}
            for medium in MATERIALS:
                with np.load(ROOT/'results'/f'{medium}_n32.npz') as full:
                    matches[medium]=relative(full['receiver_pressure'],responses[f'{medium}_full32'])
        result['full_grid32_modal_relative_difference']=matches
        if max(matches.values())>=1e-10: result['status']='fail'
    dump(ROOT/'results'/'small_grid_verification.json',result)
    print(json.dumps(result),flush=True)
    if result['status']!='pass': raise SystemExit(1)


def clock_check():
    n,name=64,'rotated_stripes'
    answer=propagate(n,name,samples=641,chunk_duration=.5)
    np.savez_compressed(ROOT/'results'/'chunk_check_n64.npz',**answer)
    with np.load(ROOT/'results'/f'{name}_n{n}.npz') as original:
        result={'grid':n,'chunk_durations':[1.,.5],'samples':641,
            'relative_trace_difference':relative(answer['receiver_pressure'],original['receiver_pressure']),
            'max_absolute_trace_difference':float(np.max(np.abs(answer['receiver_pressure']-original['receiver_pressure']))),
            'energy_relative_drift':float(answer['energy_relative_drift'])}
    dump(ROOT/'results'/'chunk_check.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--grids',nargs='+',type=int,default=EXPERIMENT['propagation_grids'])
    parser.add_argument('--materials',nargs='+',choices=MATERIALS,default=MATERIALS)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--worker',action='store_true',help='Write case arrays only; a serial pass consolidates execution metadata.')
    parser.add_argument('--clock-check',action='store_true')
    parser.add_argument('--chunk-duration',type=float,default=None)
    parser.add_argument('--cpu-affinity',nargs='+',type=int,default=None)
    parser.add_argument('--progress',action='store_true')
    args=parser.parse_args()
    if args.cpu_affinity is not None:
        if not hasattr(os,'sched_setaffinity'):
            raise RuntimeError('Explicit CPU affinity is unavailable on this platform.')
        os.sched_setaffinity(0,set(args.cpu_affinity))
    (ROOT/'results').mkdir(exist_ok=True)
    if args.verify: verify(); return
    if args.clock_check: clock_check(); return
    if not args.worker:
        dump(ROOT/'results'/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'numpy':np.__version__,'scipy':scipy.__version__,'threads':2,'precision':'float64'})
    for n in args.grids:
        for name in args.materials:
            target=ROOT/'results'/f'{name}_n{n}.npz'
            if not target.exists():
                print(json.dumps({'state':'running','material':name,'grid':n}),flush=True)
                if n>=512 or (n==384 and name!='rotated_stripes'):
                    from chebyshev_reference import propagate as chebyshev_propagate
                    answer=chebyshev_propagate(n,name,
                        tail_tolerance=EXPERIMENT.get('chebyshev_tail_tolerance',1e-17),
                        samples=EXPERIMENT['samples'],progress=args.progress)
                else:
                    chunk=args.chunk_duration
                    if chunk is None:
                        chunk=EXPERIMENT.get('extended_grid_chunk_duration',EXPERIMENT['chunk_duration']) if n>=384 else EXPERIMENT['chunk_duration']
                    answer=propagate(n,name,samples=EXPERIMENT['samples'],chunk_duration=chunk,progress=args.progress)
                np.savez_compressed(target,**answer)
            with np.load(target) as data:
                row={'material':name,'grid':n,'wall_seconds':float(data['wall_seconds']),
                    'energy_relative_drift':float(data['energy_relative_drift']),
                    'chunk_duration':float(data['chunk_duration']),
                    'propagator':str(data['propagator']) if 'propagator' in data else 'Sparse first-order exponential',
                    'energy_check_count':len(data['energy_check_times']) if 'energy_check_times' in data else len(data['time'])}
                print(json.dumps(row),flush=True)
    if args.worker: return
    rows=[]
    for n in EXPERIMENT['propagation_grids']:
        for name in MATERIALS:
            target=ROOT/'results'/f'{name}_n{n}.npz'
            if target.exists():
                with np.load(target) as data:
                    rows.append({'material':name,'grid':n,'samples':int(data['samples']),
                        'wall_seconds':float(data['wall_seconds']),
                        'energy_relative_drift':float(data['energy_relative_drift']),
                        'chunk_duration':float(data['chunk_duration']),
                        'propagator':str(data['propagator']) if 'propagator' in data else 'Sparse first-order exponential',
                        'energy_check_count':len(data['energy_check_times']) if 'energy_check_times' in data else len(data['time'])})
    dump(ROOT/'results'/'execution.json',{'completed':len(rows),'requested':len(EXPERIMENT['propagation_grids'])*len(MATERIALS),
        'case_records':rows,'all_requested_complete':len(rows)==len(EXPERIMENT['propagation_grids'])*len(MATERIALS)})


if __name__=='__main__': main()
