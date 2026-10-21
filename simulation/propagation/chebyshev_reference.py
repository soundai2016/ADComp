#!/usr/bin/env python3
"""Full-order propagation by a Chebyshev--Bessel expansion.

For A=C C^T, choose L>=lambda_max(A), X=2A/L-I and z=t*sqrt(L).
cos(t*sqrt(A)) = J_0(z) I + 2 sum_{k>=1} (-1)^k J_{2k}(z) T_k(X).
The polynomial recursion is shared across output times; the spatial
operator retains its full dimension.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import jv

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from acoustic_core import build_sparse,ports,CONFIG


def bessel_even_tail_bound(degree,zmax):
    """Uniform sum bound 2*sum_{k>degree}|J_{2k}(z)|, |z|<=zmax.

    For n>z, the generating-function Cauchy estimate is
    |J_n(z)| <= exp(sqrt(n*n-z*z)-n*acosh(n/z)).  Its log derivative
    with respect to n is -acosh(n/z), which decreases with n.
    Consecutive even orders are therefore bounded by a geometric series.
    """
    if zmax==0:return 0.
    order=2*(degree+1)
    if order<=zmax:return float('inf')
    alpha=np.arccosh(order/zmax)
    exponent=np.sqrt(order*order-zmax*zmax)-order*alpha
    return float(2*np.exp(exponent)/(-np.expm1(-2*alpha)))


def sinc_coefficient_tail_bound(degree,zmax,omega):
    """Error in all retained integrated coefficients from the finite odd sum."""
    first_order=2*(degree+100)+1
    alpha=np.arccosh(first_order/zmax)
    exponent=np.sqrt(first_order*first_order-zmax*zmax)-first_order*alpha
    odd_sum_tail=np.exp(exponent)/(-np.expm1(-2*alpha))
    return float((2+4*degree)*odd_sum_tail/omega)


def receiver_action(A,B,R,times,tail_tolerance=1e-17):
    """Full-order cosine action observed by R; no field snapshots are formed.

    Inputs are the symmetric PSD pressure matrix A, initial mass-weighted
    pressure columns B, receiver columns R and nonnegative output times.
    Assembly of A/B/R is outside this call and must be charged separately.
    The returned stage seconds cover every operation inside this function.
    """
    started=time.perf_counter()
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or np.min(times)<0:
        raise ValueError('Nonnegative one-dimensional output times required.')
    upper=float(np.asarray(abs(A).sum(axis=1)).max())*(1.+1e-14)
    omega=np.sqrt(upper)
    zmax=float(np.max(times))*omega
    degree=int(zmax/2)+1
    while bessel_even_tail_bound(degree,zmax)>tail_tolerance:degree+=1
    q0=B.copy();q1=(2/upper)*(A@B)-B
    observations=np.empty((degree+1,R.shape[1],B.shape[1]))
    observations[0]=R.T@q0
    setup_finished=time.perf_counter()
    for k in range(1,degree+1):
        observations[k]=R.T@q1
        if k<degree:q0,q1=q1,2*((2/upper)*(A@q1)-q1)-q0
    recurrence_finished=time.perf_counter()
    coeff=jv(2*np.arange(degree+1)[:,None],omega*times[None,:])
    coeff[1:]*=(2*(-1.)**np.arange(1,degree+1))[:,None]
    traces=np.einsum('kt,krs->trs',coeff,observations,optimize=True)
    finished=time.perf_counter()
    return traces,{'polynomial_degree':degree,'squared_frequency_upper_bound':upper,
        'uniform_cosine_tail_bound':bessel_even_tail_bound(degree,zmax),
        'tail_tolerance':tail_tolerance,'setup_seconds':setup_finished-started,
        'recurrence_and_observation_seconds':recurrence_finished-setup_finished,
        'coefficient_and_output_seconds':finished-recurrence_finished,
        'total_seconds':finished-started}


def propagate(n,name,tail_tolerance=1e-17,samples=641,progress=False):
    started=time.perf_counter()
    X,Y,rho,bulk,mass,C,J=build_sparse(n,name)
    del J
    A=(C@C.T).tocsr()
    B=ports(X,Y,bulk,CONFIG['source_centres'],True)
    R=ports(X,Y,bulk,CONFIG['receiver_centres'],False)
    # Symmetry and Gershgorin give a real-arithmetic upper spectral bound.
    # Relative padding limits downward rounding; it is not an interval bound.
    upper=float(np.asarray(abs(A).sum(axis=1)).max())*(1.+1e-14)
    omega=np.sqrt(upper)
    zmax=CONFIG['time_end']*omega
    degree=int(zmax/2)+1
    while bessel_even_tail_bound(degree,zmax)>tail_tolerance:
        degree+=1
    times=np.linspace(0,CONFIG['time_end'],samples)
    snap_ids=np.linspace(0,samples-1,5,dtype=int)
    snap_times=times[snap_ids]
    orders=2*np.arange(degree+1)
    sign=2*(-1.)**np.arange(1,degree+1)
    cosine_coeff=jv(orders[:,None],omega*snap_times[None,:])
    cosine_coeff[1:]*=sign[:,None]
    # Integral_0^z J_{2k}(x) dx = 2 sum_{j>=k} J_{2j+1}(z).
    # Extending the odd series by 100 orders gives a still smaller tail.
    odd=jv((2*np.arange(degree+100)+1)[:,None],omega*snap_times[None,:])
    sine_coeff=2*np.cumsum(odd[::-1],axis=0)[::-1][:degree+1]/omega
    sine_coeff[1:]*=sign[:,None]
    q0=B.copy()
    q1=(2/upper)*(A@B)-B
    pressure=cosine_coeff[0,:,None,None]*q0[None,:,:]
    primitive=sine_coeff[0,:,None,None]*q0[None,:,:]
    observations=np.empty((degree+1,R.shape[1],B.shape[1]))
    observations[0]=R.T@q0
    for k in range(1,degree+1):
        q=q1
        observations[k]=R.T@q
        for j in range(1,5):
            pressure[j]+=cosine_coeff[k,j]*q
            primitive[j]+=sine_coeff[k,j]*q
        if k<degree:
            q0,q1=q1,2*((2/upper)*(A@q1)-q1)-q0
        if progress and k%1000==0:
            print(json.dumps({'material':name,'grid':n,'degree_completed':k,
                'degree':degree,'wall_seconds':time.perf_counter()-started}),flush=True)
    coeff=jv(orders[:,None],omega*times[None,:])
    coeff[1:]*=sign[:,None]
    traces=np.einsum('kt,krs->trs',coeff,observations,optimize=True)
    velocity=np.array([C.T@w for w in primitive])
    energy=np.sum(pressure*pressure,axis=1)+np.sum(velocity*velocity,axis=1)
    energy0=np.sum(B*B,axis=0)
    tail=bessel_even_tail_bound(degree,zmax)
    coefficient_tail=sinc_coefficient_tail_bound(degree,zmax,omega)
    return {'time':times,'receiver_pressure':traces,'snapshot_times':snap_times,
        'pressure_snapshots':pressure/np.sqrt(mass[None,:,None]),'rho':rho,'bulk_modulus':bulk,
        'energy_relative_drift':np.array(float(np.max(abs(energy/energy0-1.)))),
        'energy_check_times':snap_times,'energy_at_check_times':energy,
        'wall_seconds':np.array(time.perf_counter()-started),'grid':np.array(n),
        'samples':np.array(samples),'chunk_duration':np.array(0.),
        'propagator':np.array('Chebyshev-Bessel full-order matrix function'),
        'squared_frequency_upper_bound':np.array(upper),'polynomial_degree':np.array(degree),
        'uniform_cosine_tail_bound':np.array(tail),
        'uniform_integrated_cosine_tail_bound':np.array(CONFIG['time_end']*tail+coefficient_tail),
        'uniform_sinc_coefficient_tail_bound':np.array(coefficient_tail),
        'tail_tolerance':np.array(tail_tolerance)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grids',nargs='+',type=int,default=[384])
    p.add_argument('--materials',nargs='+',default=['double_inclusion','smooth_checker'])
    p.add_argument('--cpu-affinity',nargs='+',type=int,default=None)
    p.add_argument('--check-existing',action='store_true',help='Compare against existing exponential results and write checks only.')
    p.add_argument('--check-suffix',action='store_true',help='Keep arrays under a separate _chebyshev_check suffix.')
    p.add_argument('--progress',action='store_true')
    args=p.parse_args()
    if args.cpu_affinity is not None:
        os.sched_setaffinity(0,set(args.cpu_affinity))
    records=[]
    for n in args.grids:
        for name in args.materials:
            target=ROOT/'results'/f'{name}_n{n}.npz'
            if not args.check_existing and not args.check_suffix and target.exists():
                print(json.dumps({'state':'retained','material':name,'grid':n}),flush=True)
                continue
            answer=propagate(n,name,progress=args.progress)
            row={k:float(answer[k]) for k in ('energy_relative_drift','wall_seconds','uniform_cosine_tail_bound','squared_frequency_upper_bound')}
            row.update(material=name,grid=n,polynomial_degree=int(answer['polynomial_degree']),energy_check_times=answer['energy_check_times'].tolist())
            if args.check_existing:
                with np.load(target) as reference:
                    for key in ('receiver_pressure','pressure_snapshots'):
                        row[key+'_relative_difference']=float(np.linalg.norm(answer[key]-reference[key])/np.linalg.norm(reference[key]))
                if max(row[k+'_relative_difference'] for k in ('receiver_pressure','pressure_snapshots'))>2e-10:
                    raise RuntimeError('Chebyshev/exponential verification tolerance exceeded.')
            else:
                if args.check_suffix:target=target.with_name(target.stem+'_chebyshev_check.npz')
                np.savez_compressed(target,**answer)
                row['file']=target.name
                row['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
            records.append(row)
            print(json.dumps(row),flush=True)
    suffix='verification' if args.check_existing else 'execution'
    grids='_'.join(str(n) for n in args.grids)
    (ROOT/'results'/f'chebyshev_{suffix}_n{grids}.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
