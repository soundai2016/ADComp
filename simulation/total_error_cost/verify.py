#!/usr/bin/env python3
"""Small-grid equality of the receiver-only and full-field implementations."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
if hasattr(os, 'sched_setaffinity'):
    available = os.sched_getaffinity(0)
    if 8 in available:
        os.sched_setaffinity(0, {8})
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import run


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    with threadpool_limits(limits=1):
        times = np.linspace(0, .4, 33)
        full, stages, checks = run.full_receiver(8, 'rotated_stripes', times, .1)
        mass, A, J, B, R, rho, bulk = run.assemble(8, 'rotated_stripes')
        lam, V = eigh(A.toarray())
        lam = np.maximum(lam, 0)
        exact = np.einsum('ir,tr,rs->tis', R.T@V,
                          np.cos(times[:,None]*np.sqrt(lam)[None,:]), V.T@B, optimize=True)
        result = {'full_modal_difference': relative(full['receiver_pressure'], exact),
                  'energy_relative_drift': checks['energy_relative_drift'], 'reduced_matches': {}}
        for method in ('craig_bampton','rational_krylov'):
            raw, stages, diag = run.reduced_receiver(8, 'rotated_stripes', 4, method, times)
            old, U, old_stages, old_diag = run.BASE.reduced_run(8, 'rotated_stripes', 4, method, times)
            reconstructed = np.einsum('ir,tr,rs->tis', raw['receiver_modal_matrix'],
                np.cos(times[:,None]*np.sqrt(raw['reduced_eigenvalues'])[None,:]),
                raw['initial_modal_coefficients'], optimize=True)
            result['reduced_matches'][method] = {
                'full_field_implementation_difference': relative(raw['receiver_pressure'], old['receiver_pressure']),
                'stored_receiver_representation_difference': relative(raw['receiver_pressure'], reconstructed),
                'orthonormality_frobenius': diag['orthonormality_frobenius']}
        cheb, stages, checks = run.chebyshev_receiver(8, 'rotated_stripes', times, 1e-17)
        result['chebyshev_modal_difference'] = relative(cheb['receiver_pressure'], exact)
        result['chebyshev_tail_bound'] = checks['uniform_cosine_tail_bound']
        leapfrog = []
        for cfl in (.25, .125):
            explicit, stages, diag = run.explicit_receiver(8, 'rotated_stripes', cfl, times)
            dt = diag['time_step']
            angle = np.arccos(np.clip(1-.5*dt*dt*lam, -1, 1))
            modal_explicit = np.einsum('ir,tr,rs->tis', R.T@V,
                np.cos(np.rint(times[:,None]/dt)*angle[None,:]), V.T@B, optimize=True)
            leapfrog.append({'cfl': cfl, 'time_step': dt,
                             'modal_discrete_difference': relative(explicit['receiver_pressure'], modal_explicit),
                             'same_grid_time_error': relative(explicit['receiver_pressure'], exact),
                             'sufficient_stability_product': diag['sufficient_stability_product']})
        result['stormer_verlet'] = leapfrog
        result['status'] = 'pass' if result['full_modal_difference'] < 1e-10 and result['chebyshev_modal_difference'] < 1e-10 and max(v['modal_discrete_difference'] for v in leapfrog)<1e-10 and all(
            v['full_field_implementation_difference']<1e-10 and v['stored_receiver_representation_difference']<1e-10
            for v in result['reduced_matches'].values()) else 'fail'
        (Path(__file__).resolve().parent/'results'/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if result['status']!='pass':
            raise SystemExit(1)


if __name__=='__main__':
    main()
