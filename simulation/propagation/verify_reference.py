#!/usr/bin/env python3
"""Check independent propagators, tail bounds and energy records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chebyshev_reference import bessel_even_tail_bound,sinc_coefficient_tail_bound

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'


def main():
    with np.load(RESULTS/'rotated_stripes_n384.npz') as exponential, np.load(RESULTS/'rotated_stripes_n384_chebyshev_check.npz') as polynomial:
        if not np.array_equal(exponential['time'],polynomial['time']):
            raise ValueError('Independent propagators have different observation times.')
        agreement={key:float(np.linalg.norm(exponential[key]-polynomial[key])/np.linalg.norm(exponential[key]))
            for key in ('receiver_pressure','pressure_snapshots')}
    rows=[]
    for n in (384,512):
        for name in json.loads((ROOT/'experiment_config.json').read_text())['materials']:
            file=RESULTS/f'{name}_n{n}.npz'
            with np.load(file) as z:
                shape_ok=z['receiver_pressure'].shape==(641,4,3) and z['pressure_snapshots'].shape==(5,n*n,3)
                finite=bool(np.isfinite(z['receiver_pressure']).all() and np.isfinite(z['pressure_snapshots']).all())
                row={'material':name,'grid':n,'shape_ok':shape_ok,'finite':finite,
                    'wall_seconds':float(z['wall_seconds']),
                    'energy_relative_drift':float(z['energy_relative_drift']),
                    'energy_check_count':len(z['energy_check_times']) if 'energy_check_times' in z else len(z['time']),
                    'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
                if 'polynomial_degree' in z:
                    omega=float(np.sqrt(z['squared_frequency_upper_bound']))
                    degree=int(z['polynomial_degree']);zmax=8*omega
                    tail=bessel_even_tail_bound(degree,zmax)
                    coefficient_tail=sinc_coefficient_tail_bound(degree,zmax,omega)
                    row.update(polynomial_degree=degree,recomputed_uniform_cosine_tail_bound=tail,
                        recomputed_sinc_coefficient_tail_bound=coefficient_tail,
                        recomputed_total_integrated_cosine_tail_bound=8*tail+coefficient_tail,
                        tail_criterion_pass=tail<=1e-17)
                rows.append(row)
    numerical_pass=max(agreement.values())<2e-10 and all(r['shape_ok'] and r['finite'] and r.get('tail_criterion_pass',True) for r in rows)
    passed=numerical_pass
    output={'status':'pass' if passed else 'fail',
        'numerical_checks_pass':numerical_pass,
        'grid384_exponential_chebyshev_relative_difference':agreement,
        'relative_agreement_tolerance':2e-10,
        'energy_sampling':'641 times for the exponential; snapshot times 0,2,4,6,8 for Chebyshev.',
        'tail_interpretation':'Exact-arithmetic uniform matrix-function truncation estimates; floating-point error is checked by independent propagation.',
        'verified_implementation_sha256':hashlib.sha256((ROOT/'chebyshev_reference.py').read_bytes()).hexdigest(),
        'reference_cases':rows}
    report='reference_verification.json'
    (RESULTS/report).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
