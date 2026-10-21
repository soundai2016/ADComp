#!/usr/bin/env python3
"""Check the continuum pressure norm and the streamed output-cost implementation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
if hasattr(os,'sched_getaffinity') and 8 in os.sched_getaffinity(0):
    os.sched_setaffinity(0,{8})
import json
from pathlib import Path
import run_interface as experiment
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    with threadpool_limits(limits=1):
        n=8;times=np.linspace(0,.4,33)
        raw,stages,errors,diagnostics=experiment.full_run(n,times,7,True)
        model=experiment.KERNEL.separable_setup(n)
        u=experiment.KERNEL.separable_pressure(model,times)
        u_exact=experiment.KERNEL.exact_mass_weighted_pressure(n,times)
        norm=experiment.KERNEL.exact_pressure_norm_squared(times)
        residual=np.sum((u-u_exact)**2)+np.sum(norm)-np.sum(u_exact**2)
        direct=float(np.sqrt(residual/np.sum(norm)))
        result={'streamed_continuum_norm_difference':abs(direct-errors[0]['continuous_field_error']),
                'minimum_projection_loss':diagnostics['minimum_projection_loss'],'reduced':{}}
        for method in ('craig_bampton','rational_krylov'):
            reduced,timing,field_error,diagnostic=experiment.reduced_run(n,4,method,times,7,True)
            reconstructed=np.einsum('ir,tr,rs->tis',reduced['receiver_modal_matrix'],
                np.cos(times[:,None]*np.sqrt(reduced['reduced_eigenvalues'])[None,:]),
                reduced['initial_modal_coefficients'],optimize=True)
            result['reduced'][method]={'receiver_reconstruction_relative_difference':float(np.linalg.norm(reconstructed-reduced['receiver_pressure'])/np.linalg.norm(reconstructed)),
                'continuous_field_error':field_error[0]['continuous_field_error'],
                'cell_average_field_error':field_error[0]['cell_average_field_error'],
                'all_stages_nonnegative':all(value>=0 for value in timing.values())}
        result['status']='pass' if result['streamed_continuum_norm_difference']<1e-12 and all(
            row['receiver_reconstruction_relative_difference']<1e-12 and row['all_stages_nonnegative'] for row in result['reduced'].values()) else 'fail'
        path=Path(__file__).resolve().parent/'results'/'interface_verification.json'
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
