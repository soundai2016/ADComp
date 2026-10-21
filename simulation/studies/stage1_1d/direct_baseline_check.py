"""Time the equivalent direct finite-volume solver."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,csv,time
from pathlib import Path
import numpy as np
from core import *

HERE=Path(__file__).resolve().parent

def timed(f,reps=9):
    f();elapsed=[]
    for i in range(reps):
        t=time.perf_counter();f();elapsed.append(time.perf_counter()-t)
    return float(np.median(elapsed))

def run():
    cases=json.loads((HERE/'media.json').read_text())[:4]
    w=np.load(HERE/'response_data.npz')['omega'];rows=[]
    for case in cases:
        r,k,d=case['rho'],case['bulk'],case['widths']
        for n in [4,8,16,32,64,128,256]:
            matrices=fv_matrices(r,k,d,n);model=build_modal_model(r,k,d,n)
            direct=fv_scattering_direct(w,matrices);modal=evaluate(model,w)
            error=float(np.max(abs(direct-modal)))
            setup=timed(lambda:fv_matrices(r,k,d,n));query=timed(lambda:fv_scattering_direct(w,matrices))
            rows.append({'case':case['case'],'cells':n,'max_modal_direct_difference':error,
                         'setup_seconds':setup,'query_513freq_seconds':query,'one_query_total_seconds':setup+query,
                         '100_query_mean_seconds':setup/100+query})
    with (HERE/'direct_fv_cost_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={'max_absolute_scattering_difference':max(r['max_modal_direct_difference'] for r in rows),
             'median_cost_by_cells':{str(n):{key:float(np.median([r[key] for r in rows if r['cells']==n])) for key in ['setup_seconds','query_513freq_seconds','one_query_total_seconds','100_query_mean_seconds']} for n in [4,8,16,32,64,128,256]},
             'status':'Post-stage implementation audit of the same FV equations; direct solve uses no modal eigendecomposition. Exact analytic layers remains available and faster overall.'}
    (HERE/'direct_fv_audit.json').write_text(json.dumps(summary,indent=2))
    assert summary['max_absolute_scattering_difference']<1e-8
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
