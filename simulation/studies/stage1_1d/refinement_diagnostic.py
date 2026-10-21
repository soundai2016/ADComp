#!/usr/bin/env python3
"""Compare fixed modal order and grid refinement on 28 layered media."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,csv
from pathlib import Path
import numpy as np
from core import *

HERE=Path(__file__).resolve().parent

def run():
    cases=json.loads((HERE/'media.json').read_text())
    data=np.load(HERE/'response_data.npz');w=data['omega'];held=data['heldout']
    rows=[]
    for case in cases:
        exact=exact_scattering(w,case['rho'],case['bulk'],case['widths'])
        for n in [32,64,128,256,512]:
            full=build_modal_model(case['rho'],case['bulk'],case['widths'],n)
            methods={'full_FV':full,'fixed_modal16_tail':truncate_model(full,16,True),
                     'growing_modal_tail':truncate_model(full,max(16,n//4),True)}
            for method,model in methods.items():
                pred=evaluate(model,w)
                for label,mask in [('low_band',held&(w<=4*np.pi)),('all_band',held)]:
                    rows.append({'case':case['case'],'cells':n,'method':method,'mode_count':len(model.lambdas),
                                 'band':label,**scattering_metrics(pred,exact,mask)})
    with (HERE/'refinement_diagnostic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={}
    for band in ['low_band','all_band']:
        summary[band]={}
        for method in methods:
            summary[band][method]={str(n):float(np.mean([r['rms_complex_S_error'] for r in rows if r['band']==band and r['method']==method and r['cells']==n])) for n in [32,64,128,256,512]}
    summary['status']='Exploratory diagnostic on the same 28 development cases.'
    summary['conclusion']='A fixed finite modal order has a response-error floor. Grid refinement alone cannot force this closure error to zero. Growing mode count is one sufficient numerical route, without a claim of learned convergence.'
    (HERE/'refinement_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
