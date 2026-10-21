#!/usr/bin/env python3
"""Rescore fixed modal interventions against a finer propagation reference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'evaluation_config.json').read_text())
MODELS=json.loads((HERE/'config.json').read_text())

def relative(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def main():
    nr=CFG['evaluation_reference_grid'];rows=[];provenance={}
    for material in MODELS['materials']:
        path=HERE.parent/'propagation'/'results'/f'{material}_n{nr}.npz'
        with np.load(path) as ref:
            tr=ref['time'];yref=ref['receiver_pressure']
        provenance[material]={'file':str(path.relative_to(HERE.parent)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for method,m in MODELS['models']:
            with np.load(HERE/'results'/f'{material}_{method}_m{m}_decomposition.npz') as z:
                np.testing.assert_array_equal(tr,z['time'])
                row={'material':material,'method':method,'m':m,'evaluation_reference_grid':nr,
                    f'total_error_vs_{nr}':relative(z['rom_response'],yref),
                    f'error_without_ritz_shift_vs_{nr}':relative(z['no_ritz_frequency_response'],yref),
                    f'error_with_fine_frequencies_vs_{nr}':relative(z['fine_frequency_response'],yref),
                    f'reference_256_to_{nr}_difference':relative(z['reference_256_response'],yref),
                    f'reference_128_to_{nr}_difference':relative(z['fine_response'],yref)}
                rows.append(row)
    with (HERE/'results'/'reference_evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'evaluation_reference_grid':nr,'matching_reference_grid':CFG['modal_matching_grid'],
             'evaluation_config_sha256':hashlib.sha256((HERE/'evaluation_config.json').read_bytes()).hexdigest(),
             'matching_config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'reference_provenance':provenance,'results':rows}
    (HERE/'results'/'reference_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
