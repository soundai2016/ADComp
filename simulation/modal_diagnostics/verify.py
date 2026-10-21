#!/usr/bin/env python3
"""Check stored modal components against the receiver trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
import csv
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
rows=list(csv.DictReader(open(HERE/'results'/'summary.csv')))
worst_identity=0.;worst_same_grid=0.;worst_recorded_error=0.
for r in rows:
    z=np.load(HERE/'results'/f"{r['material']}_{r['method']}_m{r['m']}_decomposition.npz")
    err=z['rom_response']-z['fine_response']
    terms=sum(z[k] for k in ['ritz_frequency','grid_frequency','source_coupling','receiver_coupling','unmatched_modes'])
    defect=float(np.linalg.norm(err-terms)/np.linalg.norm(err));worst_identity=max(worst_identity,defect)
    same=sum(z[k] for k in ['same_grid_frequency','same_grid_source','same_grid_receiver','same_grid_unmatched'])
    same_defect=float(np.linalg.norm(z['rom_response']-z['coarse_response']-same)/np.linalg.norm(z['rom_response']-z['coarse_response']))
    worst_same_grid=max(worst_same_grid,same_defect)
    total=float(np.linalg.norm(z['rom_response']-z['reference_256_response'])/np.linalg.norm(z['reference_256_response']))
    worst_recorded_error=max(worst_recorded_error,abs(total-float(r['total_error_vs_256'])))
    assert defect<1e-12 and same_defect<1e-10
    assert float(r['stored_rom_response_error'])<1e-10
    assert float(r['minmax_min_difference'])>-1e-9
checks=json.loads((HERE/'results'/'validation.json').read_text())['cases']
for c in checks:
    assert c['coarse_complete_response_error']<1e-9
    assert c['fine_response_coverage_pass']
    assert c['fine_truncated_response_error']<CFG['fine_response_coverage_limit']
    assert c['coarse_eigen_residual']<1e-8 and c['fine_eigen_residual']<1e-7
result={'models_checked':len(rows),'maximum_fine_decomposition_relative_defect':worst_identity,
        'maximum_same_grid_decomposition_relative_defect':worst_same_grid,
        'maximum_recorded_error_difference':worst_recorded_error,'passed':True}
(HERE/'results'/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
