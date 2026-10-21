#!/usr/bin/env python3
"""Check the computed candidate set, hashes, recorded costs and receiver arrays."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    config_hash=digest(HERE/'config.json')
    expected_smooth=len(cfg['smooth_materials'])*(len(cfg['smooth_full_grids'])+len(cfg['chebyshev_full_grids'])+
        len(cfg['explicit_full_grids'])*len(cfg['explicit_cfl_values'])+2*sum(len(v) for v in cfg['smooth_reduced_modes'].values()))
    expected_interface=len(cfg['interface_full_grids'])+2*len(cfg['interface_reduced_grids'])*len(cfg['interface_modes_per_patch'])
    maximum_reconstruction=0.0
    maximum_total_difference=0.0
    counts={}
    intervals=[]
    for group,expected in [('smooth',expected_smooth),('interface',expected_interface)]:
        paths=sorted((HERE/'results'/group).glob('*_timing.json'))
        if len(paths)!=expected:
            raise RuntimeError(f'{group}: expected {expected}, found {len(paths)} settings')
        counts[group]=len(paths)
        for path in paths:
            record=json.loads(path.read_text())
            if record['config_sha256']!=config_hash:
                raise RuntimeError(f'Changed configuration: {path.name}')
            if group=='smooth':
                if record['code_sha256']!=digest(HERE/'run.py'):
                    raise RuntimeError('Smooth timing implementation changed')
            else:
                sources={'run.py':HERE/'run.py','run_interface.py':HERE/'run_interface.py','kernel.py':ROOT/'interface_case'/'kernel.py'}
                if record['code_sha256']!={name:digest(source) for name,source in sources.items()}:
                    raise RuntimeError('Interface timing implementation changed')
            for relative,recorded in record['dependency_sha256'].items():
                if recorded!=digest(ROOT/relative):
                    raise RuntimeError(f'Changed dependency: {relative}')
            repetitions=record['repetitions']
            if len(repetitions)!=cfg['repetitions']:
                raise RuntimeError('Incomplete repetitions')
            if any(value<0 or not np.isfinite(value) for rep in repetitions for value in rep.values()):
                raise RuntimeError('Invalid timing')
            if group=='smooth':
                total=float(np.median([sum(rep.values()) for rep in repetitions]))
                maximum_total_difference=max(maximum_total_difference,abs(total-record['median']['total_seconds']))
            else:
                both=float(np.median([sum(rep.values()) for rep in repetitions]))
                receivers=float(np.median([sum(value for key,value in rep.items() if key!='field_reconstruction') for rep in repetitions]))
                maximum_total_difference=max(maximum_total_difference,abs(both-record['median']['both_total_seconds']),abs(receivers-record['median']['receiver_total_seconds']))
            if record['method']=='full_stormer_verlet' and not record['diagnostics']['sufficient_stability_product']<2:
                raise RuntimeError('Explicit stability test failed')
            raw_path=path.with_name(path.name.replace('_timing.json','.npz'))
            with np.load(raw_path) as data:
                times=data['time'];y=data['receiver_pressure']
                np.testing.assert_array_equal(times,np.linspace(0,cfg['timing_time_end'],cfg['samples']))
                if y.shape!=(cfg['samples'],4,3) or not np.isfinite(y).all():
                    raise RuntimeError('Invalid receiver array')
                if 'reduced_eigenvalues' in data:
                    reconstructed=np.einsum('ir,tr,rs->tis',data['receiver_modal_matrix'],
                        np.cos(times[:,None]*np.sqrt(data['reduced_eigenvalues'])[None,:]),data['initial_modal_coefficients'],optimize=True)
                    maximum_reconstruction=max(maximum_reconstruction,float(np.linalg.norm(reconstructed-y)/np.linalg.norm(y)))
            intervals.append(record['timed_interval_unix'])
    intervals.sort()
    nonoverlap=all(second[0]>=first[1] for first,second in zip(intervals[:-1],intervals[1:]))
    summaries=[json.loads((HERE/'results'/name).read_text()) for name in ('summary.json','interface_summary.json')]
    if not all(summary['complete'] for summary in summaries):
        raise RuntimeError('Final summary is incomplete')
    result={'status':'pass' if maximum_total_difference<1e-12 and maximum_reconstruction<1e-12 and nonoverlap else 'fail',
            'settings':counts,'repetitions_per_setting':cfg['repetitions'],'configuration_and_all_recorded_code_hashes_match':True,
            'maximum_receiver_reconstruction_difference':maximum_reconstruction,
            'maximum_recorded_median_total_difference':maximum_total_difference,
            'timed_setting_intervals_do_not_overlap':nonoverlap}
    (HERE/'results'/'output_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
