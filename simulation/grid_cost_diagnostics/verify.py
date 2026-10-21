#!/usr/bin/env python3
"""Small-grid solver equality, then integrity checks for the completed sweep."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import run

HERE = Path(__file__).resolve().parent


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def preflight():
    cfg = json.loads((HERE/'config.json').read_text())
    result = {'settings':len(list(run.candidates(cfg))), 'checks':{}}
    with threadpool_limits(limits=1):
        times = np.linspace(0, .4, 33)
        for material in cfg['materials']:
            mass, A, J, B, R, rho, bulk = run.SOLVER.assemble(8, material)
            lam, V = eigh(A.toarray())
            lam = np.maximum(lam, 0)
            cases = {}
            for cfl in cfg['explicit_cfl_values']:
                raw, stages, diagnostic = run.SOLVER.explicit_receiver(8, material, cfl, times)
                dt = diagnostic['time_step']
                omega = 2*np.arcsin(np.clip(.5*dt*np.sqrt(lam), 0, 1))/dt
                expected = np.einsum('ir,tr,rs->tis',R.T@V,np.cos(times[:,None]*omega[None,:]),V.T@B,optimize=True)
                error = relative(raw['receiver_pressure'],expected)
                assert error<1e-10
                assert diagnostic['sufficient_stability_product']<2
                cases[f'verlet_cfl_{cfl}'] = {'relative_error_to_discrete_modal_solution':error,
                    'sufficient_stability_product':diagnostic['sufficient_stability_product']}
            raw, stages, diagnostic = run.SOLVER.reduced_receiver(8, material, 4, 'craig_bampton', times)
            old, U, old_stages, old_diagnostic = run.SOLVER.BASE.reduced_run(8, material, 4, 'craig_bampton', times)
            reconstructed = np.einsum('ir,tr,rs->tis',raw['receiver_modal_matrix'],
                np.cos(times[:,None]*np.sqrt(raw['reduced_eigenvalues'])[None,:]),raw['initial_modal_coefficients'],optimize=True)
            difference, replay = relative(raw['receiver_pressure'],old['receiver_pressure']), relative(reconstructed,raw['receiver_pressure'])
            assert difference<1e-10 and replay<1e-12
            cases['craig_bampton'] = {'relative_error_to_full_field_implementation':difference,
                                     'relative_error_from_saved_modal_representation':replay}
            result['checks'][material] = cases
    result.update(status='pass', dependency_sha256=run.hashes())
    run.dump(HERE/'results'/'preflight.json',result)
    return result


def verify_outputs():
    cfg=json.loads((HERE/'config.json').read_text())
    expected=list(run.candidates(cfg))
    folder=HERE/'results'/'candidates'
    paths=list(folder.glob('*_timing.json'))
    assert {p.name for p in paths}=={run.stem(c)+'_timing.json' for c in expected}
    dependencies=run.hashes()
    max_reconstruction=0.
    max_repeat=0.
    intervals=[]
    stage_sum_differences=[]
    for candidate in expected:
        stem=run.stem(candidate)
        record=json.loads((folder/(stem+'_timing.json')).read_text())
        assert all(record[k]==v for k,v in candidate.items())
        assert record['dependency_sha256']==dependencies
        assert record['cpu_affinity']==cfg['cpu_affinity']
        assert len(record['repetitions'])==len(record['repetition_intervals'])==cfg['repetitions']
        totals=[]
        for stages,interval in zip(record['repetitions'],record['repetition_intervals']):
            assert set(stages)==set(run.SOLVER.STAGES)
            assert all(np.isfinite(v) and v>=0 for v in stages.values())
            total=sum(stages.values())
            assert total<=interval['end_unix']-interval['start_unix']+.01
            totals.append(total)
            intervals.append((interval['start_unix'],interval['end_unix']))
        stage_sum_differences.append(abs(float(np.median(totals))-record['median']['total_seconds']))
        path=folder/(stem+'.npz')
        assert run.digest(path)==record['raw_sha256']
        with np.load(path) as data:
            times=data['time']; y=data['receiver_pressure'];repeated=data['receiver_pressure_repetitions']
            np.testing.assert_array_equal(times,np.linspace(0,cfg['time_end'],cfg['samples']))
            assert y.shape==(cfg['samples'],4,3) and np.isfinite(y).all()
            assert repeated.shape==(cfg['repetitions'],cfg['samples'],4,3) and np.isfinite(repeated).all()
            max_repeat=max(max_repeat,max(relative(yr,y) for yr in repeated))
            if candidate['method']=='craig_bampton':
                assert len(data['reduced_eigenvalues'])==candidate['dimension']
                reconstructed=np.einsum('ir,tr,rs->tis',data['receiver_modal_matrix'],
                    np.cos(times[:,None]*np.sqrt(data['reduced_eigenvalues'])[None,:]),data['initial_modal_coefficients'],optimize=True)
                max_reconstruction=max(max_reconstruction,relative(reconstructed,y))
            else:
                assert all(d['sufficient_stability_product']<2 for d in record['diagnostics'])
    intervals.sort()
    assert all(b[0]>=a[1] for a,b in zip(intervals[:-1],intervals[1:]))
    assert max_reconstruction<1e-12 and max_repeat<1e-10
    assert max(stage_sum_differences)<1e-12
    summary=json.loads((HERE/'results'/'summary.json').read_text())
    assert summary['complete']
    result={'status':'pass','settings':len(paths),'timed_repetitions':len(intervals),
        'all_configuration_and_runtime_hashes_match':True,'all_raw_hashes_match':True,
        'maximum_receiver_reconstruction_difference':max_reconstruction,
        'maximum_repeated_receiver_difference':max_repeat,
        'maximum_recorded_cost_difference':max(stage_sum_differences),'timing_intervals_do_not_overlap':True}
    run.dump(HERE/'results'/'output_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',action='store_true');args=parser.parse_args()
    print(json.dumps(verify_outputs() if args.results else preflight(),indent=2))


if __name__=='__main__':main()
