#!/usr/bin/env python3
"""Layered-wave scattering, modal truncation, and passive realization checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import csv,json,time,hashlib,platform
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from core import *

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
W=np.linspace(P['frequency']['minimum'],P['frequency']['maximum'],P['frequency']['count'])
HELDOUT=np.arange(len(W))%4!=0
MASKS={'heldout_all':HELDOUT,'development_all':~HELDOUT,
       'heldout_low_0_to_4pi':HELDOUT&(W<=4*np.pi),
       'heldout_mid_4pi_to_8pi':HELDOUT&(W>4*np.pi)&(W<=8*np.pi),
       'heldout_high_8pi_to_12pi':HELDOUT&(W>8*np.pi)}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def timed(f,reps=5):
    f(); times=[]
    for _ in range(reps):
        t=time.perf_counter();f();times.append(time.perf_counter()-t)
    return float(np.median(times))


def make_cases():
    r,k,widths=sample_media(P['media_seed'],P['random_development_cases'])
    cases=[(f'random_{i:02d}',r[i],k[i],widths[i]) for i in range(len(r))]
    ones=np.ones(8);alt=np.array([1,4]*4);group=np.array([1]*4+[4]*4)
    x=(np.arange(8)+.5)/8
    cases += [('homogeneous',ones,ones,ones/8),('alternating_impedance_4',alt,alt,ones/8),
              ('grouped_impedance_4',group,group,ones/8),
              ('smooth_log_sinusoid_8layers',np.exp(.7*np.sin(2*np.pi*x)),np.exp(.6*np.cos(2*np.pi*x)),ones/8)]
    return cases


def models_for(r,k,d):
    models={f'full_fv_{n}':build_modal_model(r,k,d,n=n) for n in P['full_fv_cells']}
    full=models['full_fv_256']
    for order in P['orders']:
        models[f'modal_{order}']=truncate_model(full,order,False)
        models[f'modal_tail_{order}']=truncate_model(full,order,True)
        models[f'matched_fv_{order}']=build_modal_model(r,k,d,n=order)
    return models


def checks_and_time(model):
    a,b,c,d,metric=scattering_state_space(model)
    time_checks={}
    ss=[]
    for w in (.31,2.17,7.53,13.69,27.31):
        ss.append(c@np.linalg.solve(1j*w*np.eye(len(a))-a,b)+d)
    truth=evaluate(model,np.array([.31,2.17,7.53,13.69,27.31]))
    time_checks['state_frequency_max_error']=float(np.max(abs(np.asarray(ss)-truth)))
    time_checks['maximum_generator_real_part']=float(np.max(np.linalg.eigvals(a).real))
    # Exact zero-order-hold response to a left-port rectangular pulse with zero
    # initial state; this checks the realization, not layer-waveform accuracy.
    dt=.002;steps=6000;n=len(a)
    aug=np.block([[a,b],[np.zeros((2,n+2))]])
    step=expm(dt*aug); ad,bd=step[:n,:n],step[:n,n:]
    state=np.zeros(n); energies=[];out=[];incoming=[]
    # Pulse switches lie on sample times.
    for i in range(steps):
        inp=np.array([1. if 100<=i<200 else 0.,0.])
        out.append(c@state+d@inp);incoming.append(inp)
        energies.append(.5*state@metric@state)
        state=ad@state+bd@inp
    energy=np.array(energies);out=np.array(out);incoming=np.array(incoming)
    time_checks['max_preinput_output']=float(abs(out[:100]).max())
    time_checks['max_energy_increment_after_source']=float(np.diff(energy[200:]).max())
    time_checks['final_stored_energy']=float(energy[-1])
    return time_checks,np.arange(steps)*dt,incoming,out,energy


def run():
    began=time.perf_counter(); cases=make_cases(); rows=[];cost=[];all_predictions=[]
    checks={'max_unitarity':0.,'max_reciprocity':0.,'minimum_RHP_impedance_eigenvalue':float('inf'),
            'minimum_tail_eigenvalue':float('inf'),'minimum_signed_endpoint_residue':float('inf'),
            'minimum_signed_retained16_endpoint_residue':float('inf'),
            'exact_scattering_composition_max_error':0.}
    jsoncases=[]
    for ci,(name,r,k,d) in enumerate(cases):
        target=exact_scattering(W,r,k,d)
        models=models_for(r,k,d)
        pred={'uniform_effective':exact_scattering(W,[np.sum(r*d)/sum(d)],[sum(d)/np.sum(d/k)],[sum(d)])}
        pred.update({key:evaluate(model,W) for key,model in models.items()})
        jsoncases.append({'case':name,'rho':r.tolist(),'bulk':k.tolist(),'widths':d.tolist()})
        for method,arr in pred.items():
            for band,mask in MASKS.items():
                metric=scattering_metrics(arr,target,mask)
                rows.append({'case':name,'method':method,'band':band,'frequency_count':int(mask.sum()),**metric})
                checks['max_unitarity']=max(checks['max_unitarity'],metric['max_unitarity_error'])
                checks['max_reciprocity']=max(checks['max_reciprocity'],metric['max_reciprocity_error'])
        if ci<4:
            for method,model in models.items():
                query=timed(lambda:evaluate(model,W),9)
                if method.startswith('modal'):
                    order=len(model.lambdas);uset=method.startswith('modal_tail')
                    setup=timed(lambda:truncate_model(build_modal_model(r,k,d,256),order,uset),3)
                    states=2*order-1+(2 if uset else 0)
                else:
                    setup=timed(lambda:build_modal_model(r,k,d,model.cells),3);states=2*model.cells+1
                cost.append({'case':name,'method':method,'state_dimension':states,'setup_seconds':setup,
                             'query_513freq_seconds':query,'one_query_total_seconds':setup+query,
                             '100_query_mean_seconds':setup/100+query})
            for method,func in [('exact_layers',lambda:exact_scattering(W,r,k,d)),
                                ('uniform_effective',lambda:exact_scattering(W,[np.sum(r*d)/sum(d)],[sum(d)/np.sum(d/k)],[sum(d)]))]:
                query=timed(func,9)
                cost.append({'case':name,'method':method,'state_dimension':0,'setup_seconds':0.,
                             'query_513freq_seconds':query,'one_query_total_seconds':query,
                             '100_query_mean_seconds':query})
        for method,model in models.items():
            signed=model.residues[:,1]*(-1.)**np.arange(len(model.lambdas))
            checks['minimum_signed_endpoint_residue']=min(checks['minimum_signed_endpoint_residue'],float(signed.min()))
            checks['minimum_signed_retained16_endpoint_residue']=min(checks['minimum_signed_retained16_endpoint_residue'],float(signed[:min(16,len(signed))].min()))
            if method.startswith('modal_tail'):
                checks['minimum_tail_eigenvalue']=min(checks['minimum_tail_eigenvalue'],float(np.linalg.eigvalsh(model.tail).min()))
            s=np.array([.01+.3j,.1+3j,.5+10j,1+30j])
            z=modal_impedance(s,model.lambdas,model.residues,model.tail)
            eig=np.linalg.eigvalsh((z+z.conj().transpose(0,2,1))/2)
            checks['minimum_RHP_impedance_eigenvalue']=min(checks['minimum_RHP_impedance_eigenvalue'],float(eig.min()))
        all_predictions.append({'exact':target,**pred})
        print(f'{ci+1}/{len(cases)} {name}',flush=True)
    comp=[]
    for i in range(0,len(cases),2):
        a,b=cases[i],cases[i+1]
        truth=exact_scattering(W,np.r_[a[1],b[1]],np.r_[a[2],b[2]],np.r_[a[3],b[3]])
        direct=cascade_scattering(all_predictions[i]['exact'],all_predictions[i+1]['exact'])
        checks['exact_scattering_composition_max_error']=max(checks['exact_scattering_composition_max_error'],float(abs(truth-direct).max()))
        for method in all_predictions[i]:
            if method=='exact':continue
            arr=cascade_scattering(all_predictions[i][method],all_predictions[i+1][method])
            for band,mask in MASKS.items():
                comp.append({'pair':f'{a[0]}+{b[0]}','method':method,'band':band,
                             'frequency_count':int(mask.sum()),**scattering_metrics(arr,truth,mask)})
    # Fixed first random case for realization and time-domain verification.
    a=cases[0];model=truncate_model(build_modal_model(a[1],a[2],a[3],256),16,True)
    tchecks,t,inp,out,energy=checks_and_time(model);checks.update(tchecks)
    np.savez_compressed(HERE/'causal_waveform_check.npz',t=t,incoming=inp,outgoing=out,energy=energy)
    write_csv(HERE/'per_case_results.csv',rows);write_csv(HERE/'composition_results.csv',comp);write_csv(HERE/'cost_results.csv',cost)
    (HERE/'media.json').write_text(json.dumps(jsoncases,indent=2))
    names=list(all_predictions[0]);np.savez_compressed(HERE/'response_data.npz',omega=W,heldout=HELDOUT,
        case_names=np.array([x[0] for x in cases]),method_names=np.array(names),
        responses=np.array([[v[key] for key in names] for v in all_predictions]))
    summary={}
    for label,data in [('single_patch',rows),('two_patch',comp)]:
        summary[label]={}
        for band in MASKS:
            summary[label][band]={}
            for method in names[1:]:
                records=[x for x in data if x['method']==method and x['band']==band]
                vals=np.array([x['rms_complex_S_error'] for x in records])
                summary[label][band][method]={'mean_case_rms_error':float(vals.mean()),'median_case_rms_error':float(np.median(vals)),
                                             'worst_case_rms_error':float(vals.max()),'case_count':len(vals)}
    summary['cost']={}
    for method in sorted({x['method'] for x in cost}):
        records=[x for x in cost if x['method']==method]
        summary['cost'][method]={key:float(np.median([x[key] for x in records])) for key in ['state_dimension','setup_seconds','query_513freq_seconds','one_query_total_seconds','100_query_mean_seconds']}
    summary['checks']=checks;summary['elapsed_seconds']=time.perf_counter()-began
    summary['environment']={'python':platform.python_version(),'numpy':np.__version__,'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
    summary['interpretation']='Exact-geometry development study. The error is the absolute RMS per complex scattering entry.'
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2))
    assert checks['max_unitarity']<1e-8 and checks['max_reciprocity']<1e-8
    assert checks['minimum_RHP_impedance_eigenvalue']>-1e-10
    assert checks['minimum_tail_eigenvalue']>0
    assert checks['exact_scattering_composition_max_error']<1e-10
    assert checks['state_frequency_max_error']<1e-8
    assert checks['max_preinput_output']==0
    assert checks['max_energy_increment_after_source']<1e-12
    print(json.dumps({'elapsed_seconds':summary['elapsed_seconds'],'checks':checks},indent=2))


if __name__=='__main__':run()
