#!/usr/bin/env python3
"""Frequency and coupling contributions to acoustic reduction error."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from acoustic_core import build_sparse,ports,CONFIG as CORE
CFG=json.loads((HERE/'config.json').read_text())
OUT=HERE/'results'

def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def savecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def spectral_problem(n,name,complete=False):
    cache=OUT/f'{name}_n{n}_spectrum.npz'
    X,Y,rho,bulk,mass,C,J=build_sparse(n,name)
    A=(C@C.T).tocsr()
    B=ports(X,Y,bulk,CORE['source_centres'],True)
    R=ports(X,Y,bulk,CORE['receiver_centres'],False)
    if cache.exists() and zipfile.is_zipfile(cache):
        z=np.load(cache);lam=z['eigenvalues'];V=z['eigenvectors']
    else:
        if complete:
            lam,V=linalg.eigh(A.toarray(),check_finite=False,driver='evd')
        else:
            lam,V=eigsh(A,k=CFG['fine_modes'],sigma=-1e-7,which='LM',
                        tol=CFG['eigen_tolerance'],
                        v0=np.random.default_rng(673).normal(size=n*n))
            order=np.argsort(lam);lam=lam[order];V=V[:,order]
        lam[lam<1e-9]=0
        temporary=cache.with_name(cache.stem+'.tmp.npz')
        np.savez_compressed(temporary,eigenvalues=lam,eigenvectors=V,mass=mass,
                            source_coupling=V.T@B,receiver_coupling=R.T@V)
        temporary.replace(cache)
    eig_res=np.linalg.norm(A@V-V*lam[None,:],axis=0)/np.maximum(lam,1)
    return {'omega':np.sqrt(lam),'V':V,'b':V.T@B,'c':R.T@V,'mass':mass,
            'A':A,'B':B,'R':R,'eigen_residual':float(np.max(eig_res))}

def residue(b,c):
    return np.einsum('oj,js->jos',c,b)

def response(t,omega,res):
    return np.einsum('tj,jos->tos',np.cos(t[:,None]*omega[None,:]),res,optimize=True)

def gap(omega):
    left=np.r_[np.inf,np.diff(omega)]
    right=np.r_[np.diff(omega),np.inf]
    return np.minimum(left,right)/np.maximum(omega,1e-100)

def restriction(V,mass_f,mass_c,nf,nc):
    ratio=nf//nc
    physical=V/np.sqrt(mass_f[:,None])
    blocks=physical.reshape(nc,ratio,nc,ratio,V.shape[1]).mean(axis=(1,3))
    W=blocks.reshape(nc*nc,-1)*np.sqrt(mass_c[:,None])
    norms=np.linalg.norm(W,axis=0)
    return W/np.maximum(norms,1e-300),norms

def match(left,right,wleft,wright):
    overlap=left.T@right
    il,ir=linear_sum_assignment(-overlap**2)
    mac=overlap[il,ir]**2
    accepted=(mac>=CFG['mac_threshold'])&(gap(wleft)[il]>=CFG['relative_frequency_gap'])&(gap(wright)[ir]>=CFG['relative_frequency_gap'])
    accepted&=(wleft[il]>1e-7)&(wright[ir]>1e-7)
    return il,ir,mac,np.sign(overlap[il,ir]),accepted

def cluster_angles(left,right,wleft,wright,assignment):
    # Compare near-degenerate clusters as subspaces; exclude scalar shifts.
    groups=[];start=1
    for j in range(2,len(wleft)+1):
        split=j==len(wleft) or (wleft[j]-wleft[j-1])/max(wleft[j],1e-100)>=CFG['near_cluster_tolerance']
        if split:
            if j-start>1: groups.append(np.arange(start,j))
            start=j
    rows=[]
    for ids in groups:
        center=float(np.mean(wleft[ids]))
        if any(int(j) not in assignment for j in ids):
            continue
        candidates=np.array([assignment[int(j)] for j in ids],dtype=int)
        rightq,_=linalg.qr(right[:,candidates],mode='economic')
        values=linalg.svdvals(left[:,ids].T@rightq)
        rows.append({'start_mode':int(ids[0]),'end_mode':int(ids[-1]),'dimension':len(ids),
                     'comparison_dimension':len(candidates),'frequency_center':center,
                     'smallest_principal_cosine':float(values.min()),
                     'largest_principal_angle_degrees':float(np.degrees(np.arccos(np.clip(values.min(),0,1))))})
    return rows

def norm(x): return float(np.linalg.norm(x))
def cosine(a,b): return float(np.sum(a*b)/max(norm(a)*norm(b),1e-300))

def process(name):
    n=CFG['coarse_grid'];nf=CFG['fine_grid'];t=np.linspace(0,CFG['time_end'],CFG['samples'])
    coarse=spectral_problem(n,name,True);fine=spectral_problem(nf,name)
    yc=response(t,coarse['omega'],residue(coarse['b'],coarse['c']))
    yf_modes=response(t,fine['omega'],residue(fine['b'],fine['c']))
    yf=np.load(HERE.parent/'propagation'/'results'/f'{name}_n{nf}.npz')['receiver_pressure']
    yref=np.load(HERE.parent/'propagation'/'results'/f'{name}_n256.npz')['receiver_pressure']
    ycdirect=np.load(HERE.parent/'propagation'/'results'/f'{name}_n{n}.npz')['receiver_pressure']
    W,_=restriction(fine['V'],fine['mass'],coarse['mass'],nf,n)
    ic,iff,mac_cf,sign_cf,accept_cf=match(coarse['V'],W,coarse['omega'],fine['omega'])
    coarse_to_fine={int(i):(int(j),float(mac),float(sgn)) for i,j,mac,sgn,ok in zip(ic,iff,mac_cf,sign_cf,accept_cf) if ok}
    fine_coverage=norm(yf_modes-yf)/norm(yf)
    if fine_coverage>CFG['fine_response_coverage_limit']:
        print(f'Coverage criterion not met for {name}: fine modal omission {fine_coverage:.4g}; retain full remainder and do not infer complete frequency attribution.',flush=True)
    coverage={'material':name,'coarse_eigen_residual':coarse['eigen_residual'],
              'fine_eigen_residual':fine['eigen_residual'],
              'coarse_complete_response_error':norm(yc-ycdirect)/norm(ycdirect),
              'fine_truncated_response_error':fine_coverage,
              'fine_response_coverage_pass':fine_coverage<=CFG['fine_response_coverage_limit'],
              'fine_to_256_response_difference':norm(yf-yref)/norm(yref),
              'accepted_coarse_fine_pairs':len(coarse_to_fine),
              'near_cluster_diagnostics':cluster_angles(coarse['V'][:,:CFG['fine_modes']],W,coarse['omega'][:CFG['fine_modes']],fine['omega'],{int(i):int(j) for i,j in zip(ic,iff)})}
    pairrows=[];summary=[];couplingrows=[];components=[]
    np.savez_compressed(OUT/f'{name}_coarse_fine_matching.npz',coarse_indices=ic,fine_indices=iff,mac=mac_cf,sign=sign_cf,accepted=accept_cf)
    for method,m in CFG['models']:
        label=f'{method}_m{m}'
        raw=np.load(HERE.parent/'accuracy_cost'/'results'/f'{name}_n{n}_{label}.npz')
        Q=raw['orthonormal_basis'];Vr=Q@raw['reduced_eigenvectors'];wr=np.sqrt(raw['reduced_eigenvalues'])
        br=Vr.T@coarse['B'];cr=coarse['R'].T@Vr;rr=residue(br,cr)
        yr=response(t,wr,rr);direct=raw['receiver_pressure']
        ir,jc,mac_rc,sign_rc,accept_rc=match(Vr,coarse['V'],wr,coarse['omega'])
        np.savez_compressed(OUT/f'{name}_{label}_matching.npz',rom_indices=ir,coarse_indices=jc,mac=mac_rc,sign=sign_rc,accepted=accept_rc)
        sr=ir[accept_rc];sc=jc[accept_rc];ss=sign_rc[accept_rc]
        rrc=rr[sr];rc=residue(coarse['b'][sc],coarse['c'][:,sc])
        eqfreq=response(t,wr[sr],rrc)-response(t,coarse['omega'][sc],rrc)
        eqres=response(t,coarse['omega'][sc],rrc-rc)
        eqtail=yr-yc-eqfreq-eqres
        # Align mode signs before the exact symmetric source/receiver residue split.
        brsame=br[sr]*ss[:,None];crsame=cr[:,sr]*ss[None,:]
        bcsame=coarse['b'][sc];ccsame=coarse['c'][:,sc]
        eq_source=response(t,coarse['omega'][sc],residue(brsame-bcsame,(crsame+ccsame)/2))
        eq_receiver=response(t,coarse['omega'][sc],residue((brsame+bcsame)/2,crsame-ccsame))
        chain=[(int(l),int(j),coarse_to_fine[int(j)][0],float(mrc),coarse_to_fine[int(j)][1],float(src),coarse_to_fine[int(j)][2])
               for l,j,mrc,src,ok in zip(ir,jc,mac_rc,sign_rc,accept_rc) if ok and int(j) in coarse_to_fine]
        ri=np.array([a[0] for a in chain],dtype=int);ci=np.array([a[1] for a in chain],dtype=int);fi=np.array([a[2] for a in chain],dtype=int)
        # Gauge: ROM and fine modes are both oriented toward the coarse mode.
        srom=np.array([a[5] for a in chain]);sfine=np.array([a[6] for a in chain])
        brr=br[ri]*srom[:,None];crr=cr[:,ri]*srom[None,:]
        bff=fine['b'][fi]*sfine[:,None];cff=fine['c'][:,fi]*sfine[None,:]
        rrom=residue(brr,crr);rfine=residue(bff,cff)
        ritz=response(t,wr[ri],rrom)-response(t,coarse['omega'][ci],rrom)
        grid=response(t,coarse['omega'][ci],rrom)-response(t,fine['omega'][fi],rrom)
        couplings=response(t,fine['omega'][fi],rrom-rfine)
        source=response(t,fine['omega'][fi],residue(brr-bff,(crr+cff)/2))
        receiver=response(t,fine['omega'][fi],residue((brr+bff)/2,crr-cff))
        tail=yr-yf-ritz-grid-couplings
        # Frequency replacements retain modal residues and unmatched responses.
        cf_no_ritz=yr-ritz
        cf_fine_frequency=yr-ritz-grid
        denom=norm(yf)
        wgt=np.linalg.norm(rfine,axis=(1,2));wgtsum=max(float(wgt.sum()),1e-300)
        signed_grid=(coarse['omega'][ci]-fine['omega'][fi])/fine['omega'][fi]
        signed_ritz=(wr[ri]-coarse['omega'][ci])/fine['omega'][fi]
        for k,(l,j,f,mrc,mcf,sr0,sf0) in enumerate(chain):
            pairrows.append({'material':name,'method':method,'m':m,'rank':len(wr),
                'rom_mode':l,'coarse_mode':j,'fine_mode':f,'mac_rom_coarse':mrc,'mac_coarse_fine':mcf,
                'omega_rom':float(wr[l]),'omega_coarse':float(coarse['omega'][j]),'omega_fine':float(fine['omega'][f]),
                'relative_grid_shift':float(signed_grid[k]),'relative_ritz_shift':float(signed_ritz[k]),
                'fine_residue_norm':float(wgt[k]),'coarse_fine_shape_angle_degrees':float(np.degrees(np.arccos(np.sqrt(np.clip(mcf,0,1))))),
                'rom_coarse_shape_angle_degrees':float(np.degrees(np.arccos(np.sqrt(np.clip(mrc,0,1))))),
                'source_coupling_change_norm':norm(brr[k]-bff[k]),'receiver_coupling_change_norm':norm(crr[:,k]-cff[:,k]),
                'fine_source_coupling_norm':norm(bff[k]),'fine_receiver_coupling_norm':norm(cff[:,k])})
        error=yr-yf
        comp=[ritz,grid,source,receiver,tail]
        compnames=['ritz_frequency','grid_frequency','source_coupling','receiver_coupling','unmatched_modes']
        for a,ca in enumerate(comp):
            for b,cb in enumerate(comp):
                components.append({'material':name,'method':method,'m':m,'component_a':compnames[a],
                    'component_b':compnames[b],'normalized_inner_product':float(np.sum(ca*cb)/denom**2)})
        row={'material':name,'method':method,'m':m,'rank':len(wr),'accepted_same_grid_pairs':len(sr),
             'accepted_three_grid_chain':len(chain),'same_grid_error':norm(yr-yc)/norm(yc),
             'same_grid_frequency_component':norm(eqfreq)/norm(yc),
             'same_grid_residue_component':norm(eqres)/norm(yc),
             'same_grid_source_component':norm(eq_source)/norm(yc),
             'same_grid_receiver_component':norm(eq_receiver)/norm(yc),
             'same_grid_unmatched_component':norm(eqtail)/norm(yc),
             'total_error_vs_128':norm(error)/denom,'total_error_vs_256':norm(yr-yref)/norm(yref),
             'ritz_frequency_component':norm(ritz)/denom,'grid_frequency_component':norm(grid)/denom,
             'source_coupling_component':norm(source)/denom,'receiver_coupling_component':norm(receiver)/denom,
             'residue_component':norm(couplings)/denom,'unmatched_component':norm(tail)/denom,
             'ritz_grid_cosine':cosine(ritz,grid),'ritz_grid_inner_product':float(np.sum(ritz*grid)/denom**2),
             'error_without_ritz_shift_vs_128':norm(cf_no_ritz-yf)/denom,
             'error_with_fine_frequencies_vs_128':norm(cf_fine_frequency-yf)/denom,
             'error_without_ritz_shift_vs_256':norm(cf_no_ritz-yref)/norm(yref),
             'error_with_fine_frequencies_vs_256':norm(cf_fine_frequency-yref)/norm(yref),
             'matched_fine_response_omission':norm(yf-response(t,fine['omega'][fi],rfine))/denom,
             'weighted_grid_shift':float(np.sum(wgt*signed_grid)/wgtsum),
             'weighted_ritz_shift':float(np.sum(wgt*signed_ritz)/wgtsum),
             'weighted_net_shift':float(np.sum(wgt*(signed_grid+signed_ritz))/wgtsum),
             'weighted_mac_rom_coarse':float(np.sum(wgt*np.array([a[3] for a in chain]))/wgtsum),
             'weighted_mac_coarse_fine':float(np.sum(wgt*np.array([a[4] for a in chain]))/wgtsum),
             'opposite_frequency_shift_weight_fraction':float(np.sum(wgt[(signed_grid<0)&(signed_ritz>0)])/wgtsum),
             'decomposition_relative_defect':norm(error-sum(comp))/max(norm(error),1e-300),
             'coupling_split_relative_defect':norm(couplings-source-receiver)/max(norm(couplings),1e-300),
             'stored_rom_response_error':norm(yr-direct)/norm(direct),
             'minmax_min_difference':float(np.min(wr-coarse['omega'][:len(wr)])),
             'same_grid_coupling_split_defect':norm(eqres-eq_source-eq_receiver)/max(norm(eqres),1e-300)}
        summary.append(row)
        np.savez_compressed(OUT/f'{name}_{label}_decomposition.npz',time=t,rom_response=yr,
                            coarse_response=yc,fine_response=yf,reference_256_response=yref,
                            ritz_frequency=ritz,grid_frequency=grid,source_coupling=source,
                            receiver_coupling=receiver,unmatched_modes=tail,
                            same_grid_frequency=eqfreq,same_grid_source=eq_source,
                            same_grid_receiver=eq_receiver,same_grid_unmatched=eqtail,
                            no_ritz_frequency_response=cf_no_ritz,fine_frequency_response=cf_fine_frequency)
    savecsv(OUT/f'{name}_modal_pairs.csv',pairrows)
    savecsv(OUT/f'{name}_summary.csv',summary)
    savecsv(OUT/f'{name}_component_gram.csv',components)
    dump(OUT/f'{name}_validation.json',coverage)
    return summary,coverage

def main():
    p=argparse.ArgumentParser();p.add_argument('--materials',nargs='+',default=CFG['materials']);args=p.parse_args()
    if hasattr(os,'sched_getaffinity'):
        available=set(os.sched_getaffinity(0));cpus=available&set(CFG['cpu_affinity'])
        os.sched_setaffinity(0,cpus or set(sorted(available)[:CFG['threads']]))
    OUT.mkdir(exist_ok=True)
    allrows=[];validation=[]
    for name in args.materials:
        print('Computing',name,flush=True)
        rows,checks=process(name);allrows.extend(rows);validation.append(checks)
        print(json.dumps({'material':name,'fine_modal_error':checks['fine_truncated_response_error'],
              'accepted_fine_pairs':checks['accepted_coarse_fine_pairs'],
              'model_summary':[{k:r[k] for k in ['method','m','accepted_three_grid_chain','ritz_grid_cosine','total_error_vs_128','error_without_ritz_shift_vs_128']} for r in rows]}),flush=True)
    savecsv(OUT/'summary.csv',allrows)
    dump(OUT/'validation.json',{'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':validation})

if __name__=='__main__':main()
