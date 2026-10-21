#!/usr/bin/env python3
"""Explore basis capacity and material sampling at a fixed propagation grid."""
import json, time, csv
from pathlib import Path
import numpy as np
from scipy import linalg, interpolate
import run_2d as core

HERE=Path(__file__).resolve().parent
CONFIG={
 'status':'exploratory development diagnostic',
 'grid':32,'orders':[8,16,32,64,128],
 'input_grids':[8,16,24,32],
 'materials':core.CFG['materials'],
 'hypotheses':['Increasing order alone may close the gap',
   'An invariant individualized eigenbasis may outperform pressure POD at equal order',
   'Input coarsening must be separated from propagation-grid changes'],
 'no_new_test_claim':True,
 'source_time_frequency_settings':'defined by run_2d.CFG',
}

def main():
 core.save_json(HERE/'exploratory_config.json',CONFIG)
 n=32;t=np.linspace(0,8,641);tt=np.linspace(0,4,161);freq=np.linspace(.1,4.5,121)
 rows=[];sampling=[]
 for name in core.CFG['materials']:
  print(name,flush=True)
  stamp=time.perf_counter();X,Y,rho,K,C,A=core.operators(n,name);assembly=time.perf_counter()-stamp
  stamp=time.perf_counter();lam,V=core.modes(A);eigentime=time.perf_counter()-stamp
  B=core.ports(X,Y,K,core.CFG['test_source_centres'],True)
  R=core.ports(X,Y,K,core.CFG['receiver_centres'],False)
  y,u,H=core.response(lam,V,B,R,t,freq,.2,True)
  Bt=core.ports(X,Y,K,core.CFG['train_source_centres'],True)
  stamp=time.perf_counter();_,snap,_=core.response(lam,V,Bt,R,tt,freq[:1],.2,True)
  U,_,_=linalg.svd(snap.transpose(1,0,2).reshape(n*n,-1),full_matrices=False,check_finite=False)
  podtime=time.perf_counter()-stamp
  for method in ['analytic_cosine','oracle_response_pod','oracle_low_eigenmodes']:
   for r in CONFIG['orders']:
    stamp=time.perf_counter();W=core.fixed_basis(n,K,r) if method=='analytic_cosine' else (U[:,:r] if method=='oracle_response_pod' else V[:,:r])
    Ar=(C.T@W).T@(C.T@W); lr,Vr=core.modes(Ar)
    buildtime=time.perf_counter()-stamp
    AW=C@(C.T@W)
    invariance=float(np.linalg.norm(AW-W@Ar)/max(np.linalg.norm(AW),1e-30))
    yy,ur,HH=core.response(lr,Vr,W.T@B,W.T@R,t,freq,.2,True)
    uu=np.einsum('ir,trs->tis',W,ur,optimize=True)
    times=[]
    for _ in range(3):
     stamp=time.perf_counter();core.response(lr,Vr,W.T@B,W.T@R,t,freq,.2,False);times.append(time.perf_counter()-stamp)
    rows.append({'material':name,'grid':n,'method':method,'order':r,
      'pressure_field_relative_l2':core.relative(uu[1:],u[1:]),
      'receiver_relative_l2':core.relative(yy[1:],y[1:]),
      'transfer_relative_l2':core.relative(HH,H),
      'initial_pressure_energy_retained':float(np.linalg.norm(W.T@B)**2/np.linalg.norm(B)**2),
      'normalized_stiffness_invariance_defect':invariance,
      'setup_seconds':assembly+buildtime+(eigentime if method.startswith('oracle') else 0)+(podtime if method=='oracle_response_pod' else 0),
      'receiver_online_seconds':float(np.median(times)),
      'status':'development diagnostic; individualized eigenmodes require the full operator'})
  for nin in CONFIG['input_grids']:
   xi,yi,ri,ki=core.material(nin,name);xco=(np.arange(nin)+.5)/nin
   points=np.stack([X.ravel(),Y.ravel()],axis=1)
   lr=interpolate.RegularGridInterpolator((xco,xco),np.log(ri),method='linear',bounds_error=False,fill_value=None)(points).reshape(n,n)
   lk=interpolate.RegularGridInterpolator((xco,xco),np.log(ki),method='linear',bounds_error=False,fill_value=None)(points).reshape(n,n)
   rrec,krec=np.exp(lr),np.exp(lk)
   _,_,Ai,_=core.operators_from_arrays(rrec,krec);li,Vi=core.modes(Ai)
   Bi=core.ports(X,Y,krec,core.CFG['test_source_centres'],True);Ri=core.ports(X,Y,krec,core.CFG['receiver_centres'],False)
   yy,_,HH=core.response(li,Vi,Bi,Ri,t,freq,.2,False)
   sampling.append({'material':name,'input_grid':nin,'propagation_grid':n,
    'reconstruction':'positive log-bilinear interpolation, linear edge extrapolation',
    'rho_relative_l2':core.relative(rrec,rho),'K_relative_l2':core.relative(krec,K),
    'receiver_relative_l2':core.relative(yy[1:],y[1:]),'transfer_relative_l2':core.relative(HH,H)})
 for filename,rr in [('capacity_results.csv',rows),('input_sampling_results.csv',sampling)]:
  with (HERE/filename).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 agg=[]
 for m in ['analytic_cosine','oracle_response_pod','oracle_low_eigenmodes']:
  for r in CONFIG['orders']:
   rr=[x for x in rows if x['method']==m and x['order']==r]
   agg.append({'method':m,'order':r,'mean_receiver_error':float(np.mean([x['receiver_relative_l2'] for x in rr])),
   'max_receiver_error':float(np.max([x['receiver_relative_l2'] for x in rr])),
   'mean_pressure_field_error':float(np.mean([x['pressure_field_relative_l2'] for x in rr])),
   'mean_transfer_error':float(np.mean([x['transfer_relative_l2'] for x in rr]))})
 core.save_json(HERE/'capacity_summary.json',{'status':CONFIG['status'],'groups':agg})
 print(json.dumps(agg,indent=2),flush=True)

if __name__=='__main__':main()
