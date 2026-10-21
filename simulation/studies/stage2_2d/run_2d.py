#!/usr/bin/env python3
"""Passive projections of two-dimensional cavity acoustics.

Response-POD bases use full-solver trajectories for each medium.
The cosine basis is defined analytically in physical coordinates.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import json, time, hashlib, csv
from pathlib import Path
import numpy as np
from scipy import linalg, sparse

HERE=Path(__file__).resolve().parent
CFG={
 'experiment':'cavity_projection','seed':170216,
 'materials':['smooth_mild','smooth_contrast','smooth_inclusion'],
 'grids':[16,24,32],'reference_grid':48,'audit_grid':64,
 'orders':[8,16,32],'coarse_grids':[3,4,6],
 'train_source_centres':[[.24,.31],[.73,.64]],
 'test_source_centres':[[.19,.76],[.54,.47],[.81,.22]],
 'receiver_centres':[[.22,.21],[.79,.77],[.28,.69],[.68,.33]],
 'physical_gaussian_sigma':.085,
 'train_time_end':4.0,'train_samples':161,
 'test_time_end':8.0,'test_samples':641,
 'laplace_eta':.2,'frequency_cycles_per_time':[.1,4.5,121],
 'boundary':'rigid walls, normal velocity exactly zero',
 'clock':'exact semidiscrete modal propagation (no time stepping error)',
 'split':'new development media only; not the root final test',
 'material_input':'analytic medium sampled independently on every propagation grid',
 'precision':'float64','threads':2,
}

def save_json(p,x):
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def material(n,name):
 x=(np.arange(n)+.5)/n; X,Y=np.meshgrid(x,x,indexing='ij')
 if name=='smooth_mild':
  rho=np.exp(.24*np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)+.09*np.sin(4*np.pi*X+2*np.pi*Y))
  K=np.exp(.32*np.sin(2*np.pi*X)*np.sin(2*np.pi*Y)+.1*np.cos(2*np.pi*X-4*np.pi*Y))
 elif name=='smooth_contrast':
  rho=np.exp(.65*np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)+.2*np.sin(4*np.pi*X+2*np.pi*Y))
  K=np.exp(-.72*np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)+.2*np.cos(2*np.pi*X-4*np.pi*Y))
 else:
  g=np.exp(-((X-.57)**2+(Y-.43)**2)/(2*.13**2))
  rho=np.exp(1.1*g+.18*np.cos(2*np.pi*X+2*np.pi*Y))
  K=np.exp(-.8*g+.22*np.sin(2*np.pi*X)*np.cos(2*np.pi*Y))
 return X,Y,rho,K

def operators(n,name):
 X,Y,rho,K=material(n,name)
 D,C,A,G=operators_from_arrays(rho,K)
 return X,Y,rho,K,C,A

def operators_from_arrays(rho,K):
 """Return D,C,A,G. beta*p_tt=-G*p, A=sqrtK G sqrtK.

 Physical mass/stiffness with quadrature are h²*diag(1/K), h²*G.
 For static-condensation labels use this same h² convention on both sides.
 """
 n=rho.shape[0]; h=1/n
 # Columns are internal normal-velocity faces; zero wall flux is eliminated.
 row=[]; col=[]; val=[]; rf=[]; j=0
 for axis in (0,1):
  for a in range(n-(axis==0)):
   for b in range(n-(axis==1)):
    c=a*n+b; d=(a+(axis==0))*n+b+(axis==1)
    row += [c,d]; col += [j,j]; val += [1/h,-1/h]
    rf.append(.5*(rho.ravel()[c]+rho.ravel()[d])); j+=1
 D=sparse.coo_matrix((val,(row,col)),shape=(n*n,j)).tocsr()
 C=sparse.diags(np.sqrt(K.ravel()))@D@sparse.diags(1/np.sqrt(rf))
 A=(C@C.T).toarray()
 G=D@sparse.diags(1/np.array(rf))@D.T
 return D,C,A,G

def ports(X,Y,K,centres,remove_mean=False):
 # beta*p_t + div(v) = f(x)*a(t); u_p=h*sqrt(beta)*p.
 # b=h*sqrt(K)*f; collocated output b^T*u = integral f*p.
 h=1/X.shape[0]; cols=[]
 for x,y in centres:
  f=np.exp(-((X-x)**2+(Y-y)**2)/(2*CFG['physical_gaussian_sigma']**2))
  f/=np.sum(f)*h*h
  if remove_mean: f=f-np.mean(f)
  cols.append(h*np.sqrt(K.ravel())*f.ravel())
 return np.stack(cols,axis=1)

def modes(A):
 lam,V=linalg.eigh(A,driver='evd',check_finite=False)
 # The constant-pressure mode is exactly zero analytically; roundoff is clipped.
 tol=1e-11*max(float(lam[-1]),1.)
 if lam[0]<-tol: raise RuntimeError('negative stiffness')
 lam=np.maximum(lam,0)
 lam[lam<1e-12]=0
 return lam,V

def response(lam,V,B,R,t,freq,eta,field=True):
 om=np.sqrt(lam); BB=V.T@B; RR=R.T@V
 ct=np.cos(t[:,None]*om[None,:])
 yp=np.einsum('rm,tm,ms->trs',RR,ct,BB,optimize=True)
 u=np.einsum('im,tm,ms->tis',V,ct,BB,optimize=True) if field else None
 s=eta+2j*np.pi*freq
 factor=s[:,None]/(s[:,None]**2+lam[None,:])
 H=np.einsum('rm,fm,ms->frs',RR,factor,BB,optimize=True)
 return yp,u,H

def relative(a,b):
 return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def fixed_basis(n,K,r):
 x=(np.arange(n)+.5)/n; X,Y=np.meshgrid(x,x,indexing='ij')
 edge=max(10,2*int(np.ceil(np.sqrt(r)))+2)
 pairs=sorted([(a,b) for a in range(edge) for b in range(edge)],key=lambda ab:(ab[0]**2+ab[1]**2,ab[0],ab[1]))[:r]
 P=np.stack([(np.cos(np.pi*a*X)*np.cos(np.pi*b*Y)/np.sqrt(K)).ravel()/n for a,b in pairs],axis=1)
 return linalg.qr(P,mode='economic',check_finite=False)[0]

def summarize_rows(rows):
 groups={}
 for row in rows:
  if row['record_type']!='projection': continue
  key=(row['method'],row['grid'],row['order'])
  groups.setdefault(key,[]).append(row)
 out=[]
 for (m,n,r),rr in groups.items():
  out.append({'method':m,'grid':n,'order':r,'cases':len(rr),
   'mean_field_error':float(np.mean([z['field_relative_l2'] for z in rr])),
   'mean_receiver_error':float(np.mean([z['receiver_relative_l2'] for z in rr])),
   'max_receiver_error':float(np.max([z['receiver_relative_l2'] for z in rr])),
   'mean_transfer_error':float(np.mean([z['transfer_relative_l2'] for z in rr])),
   'mean_setup_seconds':float(np.mean([z['setup_seconds'] for z in rr])),
   'mean_receiver_online_seconds':float(np.mean([z['receiver_online_seconds'] for z in rr]))})
 return out

def main():
 HERE.mkdir(exist_ok=True)
 cfgpath=HERE/'frozen_config.json'
 if cfgpath.exists() and json.loads(cfgpath.read_text())!=CFG:
  raise RuntimeError('Configuration differs from existing results')
 save_json(cfgpath,CFG)
 (HERE/'config.sha256').write_text(hashlib.sha256(cfgpath.read_bytes()).hexdigest()+'  frozen_config.json\n')
 t=np.linspace(0,CFG['test_time_end'],CFG['test_samples'])
 tt=np.linspace(0,CFG['train_time_end'],CFG['train_samples'])
 freq=np.linspace(*CFG['frequency_cycles_per_time'])
 rows=[]; checks=[]; curves={}; captures=[]
 start_all=time.perf_counter()
 for name in CFG['materials']:
  print('MATERIAL',name,flush=True)
  full={}
  for n in sorted(set(CFG['grids']+[CFG['reference_grid'],CFG['audit_grid']]+CFG['coarse_grids'])):
   stamp=time.perf_counter(); X,Y,rho,K,C,A=operators(n,name); assembly=time.perf_counter()-stamp
   stamp=time.perf_counter(); lam,V=modes(A); eigtime=time.perf_counter()-stamp
   B=ports(X,Y,K,CFG['test_source_centres'],True)
   R=ports(X,Y,K,CFG['receiver_centres'],False)
   stamp=time.perf_counter(); y,u,H=response(lam,V,B,R,t,freq,CFG['laplace_eta'],field=n in CFG['grids']); online=time.perf_counter()-stamp
   # Dedicated receiver-only solve, no field reconstruction in this timing.
   ts=[]
   for _ in range(3):
    stamp=time.perf_counter(); response(lam,V,B,R,t,freq,CFG['laplace_eta'],field=False); ts.append(time.perf_counter()-stamp)
   recvtime=float(np.median(ts))
   full[n]={'y':y,'H':H,'setup':assembly+eigtime,'recvtime':recvtime}
   if n in CFG['grids']:
    print('  GRID',n,'eigh',eigtime,flush=True)
    # Oracle POD uses full trajectories of two *other* sources, t<=4.
    Bt=ports(X,Y,K,CFG['train_source_centres'],True)
    stamp=time.perf_counter()
    _,snap,_=response(lam,V,Bt,R,tt,freq[:1],CFG['laplace_eta'],field=True)
    snap=snap.transpose(1,0,2).reshape(n*n,-1)
    U,svals,_=linalg.svd(snap,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    podextra=time.perf_counter()-stamp
    for method in ['analytic_cosine','oracle_response_pod']:
     for r in CFG['orders']:
      stamp=time.perf_counter()
      W=fixed_basis(n,K,r) if method=='analytic_cosine' else U[:,:r]
      CtW=C.T@W; Ar=CtW.T@CtW
      lr,Vr=modes(Ar); Br=W.T@B; Rr=W.T@R
      buildtime=time.perf_counter()-stamp
      yy,ur,HH=response(lr,Vr,Br,Rr,t,freq,CFG['laplace_eta'],field=True)
      uu=np.einsum('ir,trs->tis',W,ur,optimize=True)
      times=[]
      for _ in range(5):
       stamp=time.perf_counter(); response(lr,Vr,Br,Rr,t,freq,CFG['laplace_eta'],field=False); times.append(time.perf_counter()-stamp)
      romtime=float(np.median(times))
      # Exact first-order passive realization: off-diagonal sqrt(Ar).
      sqrtA=(Vr*np.sqrt(lr))@Vr.T
      Lr=np.block([[np.zeros((r,r)),-sqrtA],[sqrtA,np.zeros((r,r))]])
      skew=float(np.max(np.abs(Lr+Lr.T)))
      coeff=Vr.T@Br
      energies=np.sum((np.cos(t[:,None,None]*np.sqrt(lr)[None,:,None])*coeff[None,:,:])**2+(np.sin(t[:,None,None]*np.sqrt(lr)[None,:,None])*coeff[None,:,:])**2,axis=1)
      edrift=float(np.max(np.abs(energies/energies[0]-1)))
      abc=np.array([.7,-.3,.6]); ycombo,_,_=response(lr,Vr,Br@abc[:,None],Rr,t,freq,CFG['laplace_eta'],False)
      linearity=float(np.max(np.abs(ycombo[:,:,0]-yy@abc)))
      # Collocated shifted response must be positive-real, eta>0.
      _,_,Hcol=response(lr,Vr,Br,Br,t[:1],freq,CFG['laplace_eta'],False)
      minreal=float(min(linalg.eigvalsh((h+h.conj().T)/2)[0] for h in Hcol))
      checks.append({'material':name,'grid':n,'method':method,'order':r,'skew_defect':skew,'energy_drift':edrift,'linearity_absolute':linearity,'positive_real_min_eigenvalue':minreal})
      phase=np.angle(HH*np.conj(H)); weight=np.abs(H)**2
      row={'record_type':'projection','material':name,'grid':n,'method':method,'order':r,'active_first_order_dof':2*r,
       'reference':'same_grid_full','field_relative_l2':relative(uu[1:],u[1:]),'receiver_relative_l2':relative(yy[1:],y[1:]),
       'transfer_relative_l2':relative(HH,H),'weighted_phase_error_rad':float(np.sum(weight*np.abs(phase))/np.sum(weight)),
       'initial_energy_retained':float(np.linalg.norm(Br)**2/np.linalg.norm(B)**2),
       'setup_seconds':assembly+buildtime+(eigtime+podextra if method=='oracle_response_pod' else 0),
       'receiver_online_seconds':romtime,'full_setup_seconds':assembly+eigtime,'full_receiver_online_seconds':recvtime,
       'basis_fit_extra_seconds':podextra if method=='oracle_response_pod' else 0,
       'same_medium_1query_seconds':assembly+buildtime+(eigtime+podextra if method=='oracle_response_pod' else 0)+romtime,
       'same_medium_100query_mean_seconds':(assembly+buildtime+(eigtime+podextra if method=='oracle_response_pod' else 0))/100+romtime}
      rows.append(row)
      curves[f'{name}_n{n}_{method}_r{r}']=yy
      if n==32 and r==32:
       captures.append((name,method,yy,H,HH))
    curves[f'{name}_n{n}_full']=y
   print('  n',n,'setup',round(assembly+eigtime,3),'receiver',round(recvtime,4),flush=True)
  ref=full[CFG['reference_grid']]
  aud=full[CFG['audit_grid']]
  rows.append({'record_type':'reference_audit','material':name,'grid':CFG['reference_grid'],'method':'full_fv','order':0,'reference':f"full_fv_{CFG['audit_grid']}",
   'receiver_relative_l2':relative(ref['y'][1:],aud['y'][1:]),'transfer_relative_l2':relative(ref['H'],aud['H']),
   'setup_seconds':ref['setup'],'receiver_online_seconds':ref['recvtime']})
  for n in CFG['grids']+CFG['coarse_grids']:
   row={'record_type':'discretization','material':name,'grid':n,'method':'full_fv','order':n*n,
    'active_first_order_dof':2*n*n-1,'stored_first_order_dof':3*n*n-2*n,'reference':f"full_fv_{CFG['reference_grid']}",
    'receiver_relative_l2':relative(full[n]['y'][1:],ref['y'][1:]),'transfer_relative_l2':relative(full[n]['H'],ref['H']),
    'setup_seconds':full[n]['setup'],'receiver_online_seconds':full[n]['recvtime']}
   rows.append(row)
  for row in rows:
   if row.get('material')==name and row['record_type']=='projection':
    key=f"{name}_n{row['grid']}_{row['method']}_r{row['order']}"
    row['receiver_error_vs_48']=relative(curves[key][1:],ref['y'][1:])
    cg={8:3,16:4,32:6}[row['order']]
    row['near_matched_coarse_grid']=cg
    row['near_matched_coarse_pressure_dof']=cg*cg
    row['near_matched_coarse_receiver_error_vs_48']=relative(full[cg]['y'][1:],ref['y'][1:])
  curves[f'{name}_n48_full']=ref['y'];curves[f'{name}_n64_audit']=aud['y']
  keys=sorted(set().union(*(r.keys() for r in rows)))
  with (HERE/'all_results.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
  save_json(HERE/'structural_checks.json',checks)
 np.savez_compressed(HERE/'receiver_traces.npz',time=t,frequency=freq,**curves)
 summaries=summarize_rows(rows); save_json(HERE/'summary.json',{'config_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-start_all,'groups':summaries,'reference_audits':[r for r in rows if r['record_type']=='reference_audit'],'all_checks_pass':all(c['skew_defect']<1e-12 and c['energy_drift']<1e-12 and c['linearity_absolute']<1e-10 and c['positive_real_min_eigenvalue']>-1e-10 for c in checks)})

if __name__=='__main__': main()
