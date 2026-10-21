#!/usr/bin/env python3
"""Same-grid a posteriori bound with logged float64 roundoff guards.

The analytic bound is exact. Quadrature checks validate the numerical
evaluation; the guards do not provide an interval enclosure.
"""
import json, csv, time, hashlib
from pathlib import Path
import numpy as np
from scipy import linalg, special
import run_2d as core

HERE=Path(__file__).resolve().parent
CFG={'status':'exploratory development diagnostic',
 'grids':[16,32],'orders':[8,16,32,64,128],
 'materials':core.CFG['materials'],
 'methods':['analytic_cosine','oracle_response_pod','oracle_low_eigenmodes'],
 'times':[1.,2.,4.,8.],
 'adaptive_relative_energy_thresholds':[.1,.2],
 'scope':'same discrete whitened first-order system; no continuum certificate',
 'input_sources':'same three held-out physical source positions as primary development experiment',
 'roundoff_policy':'symmetrize exact sine Gram, clip negative eigenvalues upward; clip negative residual-Gram eigenvalues upward; positive diagonal epsilon guard; retain all guard sizes',
 'zero_mode_policy':'Ritz eigenvalues below 100*eps*r*max(1,lambda_max) are set to zero in proposed ROM; resulting F includes that change, g0(t)=t exactly',
 'quadrature_checks':'smooth_mild, grid16/order16 and grid32/order128, all three methods, T=8',
 'warning':'float64 with checks, not machine-verified interval bounds'}

def gfun(w,t):
 return t[:,None]*np.sinc(t[:,None]*w[None,:]/np.pi)

def integrated_g_gram(w,T):
 """Exact analytic integral of g_i(t)g_j(t), g_i=sin(w_i*t)/w_i."""
 r=len(w);I=np.empty((r,r));z=w==0; nz=~z
 a=w[nz][:,None];b=w[nz][None,:]
 I[np.ix_(nz,nz)]=.5*T*(np.sinc((a-b)*T/np.pi)-np.sinc((a+b)*T/np.pi))/(a*b)
 I[np.ix_(z,z)]=T**3/3
 if np.any(z) and np.any(nz):
  x=w[nz]*T
  q=(np.sin(x)-x*np.cos(x))/w[nz]**3
  I[np.ix_(z,nz)]=q[None,:];I[np.ix_(nz,z)]=q[:,None]
 I=(I+I.T)*.5
 ev,U=linalg.eigh(I,check_finite=False)
 min_ev=float(ev[0]);neg=float(np.maximum(-ev,0).sum())
 # PSD projection raises the quadratic form; log the upward correction.
 Ip=(U*np.maximum(ev,0))@U.T if min_ev<0 else I
 return I,Ip,{'sine_gram_min_eigenvalue':min_ev,'sine_gram_upward_trace_correction':neg,'sine_gram_fro_norm':float(np.linalg.norm(I))}

def write_csv(name,rows):
 with (HERE/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
  writer.writeheader();writer.writerows(rows)

def main():
 core.save_json(HERE/'certificate_config.json',CFG)
 rows=[];guards=[];quad=[];adaptive=[]
 all_start=time.perf_counter()
 for name in CFG['materials']:
  for n in CFG['grids']:
   print(name,n,flush=True)
   X,Y,rho,K,C,A=core.operators(n,name);lf,Vf=core.modes(A);wf=np.sqrt(lf)
   B=core.ports(X,Y,K,core.CFG['test_source_centres'],True)
   R=core.ports(X,Y,K,core.CFG['receiver_centres'],False)
   Bt=core.ports(X,Y,K,core.CFG['train_source_centres'],True)
   _,snap,_=core.response(lf,Vf,Bt,R,np.linspace(0,4,161),np.array([.1]),.2,True)
   Upod,_,_=linalg.svd(snap.transpose(1,0,2).reshape(n*n,-1),full_matrices=False,check_finite=False)
   coeff=Vf.T@B
   full={}
   for T in CFG['times']:
    pf=Vf@(np.cos(wf*T)[:,None]*coeff)
    vf=C.T@(Vf@(gfun(wf,np.array([T]))[0,:,None]*coeff))
    full[T]=(pf,vf)
   for method in CFG['methods']:
    for r in CFG['orders']:
     V=core.fixed_basis(n,K,r) if method=='analytic_cosine' else (Upod[:,:r] if method=='oracle_response_pod' else Vf[:,:r])
     Cr=C.T@V;Kr=Cr.T@Cr;lam,U=core.modes(Kr)
     zerotol=100*np.finfo(float).eps*r*max(1.,float(lam[-1]))
     clipped=lam<zerotol;largest_clipped=float(lam[clipped].max(initial=0))
     lam[clipped]=0;w=np.sqrt(lam);W=V@U;CW=C.T@W;a=W.T@B
     stamp=time.perf_counter();F=C@CW-W*lam[None,:];G=F.T@F;certificate_precompute=time.perf_counter()-stamp
     p0=W@a;e0=np.linalg.norm(B-p0,axis=0);bnorm=np.linalg.norm(B,axis=0)
     initial_orthogonal_error=np.max(np.abs(W.T@(B-p0)))
     for T in CFG['times']:
      stamp=time.perf_counter();I,Ip,glog=integrated_g_gram(w,T)
      E=(G*Ip+G.T*Ip.T)/2
      ee,EU=linalg.eigh(E,check_finite=False);emin=float(ee[0]);eneg=float(np.maximum(-ee,0).sum())
      # The Schur product is PSD; clip roundoff-negative modes and add diagonal padding.
      E=(EU*np.maximum(ee,0))@EU.T if emin<0 else E
      padding=64*np.finfo(float).eps*r*max(float(np.linalg.norm(E,2)),np.finfo(float).tiny)
      E=E+padding*np.eye(r)
      integral=np.einsum('is,ij,js->s',a,E,a,optimize=True)
      rawintegral=np.einsum('is,ij,js->s',a,G*I,a,optimize=True)
      if np.min(integral)<0: raise RuntimeError('PSD guard failed')
      bound=e0+np.sqrt(T*integral)
      # Keep the evaluation guard separate from the analytic bound.
      bound_guarded=bound+1e-11*bnorm
      evaltime=time.perf_counter()-stamp
      pr=W@(np.cos(w*T)[:,None]*a)
      vr=CW@(gfun(w,np.array([T]))[0,:,None]*a)
      pf,vf=full[T]
      actual=np.sqrt(np.sum((pf-pr)**2,axis=0)+np.sum((vf-vr)**2,axis=0))
      # Zero-mode clipping can change the reconstructed norm; evaluate it directly.
      triangle=bnorm+np.sqrt(np.sum(pr**2,axis=0)+np.sum(vr**2,axis=0))
      gid=f'{name}_n{n}_{method}_r{r}_T{T:g}'
      guards.append({'id':gid,**glog,'residual_gram_min_eigenvalue':emin,'residual_gram_upward_trace_correction':eneg,
       'residual_gram_diagonal_padding':padding,'zero_mode_threshold':zerotol,'zero_modes':int(np.sum(clipped)),
       'largest_clipped_ritz_eigenvalue':largest_clipped,'initial_projection_orthogonality_defect':float(initial_orthogonal_error),
       'min_raw_integrated_residual':float(rawintegral.min()),'max_integral_guard_increment':float(np.max(integral-rawintegral))})
      for j in range(B.shape[1]):
       rows.append({'material':name,'grid':n,'method':method,'order':r,'source':j,'time':T,
        'actual_relative_full_energy_error':float(actual[j]/bnorm[j]),
        'initial_relative_energy_error':float(e0[j]/bnorm[j]),
        'integrated_defect_squared':float(integral[j]),
        'residual_certificate_relative':float(bound[j]/bnorm[j]),
        'guarded_certificate_relative':float(bound_guarded[j]/bnorm[j]),
        'guarded_certificate_with_triangle_cap':float(min(bound_guarded[j],triangle[j]+1e-11*bnorm[j])/bnorm[j]),
        'certificate_minus_actual':float((bound_guarded[j]-actual[j])/bnorm[j]),
        'certificate_over_actual':float(bound_guarded[j]/max(actual[j],1e-30)),
        'certificate_precompute_seconds':certificate_precompute,'certificate_evaluate_three_sources_seconds':evaltime,
        'squared_defect_fro_norm':float(np.linalg.norm(F)**2),
        'certificate_pass':bool(actual[j]<=bound_guarded[j]),
        'roundoff_guard_relative':1e-11})
      if name=='smooth_mild' and (n,r) in [(16,16),(32,128)] and T==8:
       nq=max(256,int(np.ceil(2*max(w)*T))+64)
       x,qw=special.roots_legendre(nq);tq=(x+1)*T/2;qw=qw*T/2
       gq=gfun(w,tq);Iq=gq.T@(qw[:,None]*gq)
       iq=np.einsum('is,ij,js->s',a,G*Iq,a,optimize=True)
       rel=np.max(np.abs(rawintegral-iq)/np.maximum(np.abs(iq),1e-30))
       quad.append({'id':gid,'quadrature_order':nq,'sine_gram_relative_fro_error':float(np.linalg.norm(I-Iq)/np.linalg.norm(Iq)),
        'max_relative_integrated_defect_difference':float(rel),'max_absolute_integrated_defect_difference':float(np.max(np.abs(rawintegral-iq)))})
   # Simulate stopping from certificate values only; reference errors are audit outputs.
   for method in CFG['methods']:
    for threshold in CFG['adaptive_relative_energy_thresholds']:
     for j in range(3):
      rr=[z for z in rows if z['material']==name and z['grid']==n and z['method']==method and z['source']==j and z['time']==8]
      eligible=[z for z in rr if z['guarded_certificate_relative']<=threshold]
      selected=min(eligible,key=lambda z:z['order']) if eligible else None
      adaptive.append({'material':name,'grid':n,'method':method,'source':j,'relative_energy_threshold':threshold,
       'selected_order':selected['order'] if selected else None,'certified_in_tested_orders':selected is not None,
       'selected_certificate':selected['guarded_certificate_relative'] if selected else None,
       'selected_actual_error_for_audit_only':selected['actual_relative_full_energy_error'] if selected else None})
 write_csv('energy_certificate_results.csv',rows);write_csv('energy_certificate_adaptive.csv',adaptive)
 core.save_json(HERE/'energy_certificate_guards.json',guards);core.save_json(HERE/'energy_certificate_quadrature.json',quad)
 groups=[]
 for method in CFG['methods']:
  for n in CFG['grids']:
   for r in CFG['orders']:
    rr=[z for z in rows if z['method']==method and z['grid']==n and z['order']==r and z['time']==8]
    groups.append({'method':method,'grid':n,'order':r,'cases':len(rr),
     'mean_actual_relative_full_energy_error':float(np.mean([z['actual_relative_full_energy_error'] for z in rr])),
     'mean_certificate_relative':float(np.mean([z['guarded_certificate_relative'] for z in rr])),
     'max_certificate_relative':float(np.max([z['guarded_certificate_relative'] for z in rr])),
     'mean_certificate_with_triangle_cap':float(np.mean([z['guarded_certificate_with_triangle_cap'] for z in rr]))})
 summary={'status':CFG['status'],'mathematical_scope':CFG['scope'],'rows':len(rows),'model_combinations':len(rows)//12,
  'all_computed_bounds_hold':all(z['certificate_pass'] for z in rows),'minimum_guarded_bound_minus_error':min(z['certificate_minus_actual'] for z in rows),
  'max_quadrature_relative_gram_error':max(z['sine_gram_relative_fro_error'] for z in quad),
  'max_quadrature_relative_integrated_defect_error':max(z['max_relative_integrated_defect_difference'] for z in quad),
  'wall_seconds':time.perf_counter()-all_start,'groups':groups,
  'adaptive_counts':[{'method':m,'threshold':th,'accepted':sum(z['certified_in_tested_orders'] for z in adaptive if z['method']==m and z['relative_energy_threshold']==th),'total':18} for m in CFG['methods'] for th in CFG['adaptive_relative_energy_thresholds']]}
 core.save_json(HERE/'energy_certificate_summary.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2),flush=True)

if __name__=='__main__':main()
