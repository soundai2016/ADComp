#!/usr/bin/env python3
"""Independent small-grid algebra checks for patch congruence/assembly."""
from run_patch import *

def main():
 results=[]
 for name in CFG['materials']:
  n=8;X,Y,rho,bulk,mass,K,A=build(n,name)
  T0,_,_,bb,interiors=component_basis(n,mass,K,1)
  blocks=[T0]
  for ii in interiors:
   vals,vec=linalg.eigh(K[np.ix_(ii,ii)],np.diag(mass[ii]),check_finite=False)
   block=np.zeros((n*n,len(ii)));block[ii]=vec;blocks.append(block)
  T=np.column_stack(blocks);Mr=T.T@(mass[:,None]*T);Kr=T.T@K@T
  vals,vec=linalg.eigh(Kr,Mr,check_finite=False); vals=np.maximum(vals,0);vals[vals<1e-10]=0
  V=np.sqrt(mass[:,None])*(T@vec)
  lf,Vf=eig(A);B=port(X,Y,bulk,CFG['source_centres'],True);R=port(X,Y,bulk,CFG['receiver_centres'],False)
  t=np.linspace(0,8,97);freq=np.linspace(.1,4.5,37)
  ya,ua,Ha=response(vals,V,B,R,t,freq,True);yf,uf,Hf=response(lf,Vf,B,R,t,freq,True)
  ii=np.concatenate(interiors)
  Schur=K[np.ix_(bb,bb)]-K[np.ix_(bb,ii)]@linalg.solve(K[np.ix_(ii,ii)],K[np.ix_(ii,bb)],assume_a='pos')
  static=T0.T@K@T0
  # Dynamic elimination uses the *full* transformed mass, including cross blocks.
  source=np.random.default_rng(171717).normal(size=(n*n,2))
  sb=T.T@source; errs=[]
  for omega in [0.3,3.,9.]:
   s=.2+1j*omega;D=K+s*s*np.diag(mass);Dr=Kr+s*s*Mr
   p=linalg.solve(D,source);pr=T@linalg.solve(Dr,sb)
   errs.append(relative(pr,p))
  row={'material':name,'full_dimension':n*n,'transformed_dimension':T.shape[1],
    'full_basis_eigenvalue_relative_error':relative(vals,lf),
    'full_basis_receiver_relative_error':relative(ya,yf),'full_basis_field_relative_error':relative(ua,uf),
    'full_basis_transfer_relative_error':relative(Ha,Hf),'static_schur_relative_error':relative(static,Schur),
    'dynamic_system_equivalence_max_relative_error':max(errs),
    'interface_indices_unique':len(set(bb.tolist()))==len(bb),
    'mass_orthonormality_max_error':float(np.max(np.abs(V.T@V-np.eye(n*n))))}
  row['pass']=all(row[k]<1e-9 for k in ['full_basis_eigenvalue_relative_error','full_basis_receiver_relative_error','full_basis_field_relative_error','full_basis_transfer_relative_error','static_schur_relative_error','dynamic_system_equivalence_max_relative_error','mass_orthonormality_max_error']) and row['interface_indices_unique']
  results.append(row)
 dump(HERE/'independent_algebra_audit.json',{'all_pass':all(x['pass'] for x in results),'cases':results})
 print(json.dumps(results,indent=2))

if __name__=='__main__':main()
