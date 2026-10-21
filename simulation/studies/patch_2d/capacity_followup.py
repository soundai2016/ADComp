#!/usr/bin/env python3
"""Explore local capacity, global eigenmodes and exact source enrichment."""
from run_patch import *

EXPLORATORY={
 'status':'exploratory capacity comparison',
 'grid':32,'materials':CFG['materials'],'modes_per_patch':[32,64,96,128],
 'comparators':['fixed_interface_modes','global_low_eigenmode_oracle','fixed_interface_plus_exact_source_span'],
 'global_oracle':'lowest full-grid eigenmodes, same pressure dimension, individualized full eigensolve',
 'source_span':'adds all three evaluated source vectors; query-aware favorable diagnostic, not source-independent',
 'response_settings':'defined by run_patch.CFG; full same-grid and 48-square references',
 'no_neural_training':True,
}

def main():
 p=HERE/'capacity_followup_config.json'
 if p.exists() and json.loads(p.read_text())!=EXPLORATORY:raise RuntimeError('Configuration differs from existing capacity results')
 dump(p,EXPLORATORY);(HERE/'capacity_followup_config.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  capacity_followup_config.json\n')
 t=np.linspace(0,CFG['time_end'],CFG['samples']);freq=np.linspace(*CFG['frequency_cycles_per_time']);nz=t>0
 prior=np.load(HERE/'receiver_and_transfer_responses.npz');rows=[];raw={}
 for name in EXPLORATORY['materials']:
  n=EXPLORATORY['grid'];X,Y,rho,bulk,mass,K,A=build(n,name)
  lam,V=eig(A);B=port(X,Y,bulk,CFG['source_centres'],True);R=port(X,Y,bulk,CFG['receiver_centres'],False)
  y,u,H=response(lam,V,B,R,t,freq,True);ref=prior[f'{name}_n48_full_y']
  for r in EXPLORATORY['modes_per_patch']:
   t0=time.perf_counter();T0,pieces,_,_,_=component_basis(n,mass,K,r)
   T=np.column_stack([T0]+pieces);W=np.sqrt(mass[:,None])*T;Q,_=linalg.qr(W,mode='economic',check_finite=False)
   buildsec=time.perf_counter()-t0
   bases=[('fixed_interface_modes',Q),('global_low_eigenmode_oracle',V[:,:Q.shape[1]])]
   if r==32:
    Qe,_=linalg.qr(np.column_stack([Q,B]),mode='economic',check_finite=False)
    bases.append(('fixed_interface_plus_exact_source_span',Qe))
   for method,Qa in bases:
    Ar=Qa.T@A@Qa;lr,Vr=eig(Ar);Br=Qa.T@B;Rr=Qa.T@R
    yy,ur,HH=response(lr,Vr,Br,Rr,t,freq,True)
    uu=np.einsum('ir,trs->tis',Qa,ur,optimize=True)
    row={'material':name,'method':method,'modes_per_patch':r,'pressure_dof':Qa.shape[1],
     'same_grid_receiver_error':relative(yy[nz],y[nz]),'same_grid_field_error':relative(uu[nz],u[nz]),
     'same_grid_transfer_error':relative(HH,H),'receiver_error_vs_48':relative(yy[nz],ref[nz]),
     'initial_energy_retained':float(np.linalg.norm(Br)**2/np.linalg.norm(B)**2),
     'receiver_online_seconds':online_time(lr,Vr,Br,Rr,t,freq),
     'component_basis_build_seconds':buildsec if method!='global_low_eigenmode_oracle' else None}
    rows.append(row);raw[f'{name}_{method}_r{r}_y']=yy;raw[f'{name}_{method}_r{r}_H']=HH
    print(name,method,r,Qa.shape[1],row['same_grid_receiver_error'],row['same_grid_field_error'],flush=True)
 with (HERE/'capacity_followup_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(HERE/'capacity_followup_responses.npz',**raw)
 summary=[]
 for method in EXPLORATORY['comparators']:
  for r in EXPLORATORY['modes_per_patch']:
   rr=[x for x in rows if x['method']==method and x['modes_per_patch']==r]
   if not rr:continue
   summary.append({'method':method,'modes_per_patch':r,'pressure_dof':rr[0]['pressure_dof'],**{k:float(np.mean([z[k] for z in rr])) for k in ['same_grid_receiver_error','same_grid_field_error','same_grid_transfer_error','receiver_error_vs_48','initial_energy_retained','receiver_online_seconds']}})
 dump(HERE/'capacity_followup_summary.json',{'status':EXPLORATORY['status'],'summary':summary})
 print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
