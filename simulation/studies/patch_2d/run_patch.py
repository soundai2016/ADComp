#!/usr/bin/env python3
"""Local dynamic condensation for heterogeneous cavity acoustics.

All spaces project the same physical mass and stiffness matrices.
Craig--Bampton spaces retain the interface coordinates and fixed-interface modes.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import json, hashlib, time, csv
from pathlib import Path
import numpy as np
from scipy import linalg, sparse

HERE=Path(__file__).resolve().parent
CFG={
 'experiment':'local_dynamic_condensation',
 'status':'development diagnostic with analytic material fields',
 'materials':['rotated_stripes','double_inclusion','smooth_checker'],
 'grids':[16,24,32], 'reference_grid':48, 'audit_grid':64,
 'audit_material':'rotated_stripes', 'internal_modes_per_patch':[0,4,8,16,32],
 'interface':'one exact pressure row x_index=n/2-1; all n nodes retained once',
 'source_centres':[[.19,.76],[.54,.47],[.81,.22]],
 'receiver_centres':[[.22,.21],[.79,.77],[.28,.69],[.68,.33]],
 'source_receiver_sigma':.07, 'time_end':8., 'samples':641,
 'frequency_cycles_per_time':[.1,4.5,121], 'laplace_eta':.2,
 'short_time_end':2., 'boundary':'rigid walls',
 'material_information':'analytic rho and K resampled independently on every grid',
 'clock':'exact semidiscrete modal propagation; initial impulse',
 'coarse_baseline':'full FV on round(sqrt(retained pressure DOF)) cells per side',
 'precision':'float64', 'threads':2,
}

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def material(n,name):
    x=(np.arange(n)+.5)/n; X,Y=np.meshgrid(x,x,indexing='ij')
    if name=='rotated_stripes':
        rho=np.exp(.65*np.sin(4*np.pi*(.8*X+.6*Y))+.1*np.cos(2*np.pi*Y))
        bulk=np.exp(-.45*np.sin(4*np.pi*(.8*X+.6*Y))+.12*np.sin(2*np.pi*X))
    elif name=='double_inclusion':
        a=np.exp(-((X-.34)**2+(Y-.35)**2)/(2*.1**2))
        b=np.exp(-((X-.70)**2+(Y-.67)**2)/(2*.12**2))
        rho=np.exp(1.2*a-.55*b+.1*np.cos(2*np.pi*X))
        bulk=np.exp(-.8*a+.9*b+.1*np.sin(2*np.pi*Y))
    elif name=='smooth_checker':
        z=np.cos(4*np.pi*X)*np.cos(4*np.pi*Y)
        rho=np.exp(.8*z+.1*np.sin(2*np.pi*X))
        bulk=np.exp(-.7*z+.15*np.cos(2*np.pi*Y))
    else: raise KeyError(name)
    return X,Y,rho,bulk

def build(n,name):
    X,Y,rho,bulk=material(n,name); h=1/n
    rr=[];cc=[];vv=[];rf=[];f=0
    for axis in (0,1):
        for i in range(n-(axis==0)):
            for j in range(n-(axis==1)):
                a=i*n+j; b=(i+(axis==0))*n+j+(axis==1)
                rr.extend([a,b]);cc.extend([f,f]);vv.extend([1/h,-1/h])
                rf.append(.5*(rho.ravel()[a]+rho.ravel()[b])); f+=1
    D=sparse.coo_matrix((vv,(rr,cc)),shape=(n*n,f)).tocsr()
    K=(h*h*(D@sparse.diags(1/np.asarray(rf))@D.T)).toarray()
    mass=(h*h/bulk).ravel()
    A=K/np.sqrt(mass[:,None]*mass[None,:])
    return X,Y,rho,bulk,mass,K,A

def port(X,Y,bulk,centres,zero_mean):
    n=X.shape[0]; out=[]
    for x,y in centres:
        f=np.exp(-((X-x)**2+(Y-y)**2)/(2*CFG['source_receiver_sigma']**2))
        f*=n*n/f.sum()
        if zero_mean:f-=f.mean()
        out.append((np.sqrt(bulk)*f/n).ravel())
    return np.array(out).T

def eig(A):
    lam,V=linalg.eigh((A+A.T)*.5,driver='evd',check_finite=False)
    if lam[0]<-1e-10*max(lam[-1],1):raise RuntimeError('negative stiffness')
    lam=np.maximum(lam,0); lam[lam<1e-10]=0
    return lam,V

def response(lam,V,B,R,t,freq,field=False):
    b=V.T@B; r=R.T@V
    ct=np.cos(t[:,None]*np.sqrt(lam)[None,:])
    y=np.einsum('rm,tm,ms->trs',r,ct,b,optimize=True)
    u=np.einsum('im,tm,ms->tis',V,ct,b,optimize=True) if field else None
    s=CFG['laplace_eta']+2j*np.pi*freq
    f=s[:,None]/(s[:,None]**2+lam[None,:])
    H=np.einsum('rm,fm,ms->frs',r,f,b,optimize=True)
    return y,u,H

def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def component_basis(n,mass,K,max_modes):
    """One globally assembled interface: there is no duplicated interface mass.

    T=[I_G; -K_ii^-1 K_iG] plus fixed-interface generalized eigenmodes.
    Each interior block has homogeneous Dirichlet data at the shared interface.
    All outer boundaries retain the original Neumann conditions.
    """
    mid=n//2-1
    interface=np.arange(mid*n,(mid+1)*n)
    interiors=[np.arange(0,mid*n),np.arange((mid+1)*n,n*n)]
    T0=np.zeros((n*n,n)); T0[interface]=np.eye(n)
    pieces=[]; raw_eigs=[]
    for ii in interiors:
        kii=K[np.ix_(ii,ii)]; kib=K[np.ix_(ii,interface)]
        lift=-linalg.solve(kii,kib,assume_a='pos',check_finite=False)
        T0[ii]=lift
        mi=mass[ii]
        ai=kii/np.sqrt(mi[:,None]*mi[None,:])
        vals,vec=linalg.eigh(ai,subset_by_index=[0,max_modes-1],driver='evr',check_finite=False)
        modes=np.zeros((n*n,max_modes));modes[ii]=vec/np.sqrt(mi[:,None])
        pieces.append(modes);raw_eigs.append(vals)
    return T0,pieces,raw_eigs,interface,interiors

def online_time(lam,V,B,R,t,freq):
    response(lam,V,B,R,t,freq)
    times=[]
    for _ in range(3):
        start=time.perf_counter();response(lam,V,B,R,t,freq);times.append(time.perf_counter()-start)
    return float(np.median(times))

def main():
    start_all=time.perf_counter(); cfg=HERE/'frozen_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=CFG:raise RuntimeError('Configuration differs from existing results')
    dump(cfg,CFG);(HERE/'config.sha256').write_text(hashlib.sha256(cfg.read_bytes()).hexdigest()+'  frozen_config.json\n')
    t=np.linspace(0,CFG['time_end'],CFG['samples']);freq=np.linspace(*CFG['frequency_cycles_per_time'])
    short=(t>0)&(t<=CFG['short_time_end']);long=t>0
    rows=[];checks=[];raw={'time':t,'frequency':freq}; fields={}; setup_rows=[]
    for name in CFG['materials']:
        print('MATERIAL',name,flush=True);cache={}
        coarse=sorted(set(int(np.floor(np.sqrt(n+2*r)+.5)) for n in CFG['grids'] for r in CFG['internal_modes_per_patch']))
        ns=sorted(set(CFG['grids']+coarse+[CFG['reference_grid']]+([CFG['audit_grid']] if name==CFG['audit_material'] else [])))
        for n in ns:
            t0=time.perf_counter();X,Y,rho,bulk,mass,K,A=build(n,name);assembly=time.perf_counter()-t0
            t0=time.perf_counter();lam,V=eig(A);esec=time.perf_counter()-t0
            B=port(X,Y,bulk,CFG['source_centres'],True);R=port(X,Y,bulk,CFG['receiver_centres'],False)
            y,u,H=response(lam,V,B,R,t,freq,field=n in CFG['grids'])
            on=online_time(lam,V,B,R,t,freq)
            cache[n]={'y':y,'H':H,'setup':assembly+esec,'online':on}
            raw[f'{name}_n{n}_full_y']=y;raw[f'{name}_n{n}_full_H']=H
            if n in CFG['grids']:
                fields[f'{name}_n{n}_full_snapshots']=u[[0,160,320,480,640]]
                raw[f'{name}_n{n}_rho']=rho;raw[f'{name}_n{n}_bulk']=bulk
                t0=time.perf_counter();T0,pieces,evals,interface,interiors=component_basis(n,mass,K,max(CFG['internal_modes_per_patch']));basis_build=time.perf_counter()-t0
                constant_error=float(np.max(np.abs(T0@np.ones(n)-1)))
                interior_disconnect=float(np.max(np.abs(K[np.ix_(interiors[0],interiors[1])]),initial=0))
                for r in CFG['internal_modes_per_patch']:
                    t0=time.perf_counter()
                    T=np.column_stack([T0]+[p[:,:r] for p in pieces])
                    W=np.sqrt(mass[:,None])*T
                    Q,_=linalg.qr(W,mode='economic',check_finite=False)
                    Ar=Q.T@A@Q;lr,Vr=eig(Ar);Br=Q.T@B;Rr=Q.T@R
                    extra=time.perf_counter()-t0
                    yy,ur,HH=response(lr,Vr,Br,Rr,t,freq,field=True)
                    uu=np.einsum('ir,trs->tis',Q,ur,optimize=True)
                    ron=online_time(lr,Vr,Br,Rr,t,freq)
                    phase=np.angle(HH*np.conj(H));wt=np.abs(H)**2
                    per_source=[relative(yy[long,:,s],y[long,:,s]) for s in range(B.shape[1])]
                    row={'material':name,'grid':n,'method':'static_condensation' if r==0 else 'dynamic_condensation','modes_per_patch':r,'pressure_dof':Q.shape[1],
                        'same_grid_receiver_error':relative(yy[long],y[long]),'same_grid_short_receiver_error':relative(yy[short],y[short]),
                        'same_grid_field_error':relative(uu[long],u[long]),'same_grid_transfer_error':relative(HH,H),
                        'weighted_phase_error_rad':float(np.sum(wt*np.abs(phase))/wt.sum()),
                        'worst_source_receiver_error':max(per_source),'source1_receiver_error':per_source[0], 'source2_receiver_error':per_source[1], 'source3_receiver_error':per_source[2],
                        'initial_energy_retained':float(np.linalg.norm(Br)**2/np.linalg.norm(B)**2),
                        'setup_seconds_shared_max32_basis':assembly+basis_build+extra,'receiver_online_seconds':ron,
                        'full_setup_seconds':assembly+esec,'full_receiver_online_seconds':on,
                        'online_speedup':on/ron,'same_medium_1query_seconds_shared_setup':assembly+basis_build+extra+ron,
                        'same_medium_100query_seconds_shared_setup':(assembly+basis_build+extra)/100+ron,
                        'near_matched_coarse_grid':int(np.floor(np.sqrt(Q.shape[1])+.5))}
                    rows.append(row)
                    coeff=Vr.T@Br; om=np.sqrt(lr)
                    energy=np.sum((np.cos(t[:,None,None]*om[None,:,None])*coeff[None,:,:])**2+(np.sin(t[:,None,None]*om[None,:,None])*coeff[None,:,:])**2,axis=1)
                    _,_,Hcol=response(lr,Vr,Br,Br,t[:1],freq)
                    minreal=min(linalg.eigvalsh((z+z.conj().T)/2)[0] for z in Hcol)
                    Mr=T.T@(mass[:,None]*T);Kr=T.T@K@T
                    sqrtA=(Vr*np.sqrt(lr))@Vr.T
                    checks.append({'material':name,'grid':n,'modes_per_patch':r,'static_constant_error':constant_error,'interior_disconnect':interior_disconnect,
                        'mass_min_eigenvalue':float(linalg.eigvalsh(Mr)[0]),'stiffness_relative_symmetry':relative(Kr,Kr.T),
                        'orthonormality_defect':float(np.max(np.abs(Q.T@Q-np.eye(Q.shape[1])))),
                        'min_stiffness_eigenvalue':float(linalg.eigvalsh(Ar)[0]),
                        'passive_generator_skew_defect':float(np.max(np.abs(sqrtA-sqrtA.T))),
                        'energy_drift':float(np.max(np.abs(energy/energy[0]-1))),
                        'collocated_positive_real_min_eigenvalue':float(minreal)})
                    raw[f'{name}_n{n}_r{r}_y']=yy;raw[f'{name}_n{n}_r{r}_H']=HH
                    fields[f'{name}_n{n}_r{r}_snapshots']=uu[[0,160,320,480,640]]
                    print(' n',n,'r',r,'DOF',Q.shape[1],'receiver',round(row['same_grid_receiver_error'],5),'retained',round(row['initial_energy_retained'],5),flush=True)
            print(' full',n,'setup',round(assembly+esec,3),'online',round(on,4),flush=True)
        ref=cache[CFG['reference_grid']]
        for row in rows:
            if row['material']!=name:continue
            n=row['grid'];r=row['modes_per_patch'];c=row['near_matched_coarse_grid']
            yy=raw[f'{name}_n{n}_r{r}_y'];HH=raw[f'{name}_n{n}_r{r}_H']
            row.update(receiver_error_vs_48=relative(yy[long],ref['y'][long]),transfer_error_vs_48=relative(HH,ref['H']),
                full_grid_receiver_error_vs_48=relative(cache[n]['y'][long],ref['y'][long]),
                matched_coarse_pressure_dof=c*c,matched_coarse_receiver_error_vs_48=relative(cache[c]['y'][long],ref['y'][long]),
                matched_coarse_transfer_error_vs_48=relative(cache[c]['H'],ref['H']),
                matched_coarse_setup_seconds=cache[c]['setup'],matched_coarse_receiver_online_seconds=cache[c]['online'])
        for n in ns:
            setup_rows.append({'material':name,'grid':n,'pressure_dof':n*n,'setup_seconds':cache[n]['setup'],'receiver_online_seconds':cache[n]['online'],
                'receiver_error_vs_48':relative(cache[n]['y'][long],ref['y'][long]),'transfer_error_vs_48':relative(cache[n]['H'],ref['H'])})
        dump(HERE/'structural_checks.json',checks)
        for path,rr in [(HERE/'all_results.csv',rows),(HERE/'full_grid_results.csv',setup_rows)]:
            with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        np.savez_compressed(HERE/'receiver_and_transfer_responses.npz',**raw)
        np.savez_compressed(HERE/'field_snapshots.npz',**fields)
    summary=[]
    for n in CFG['grids']:
        for r in CFG['internal_modes_per_patch']:
            rr=[x for x in rows if x['grid']==n and x['modes_per_patch']==r]
            summary.append({'grid':n,'modes_per_patch':r,'pressure_dof':rr[0]['pressure_dof'],**{key:float(np.mean([x[key] for x in rr])) for key in ['same_grid_receiver_error','same_grid_field_error','same_grid_transfer_error','receiver_error_vs_48','matched_coarse_receiver_error_vs_48','initial_energy_retained','online_speedup','receiver_online_seconds','matched_coarse_receiver_online_seconds']}})
    audit=[x for x in setup_rows if x['material']==CFG['audit_material'] and x['grid']==CFG['audit_grid']][0]
    result={'status':'development diagnostic','summary':summary,'reference_48_vs_64_selected_material':audit,
        'structural_all_pass':all(x['mass_min_eigenvalue']>0 and x['static_constant_error']<1e-10 and x['interior_disconnect']==0 and x['energy_drift']<1e-12 and x['collocated_positive_real_min_eigenvalue']>-1e-10 for x in checks),
        'elapsed_seconds':time.perf_counter()-start_all}
    dump(HERE/'summary.json',result)

if __name__=='__main__':main()
