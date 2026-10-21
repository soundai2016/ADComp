#!/usr/bin/env python3
"""Modal matching, reference sensitivity, residue paths, and finite-time diagnostics."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time as clock
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
OUT = HERE / "results"
CACHE = HERE / "cache"
CFG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, str(SOURCE))
from acoustic_core import build_sparse, ports, CONFIG as CORE


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


MODAL = module("original_modal_diagnostics", SOURCE / "modal_diagnostics/run.py")
ACCURACY = module("original_accuracy_cost", SOURCE / "accuracy_cost/run.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csvwrite(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(value):
    return float(np.asarray(value))


def norm(value):
    return float(np.linalg.norm(value))


def residue(b, c):
    return MODAL.residue(b, c)


def response(t, w, h):
    if len(w) == 0:
        return np.zeros((len(t), 4, 3))
    return MODAL.response(t, w, h)


def cached_spectrum(n, name, checks):
    """Read supplied spectrum first; create new fine references locally."""
    old = SOURCE / "modal_diagnostics/results" / f"{name}_n{n}_spectrum.npz"
    own = CACHE / f"{name}_n{n}_spectrum.npz"
    path = old if old.exists() else own
    X, Y, rho, bulk, mass, C, _ = build_sparse(n, name)
    A = (C @ C.T).tocsr()
    B = ports(X, Y, bulk, CORE["source_centres"], True)
    R = ports(X, Y, bulk, CORE["receiver_centres"], False)
    start = clock.perf_counter()
    if path.exists():
        with np.load(path) as z:
            lam, V = z["eigenvalues"].copy(), z["eigenvectors"].copy()
        origin = "supplied" if path == old else "new_cached"
    else:
        print(f"Eigensolve {name} n={n}", flush=True)
        if n == CFG["coarse_grid"]:
            lam, V = linalg.eigh(A.toarray(), check_finite=False, driver="evd")
        else:
            lam, V = eigsh(A, k=CFG["fine_modes"], sigma=-1e-7, which="LM",
                tol=CFG["eigen_tolerance"],
                v0=np.random.default_rng(673).normal(size=n*n))
            order = np.argsort(lam)
            lam, V = lam[order], V[:, order]
        lam[lam < 1e-9] = 0.0
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, eigenvalues=lam, eigenvectors=V, mass=mass,
            source_coupling=V.T@B, receiver_coupling=R.T@V)
        temporary.replace(path)
        origin = "newly_computed"
    assert len(lam) == (n*n if n == CFG["coarse_grid"] else CFG["fine_modes"])
    defect = np.linalg.norm(A@V-V*lam[None, :], axis=0)/np.maximum(lam, 1.0)
    ortho = norm(V.T@V-np.eye(V.shape[1]))
    record = dict(material=name, grid=n, modes=len(lam), origin=origin,
        path=str(path.relative_to(SOURCE)), sha256=digest(path),
        elapsed_seconds=clock.perf_counter()-start,
        maximum_scaled_eigen_residual=float(defect.max()), orthogonality_defect=ortho)
    checks.append(record)
    print(json.dumps(record), flush=True)
    return dict(omega=np.sqrt(lam), V=V, b=V.T@B, c=R.T@V, mass=mass, A=A, B=B, R=R)


def reduced_model(name, coarse, method, m, t, checks):
    n = CFG["coarse_grid"]
    old = SOURCE / "accuracy_cost/results" / f"{name}_n{n}_{method}_m{m}.npz"
    own = CACHE / f"{name}_n{n}_{method}_m{m}.npz"
    path = old if old.exists() else own
    if path.exists():
        with np.load(path) as z:
            Q = z["orthonormal_basis"].copy()
            Vsmall = z["reduced_eigenvectors"].copy()
            lam = z["reduced_eigenvalues"].copy()
            ystored = z["receiver_pressure"].copy()
            np.testing.assert_array_equal(z["time"], t)
        origin = "supplied" if path == old else "new_cached"
    else:
        print(f"Rebuilding {name} {method} m={m}", flush=True)
        raw, _, _, _ = ACCURACY.reduced_run(n, name, m, method, t)
        np.savez_compressed(path, **raw)
        Q, Vsmall, lam = raw["orthonormal_basis"], raw["reduced_eigenvectors"], raw["reduced_eigenvalues"]
        ystored = raw["receiver_pressure"]
        origin = "newly_computed"
    V = Q@Vsmall
    w = np.sqrt(lam)
    b, c = V.T@coarse["B"], coarse["R"].T@V
    H = residue(b, c)
    y = response(t, w, H)
    checks.append(dict(material=name, method=method, m=m, origin=origin,
        path=str(path.relative_to(SOURCE)), sha256=digest(path),
        stored_response_relative_difference=norm(y-ystored)/norm(ystored),
        basis_orthogonality_defect=norm(V.T@V-np.eye(len(w)))))
    return dict(V=V, omega=w, b=b, c=c, H=H, y=y, method=method, m=m)


def assignment(left, right, wl, wr):
    overlap = left.T@right
    il, ir = linear_sum_assignment(-overlap**2)
    return dict(left=il, right=ir, mac=overlap[il, ir]**2,
        sign=np.sign(overlap[il, ir]), gap_left=MODAL.gap(wl)[il],
        gap_right=MODAL.gap(wr)[ir], omega_left=wl[il], omega_right=wr[ir])


def accepted(a, mac, gap):
    return ((a["mac"] >= mac) & (a["gap_left"] >= gap) &
        (a["gap_right"] >= gap) & (a["omega_left"] > 1e-7) & (a["omega_right"] > 1e-7))


def chain(ar, af, mac, gap):
    cf = {int(i): int(j) for i, j, good in zip(af["left"], af["right"], accepted(af, mac, gap)) if good}
    return [(int(i), int(j), cf[int(j)])
        for i, j, good in zip(ar["left"], ar["right"], accepted(ar, mac, gap))
        if good and int(j) in cf]


def windows(t):
    for end in CFG["cumulative_end_times"]:
        yield "cumulative", 0.0, end, (t >= 0)&(t <= end)
    for start, end in zip(CFG["successive_edges"][:-1], CFG["successive_edges"][1:]):
        yield "successive", start, end, ((t >= start) if start == 0 else (t > start))&(t <= end)


def groups():
    yield "all", 0, 0, np.ones((4, 3), dtype=bool)
    for s in range(3):
        for r in range(4):
            select = np.zeros((4, 3), dtype=bool)
            select[r, s] = True
            yield "channel", s+1, r+1, select


def selected_arrays(triples, model, coarse, fine):
    if not triples:
        empty = np.array([], dtype=int)
        ri = ci = fi = empty
    else:
        ri, ci, fi = np.array(triples, dtype=int).T
    return dict(ri=ri, ci=ci, fi=fi, wr=model["omega"][ri],
        wh=coarse["omega"][ci], wf=fine["omega"][fi],
        Hr=model["H"][ri], Hf=residue(fine["b"][fi], fine["c"][:, fi]))


def path_arrays(t, a, path):
    zr, zh, zf = (np.cos(t[:, None]*a[k][None, :]) for k in ("wr", "wh", "wf"))
    if path == "rom":
        H, zc = a["Hr"], zf
    elif path == "fine":
        H, zc = a["Hf"], zr
    elif path == "symmetric":
        H, zc = (a["Hr"]+a["Hf"])/2, (zr+zf)/2
    else:
        raise ValueError(path)
    am = (zr-zh)[:, :, None, None]*H[None, :, :, :]
    bm = (zh-zf)[:, :, None, None]*H[None, :, :, :]
    coupling = np.einsum("tj,jrs->trs", zc, a["Hr"]-a["Hf"])
    return am, bm, coupling


def threshold_paths(name, grid, t, model, coarse, fine, yref, yfine, ar, af):
    settings = [(m, g) for m in CFG["mac_thresholds"] for g in CFG["relative_frequency_gaps"]]
    selections = {(m, g):chain(ar, af, m, g) for m, g in settings}
    common_keys = set.intersection(*(set(v) for v in selections.values()))
    common = sorted(common_keys)
    rows, detail, pairrows = [], [], []
    error = model["y"]-yref
    den = norm(yref)
    for mac, gap in settings:
        for scope, triples in (("own", selections[(mac, gap)]), ("common", common)):
            a = selected_arrays(triples, model, coarse, fine)
            weights = np.linalg.norm(a["Hf"], axis=(1, 2))
            weights /= max(float(weights.sum()), 1e-300)
            dg, dr = a["wh"]-a["wf"], a["wr"]-a["wh"]
            wg = float(np.sum(weights*dg/a["wf"]))
            wr = float(np.sum(weights*dr/a["wf"]))
            selected_difference = response(t, a["wr"], a["Hr"])-response(t, a["wf"], a["Hf"])
            remaining = error-selected_difference
            fine_selected = response(t, a["wf"], a["Hf"])
            interaction=response(t,a["wr"],a["Hr"]-a["Hf"])-response(t,a["wf"],a["Hr"]-a["Hf"])
            for path in ("rom", "fine", "symmetric"):
                am, bm, cp = path_arrays(t, a, path)
                A, B = am.sum(axis=1), bm.sum(axis=1)
                changed = model["y"]-A
                row = dict(material=name, fine_grid=grid, method=model["method"], m=model["m"],
                    mac_threshold=mac, relative_gap=gap, selection=scope, path=path,
                    matched_count=len(triples), common_count=len(common),
                    weighted_grid_shift=wg, weighted_ritz_shift=wr,
                    frequency_compensation_ratio=(-wr/wg if abs(wg)>1e-30 else 0.0),
                    original_error=norm(error)/den, replaced_error=norm(changed-yref)/den,
                    error_change=(norm(changed-yref)-norm(error))/den,
                    squared_error_change=(norm(changed-yref)**2-norm(error)**2)/den**2,
                    ritz_norm=norm(A)/den, grid_norm=norm(B)/den,
                    coupling_norm=norm(cp)/den, remaining_norm=norm(remaining)/den,
                    path_interaction_norm=norm(interaction)/den,
                    ritz_grid_cosine=float(np.sum(A*B)/max(norm(A)*norm(B),1e-300)),
                    decomposition_defect=norm(error-A-B-cp-remaining)/den,
                    matched_fine_response_omission=norm(yfine-fine_selected)/norm(yfine),
                    eigenmode_coverage_norm=norm(yfine-response(t,fine["omega"],residue(fine["b"],fine["c"])))/norm(yfine))
                rows.append(row)
                for kind, start, end, use in windows(t):
                    eref = np.sum(yref[use]**2, axis=0)
                    old = np.sum(error[use]**2, axis=0)
                    new = np.sum((error[use]-A[use])**2, axis=0)
                    for group, source, receiver, select in groups():
                        d = float(eref[select].sum())
                        s0, s1 = float(old[select].sum()), float(new[select].sum())
                        delta = (s1-s0)/d
                        detail.append(dict(material=name, fine_grid=grid, mac_threshold=mac,
                            relative_gap=gap, selection=scope, path=path, matched_count=len(triples),
                            window_kind=kind, start=start, end=end, group=group, source=source,
                            receiver=receiver, original_error=np.sqrt(s0/d), replaced_error=np.sqrt(s1/d),
                            error_change=(np.sqrt(s1)-np.sqrt(s0))/np.sqrt(d),
                            squared_error_change=delta, sign=int(np.sign(delta)) if abs(delta)>CFG["change_zero_tolerance"] else 0))
            if scope == "own":
                for j, (ri, ci, fi) in enumerate(triples):
                    pairrows.append(dict(material=name, fine_grid=grid, mac_threshold=mac, relative_gap=gap,
                        rom_mode=ri, coarse_mode=ci, fine_mode=fi, weight=float(weights[j]),
                        omega_rom=float(a["wr"][j]), omega_coarse=float(a["wh"][j]), omega_fine=float(a["wf"][j]),
                        in_common_set=(ri,ci,fi) in common_keys))
    return rows, detail, pairrows


def cluster_diagnostics(name, grid, coarse, restricted, fine, af):
    """Cluster angles only; no arbitrary scalar shifts assigned to a cluster."""
    w = coarse["omega"][:CFG["fine_modes"]]
    threshold = CFG["cluster_gap"]
    assignment_map = {int(i):int(j) for i,j in zip(af["left"],af["right"])}
    clusters, start = [], 1
    for stop in range(2, len(w)+1):
        if stop == len(w) or (w[stop]-w[stop-1])/max(w[stop],1e-300) >= threshold:
            if stop-start > 1:
                clusters.append(np.arange(start,stop))
            start=stop
    rows=[]
    for ids in clusters:
        if not all(int(j) in assignment_map for j in ids):
            continue
        fi=np.array([assignment_map[int(j)] for j in ids])
        Qc,_=linalg.qr(coarse["V"][:,ids],mode="economic")
        Qf,_=linalg.qr(restricted[:,fi],mode="economic")
        singular=linalg.svdvals(Qc.T@Qf)
        projector_distance=np.sqrt(max(0.0,2*len(ids)-2*np.sum(singular**2)))
        # Orthogonal change of basis must rotate the whole frequency matrix.
        rng=np.random.default_rng(1700+int(ids[0]))
        U,_=linalg.qr(rng.normal(size=(len(ids),len(ids))))
        lam=fine["omega"][fi]**2
        K=U.T@np.diag(lam)@U
        lv,Z=linalg.eigh(K)
        b0,c0=fine["b"][fi],fine["c"][:,fi]
        b1=Z.T@(U.T@b0);c1=(c0@U)@Z
        test_t=np.linspace(0,8,65)
        original=response(test_t,np.sqrt(lam),residue(b0,c0))
        transformed=response(test_t,np.sqrt(np.maximum(lv,0)),residue(b1,c1))
        rows.append(dict(material=name,fine_grid=grid,start_mode=int(ids[0]),end_mode=int(ids[-1]),
            dimension=len(ids),smallest_principal_cosine=float(singular.min()),
            largest_principal_angle_degrees=float(np.degrees(np.arccos(np.clip(singular.min(),0,1)))),
            projector_frobenius_distance=float(projector_distance),
            block_response_basis_rotation_relative_defect=norm(original-transformed)/max(norm(original),1e-300),
            individually_attributed=False))
    return rows


def finite_time(name, grid, t, model, coarse, fine, yref, triples):
    with np.load(SOURCE/"propagation/results"/f"{name}_n384.npz") as z:
        np.testing.assert_array_equal(t,z["time"])
        alternate_reference=z["receiver_pressure"].copy()
    a=selected_arrays(triples,model,coarse,fine)
    am,bm,coupling=path_arrays(t,a,"rom")
    A,B=am.sum(axis=1),bm.sum(axis=1)
    error=model["y"]-yref
    q=error-A-B
    remainder=q-coupling
    dg,dr=a["wh"]-a["wf"],a["wr"]-a["wh"]
    base=-t[:,None]*np.sin(t[:,None]*a["wf"][None,:])
    H=a["Hr"]
    psi=base[:,:,None,None]*H[None,:,:,:]
    alin=np.einsum("tjrs,j->trs",psi,dr)
    blin=np.einsum("tjrs,j->trs",psi,dg)
    rows=[];saved={"rom_modes":a["ri"],"coarse_modes":a["ci"],"fine_modes":a["fi"],"delta_grid":dg,"delta_ritz":dr}
    analysis_windows=list(windows(t))
    analysis_windows.extend(("short_cumulative",0.0,end,(t>=0)&(t<=end))
        for end in CFG["linearization_short_end_times"])
    for kind,start,end,use in analysis_windows:
        time_gram=base[use].T@base[use]
        for group,source,receiver,select in groups():
            den2=float(np.sum(yref[use][:,select]**2))
            aa=A[use][:,select];bb=B[use][:,select];qq=q[use][:,select]
            ee=error[use][:,select]
            a_modes=am[use][:,:,select];b_modes=bm[use][:,:,select]
            exact=(np.sum((ee-aa)**2)-np.sum(ee**2))/den2
            reference_difference=(yref-alternate_reference)[use][:,select]
            alternate_error=ee+reference_difference
            alternate_exact=(np.sum((alternate_error-aa)**2)-np.sum(alternate_error**2))/den2
            reference_delta=2*float(np.sum(aa*reference_difference))/den2
            reference_bound=2*norm(aa)*norm(reference_difference)/den2
            ritz_square=-float(np.sum(aa**2))/den2
            grid_cross=-2*float(np.sum(aa*bb))/den2
            q_cross=-2*float(np.sum(aa*qq))/den2
            diagonal=-float(np.sum(a_modes**2)+2*np.sum(a_modes*b_modes))/den2
            offdiagonal=ritz_square+grid_cross-diagonal
            Hselected=H[:,select]
            G=time_gram*(Hselected@Hselected.T)
            cq=np.sum(Hselected*(base[use].T@qq),axis=1)
            linear=-(2*dr@G@dg+dr@G@dr+2*dr@cq)/den2
            linear_noq=-(2*dr@G@dg+dr@G@dr)/den2
            linear_diag=-np.sum((2*dr*dg+dr*dr)*np.diag(G))/den2
            approx=(np.sum((blin[use][:,select]+qq)**2)-np.sum((alin[use][:,select]+blin[use][:,select]+qq)**2))/den2
            single_mode_negative_grid=(dg<0)&(dr>0)&(dr < -2*dg)
            weight=np.linalg.norm(Hselected,axis=1)
            eigenvalues=linalg.eigvalsh((G+G.T)/2)
            scale=max(float(np.max(np.abs(eigenvalues))),1e-300)
            row=dict(material=name,fine_grid=grid,method=model["method"],m=model["m"],matched_count=len(triples),
                window_kind=kind,start=start,end=end,group=group,source=source,receiver=receiver,
                original_error=norm(ee)/np.sqrt(den2),replaced_error=norm(ee-aa)/np.sqrt(den2),
                exact_squared_error_change=exact,negative_ritz_square=ritz_square,
                negative_twice_ritz_grid=grid_cross,negative_twice_ritz_remainder=q_cross,
                exact_same_mode_frequency_terms=diagonal,exact_cross_mode_frequency_terms=offdiagonal,
                coupling_cross=-2*float(np.sum(aa*coupling[use][:,select]))/den2,
                unmatched_reference_cross=-2*float(np.sum(aa*remainder[use][:,select]))/den2,
                linear_full=scalar(linear),linear_without_q=scalar(linear_noq),linear_diagonal_without_q=scalar(linear_diag),
                exact_sign=int(np.sign(exact)) if abs(exact)>CFG["change_zero_tolerance"] else 0,
                linear_sign=int(np.sign(linear)) if abs(linear)>CFG["change_zero_tolerance"] else 0,
                alternate_reference_squared_change_same_denominator=float(alternate_exact),
                reference_change_exact_cross=reference_delta,reference_change_bound=reference_bound,
                reference_sign_margin=abs(exact)/max(reference_bound,1e-300),
                reference_change_identity_defect=abs(exact-alternate_exact-reference_delta),
                exact_budget_defect=abs(exact-ritz_square-grid_cross-q_cross),
                linear_gram_identity_defect=abs(scalar(linear)-scalar(approx)),
                gram_relative_minimum_eigenvalue=float(eigenvalues.min()/scale),
                gram_relative_symmetry_defect=norm(G-G.T)/max(norm(G),1e-300),
                ritz_linearization_error=norm(alin[use][:,select]-aa)/max(norm(aa),1e-300),
                grid_linearization_error=norm(blin[use][:,select]-bb)/max(norm(bb),1e-300),
                maximum_phase_offset=float(end*np.max(np.maximum(np.abs(dg),np.abs(dg+dr)),initial=0)),
                relative_signal_energy=float(den2/np.sum(yref[use]**2)),
                single_mode_condition_residue_fraction=float(weight[single_mode_negative_grid].sum()/max(weight.sum(),1e-300)))
            rows.append(row)
            if group=="all":
                key=f"{kind}_{start:g}_{end:g}"
                saved[key+"_gram_normalized"]=G/den2
                saved[key+"_q_inner_normalized"]=cq/den2
    np.savez_compressed(OUT/f"{name}_n{grid}_{model['method']}_m{model['m']}_gram.npz",**saved)
    return rows


def original_comparison(name,t,model,coarse,fine,yfine,triples):
    a=selected_arrays(triples,model,coarse,fine)
    am,bm,coupling=path_arrays(t,a,"rom")
    A,B=am.sum(axis=1),bm.sum(axis=1)
    weight=np.linalg.norm(a["Hf"],axis=(1,2));weight/=max(float(weight.sum()),1e-300)
    values=dict(accepted_three_grid_chain=len(triples),
        weighted_grid_shift=float(np.sum(weight*(a["wh"]-a["wf"])/a["wf"])),
        weighted_ritz_shift=float(np.sum(weight*(a["wr"]-a["wh"])/a["wf"])),
        ritz_grid_cosine=float(np.sum(A*B)/max(norm(A)*norm(B),1e-300)),
        total_error_vs_128=norm(model["y"]-yfine)/norm(yfine),
        error_without_ritz_shift_vs_128=norm(model["y"]-A-yfine)/norm(yfine))
    with (SOURCE/"modal_diagnostics/results/summary.csv").open() as stream:
        old=next(r for r in csv.DictReader(stream) if r["material"]==name and r["method"]==model["method"] and int(r["m"])==model["m"])
    Aold=response(t,a["wr"],a["Hr"])-response(t,a["wh"],a["Hr"])
    Bold=response(t,a["wh"],a["Hr"])-response(t,a["wf"],a["Hr"])
    old_order_cosine=float(np.sum(Aold*Bold)/max(norm(Aold)*norm(Bold),1e-300))
    return [dict(material=name,method=model["method"],m=model["m"],quantity=k,
        original=float(old[k]),recomputed=float(v),absolute_difference=abs(float(v)-float(old[k])),
        original_order_recomputed=old_order_cosine if k=="ritz_grid_cosine" else float(v),
        original_order_absolute_difference=abs((old_order_cosine if k=="ritz_grid_cosine" else float(v))-float(old[k])),
        ritz_relative_norm=norm(A)/norm(yfine),
        angle_near_zero_component=(k=="ritz_grid_cosine" and norm(A)/norm(yfine)<1e-6)) for k,v in values.items()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--materials",nargs="+",default=CFG["materials"])
    p.add_argument("--fine-grids",nargs="+",type=int,default=CFG["fine_grids"])
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--skip-controls",action="store_true")
    args=p.parse_args()
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    if hasattr(os,"sched_getaffinity"):
        available=set(os.sched_getaffinity(0));chosen=available&set(CFG["cpu_affinity"])
        os.sched_setaffinity(0,chosen or set(sorted(available)[:CFG["threads"]]))
    start=clock.perf_counter()
    for name in args.materials:
        print("Processing "+name,flush=True)
        checks=[];baseline=[];sensitivity=[];channel=[];pairs=[];cluster=[];finite=[];reference=[]
        n=CFG["coarse_grid"]
        coarse=cached_spectrum(n,name,checks)
        with np.load(SOURCE/"propagation/results"/f"{name}_n{CFG['evaluation_reference_grid']}.npz") as z:
            t,yref=z["time"].copy(),z["receiver_pressure"].copy()
        if args.prepare_only:
            for grid in args.fine_grids:
                cached_spectrum(grid,name,checks)
            dump(OUT/f"{name}_preparation.json",checks)
            continue
        model=reduced_model(name,coarse,*CFG["primary_model"],t,checks)
        ar=assignment(model["V"],coarse["V"],model["omega"],coarse["omega"])
        baseline_selected={}
        for grid in args.fine_grids:
            assert grid % n == 0,"Physical block averaging requires an integer grid ratio."
            fine=cached_spectrum(grid,name,checks)
            with np.load(SOURCE/"propagation/results"/f"{name}_n{grid}.npz") as z:
                np.testing.assert_array_equal(t,z["time"])
                yfine=z["receiver_pressure"].copy()
            W,_=MODAL.restriction(fine["V"],fine["mass"],coarse["mass"],grid,n)
            af=assignment(coarse["V"],W,coarse["omega"],fine["omega"])
            rr,dd,pp=threshold_paths(name,grid,t,model,coarse,fine,yref,yfine,ar,af)
            sensitivity.extend(rr);channel.extend(dd);pairs.extend(pp)
            cluster.extend(cluster_diagnostics(name,grid,coarse,W,fine,af))
            triples=chain(ar,af,CFG["baseline_mac"],CFG["baseline_gap"])
            baseline_selected[grid]=triples
            finite.extend(finite_time(name,grid,t,model,coarse,fine,yref,triples))
            if grid==128:
                baseline.extend(original_comparison(name,t,model,coarse,fine,yfine,triples))
                if not args.skip_controls:
                    for method,m in CFG["control_models"]:
                        control=reduced_model(name,coarse,method,m,t,checks)
                        arc=assignment(control["V"],coarse["V"],control["omega"],coarse["omega"])
                        tc=chain(arc,af,CFG["baseline_mac"],CFG["baseline_gap"])
                        finite.extend(finite_time(name,grid,t,control,coarse,fine,yref,tc))
                        baseline.extend(original_comparison(name,t,control,coarse,fine,yfine,tc))
            np.savez_compressed(OUT/f"{name}_n{grid}_assignments.npz",
                **{"rom_coarse_"+k:v for k,v in ar.items()},**{"coarse_fine_"+k:v for k,v in af.items()})
            print(json.dumps(dict(material=name,grid=grid,baseline_count=len(triples),
                baseline_original_error=norm(model["y"]-yref)/norm(yref))),flush=True)
        if len(args.fine_grids)>1:
            first=args.fine_grids[0]
            keys0={(ri,ci) for ri,ci,_ in baseline_selected[first]}
            for grid in args.fine_grids[1:]:
                keys1={(ri,ci) for ri,ci,_ in baseline_selected[grid]}
                reference.append(dict(material=name,base_grid=first,comparison_grid=grid,
                    base_count=len(keys0),comparison_count=len(keys1),intersection=len(keys0&keys1),
                    lost=len(keys0-keys1),gained=len(keys1-keys0)))
        for filename,rows in (("sensitivity",sensitivity),("channel_sensitivity",channel),("pairs",pairs),
            ("clusters",cluster),("finite_time",finite),("baseline_check",baseline),("reference_matching",reference)):
            csvwrite(OUT/f"{name}_{filename}.csv",rows)
        dump(OUT/f"{name}_verification.json",dict(spectral_and_reduced_checks=checks,
            propagation_input_sha256={str(path.relative_to(SOURCE)):digest(path) for path in
                [SOURCE/"propagation/results"/f"{name}_n{g}.npz" for g in sorted(set(args.fine_grids+[384,CFG['evaluation_reference_grid']]))]},
            maximum_baseline_difference=max((r["absolute_difference"] for r in baseline),default=0),
            maximum_decomposition_defect=max(r["decomposition_defect"] for r in sensitivity),
            maximum_exact_budget_defect=max(r["exact_budget_defect"] for r in finite),
            maximum_gram_identity_defect=max(r["linear_gram_identity_defect"] for r in finite),
            minimum_gram_relative_eigenvalue=min(r["gram_relative_minimum_eigenvalue"] for r in finite),
            maximum_cluster_rotation_defect=max((r["block_response_basis_rotation_relative_defect"] for r in cluster),default=0),
            config_sha256=digest(HERE/"config.json"),analysis_sha256=digest(Path(__file__))))
    dump(OUT/"environment.json",dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),threads=CFG["threads"],affinity=sorted(os.sched_getaffinity(0)),
        arguments=vars(args),elapsed_seconds=clock.perf_counter()-start,
        input_source_hashes={str(path.relative_to(SOURCE)):digest(path) for path in
            (SOURCE/"acoustic_core.py",SOURCE/"modal_diagnostics/run.py",SOURCE/"accuracy_cost/run.py")},
        config_sha256=digest(HERE/"config.json"),analysis_sha256=digest(Path(__file__))))


if __name__=="__main__":
    main()
