"""Passive two-port acoustics on a one-dimensional layered interval.

Time convention exp(+i omega t); exterior rho=K=Z=1 at both ports.
Pressure p and inward velocity f supply power p^T f. The compatible
finite-volume / lumped FE grid includes both endpoints with half-cell dual
volumes. Integrals of rho on primal edges and beta=1/K on dual cells are exact
across material interfaces.

The Neumann impedance is Z(s)=sum s b_j b_j^T/(s^2+lambda_j), with lambda_0=0.
PSD residues ensure passivity under modal truncation. For T PSD, the optional
s*T term is a passive inertance matching the leading low-frequency omitted
impedance; its accuracy is local to that expansion.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal


@dataclass
class ModalModel:
    lambdas: np.ndarray
    residues: np.ndarray
    tail: np.ndarray
    cells: int
    compliance: float


def _medium(rho, bulk, widths=None):
    rho, bulk = np.asarray(rho, float), np.asarray(bulk, float)
    widths = np.ones(len(rho))/len(rho) if widths is None else np.asarray(widths, float)
    if not (rho.shape == bulk.shape == widths.shape and np.all(rho>0) and np.all(bulk>0) and np.all(widths>0)):
        raise ValueError('Positive equal-sized one-dimensional medium arrays required')
    return rho, bulk, widths


def exact_scattering(omega, rho, bulk, widths=None):
    """Analytic layer propagators converted to power-normalized scattering.

    Only eight or sixteen moderate-contrast layers are used in this study;
    very long/high-contrast cascades should use scattering composition.
    """
    rho, bulk, widths = _medium(rho, bulk, widths)
    w=np.atleast_1d(omega)
    matrix=np.broadcast_to(np.eye(2,dtype=complex),(len(w),2,2)).copy()
    for r,k,d in zip(rho,bulk,widths):
        z=np.sqrt(r*k); phase=w*d*np.sqrt(r/k)
        layer=np.empty_like(matrix)
        layer[:,0,0]=layer[:,1,1]=np.cos(phase)
        layer[:,0,1]=-1j*z*np.sin(phase)
        layer[:,1,0]=-1j*np.sin(phase)/z
        matrix=layer@matrix
    a,b,c,d=matrix[:,0,0],matrix[:,0,1],matrix[:,1,0],matrix[:,1,1]
    denominator=a-b-c+d
    out=np.empty_like(matrix)
    out[:,0,0]=(c+d-a-b)/denominator
    out[:,1,1]=(a-b+c-d)/denominator
    out[:,0,1]=out[:,1,0]=2/denominator
    return out


def piecewise_integral(values,widths,points):
    """Integral from the left endpoint to every clipped point."""
    bounds=np.r_[0,np.cumsum(widths)]
    x=np.clip(points,0,bounds[-1])
    indices=np.minimum(np.searchsorted(bounds,x,side='right')-1,len(values)-1)
    integral=np.r_[0,np.cumsum(values*widths)]
    return integral[indices]+values[indices]*(x-bounds[indices])


def build_modal_model(rho,bulk,widths=None,n=256):
    """Neumann modes with left-positive signs and compliance normalization.
    
    The residues array stores factors b_j of shape (n+1,2); b_j b_j^T is PSD.
    """
    rho,bulk,widths=_medium(rho,bulk,widths)
    length=widths.sum(); x=np.linspace(0,length,n+1)
    masses=np.diff(piecewise_integral(rho,widths,x))
    dual=np.r_[0,(x[1:]+x[:-1])/2,length]
    compliance=np.diff(piecewise_integral(1/bulk,widths,dual))
    diagonal=np.r_[1/masses[0],1/masses[:-1]+1/masses[1:],1/masses[-1]]/compliance
    off=-1/masses/np.sqrt(compliance[:-1]*compliance[1:])
    vals,vec=eigh_tridiagonal(diagonal,off)
    # The constant mode is known exactly; avoid tiny signed numerical eigenvalues.
    vals[0]=0
    b=np.stack((vec[0]/np.sqrt(compliance[0]),vec[-1]/np.sqrt(compliance[-1])),axis=1)
    b*=np.where(b[:,0]>=0,1,-1)[:,None]
    b[0]=1/np.sqrt(compliance.sum())
    if np.any(vals[1:]<=0): raise ArithmeticError('Nonpositive nonzero mode')
    return ModalModel(vals,b,np.zeros((2,2)),n,float(compliance.sum()))


def fv_matrices(rho,bulk,widths=None,n=256):
    """Diagonal C and tridiagonal L for an independent direct FV solve."""
    rho,bulk,widths=_medium(rho,bulk,widths)
    x=np.linspace(0,widths.sum(),n+1)
    masses=np.diff(piecewise_integral(rho,widths,x))
    dual=np.r_[0,(x[1:]+x[:-1])/2,widths.sum()]
    compliance=np.diff(piecewise_integral(1/bulk,widths,dual))
    diag=np.r_[1/masses[0],1/masses[:-1]+1/masses[1:],1/masses[-1]]
    return compliance,diag,-1/masses


def fv_scattering_direct(omega,matrices):
    """Batched Thomas solve of the terminated tridiagonal FV system.
    
    (s^2 C + L + s B B^T)p = 2s B a; outgoing = B^T p - a.
    """
    compliance,diag,off=matrices; s=1j*np.atleast_1d(omega)
    d=diag[None,:]+s[:,None]**2*compliance[None,:]
    d[:,0]+=s;d[:,-1]+=s
    rhs=np.zeros((len(s),len(diag),2),complex)
    rhs[:,0,0]=2*s;rhs[:,-1,1]=2*s
    for i in range(1,len(diag)):
        factor=off[i-1]/d[:,i-1]
        d[:,i]-=factor*off[i-1]
        rhs[:,i]-=factor[:,None]*rhs[:,i-1]
    rhs[:,-1]/=d[:,-1,None]
    for i in range(len(diag)-2,-1,-1):
        rhs[:,i]=(rhs[:,i]-off[i]*rhs[:,i+1])/d[:,i,None]
    return np.stack((rhs[:,0],rhs[:,-1]),axis=1)-np.eye(2)


def truncate_model(model,order,tail=True):
    """Retain order modes including zero; optionally retain the PSD s*T moment."""
    if order<1 or order>len(model.lambdas): raise ValueError('Invalid mode count')
    moment=np.einsum('ji,jk,j->ik',model.residues[order:],model.residues[order:],1/model.lambdas[order:]) if tail and order<len(model.lambdas) else np.zeros((2,2))
    return ModalModel(model.lambdas[:order].copy(),model.residues[:order].copy(),moment,model.cells,model.compliance)


def modal_impedance(s,lambdas,residues,tail=None):
    """Complex-frequency positive-real modal impedance; s must not be zero/pole."""
    s=np.atleast_1d(s).astype(complex)
    factors=s[:,None]/(s[:,None]**2+np.asarray(lambdas)[None,:])
    z=np.einsum('fj,ji,jk->fik',factors,residues,residues)
    if tail is not None: z+=s[:,None,None]*tail
    return z


def impedance_to_scattering(z):
    """Cayley map in identical unit-impedance exterior media."""
    eye=np.eye(2)
    # Z commutes with Z+I, so a left solve gives the Cayley transform.
    return np.linalg.solve(z+eye,z-eye)


def modal_scattering(omega,lambdas,residues,tail=None):
    """Evaluate scattering for real omega with the exp(+i omega t) convention.
    
    At poles, shift into the right half-plane by eta=1e-12*max(1,|omega|).
    The zero-frequency limit is perfect transmission.
    """
    w=np.atleast_1d(omega).astype(float); out=np.empty((len(w),2,2),complex)
    zero=w==0; out[zero]=[[0,1],[1,0]]
    if np.any(~zero):
        ww=w[~zero]; s=1j*ww
        close=np.min(abs(np.asarray(lambdas)[None,:]-ww[:,None]**2),axis=1)<1e-12*np.maximum(1,ww**2)
        s[close]+=1e-12*np.maximum(1,abs(ww[close]))
        out[~zero]=impedance_to_scattering(modal_impedance(s,lambdas,residues,tail))
    return out


def evaluate(model,omega):
    return modal_scattering(omega,model.lambdas,model.residues,model.tail)


def cascade_scattering(left,right):
    """Lossless reciprocal port connection: right output of left to left of right."""
    den=1-left[:,1,1]*right[:,0,0]
    out=np.empty_like(left)
    out[:,0,0]=left[:,0,0]+left[:,0,1]*right[:,0,0]*left[:,1,0]/den
    out[:,1,1]=right[:,1,1]+right[:,1,0]*left[:,1,1]*right[:,0,1]/den
    out[:,0,1]=left[:,0,1]*right[:,0,1]/den
    out[:,1,0]=right[:,1,0]*left[:,1,0]/den
    return out


def sample_media(seed,count,layers=8):
    """Sample ordered, equal-width layers with log-uniform density and bulk modulus."""
    rng=np.random.default_rng(seed)
    log_parameters=rng.uniform(-.8,.8,size=(count,2,layers))
    return np.exp(log_parameters[:,0]),np.exp(log_parameters[:,1]),np.full((count,layers),1/layers)


def scattering_metrics(prediction,target,mask=None):
    if mask is not None: prediction,target=prediction[mask],target[mask]
    error=np.linalg.norm(prediction-target,axis=(-2,-1))/2
    weight=np.abs(target[:,1,0])**2
    phase=np.angle(prediction[:,1,0]*target[:,1,0].conj())
    unit=prediction.conj().transpose(0,2,1)@prediction-np.eye(2)
    return {'rms_complex_S_error':float(np.sqrt(np.mean(error**2))),
            'max_complex_S_error':float(error.max()),
            'transmission_phase_rms_rad':float(np.sqrt(np.sum(weight*phase**2)/max(1e-30,weight.sum()))),
            'max_unitarity_error':float(np.linalg.norm(unit,axis=(-2,-1)).max()),
            'max_reciprocity_error':float(np.abs(prediction[:,0,1]-prediction[:,1,0]).max())}


def scattering_state_space(model):
    """Real causal state-space realization for unit exterior terminations.
    
    Input a and output b are incoming and outgoing pressure waves. The transfer
    is the modal scattering matrix. The energy metric is I without a tail,
    and diag(I,T) with a full-rank tail.
    """
    r=len(model.lambdas); d=2*r-1
    j=np.zeros((d,d)); c=np.zeros((2,d)); c[:,0]=model.residues[0]
    for mode in range(1,r):
        x,y=2*mode-1,2*mode
        freq=np.sqrt(model.lambdas[mode]); j[x,y]=-freq; j[y,x]=freq
        c[:,x]=model.residues[mode]
    if np.linalg.norm(model.tail)==0:
        return j-c.T@c,2*c.T,c,-np.eye(2),np.eye(d)
    eigen=np.linalg.eigvalsh(model.tail)
    if eigen[0]<=0: raise ValueError('Time realization requires full-rank PSD tail')
    ti=np.linalg.inv(model.tail)
    a=np.block([[j,c.T],[-ti@c,-ti]])
    b=np.vstack((np.zeros((d,2)),2*ti))
    output=np.hstack((np.zeros((2,d)),-np.eye(2)))
    metric=np.zeros_like(a); metric[:d,:d]=np.eye(d); metric[d:,d:]=model.tail
    return a,b,output,np.eye(2),metric
