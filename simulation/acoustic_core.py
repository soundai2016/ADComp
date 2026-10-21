"""Compatible finite-volume operators and physical acoustic ports."""
import numpy as np
from scipy import sparse

MATERIALS = ['rotated_stripes', 'double_inclusion', 'smooth_checker']
CONFIG = {
    'source_centres': [[.19,.76],[.54,.47],[.81,.22]],
    'receiver_centres': [[.22,.21],[.79,.77],[.28,.69],[.68,.33]],
    'sigma': .07, 'time_end': 8., 'samples': 641, 'threads': 2,
}

def material(n, name):
    x = (np.arange(n)+.5)/n
    X,Y = np.meshgrid(x,x,indexing='ij')
    if name == 'homogeneous':
        rho = np.ones_like(X); bulk = np.ones_like(X)
    elif name == 'rotated_stripes':
        rho = np.exp(.65*np.sin(4*np.pi*(.8*X+.6*Y))+.1*np.cos(2*np.pi*Y))
        bulk = np.exp(-.45*np.sin(4*np.pi*(.8*X+.6*Y))+.12*np.sin(2*np.pi*X))
    elif name == 'double_inclusion':
        a = np.exp(-((X-.34)**2+(Y-.35)**2)/(2*.1**2))
        b = np.exp(-((X-.70)**2+(Y-.67)**2)/(2*.12**2))
        rho = np.exp(1.2*a-.55*b+.1*np.cos(2*np.pi*X))
        bulk = np.exp(-.8*a+.9*b+.1*np.sin(2*np.pi*Y))
    elif name == 'smooth_checker':
        z = np.cos(4*np.pi*X)*np.cos(4*np.pi*Y)
        rho = np.exp(.8*z+.1*np.sin(2*np.pi*X))
        bulk = np.exp(-.7*z+.15*np.cos(2*np.pi*Y))
    else:
        raise ValueError(name)
    return X,Y,rho,bulk


def build_sparse(n, name):
    X,Y,rho,bulk = material(n,name)
    idx = np.arange(n*n).reshape(n,n)
    a = np.concatenate([idx[:-1,:].ravel(),idx[:,:-1].ravel()])
    b = np.concatenate([idx[1:,:].ravel(),idx[:,1:].ravel()])
    faces = np.arange(len(a))
    face_rho = .5*(rho.ravel()[a]+rho.ravel()[b])
    D = sparse.coo_matrix((np.concatenate([np.full(len(a),n),np.full(len(a),-n)]),
                          (np.concatenate([a,b]),np.concatenate([faces,faces]))),
                         shape=(n*n,len(a))).tocsr()
    mass = (1/(n*n*bulk)).ravel()
    # K_h=h^2 D rho_face^-1 D^T, A=M^-1/2 K_h M^-1/2=C C^T.
    C = sparse.diags(1/np.sqrt(mass)) @ (D/n) @ sparse.diags(1/np.sqrt(face_rho))
    J = sparse.bmat([[None,-C],[C.T,None]],format='csr')
    return X,Y,rho,bulk,mass,C,J


def ports(X,Y,bulk,centres,zero_mean):
    n = X.shape[0]
    columns = []
    for x,y in centres:
        f = np.exp(-((X-x)**2+(Y-y)**2)/(2*CONFIG['sigma']**2))
        f *= n*n/f.sum()
        if zero_mean:
            f -= f.mean()
        columns.append((np.sqrt(bulk)*f/n).ravel())
    return np.array(columns).T

