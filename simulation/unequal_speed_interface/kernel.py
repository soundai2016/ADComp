"""Exact piecewise acoustic modes for a planar unequal-speed interface.

The finite continuous modal initial field is fixed before grid projection.
Continuous roots, full discrete modes, and reduced modes are constructed
independently. The separated solvers evaluate modal responses without time stepping.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import linalg, sparse, special
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())


def gauss_interval(order, a, b):
    x, w = leggauss(order)
    return (a+b)/2 + (b-a)*x/2, (b-a)*w/2


def cosine_q(q2, d):
    q2, d = np.broadcast_arrays(np.asarray(q2, dtype=float), np.asarray(d, dtype=float))
    ans = np.empty(q2.shape)
    positive = q2 >= 0
    ans[positive] = np.cos(np.sqrt(q2[positive]) * d[positive])
    ans[~positive] = np.cosh(np.sqrt(-q2[~positive]) * d[~positive])
    return ans


def q_sine(q2, d):
    q2, d = np.broadcast_arrays(np.asarray(q2, dtype=float), np.asarray(d, dtype=float))
    ans = np.empty(q2.shape)
    positive = q2 >= 0
    qq = np.sqrt(q2[positive])
    ans[positive] = qq * np.sin(qq * d[positive])
    qq = np.sqrt(-q2[~positive])
    ans[~positive] = -qq * np.sinh(qq * d[~positive])
    return ans


def mean_cosine(q2, centre, width):
    q2, centre = np.broadcast_arrays(np.asarray(q2, dtype=float), np.asarray(centre, dtype=float))
    answer = cosine_q(q2, centre)
    positive = q2 >= 0
    answer[positive] *= np.sinc(np.sqrt(q2[positive]) * width / (2*np.pi))
    z = np.sqrt(-q2[~positive]) * width/2
    answer[~positive] *= np.sinh(z)/z
    return answer


def integral_cosine_squared(q2, length):
    if q2 >= 0:
        return length/2 * (1 + np.sinc(2*np.sqrt(q2)*length/np.pi))
    z = 2*np.sqrt(-q2)*length
    return length/2 * (1 + np.sinh(z)/z)


def material_parameters(density=None, bulk=None):
    rho = np.asarray(CFG["density"] if density is None else density, dtype=float)
    kk = np.asarray(CFG["bulk_modulus"] if bulk is None else bulk, dtype=float)
    return rho, kk, kk/rho


def characteristic(omega, ell, density=None, bulk=None, interface=None):
    rho, kk, c2 = material_parameters(density, bulk)
    a = CFG["interface_x"] if interface is None else interface
    omega = np.asarray(omega)
    qq = omega[..., None]**2/c2 - (ell*np.pi)**2
    cc = cosine_q(qq, [a, 1-a])
    ss = q_sine(qq, [a, 1-a])/rho
    return cc[..., 0]*ss[..., 1] + cc[..., 1]*ss[..., 0]


@lru_cache(maxsize=256)
def roots(ell, count, density=None, bulk=None, scan_step=None):
    rho, kk, c2 = material_parameters(density, bulk)
    step = CFG["root_scan_step"] if scan_step is None else scan_step
    lower = np.sqrt(c2.min()) * ell*np.pi
    upper = np.sqrt(c2.max()) * np.pi*np.sqrt((count+2)**2 + ell**2)
    exact_lowest = ell == 0 or np.isclose(c2[0], c2[1], rtol=1e-14, atol=0.)
    first = lower+1e-8 if exact_lowest else lower
    grid = np.linspace(first, upper, int(np.ceil((upper-lower)/step))+1)
    value = characteristic(grid, ell, tuple(rho), tuple(kk))
    indices = np.flatnonzero(np.signbit(value[:-1]) != np.signbit(value[1:]))
    found = [float(lower)] if exact_lowest else []
    for j in indices:
        root = brentq(characteristic, grid[j], grid[j+1],
                      args=(ell, tuple(rho), tuple(kk)), xtol=5e-13, rtol=2e-14)
        if not found or root-found[-1] > 1e-7:
            found.append(root)
        if len(found) == count:
            break
    if len(found) != count:
        raise RuntimeError(f"Root count failed for ell={ell}: {len(found)} instead of {count}")
    return np.asarray(found)


def mode_parameters(omega, ell, density=None, bulk=None):
    rho, kk, c2 = material_parameters(density, bulk)
    a = CFG["interface_x"]
    q2 = omega**2/c2 - (ell*np.pi)**2
    cc = cosine_q(q2, [a, 1-a])
    ss = q_sine(q2, [a, 1-a])/rho
    matrix = np.array([[cc[0], -cc[1]], [ss[0], ss[1]]])
    # Scale the flux equation by a physical wavenumber, rather than normalizing
    # each row to unit norm: both sine values vanish at some exact roots, and
    # amplifying that roundoff-sized row would corrupt the pressure condition.
    flux_scale = max(1., np.sqrt(omega**2/c2.min()+(ell*np.pi)**2)/rho.min())
    scaled = matrix.copy()
    scaled[1] /= flux_scale
    _, _, vh = linalg.svd(scaled, check_finite=False)
    amplitude = vh[-1]
    if amplitude[0] < 0:
        amplitude = -amplitude
    norm2 = sum(amplitude[i]**2/kk[i] * integral_cosine_squared(q2[i], d)
                for i, d in enumerate((a, 1-a)))
    amplitude /= np.sqrt(norm2)
    return q2, amplitude


def x_values(x, q2, amplitude, average_width=None):
    x = np.asarray(x)
    left = x < CFG["interface_x"]
    ans = np.empty((len(x), len(q2)))
    for side, active, distance in ((0, left, x[left]), (1, ~left, 1-x[~left])):
        if average_width is None:
            vals = cosine_q(q2[None, :, side], distance[:, None])
        else:
            vals = mean_cosine(q2[None, :, side], distance[:, None], average_width)
        ans[active] = vals * amplitude[None, :, side]
    return ans


def y_values(y, maximum):
    ll = np.arange(maximum+1)
    return np.cos(np.pi*np.asarray(y)[:, None]*ll) * np.where(ll == 0, 1., np.sqrt(2.))


def gaussian_integrals(edges, centre):
    z = (np.asarray(edges)-centre)/(np.sqrt(2.)*CFG["sigma"])
    return np.sqrt(np.pi/2)*CFG["sigma"]*np.diff(special.erf(z))


@lru_cache(maxsize=8)
def continuous_data(order=None, scan_step=None):
    order = CFG["quadrature_order"] if order is None else order
    rho, kk, c2 = material_parameters()
    nx = CFG["max_x_mode"]+1
    nl = CFG["max_y_mode"]+1
    omega = np.stack([roots(l, nx, scan_step=scan_step) for l in range(nl)])
    q2 = np.empty((nl, nx, 2))
    amplitudes = np.empty_like(q2)
    for l in range(nl):
        for m in range(nx):
            q2[l, m], amplitudes[l, m] = mode_parameters(omega[l, m], l)
    a = CFG["interface_x"]
    xl, wl = gauss_interval(order, 0., a)
    xr, wr = gauss_interval(order, a, 1.)
    x, wx = np.r_[xl, xr], np.r_[wl, wr]
    beta = np.r_[np.full(order, 1/kk[0]), np.full(order, 1/kk[1])]
    y, wy = gauss_interval(order, 0., 1.)
    xx = np.stack([x_values(x, q2[l], amplitudes[l]) for l in range(nl)])
    yy = y_values(y, nl-1)
    coefficients = np.empty((nl, nx, 3))
    receiver_modes = np.empty((nl, nx, 4))
    gram = np.stack([xx[l].T @ ((wx*beta)[:, None]*xx[l]) for l in range(nl)])
    for s, (cx, cy) in enumerate(CFG["source_centres"]):
        gx = np.exp(-.5*((x-cx)/CFG["sigma"])**2)
        gy = np.exp(-.5*((y-cy)/CFG["sigma"])**2)
        sx = np.einsum("lim,i->lm", xx, wx*beta*gx)
        sy = yy.T @ (wy*gy)
        coeff = sx*sy[:, None]
        coeff[0, 0] = 0.
        coefficients[:, :, s] = coeff*np.sqrt(2*CFG["initial_energy"])/np.linalg.norm(coeff)
    for r, (cx, cy) in enumerate(CFG["receiver_centres"]):
        gx = np.exp(-.5*((x-cx)/CFG["sigma"])**2)
        gy = np.exp(-.5*((y-cy)/CFG["sigma"])**2)
        rx = np.einsum("lim,i->lm", xx, wx*gx)/gaussian_integrals([0., 1.], cx)[0]
        ry = (yy.T @ (wy*gy))/gaussian_integrals([0., 1.], cy)[0]
        receiver_modes[:, :, r] = rx*ry[:, None]
    return {"omega": omega, "q_squared": q2, "amplitudes": amplitudes,
            "coefficients": coefficients, "receiver_modes": receiver_modes,
            "x_gram": gram, "y_gram": yy.T @ (wy[:, None]*yy)}


def exact_receiver(times, order=None, scan_step=None):
    data = continuous_data(order, scan_step)
    phase = np.cos(np.asarray(times)[:, None, None]*data["omega"][None])
    return np.einsum("tlm,lms,lmr->trs", phase, data["coefficients"],
                     data["receiver_modes"], optimize=True)


def exact_pressure_norm_squared(times):
    data = continuous_data()
    phase = np.cos(np.asarray(times)[:, None, None]*data["omega"][None])
    return np.einsum("tlm,lms->ts", phase**2, data["coefficients"]**2, optimize=True)


@lru_cache(maxsize=8)
def averaged_modes(n):
    if n % 2:
        raise ValueError("The interface requires an even grid")
    data = continuous_data()
    x = (np.arange(n)+.5)/n
    xx = np.stack([x_values(x, data["q_squared"][l], data["amplitudes"][l], 1/n)
                   for l in range(CFG["max_y_mode"]+1)])
    ll = np.arange(CFG["max_y_mode"]+1)
    yy = y_values(x, CFG["max_y_mode"])*np.sinc(ll/(2*n))[None, :]
    return xx, yy


def mass_vector(n):
    x = (np.arange(n)+.5)/n
    bulk = np.where(x < CFG["interface_x"], *CFG["bulk_modulus"])
    return np.repeat(1/bulk, n)/n**2


def synthesize(x_modes, y_modes, omega, coefficients, times):
    """Return physical or normalized pressure for factorized modal vectors.

    x_modes has shape (l,i,m), y_modes (j,l), omega (l,m).
    """
    times = np.asarray(times)
    modal = np.cos(times[:, None, None]*omega[None])[:, :, :, None]*coefficients[None]
    nt, nl, nm, ns = modal.shape
    temporary = x_modes @ modal.transpose(1, 2, 0, 3).reshape(nl, nm, nt*ns)
    temporary = temporary.reshape(nl, len(x_modes[0]), nt, ns).transpose(2, 1, 0, 3)
    return np.einsum("tils,jl->tijs", temporary, y_modes, optimize=True).reshape(nt, -1, ns)


def exact_mass_weighted_pressure(n, times):
    data = continuous_data()
    xx, yy = averaged_modes(n)
    return synthesize(xx, yy, data["omega"], data["coefficients"], times)*np.sqrt(mass_vector(n))[None, :, None]


def source_receiver(n, mass):
    B = exact_mass_weighted_pressure(n, [0.])[0]
    edges = np.linspace(0., 1., n+1)
    windows = []
    for cx, cy in CFG["receiver_centres"]:
        gx, gy = gaussian_integrals(edges, cx), gaussian_integrals(edges, cy)
        windows.append(np.outer(gx/gx.sum(), gy/gy.sum()).ravel()/np.sqrt(mass))
    return B, np.stack(windows, axis=-1)


def assembly(n):
    if n % 2:
        raise ValueError("The interface requires an even grid")
    x = (np.arange(n)+.5)/n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = np.where(X < CFG["interface_x"], *CFG["density"])
    bulk = np.where(X < CFG["interface_x"], *CFG["bulk_modulus"])
    idx = np.arange(n*n).reshape(n, n)
    a = np.r_[idx[:-1].ravel(), idx[:, :-1].ravel()]
    b = np.r_[idx[1:].ravel(), idx[:, 1:].ravel()]
    faces = np.arange(len(a))
    face_rho = .5*(rho.ravel()[a]+rho.ravel()[b])
    D = sparse.coo_matrix((np.r_[np.ones(len(a)), -np.ones(len(a))],
                          (np.r_[a, b], np.r_[faces, faces])), shape=(n*n, len(a))).tocsr()
    mass = (1/(n*n*bulk)).ravel()
    C = sparse.diags(1/np.sqrt(mass)) @ D @ sparse.diags(1/np.sqrt(face_rho))
    J = sparse.bmat([[None, -C], [C.T, None]], format="csr")
    B, R = source_receiver(n, mass)
    return mass, (C@C.T).tocsr(), J, B, R, rho, bulk


def x_tridiagonal(n, density=None, bulk=None):
    rho, kk, c2 = material_parameters(density, bulk)
    x = (np.arange(n)+.5)/n
    rr = np.where(x < CFG["interface_x"], *rho)
    bb = np.where(x < CFG["interface_x"], *kk)
    inv = 2/(rr[:-1]+rr[1:])
    diagonal = n*n*bb*np.r_[inv[0], inv[:-1]+inv[1:], inv[-1]]
    off = -n*n*np.sqrt(bb[:-1]*bb[1:])*inv
    return diagonal, off, bb/rr


def separable_setup(n):
    dx, ex, speed2 = x_tridiagonal(n)
    dy = np.r_[1., np.full(n-2, 2.), 1.]*n*n
    ly, Vy = linalg.eigh_tridiagonal(dy, np.full(n-1, -n*n, dtype=float))
    ly[np.abs(ly) < 1e-8] = 0.
    lam = np.empty((n, n))
    Vx = np.empty((n, n, n))
    for ell in range(n):
        lam[ell], Vx[ell] = linalg.eigh_tridiagonal(dx+speed2*ly[ell], ex)
    lam[np.abs(lam) < 1e-8] = 0.
    if np.min(lam) < 0:
        raise RuntimeError("Indefinite separated stiffness")
    mass = mass_vector(n)
    B, R = source_receiver(n, mass)
    By = np.einsum("ijs,jl->lis", B.reshape(n, n, 3), Vy, optimize=True)
    Ry = np.einsum("ijr,jl->lir", R.reshape(n, n, 4), Vy, optimize=True)
    coeff = np.matmul(Vx.transpose(0, 2, 1), By)
    observation = np.matmul(Vx.transpose(0, 2, 1), Ry)
    return {"omega": np.sqrt(lam), "x_eigenvectors": Vx, "y_eigenvectors": Vy,
            "initial_modal_coefficients": coeff, "receiver_modes": observation, "mass": mass}


def separable_receiver(model, times):
    phase = np.cos(np.asarray(times)[:, None, None]*model["omega"][None])
    return np.einsum("tlm,lms,lmr->trs", phase, model["initial_modal_coefficients"],
                     model["receiver_modes"], optimize=True)


def separable_pressure(model, times):
    return synthesize(model["x_eigenvectors"], model["y_eigenvectors"],
                      model["omega"], model["initial_modal_coefficients"], times)


def full_eigenvectors(model):
    n = len(model["mass"])**.5
    n = int(n)
    vectors = np.einsum("lim,jl->ijlm", model["x_eigenvectors"], model["y_eigenvectors"],
                        optimize=True).reshape(n*n, n*n)
    order = np.argsort(model["omega"].ravel())
    return model["omega"].ravel()[order], vectors[:, order], order
