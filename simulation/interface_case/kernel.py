"""An interface-aligned impedance step with an exact acoustic modal solution.

The finite analytic modal initial field is fixed in physical coordinates,
independently of the finite-volume grid.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy import sparse, linalg, special
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())


def _gauss_interval(order, a, b):
    x, w = leggauss(order)
    return (a + b) / 2 + (b - a) * x / 2, w * (b - a) / 2


def x_modes(x, maximum=None):
    """Beta-orthonormal x modes; beta=1 on the left and 1/8 on the right."""
    maximum = CFG["max_x_mode"] if maximum is None else maximum
    m = np.arange(maximum + 1)
    norm2 = np.where(m == 0, 9 / 16, np.where(m % 2 == 0, 9 / 32, 9 / 4))
    scale = np.where((np.asarray(x)[:, None] > .5) & (m[None, :] % 2 == 1), 8., 1.)
    return scale * np.cos(np.pi * np.asarray(x)[:, None] * m[None, :]) / np.sqrt(norm2)


def y_modes(y, maximum=None):
    maximum = CFG["max_y_mode"] if maximum is None else maximum
    modes = np.arange(maximum + 1)
    return np.cos(np.pi * np.asarray(y)[:, None] * modes[None, :]) * np.where(modes == 0, 1., np.sqrt(2.))


def gaussian_integrals(edges, centre):
    z = (np.asarray(edges) - centre) / (np.sqrt(2.) * CFG["sigma"])
    return np.sqrt(np.pi / 2) * CFG["sigma"] * np.diff(special.erf(z))


@lru_cache(maxsize=4)
def analytic_data(order=None):
    order = CFG["quadrature_order"] if order is None else order
    xl, wl = _gauss_interval(order, 0., .5)
    xr, wr = _gauss_interval(order, .5, 1.)
    x = np.r_[xl, xr]
    wx = np.r_[wl, wr]
    beta = np.r_[np.ones(order), np.full(order, 1 / 8)]
    y, wy = _gauss_interval(order, 0., 1.)
    X = x_modes(x)
    Y = y_modes(y)
    coefficients = []
    for cx, cy in CFG["source_centres"]:
        gx = np.exp(-.5 * ((x - cx) / CFG["sigma"]) ** 2)
        gy = np.exp(-.5 * ((y - cy) / CFG["sigma"]) ** 2)
        aa = np.outer(X.T @ (wx * beta * gx), Y.T @ (wy * gy))
        aa[0, 0] = 0.
        aa *= np.sqrt(2 * CFG["initial_energy"]) / np.linalg.norm(aa)
        coefficients.append(aa)
    coefficients = np.stack(coefficients, axis=-1)
    receivers = []
    for cx, cy in CFG["receiver_centres"]:
        gx = np.exp(-.5 * ((x - cx) / CFG["sigma"]) ** 2)
        gy = np.exp(-.5 * ((y - cy) / CFG["sigma"]) ** 2)
        nx = gaussian_integrals([0., 1.], cx)[0]
        ny = gaussian_integrals([0., 1.], cy)[0]
        receivers.append(np.outer(X.T @ (wx * gx) / nx, Y.T @ (wy * gy) / ny))
    m = np.arange(CFG["max_x_mode"] + 1)
    ll = np.arange(CFG["max_y_mode"] + 1)
    omega = np.pi * np.sqrt(m[:, None] ** 2 + ll[None, :] ** 2)
    return {"coefficients": coefficients, "receiver_modes": np.stack(receivers, axis=-1),
            "omega": omega, "x_gram": X.T @ ((wx * beta)[:, None] * X),
            "y_gram": Y.T @ (wy[:, None] * Y)}


def averaged_modes(n):
    if n % 2:
        raise ValueError("An even grid is required for interface alignment")
    centres = (np.arange(n) + .5) / n
    m = np.arange(CFG["max_x_mode"] + 1)
    ll = np.arange(CFG["max_y_mode"] + 1)
    return (x_modes(centres) * np.sinc(m / (2 * n))[None, :],
            y_modes(centres) * np.sinc(ll / (2 * n))[None, :])


def exact_receiver(times, order=None):
    data = analytic_data(order)
    modal = np.cos(np.asarray(times)[:, None, None] * data["omega"][None, :, :])
    return np.einsum("tml,mls,mlr->trs", modal, data["coefficients"],
                     data["receiver_modes"], optimize=True)


def exact_pressure_norm_squared(times):
    """Continuous integral of beta*p(t)^2, separately for the three sources."""
    data = analytic_data()
    phase = np.cos(np.asarray(times)[:, None, None] * data["omega"][None, :, :])
    return np.einsum("tml,mls->ts", phase ** 2, data["coefficients"] ** 2, optimize=True)


def cell_projection_norm_squared(n, times):
    """Norm of the exact cell averages (n exceeds the largest modal index)."""
    if n <= max(CFG["max_x_mode"], CFG["max_y_mode"]):
        return np.sum(exact_mass_weighted_pressure(n, times) ** 2, axis=1)
    data = analytic_data()
    m = np.arange(CFG["max_x_mode"] + 1)
    ll = np.arange(CFG["max_y_mode"] + 1)
    filter2 = np.sinc(m / (2 * n))[:, None] ** 2 * np.sinc(ll / (2 * n))[None, :] ** 2
    phase = np.cos(np.asarray(times)[:, None, None] * data["omega"][None, :, :])
    return np.einsum("tml,mls,ml->ts", phase ** 2, data["coefficients"] ** 2, filter2, optimize=True)


def exact_mass_weighted_pressure(n, times):
    data = analytic_data()
    X, Y = averaged_modes(n)
    modal = np.cos(np.asarray(times)[:, None, None] * data["omega"][None, :, :])[:, :, :, None]
    modal = modal * data["coefficients"][None, :, :, :]
    pressure = np.einsum("im,tmls,jl->tijs", X, modal, Y, optimize=True).reshape(len(times), n * n, 3)
    mass = np.repeat(np.where((np.arange(n) + .5) / n < .5, 1., 1 / 8), n) / n ** 2
    return pressure * np.sqrt(mass)[None, :, None]


def exact_solution(n, times):
    return {"time": np.asarray(times), "mass_weighted_pressure": exact_mass_weighted_pressure(n, times),
            "receiver_pressure": exact_receiver(times)}


def build_sparse(n):
    if n % 2:
        raise ValueError("An even grid is required for interface alignment")
    x = (np.arange(n) + .5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = np.where(X < .5, 1., 8.)
    bulk = rho.copy()
    index = np.arange(n * n).reshape(n, n)
    a = np.r_[index[:-1, :].ravel(), index[:, :-1].ravel()]
    b = np.r_[index[1:, :].ravel(), index[:, 1:].ravel()]
    faces = np.arange(len(a))
    face_rho = .5 * (rho.ravel()[a] + rho.ravel()[b])
    incidence = sparse.coo_matrix((np.r_[np.ones(len(a)), -np.ones(len(a))],
                                  (np.r_[a, b], np.r_[faces, faces])), shape=(n * n, len(a))).tocsr()
    mass = (1 / (n * n * bulk)).ravel()
    C = sparse.diags(1 / np.sqrt(mass)) @ incidence @ sparse.diags(1 / np.sqrt(face_rho))
    J = sparse.bmat([[None, -C], [C.T, None]], format="csr")
    return X, Y, rho, bulk, mass, C, J


def source_receiver(n, mass):
    B = exact_mass_weighted_pressure(n, [0.])[0]
    edges = np.linspace(0., 1., n + 1)
    windows = []
    for cx, cy in CFG["receiver_centres"]:
        gx = gaussian_integrals(edges, cx)
        gy = gaussian_integrals(edges, cy)
        weights = np.outer(gx / gx.sum(), gy / gy.sum()).ravel()
        windows.append(weights / np.sqrt(mass))
    return B, np.stack(windows, axis=-1)


def assembly(n, need_stiffness=True):
    X, Y, rho, bulk, mass, C, J = build_sparse(n)
    B, R = source_receiver(n, mass)
    A = (C @ C.T).tocsr() if need_stiffness else None
    return mass, A, J, B, R, rho, bulk


def separable_setup(n):
    """Complete n^2 eigensystem from two n-dimensional tridiagonal problems.
    
    Setup includes the physical initial field and receiver construction.
    """
    xx = (np.arange(n) + .5) / n
    bulk_x = np.where(xx < .5, 1., 8.)
    face_rho = .5 * (bulk_x[:-1] + bulk_x[1:])
    inv_face = 1 / face_rho
    dx = n * n * bulk_x * np.r_[inv_face[0], inv_face[:-1] + inv_face[1:], inv_face[-1]]
    ex = -n * n * np.sqrt(bulk_x[:-1] * bulk_x[1:]) * inv_face
    dy = np.r_[1., np.full(n - 2, 2.), 1.] * n * n
    ey = np.full(n - 1, -n * n, dtype=float)
    lx, Vx = linalg.eigh_tridiagonal(dx, ex)
    ly, Vy = linalg.eigh_tridiagonal(dy, ey)
    lx[np.abs(lx) < 1e-8] = 0.
    ly[np.abs(ly) < 1e-8] = 0.
    mass = np.repeat(1 / bulk_x, n) / n ** 2
    B, R = source_receiver(n, mass)
    source_modes = np.einsum("im,ijs,jl->mls", Vx, B.reshape(n, n, 3), Vy, optimize=True)
    receiver_modes = np.einsum("im,ijr,jl->mlr", Vx, R.reshape(n, n, 4), Vy, optimize=True)
    return {"x_eigenvalues": lx, "y_eigenvalues": ly, "x_eigenvectors": Vx,
            "y_eigenvectors": Vy, "initial_modal_coefficients": source_modes,
            "receiver_modes": receiver_modes, "mass": mass,
            "omega": np.sqrt(lx[:, None] + ly[None, :])}


def separable_receiver(model, times):
    n = len(model["x_eigenvalues"])
    weights = np.einsum("mlr,mls->mlrs", model["receiver_modes"], model["initial_modal_coefficients"]).reshape(n * n, 12)
    result = np.empty((len(times), 4, 3))
    for start in range(0, len(times), 32):
        tt = np.asarray(times)[start:start + 32]
        phase = np.cos(tt[:, None] * model["omega"].ravel()[None, :])
        result[start:start + len(tt)] = (phase @ weights).reshape(len(tt), 4, 3)
    return result


def separable_pressure(model, times):
    n = len(model["x_eigenvalues"])
    modal = np.cos(np.asarray(times)[:, None, None] * model["omega"][None, :, :])[:, :, :, None]
    modal = modal * model["initial_modal_coefficients"][None, :, :, :]
    return np.einsum("im,tmls,jl->tijs", model["x_eigenvectors"], modal,
                     model["y_eigenvectors"], optimize=True).reshape(len(times), n * n, 3)


def trace_only(n, times):
    return separable_receiver(separable_setup(n), times)


def separable_full(n, times):
    model = separable_setup(n)
    return {"time": np.asarray(times), "mass_weighted_pressure": separable_pressure(model, times),
            "receiver_pressure": separable_receiver(model, times), **model}
