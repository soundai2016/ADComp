"""Modal matching and finite-record algebra shared by extraction and analysis."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GRIDS = (64, 128)
RANKS = (32, 64, 128)
WINDOWS = (0.5, 2.0, 8.0)
MACS = (0.85, 0.90, 0.95)
GAPS = (0.0025, 0.005, 0.01)
DEFAULT_ID = "mac0.90_gap0.0050"
SIGN_TOLERANCE = 1e-12  # Applied to normalized squared-error changes.
ASSIGNMENT_KEYS = ("left", "right", "mac", "sign", "gap_left", "gap_right", "omega_left", "omega_right")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_gaps(omega):
    return np.minimum(np.r_[np.inf, np.diff(omega)], np.r_[np.diff(omega), np.inf]) / np.maximum(omega, 1e-100)


def assignment(overlap, omega_left, omega_right):
    """Global MAC-maximizing assignment before applying acceptance filters."""
    left, right = linear_sum_assignment(-(overlap ** 2))
    return dict(left=left, right=right, mac=overlap[left, right] ** 2,
                sign=np.sign(overlap[left, right]),
                gap_left=relative_gaps(omega_left)[left],
                gap_right=relative_gaps(omega_right)[right],
                omega_left=omega_left[left], omega_right=omega_right[right])


def accepted(a, mac, gap):
    return ((a["mac"] >= mac) & (a["gap_left"] >= gap) & (a["gap_right"] >= gap)
            & (a["omega_left"] > 1e-7) & (a["omega_right"] > 1e-7))


def chain(ar, af, omega_f, active_max, norms, mac, gap):
    """Return (reduced, full, continuous, reduced-pair, continuous-pair)."""
    ok = accepted(af, mac, gap)
    jj = af["right"]
    ok &= (omega_f[jj] <= active_max + 1e-9) & (norms[jj] > 1e-8)
    continuum = {int(i): (int(j), k) for k, (i, j, yes) in enumerate(zip(af["left"], jj, ok)) if yes}
    triples = []
    for k, (i, j, yes) in enumerate(zip(ar["left"], ar["right"], accepted(ar, mac, gap))):
        if yes and int(j) in continuum:
            f, q = continuum[int(j)]
            triples.append((int(i), int(j), f, k, q))
    return triples


def store_assignment(arrays, prefix, values):
    for key in ASSIGNMENT_KEYS:
        arrays[f"{prefix}_{key}"] = values[key]


def get_assignment(arrays, prefix):
    return {key: arrays[f"{prefix}_{key}"] for key in ASSIGNMENT_KEYS}


def modal_response(times, omega, coefficients, observation):
    residues = np.einsum("rj,js->jrs", observation, coefficients)
    return np.einsum("tj,jrs->trs", np.cos(np.outer(times, omega)), residues, optimize=True)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def outcome_groups():
    yield "aggregate", 0, 0, np.ones((4, 3), dtype=bool)
    for s in range(1, 4):
        for r in range(1, 5):
            mask = np.zeros((4, 3), dtype=bool)
            mask[r - 1, s - 1] = True
            yield "channel", s, r, mask


def analysis_rows(n, m, times, omega_r, omega_h, omega_f, br_all, cr_all,
                  bf_all, cf_all, yr_all, yf_all, triples, ar, af,
                  selection_id, selection_type, mac, gap):
    """All four port layouts, three windows and all thirteen observation groups.

    Only the ROM-residue intervention is used. The unmatched response is retained
    exactly; its contribution is recomputed from the complete ROM/reference sums.
    """
    indices = np.asarray(triples, dtype=int).reshape(-1, 5)
    if len(indices) == 0:
        raise ValueError("The diagnostic requires at least one matched triple")
    ri, hi, fi, kr, kf = indices.T
    zr, zh, zf = (np.cos(np.outer(times, w)) for w in (omega_r[ri], omega_h[hi], omega_f[fi]))
    phase_a, phase_b = zr - zh, zh - zf
    signs_r, signs_f = ar["sign"][kr], af["sign"][kf]
    rows = []
    for a in range(2):
        for b in range(2):
            br = br_all[a, ri] * signs_r[:, None]
            cr = cr_all[b][:, ri] * signs_r[None]
            bf = bf_all[a, fi] * signs_f[:, None]
            cf = cf_all[b][:, fi] * signs_f[None]
            hr = np.einsum("rj,js->jrs", cr, br)
            hf = np.einsum("rj,js->jrs", cf, bf)
            aa = np.einsum("tj,jrs->trs", phase_a, hr, optimize=True)
            bb = np.einsum("tj,jrs->trs", phase_b, hr, optimize=True)
            source = np.einsum("tj,rj,js->trs", zf, (cr + cf) / 2, br - bf, optimize=True)
            receiver = np.einsum("tj,rj,js->trs", zf, cr - cf, (br + bf) / 2, optimize=True)
            reference = yf_all[a, b]
            error = yr_all[a, b] - reference
            selected_difference = np.einsum("tj,jrs->trs", zr, hr, optimize=True) - np.einsum("tj,jrs->trs", zf, hf, optimize=True)
            unmatched = error - selected_difference
            qq = source + receiver + unmatched
            omitted = reference - np.einsum("tj,jrs->trs", zf, hf, optimize=True)
            closure = relative(error, aa + bb + qq)
            for end in WINDOWS:
                active = times <= end + 1e-12
                same_per_mode = np.sum(-phase_a[active] ** 2 - 2 * phase_a[active] * phase_b[active], axis=0)
                aggregate_den = float(np.sum(reference[active] ** 2))
                for group, s, r, mask in outcome_groups():
                    ref = reference[active][:, mask]
                    er = error[active][:, mask]
                    ra, gb, rem = (v[active][:, mask] for v in (aa, bb, qq))
                    den = float(np.sum(ref ** 2))
                    if den <= 0:
                        raise ValueError("A reference observation has zero squared norm")
                    q0, q1 = float(np.sum(er ** 2)), float(np.sum((er - ra) ** 2))
                    total, replaced = np.sqrt(q0 / den), np.sqrt(q1 / den)
                    direct = (q1 - q0) / den
                    same = float(same_per_mode @ np.sum(hr[:, mask] ** 2, axis=1) / den)
                    freq = float((-np.sum(ra ** 2) - 2 * np.sum(ra * gb)) / den)
                    qcross = float(-2 * np.sum(ra * rem) / den)
                    rows.append(dict(grid=n, modes_per_patch=m, dimension=len(omega_r),
                        selection_id=selection_id, selection_type=selection_type,
                        mac_threshold=mac, relative_gap=gap, matched_count=len(ri),
                        source_layout=a, receiver_layout=b, time_end=end,
                        group=group, source=s, receiver=r,
                        total_receiver_error=float(total), coarse_replacement_error=float(replaced),
                        change_percentage_points=float(100 * (replaced - total)),
                        delta_squared_error=float(direct), raw_squared_error_change=q1 - q0,
                        same_mode_term=same, cross_mode_term=freq - same, remainder_cross_term=qcross,
                        source_cross_term=float(-2 * np.sum(ra * source[active][:, mask]) / den),
                        receiver_cross_term=float(-2 * np.sum(ra * receiver[active][:, mask]) / den),
                        unmatched_cross_term=float(-2 * np.sum(ra * unmatched[active][:, mask]) / den),
                        reference_squared_norm=den,
                        # Channel share of the sampled squared-pressure norm.
                        reference_energy_fraction=den / aggregate_den,
                        matched_reference_omission=float(np.linalg.norm(omitted[active][:, mask]) / np.sqrt(den)),
                        budget_absolute_defect=abs(direct - freq - qcross),
                        decomposition_relative_defect=closure))
    return rows
