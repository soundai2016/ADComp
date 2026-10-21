"""Signed error changes over coherent combinations of the prescribed sources."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from core import DEFAULT_ID, SIGN_TOLERANCE, WINDOWS, modal_response


def signature(values):
    low, high = float(values[0]), float(values[-1])
    if low > SIGN_TOLERANCE:
        return "positive"
    if high < -SIGN_TOLERANCE:
        return "negative"
    if low < -SIGN_TOLERANCE and high > SIGN_TOLERANCE:
        return "indefinite"
    return "borderline"


def source_subspace_rows(n, m, times, omega_r, omega_h, omega_f, br, cr, bf, cf,
                         reduced, reference, triples, selection_id, selection_type):
    ri, hi = np.asarray(triples, dtype=int)[:, :2].T
    phase = np.cos(np.outer(times, omega_r[ri])) - np.cos(np.outer(times, omega_h[hi]))
    rows, matrices = [], []
    for source_layout in range(2):
        for receiver_layout in range(2):
            a, b = source_layout, receiver_layout
            removed = np.einsum("tj,rj,js->trs", phase, cr[b][:, ri], br[a, ri], optimize=True)
            for end in WINDOWS:
                active = times <= end + 1e-12
                F = reference[a, b, active].reshape(-1, 3)
                E = (reduced[a, b, active] - reference[a, b, active]).reshape(-1, 3)
                A = removed[active].reshape(-1, 3)
                G = F.T @ F
                D = A.T @ A - E.T @ A - A.T @ E
                G, D = (G + G.T) / 2, (D + D.T) / 2
                gram_values = np.linalg.eigvalsh(G)
                if gram_values[0] <= 1e-12 * gram_values[-1]:
                    raise ValueError("The prescribed source subspace is not resolved by these observations.")
                values, vectors = eigh(D, G)
                norm_g = float(np.linalg.norm(G, 2))
                direct_matrix = (E - A).T @ (E - A) - E.T @ E
                direct_defect = float(np.linalg.norm(D - direct_matrix, 2) / norm_g)
                diagonal = np.diag(D) / np.diag(G)
                row = dict(grid=n, modes_per_patch=m, dimension=len(omega_r),
                    source_layout=a, receiver_layout=b, time_end=end,
                    selection_id=selection_id, selection_type=selection_type,
                    matched_count=len(ri), signature=signature(values),
                    lambda_min=float(values[0]), lambda_mid=float(values[1]), lambda_max=float(values[2]),
                    reference_gram_condition=float(gram_values[-1] / gram_values[0]),
                    aggregate_delta_squared_error=float(np.trace(D) / np.trace(G)),
                    all_individual_positive=bool(np.min(diagonal) > SIGN_TOLERANCE),
                    all_individual_negative=bool(np.max(diagonal) < -SIGN_TOLERANCE),
                    matrix_identity_defect=direct_defect,
                    eigenvalue_direct_defect=0., modal_witness_defect=0.,
                    modal_witness_checked=selection_id == DEFAULT_ID)
                for j in range(3):
                    denominator = float(G[j, j])
                    original = float(np.linalg.norm(E[:, j]) / np.sqrt(denominator))
                    substituted = float(np.linalg.norm(E[:, j] - A[:, j]) / np.sqrt(denominator))
                    row.update({f"source_{j + 1}_delta_squared_error": float(diagonal[j]),
                                f"source_{j + 1}_change_pp": 100 * (substituted - original)})
                for index, side in ((0, "min"), (-1, "max")):
                    weights = vectors[:, index] / np.linalg.norm(vectors[:, index])
                    weights *= np.sign(weights[np.argmax(np.abs(weights))])
                    denominator = float(np.linalg.norm(F @ weights) ** 2)
                    original = float(np.linalg.norm(E @ weights) / np.sqrt(denominator))
                    substituted = float(np.linalg.norm((E - A) @ weights) / np.sqrt(denominator))
                    direct = substituted ** 2 - original ** 2
                    row["eigenvalue_direct_defect"] = max(row["eigenvalue_direct_defect"],
                                                           abs(direct - values[index]))
                    row.update({f"{side}_original_error": original, f"{side}_substituted_error": substituted,
                                f"{side}_change_pp": 100 * (substituted - original)})
                    for j, weight in enumerate(weights, 1):
                        row[f"{side}_source_weight_{j}"] = float(weight)
                    if selection_id == DEFAULT_ID:
                        mixed_r = modal_response(times[active], omega_r, br[a] @ weights[:, None], cr[b])
                        mixed_f = modal_response(times[active], omega_f, bf[a] @ weights[:, None], cf[b])
                        mixed_a = np.einsum("tj,rj,j->tr", phase[active], cr[b][:, ri],
                                            br[a, ri] @ weights, optimize=True)[..., None]
                        mixed_error = mixed_r - mixed_f
                        witnessed = (np.sum((mixed_error - mixed_a) ** 2) - np.sum(mixed_error ** 2)) / np.sum(mixed_f ** 2)
                        row["modal_witness_defect"] = max(row["modal_witness_defect"],
                                                          float(abs(witnessed - values[index])))
                rows.append(row)
                matrices.append((G, D))
    return rows, matrices


def summarize(rows, budgets):
    def counts(subset):
        return dict(cases=len(subset),
            **{name: sum(row["signature"] == name for row in subset)
               for name in ("positive", "negative", "indefinite", "borderline")},
            positive_aggregate_indefinite=sum(row["aggregate_delta_squared_error"] > SIGN_TOLERANCE
                                              and row["signature"] == "indefinite" for row in subset),
            all_individual_positive_indefinite=sum(row["all_individual_positive"]
                                                   and row["signature"] == "indefinite" for row in subset),
            all_individual_negative_indefinite=sum(row["all_individual_negative"]
                                                   and row["signature"] == "indefinite" for row in subset))

    keys = ("grid", "modes_per_patch", "source_layout", "receiver_layout", "time_end", "selection_id")
    aggregate = {tuple(row[key] for key in keys): row for row in budgets if row["group"] == "aggregate"}
    replay_defect = max(abs(row["aggregate_delta_squared_error"] -
                           aggregate[tuple(row[key] for key in keys)]["delta_squared_error"]) for row in rows)
    checks = dict(cases=len(rows), default_cases=sum(row["selection_id"] == DEFAULT_ID for row in rows),
        maximum_aggregate_replay_defect=replay_defect,
        maximum_matrix_identity_defect=max(row["matrix_identity_defect"] for row in rows),
        maximum_eigenvalue_direct_defect=max(row["eigenvalue_direct_defect"] for row in rows),
        maximum_modal_witness_defect=max(row["modal_witness_defect"] for row in rows),
        modal_witness_cases=sum(row["modal_witness_checked"] for row in rows),
        maximum_reference_gram_condition=max(row["reference_gram_condition"] for row in rows))
    if checks["cases"] != 720 or checks["default_cases"] != 72:
        raise ValueError("Incomplete source-subspace comparison set.")
    for name, value in checks.items():
        if name.endswith("defect") and value >= 1e-10:
            raise ValueError(f"Source-subspace verification failed: {name}={value}")
    defaults = [row for row in rows if row["selection_id"] == DEFAULT_ID]
    example = next(row for row in defaults if (row["grid"], row["modes_per_patch"],
        row["source_layout"], row["receiver_layout"], row["time_end"]) == (64, 64, 1, 1, 2.))
    return dict(scope="Real coherent combinations of three prescribed initial pressure fields; four receivers jointly; fixed layouts, spectra, basis and matching set.",
        sign_tolerance=SIGN_TOLERANCE, default=counts(defaults), all_selections=counts(rows),
        by_time={str(end): counts([row for row in defaults if row["time_end"] == end]) for end in WINDOWS},
        example=example, verification=checks)
