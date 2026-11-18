"""Ordered selection of positions (SS) and of position pairs (E)."""
from typing import List, NamedTuple

import numpy as np
from scipy import stats as scipy_stats

from .columns import apply_delta_pair_from_cols, build_column_csc
from .stats import (center_ranks, fast_spearman_from_centered_x, sort_desc_with_tiebreak,
                    spearman_and_pairs, spearman_binary_columns)


class OrderingResult(NamedTuple):
    order: List[int]              # indices (columns of X, or pairs) sorted by individual ρ, highest first
    individual_rho: np.ndarray    # individual ρ of each index (NaN -> 0)
    cumulative_rho: List[float]   # ρ when adding the indices in that order
    selection: List[int]          # prefix with the highest cumulative ρ
    rho: float                    # cumulative ρ of the selection


def order_positions(X, dif) -> OrderingResult:
    """
    Single-site selection (SS): sorts the positions by their individual ρ with |ΔPh|, adds them
    up in that order summing their divergences, and keeps the prefix with the highest
    cumulative ρ. Ties: by index (highest first), as in the original version.
    """
    rho, _, _ = spearman_binary_columns(X, dif)
    values = np.nan_to_num(rho, nan=0.0)

    def original(k):
        r = scipy_stats.spearmanr(X[:, k], dif, nan_policy='omit').correlation
        return r if r == r else 0
    order = sort_desc_with_tiebreak(values, lambda k: X[:, k].tobytes(), original)

    cumulative = np.zeros(X.shape[0])
    cum_rho = []
    for k in order:
        cumulative = cumulative + X[:, k]
        cum_rho.append(scipy_stats.spearmanr(cumulative, dif, nan_policy='omit').correlation)
    best = int(np.nanargmax(np.array(cum_rho, dtype=np.float64)))
    return OrderingResult(order, values, cum_rho, order[:best + 1], cum_rho[best])


def order_pairs(X, pairs, dif, round_decimals=10) -> OrderingResult:
    """
    Same as order_positions with position pairs: a pair differs if both positions differ
    (AND variable). The ρ of each pair is computed with matrix products.
    """
    x_centered, std_x = center_ranks(dif)
    R, _, _ = spearman_and_pairs(X, dif)
    pi, pj = pairs[:, 0], pairs[:, 1]
    values = R[pi, pj]

    def and_pair(k):
        return X[:, pi[k]] & X[:, pj[k]]

    def original(k):
        return fast_spearman_from_centered_x(x_centered, std_x, and_pair(k).astype(np.float64),
                                             round_decimals=round_decimals)
    order = sort_desc_with_tiebreak(values, lambda k: and_pair(k).tobytes(), original)

    col_indptr, col_row_indices = build_column_csc(X)
    s = np.zeros(X.shape[0], dtype=np.float64)
    cum_rho = []
    for idx in order:
        apply_delta_pair_from_cols(s, col_row_indices, col_indptr, int(pi[idx]), int(pj[idx]), 1.0)
        cum_rho.append(fast_spearman_from_centered_x(x_centered, std_x, s, round_decimals=round_decimals))
    best = int(np.argmax(np.array(cum_rho)))
    return OrderingResult(order, values, cum_rho, order[:best + 1], cum_rho[best])
