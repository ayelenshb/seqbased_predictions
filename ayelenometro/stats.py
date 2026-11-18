"""Spearman ranks and correlations (fast, vectorized versions)."""
import numba
import numpy as np


def rankdata_average(values, round_decimals=10):
    """
    Average ranks (1..n) as float64, with optional rounding to stabilize near-ties.
    The order within each group of ties does not change the average rank, so the default
    argsort is enough (about 5 times faster than the stable mergesort used before).
    """
    v = np.asarray(values, dtype=np.float64)
    if round_decimals is not None:
        v = np.round(v, decimals=round_decimals)
    order = np.argsort(v)
    return _rank_average_from_order(v, order)

@numba.njit
def _rank_average_from_order(v, order):
    n = v.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        val = v[order[i]]
        while j < n and v[order[j]] == val:
            j += 1
        avg_rank = 0.5 * (i + j + 1)      # average of the 1-based ranks (i+1) .. j
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def center_ranks(values):
    """Centered ranks of a fixed vector and their (population) SD, to reuse in many Spearman calls."""
    ranks = rankdata_average(values, round_decimals=None)
    n = ranks.size
    x_centered = ranks - 0.5 * (n + 1)
    std_x = np.sqrt(float(np.dot(x_centered, x_centered)) / n)
    if std_x == 0.0:
        std_x = 1.0
    return x_centered, std_x


def fast_spearman_from_centered_x(x_centered, std_x, y_values, round_decimals=10):
    """Spearman between a fixed x (ranks already centered) and y_values."""
    y_ranks = rankdata_average(y_values, round_decimals=round_decimals)
    n = y_ranks.size
    y_centered = y_ranks - 0.5 * (n + 1)
    dot_xy = float(np.dot(x_centered, y_centered))
    var_y = float(np.dot(y_centered, y_centered)) / n
    if var_y <= 0.0 or std_x <= 0.0:
        return 0.0
    return dot_xy / (n * std_x * np.sqrt(var_y))


def calculate_spearman_regularized_correlation_fast(
    x_centered, std_x, sum_rows_vector, coefs, lambda_reg, total_positions, round_decimals=10
):
    """ρ_reg = ρ - λ·Σ|w| / total_positions. Returns (ρ_reg, ρ, penalty)."""
    spearman = fast_spearman_from_centered_x(x_centered, std_x, sum_rows_vector, round_decimals)
    reg = lambda_reg * (np.nansum(np.abs(coefs)) / total_positions)
    return spearman - reg, spearman, reg


def spearman_binary_columns(X, values):
    """
    Spearman of each 0/1 column of X against `values`, all at once.

    For a binary variable, Spearman = Pearson with the ranks of `values`, which gives
        ρ = S / sqrt(n1 · n0 · B / n)
    with S = sum of the centered ranks over the rows with 1, n1/n0 = number of 1s/0s and
    B = sum of the squared centered ranks. It is computed with integers (2·rank), so two
    identical columns give exactly the same ρ.
    Returns (rho, S2, n1): rho is NaN if the column is constant.
    """
    n = X.shape[0]
    r2 = np.rint(2 * rankdata_average(values, round_decimals=None)).astype(np.int64) - (n + 1)
    B4 = float(np.dot(r2, r2))
    # Products in float64: all values are integers smaller than 2^53, so the result is
    # exact (no rounding) and BLAS can be used.
    S2 = r2.astype(np.float64) @ X.astype(np.float64)
    n1 = X.sum(axis=0, dtype=np.int64).astype(np.float64)
    n0 = n - n1
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = S2 / np.sqrt(n1 * n0 * B4 / n)
    rho[(n1 == 0) | (n0 == 0)] = np.nan
    return rho, S2, n1


def spearman_and_pairs(X, values):
    """
    Spearman of all column pairs (i < j) using the AND variable (both differ), with matrix
    products instead of a loop over pairs. Same formula as spearman_binary_columns.
    Returns (npos, npos) matrices: rho (0 if the AND is constant, like
    fast_spearman_from_centered_x), S2 and n1.
    """
    n = X.shape[0]
    r2 = np.rint(2 * rankdata_average(values, round_decimals=None)).astype(np.int64) - (n + 1)
    B4 = float(np.dot(r2, r2))
    # Integers in float64 (< 2^53): exact products with BLAS.
    Xf = X.astype(np.float64)
    n1 = Xf.T @ Xf
    S2 = Xf.T @ (Xf * r2.astype(np.float64)[:, None])
    n0 = n - n1
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = S2 / np.sqrt(n1 * n0 * B4 / n)
    rho[(n1 == 0) | (n0 == 0)] = 0.0
    return rho, S2, n1


def sort_desc_with_tiebreak(values, support, recompute, tol=1e-12):
    """
    Indices sorted as `sorted(..., key=lambda k: [value, index], reverse=True)`.

    `values` were computed with a vectorized formula that can differ from the original one in
    the last bit. To get exactly the same order as the earlier version: in each group of
    nearly equal values (|Δ| < tol), if the elements are not all the same variable
    (`support(k)` returns the bytes of the 0/1 column), their value is recomputed with the
    original function (`recompute(k)`) and the group is sorted with those values. If they are
    the same variable, the original function also gives identical values and the tie is
    broken by index.
    """
    values = np.asarray(values, dtype=np.float64)
    order = sorted(range(values.size), key=lambda k: (values[k], k), reverse=True)
    result, a = [], 0
    while a < len(order):
        b = a + 1
        while b < len(order) and abs(values[order[b - 1]] - values[order[b]]) < tol:
            b += 1
        group = order[a:b]
        if len(group) > 1 and len({hash(support(k)) for k in group}) > 1:
            originals = {k: recompute(k) for k in group}
            group = sorted(group, key=lambda k: (originals[k], k), reverse=True)
        result.extend(group)
        a = b
    return result
