"""The variants of the method: each one returns a Selection with everything the prediction and the controls need."""
from typing import List, NamedTuple, Optional

import numpy as np

from .columns import get_pairs, get_singletons_as_pairs, vector_from_pairs
from .mcmc import mcmc
from .selection import order_pairs, order_positions
from .stats import center_ranks, fast_spearman_from_centered_x


class Selection(NamedTuple):
    name: str
    encoding: str                # "identity" or "alphabet"
    mode: str                    # "singles" or "pairs"
    method: str                  # "ordering", "mcmc" or "fixed"
    vector: np.ndarray           # divergence per sequence pair with the selection
    idx: np.ndarray              # singles: column indices in X; pairs: (n, 2) indices in X
    columns: list                # singles: MSA columns; pairs: [[a, b], ...]
    weights: np.ndarray          # 1 for ordering and fixed positions
    rho: Optional[float]         # ρ between the divergence of the selection and |ΔPh|
    individual_rho: Optional[np.ndarray] = None   # ordering: ρ of each column of X (or pair)
    cumulative_rho: Optional[List[float]] = None  # ordering: ρ when adding in order
    order: Optional[List[int]] = None             # ordering: order of the indices

    @property
    def distinct_positions(self):
        if self.mode == "singles":
            return sorted(set(int(c) for c in self.columns))
        return sorted({int(c) for pair in self.columns for c in pair})


def singles_vector(X, idx, weights):
    return X[:, idx].astype(np.float64) @ np.asarray(weights, dtype=np.float64) if len(idx) else np.zeros(X.shape[0])


def select(name, encoding, mode, method, data, case, rng_mcmc=None, verbose=True):
    """Runs the selection of a variant on `data` (PairData of its encoding)."""
    X, dif = data.X, data.dif
    if method == "ordering" and mode == "singles":
        res = order_positions(X, dif)
        idx = np.array(res.selection, dtype=np.int64)
        weights = np.ones(len(idx))
        return Selection(name, encoding, mode, method, singles_vector(X, idx, weights), idx,
                         [data.col_list[k] for k in idx], weights, float(res.rho),
                         res.individual_rho, res.cumulative_rho, res.order)

    if method == "ordering" and mode == "pairs":
        pairs = get_pairs(data.npos)
        res = order_pairs(X, pairs, dif)
        idx = pairs[np.array(res.selection, dtype=np.int64)].astype(np.int64)
        weights = np.ones(len(idx))
        return Selection(name, encoding, mode, method, vector_from_pairs(X, idx), idx,
                         [[data.col_list[a], data.col_list[b]] for a, b in idx], weights, float(res.rho),
                         None, res.cumulative_rho, res.order)

    if method == "mcmc":
        options = dict(case.mcmc)
        if mode == "singles":
            pairs = get_singletons_as_pairs(data.npos)
            options = {"max_changes_per_step": 1, **options}
        else:
            pairs = get_pairs(data.npos)
            options = {"initial": 0.3, "step_mcmc": 0.1, **options}
        res = mcmc(case.lambda_reg, X, pairs, dif, rng_mcmc, verbose=verbose, **options)
        chosen = pairs[res.indices].astype(np.int64)
        if mode == "singles":
            idx = chosen[:, 0]
            columns = [data.col_list[k] for k in idx]
        else:
            idx = chosen
            columns = [[data.col_list[a], data.col_list[b]] for a, b in idx]
        return Selection(name, encoding, mode, method, res.sum_rows, idx, columns, res.weights, float(res.rho))

    raise ValueError(f"Unknown combination: {mode}/{method}")


def fixed_selection(name, encoding, columns, data):
    """Positions given in advance (literature). Those outside the universe or without variation are skipped."""
    idx = np.array(data.column_indices(columns), dtype=np.int64)
    weights = np.ones(len(idx))
    vector = singles_vector(data.X, idx, weights)
    xc, sx = center_ranks(data.dif)
    rho = fast_spearman_from_centered_x(xc, sx, vector) if len(idx) else None
    return Selection(name, encoding, "singles", "fixed", vector, idx,
                     [data.col_list[k] for k in idx], weights, rho)
