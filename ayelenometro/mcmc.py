"""Selection of positions or position pairs by Monte Carlo (Metropolis with adaptive temperature)."""
import sys
import time
from typing import NamedTuple

import numpy as np

from .columns import apply_delta_pair_from_cols, build_column_csc, rebuild_sum_from_selection_fast
from .stats import calculate_spearman_regularized_correlation_fast, center_ranks


class _SelectionState:
    """
    Current selection: indices and weights, the sum vector per sequence pair, and the weights in
    an array (for Σ|w| without converting lists). Adding and removing are O(1).
    """
    def __init__(self, pairs, col_indptr, col_row_indices, rows, capacity):
        self.pairs = pairs
        self.col_indptr = col_indptr
        self.col_row_indices = col_row_indices
        self.sum = np.zeros(rows, dtype=np.float64)
        self.indices = []
        self.weights = []
        self._buf = np.empty(capacity, dtype=np.float64)
        self.pos = {}
        self.members = set()

    def _delta(self, idx, w):
        apply_delta_pair_from_cols(self.sum, self.col_row_indices, self.col_indptr,
                                   int(self.pairs[idx, 0]), int(self.pairs[idx, 1]), w)

    def add(self, idx, w):
        idx = int(idx)
        self._delta(idx, w)
        p = len(self.indices)
        self.indices.append(idx)
        self.weights.append(w)
        self._buf[p] = w
        self.pos[idx] = p
        self.members.add(idx)

    def remove(self, idx):
        """Removes idx and returns its weight (None if it was not selected)."""
        p = self.pos.get(idx, None)
        if p is None:
            return None
        w = self.weights[p]
        self._delta(idx, -w)
        last = len(self.indices) - 1
        if p != last:
            self.indices[p] = self.indices[last]
            self.weights[p] = self.weights[last]
            self._buf[p] = self._buf[last]
            self.pos[self.indices[p]] = p
        self.indices.pop()
        self.weights.pop()
        self.pos.pop(idx, None)
        self.members.discard(idx)
        return w

    def current_weights(self):
        return self._buf[:len(self.indices)]


def propose_changes(state, intervals, iteration, total_pairs, rng, max_changes=None):
    """
    Conflict-free batch of changes: each index is added or removed at most once.
    max_changes: None -> up to max(1, 10 - iteration // 1000) changes per step (the batch shrinks
    as the iterations go on); 1 -> a single change.
    """
    if max_changes is None:
        # earlier version (growing batch): max_changes = min(10, 1 + iteration // 1000)
        max_changes = max(1, 10 - iteration // 1000)
    n_changes = int(rng.integers(1, max_changes + 1))
    n_sel = len(state.indices)
    removed, added, changes = set(), set(), []
    for _ in range(n_changes):
        can_remove = len(removed) < n_sel
        can_add = len(added) + n_sel < total_pairs
        if not can_remove and not can_add:
            break
        if not can_remove:
            add = True
        elif not can_add:
            add = False
        else:
            add = bool(rng.integers(2) == 0)
        if add:
            while True:
                idx = int(rng.integers(total_pairs))
                if idx not in state.members and idx not in added:
                    break
            changes.append(("add", idx, float(rng.choice(intervals))))
            added.add(idx)
        else:
            while True:
                idx = state.indices[int(rng.integers(n_sel))]
                if idx not in removed:
                    break
            changes.append(("remove", idx, state.weights[state.pos[idx]]))
            removed.add(idx)
    return changes


def acceptance_probability(delta_rho, iteration, iterations_without_change, alpha, kk):
    """
    Probability of accepting a change that does not improve ρ_reg: exp(-|Δρ_reg|·iteration / (α·kk)).
    Right after an acceptance (iterations_without_change == 0) it is 0.
    """
    if iterations_without_change == 0:
        return 0.0
    return np.exp(((-abs(delta_rho)) * iteration) / (alpha * kk))


class MCMCResult(NamedTuple):
    sum_rows: np.ndarray        # divergence per sequence pair of the best selection
    indices: np.ndarray         # selected indices (in `pairs`)
    weights: np.ndarray         # weight of each one
    rho_reg: float              # regularized ρ of the best selection
    rho: float                  # unregularized ρ of the best selection
    iterations: int


def mcmc(
    lambda_reg,
    X,
    pairs,
    dif,
    rng,
    initial=0.1,
    step_mcmc=0.0005,
    maxval=1.0,
    max_iterations=200000,
    patience=40000,
    init_fraction=0.25,
    round_decimals=10,
    max_changes_per_step=None,
    weighted=True,
    best_criterion="unregularized",
    patience_mode="best",
    reg_denominator=None,
    debug=False,
    rebuild_interval_debug=500,
    verbose=True,
    print_every=1000,
):
    """
    Monte Carlo over (weighted) subsets of positions or position pairs.

    pairs        (i, j) with i != j for epistasis, or (i, i) for single positions.
    lambda_reg   λ of ρ_reg = ρ - λ·Σ|w| / total (total = number of candidates, or reg_denominator).
    rng          np.random.Generator: all the randomness of the run comes from it.

    Start: a fraction `init_fraction` of the candidates, with random weights from the grid
    [initial, maxval] (step step_mcmc). At each iteration a batch of changes is proposed (see
    propose_changes); it is accepted if ρ_reg improves, or with acceptance_probability() if not.
    kk increases by 10 every 5 consecutive rejections and goes back to 1 on acceptance.
    The best selection according to best_criterion is returned ("unregularized": highest ρ;
    "regularized": highest ρ_reg). It stops when the best does not improve for `patience`
    iterations (patience_mode="best") or when no change is accepted for `patience` iterations
    ("unchanged"), or when max_iterations is reached.
    """
    if best_criterion not in ("unregularized", "regularized"):
        raise ValueError("best_criterion must be 'unregularized' or 'regularized'")
    if patience_mode not in ("best", "unchanged"):
        raise ValueError("patience_mode must be 'best' or 'unchanged'")
    rows = X.shape[0]
    total_pairs = pairs.shape[0]
    col_indptr, col_row_indices = build_column_csc(X)

    intervals = (np.arange(initial, maxval + 1e-12, step_mcmc, dtype=np.float64) if weighted
                 else np.array([1.0], dtype=np.float64))
    reg_total = total_pairs if reg_denominator is None else float(reg_denominator)

    init_count = max(1, int(init_fraction * total_pairs))
    init_indices = rng.choice(total_pairs, size=init_count, replace=False)
    init_weights = rng.choice(intervals, size=init_count, replace=True)

    state = _SelectionState(pairs, col_indptr, col_row_indices, rows, total_pairs)
    for idx, w in zip(init_indices, init_weights):
        state.add(idx, float(w))

    x_centered, std_x = center_ranks(dif)

    def evaluate():
        return calculate_spearman_regularized_correlation_fast(
            x_centered, std_x, state.sum, state.current_weights(), lambda_reg, reg_total,
            round_decimals=round_decimals)

    rho_reg, rho, penalty = evaluate()
    best = (state.sum.copy(), list(state.indices), list(state.weights), rho_reg, rho)

    iteration = best_count = without_change = 0
    alpha, kk, maxitsc = 1.0, 1, 5
    last_print = 0.0
    while iteration < max_iterations and best_count < patience:
        iteration += 1
        changes = propose_changes(state, intervals, iteration, total_pairs, rng, max_changes_per_step)

        applied = []
        for kind, idx, w in changes:
            if kind == "add":
                state.add(idx, w)
                applied.append((kind, idx, w))
            else:
                w_old = state.remove(idx)
                if w_old is not None:
                    applied.append((kind, idx, w_old))

        if debug and iteration % rebuild_interval_debug == 0:
            rebuilt = rebuild_sum_from_selection_fast(rows, col_row_indices, col_indptr, pairs,
                                                      state.indices, state.weights)
            if np.max(np.abs(rebuilt - state.sum)) > 1e-9:
                state.sum[:] = rebuilt

        rho_reg_n, rho_n, penalty = evaluate()
        accepted = rho_reg_n > rho_reg
        if not accepted:
            p = acceptance_probability(rho_reg - rho_reg_n, iteration, without_change, alpha, kk)
            accepted = rng.random() < p

        if accepted:
            rho_reg, rho = rho_reg_n, rho_n
            without_change = 0
            best_count = 0 if patience_mode == "unchanged" else best_count + 1
            kk = 1
            new_best = (rho > best[4]) if best_criterion == "unregularized" else (rho_reg > best[3])
            if new_best:
                best = (state.sum.copy(), list(state.indices), list(state.weights), rho_reg, rho)
                best_count = 0
        else:
            for kind, idx, w in reversed(applied):
                if kind == "add":
                    state.remove(idx)
                else:
                    state.add(idx, w)
            without_change += 1
            best_count += 1
            if without_change == maxitsc:
                kk += 10
                without_change = 0

        if verbose and (iteration % print_every == 0 or time.time() - last_print > 1.0):
            last_print = time.time()
            sys.stdout.write(f"\r    iteration {iteration}: selected {len(state.indices)}, "
                             f"ρ_reg {rho_reg:.4f}, ρ {rho:.4f}, best ρ {best[4]:.4f}, without improvement {best_count}   ")
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
    return MCMCResult(best[0], np.array(best[1], dtype=np.int64), np.array(best[2], dtype=np.float64),
                      best[3], best[4], iteration)
