"""
Controls.

Random positions (default): for each variant, random sets of the same size as the selection,
drawn from the columns that vary in its encoding, evaluated with the same partitions. In the
pair variants, each position of the selection is replaced by a random column: the number of
pairs, the number of distinct positions and how many pairs each position shares are
preserved. The weights are those of the variant, randomly reassigned.

Permuted phenotypes (optional): the phenotypes are permuted among sequences and the full
selection of the variant is repeated; it measures the r² obtained when there is no relation
between sequence and phenotype. The alphabet is not recomputed (it was built with the real
phenotypes).
"""
from dataclasses import replace
from typing import List, NamedTuple

import numpy as np

from .columns import vector_from_pairs
from .prediction import _safe_linregress, metrics, predict
from .variants import select, singles_vector


class ControlResult(NamedTuple):
    control: int
    columns: list
    r2_mean: float
    r2_sd: float
    rmse: float
    slope: float            # predicted ~ observed line with all replicates pooled
    intercept: float
    replicates: list


def _evaluate(k, vector, columns, data, partitions):
    reps = predict(vector, data, partitions)
    r2m, r2s, rmse = metrics(reps)
    obs = np.concatenate([r.observed for r in reps])
    pred = np.concatenate([r.predicted for r in reps])
    slope, intercept = _safe_linregress(obs, pred)[:2]
    return ControlResult(k, columns, r2m, r2s, rmse, slope, intercept, reps)


def random_set(sel, data, rng):
    """A random set with the same shape as the selection `sel`. Returns (vector, columns)."""
    weights = rng.permutation(np.asarray(sel.weights, dtype=np.float64))
    if sel.mode == "singles":
        idx = rng.choice(data.npos, size=len(sel.idx), replace=False)
        return singles_vector(data.X, idx, weights), [data.col_list[k] for k in idx]
    # Pairs: each distinct position of the selection is replaced by a random column (without
    # repetition). The control then has the same number of pairs, of distinct positions and the
    # same structure (how many pairs each position shares); only which positions they are changes.
    sel_pairs = np.asarray(sel.idx)
    distinct = np.unique(sel_pairs.ravel())
    new = rng.choice(data.npos, size=len(distinct), replace=False)
    mapping = dict(zip(distinct.tolist(), new.tolist()))
    pairs = np.array([sorted((mapping[int(i)], mapping[int(j)])) for i, j in sel_pairs], dtype=np.int64)
    return vector_from_pairs(data.X, pairs, weights), [[data.col_list[i], data.col_list[j]] for i, j in pairs]


def position_controls(sel, data, partitions, n, rng) -> List[ControlResult]:
    results = []
    if len(sel.idx) == 0:
        return results
    for k in range(n):
        vector, columns = random_set(sel, data, rng)
        results.append(_evaluate(k, vector, columns, data, partitions))
    return results


def permute_phenotypes(data, rng):
    perm = rng.permutation(data.n_seq)
    phen = data.phen[perm]
    return replace(data, phen=phen, dif=np.abs(phen[data.iu] - phen[data.ju]))


def permutation_controls(name, seed_name, encoding, mode, method, data, case, partitions, n, stage_seed):
    """Repeats the selection of the variant with permuted phenotypes (n times)."""
    results = []
    for k in range(n):
        # the seed key keeps its original (Spanish) text so that results do not change
        rng = np.random.default_rng(stage_seed(f"permutacion:{seed_name}:{k}"))
        data_p = permute_phenotypes(data, rng)
        sel = select(name, encoding, mode, method, data_p, case, rng_mcmc=rng, verbose=False)
        results.append(_evaluate(k, sel.vector, sel.columns, data_p, partitions))
    return results


def empirical_p(value, controls):
    """(1 + number of controls with r² >= value) / (n + 1)."""
    if not controls:
        return np.nan
    r2 = np.array([c.r2_mean for c in controls])
    return float((1 + np.sum(r2 >= value)) / (len(r2) + 1))
