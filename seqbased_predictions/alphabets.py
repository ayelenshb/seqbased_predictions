"""
Per-position alphabet (AC): for each alignment column, the grouping of residues that maximizes
the Spearman correlation between group divergence and |ΔPh|, searched with a Monte Carlo
(Metropolis-Hastings with cooling) that merges and splits groups.
"""
import math
import random
import sys
import time
from functools import lru_cache

import numpy as np
from scipy.stats import spearmanr

from .labels import Labels, labels_from_groups


@lru_cache(maxsize=8)
def _pairs(n):
    return np.triu_indices(n, k=1)


def _group_divergence(column, groups):
    mapping = {aa: i for i, group in enumerate(groups) for aa in group}
    g = np.array([mapping.get(aa, -1) for aa in column])
    iu, ju = _pairs(len(g))
    return np.where((g[iu] == g[ju]) & (g[iu] != -1), 0, 1).astype(np.int8)


def grouping_correlation(column, dphen, groups):
    """Spearman between the group divergence of a column and |ΔPh| (0 if either is constant)."""
    d = _group_divergence(column, groups)
    if len(np.unique(d)) <= 1 or len(np.unique(dphen)) <= 1:
        return 0.0
    r, _ = spearmanr(d, dphen)
    return r if not np.isnan(r) else 0.0


def _merge(groups, rng):
    if len(groups) < 2:
        return groups
    i, j = rng.sample(range(len(groups)), 2)
    new = [g for k, g in enumerate(groups) if k not in (i, j)]
    new.append(groups[i] + groups[j])
    return new


def _split(groups, rng):
    splittable = [k for k, g in enumerate(groups) if len(g) > 1]
    if not splittable:
        return groups
    k = rng.choice(splittable)
    group = groups[k]
    left = rng.sample(group, rng.randint(1, len(group) - 1))
    right = [aa for aa in group if aa not in left]
    new = [g for m, g in enumerate(groups) if m != k]
    new.append(left)
    new.append(right)
    return new


def optimize_grouping(column, dphen, rng, iterations=1000, alpha=0.1):
    """
    Monte Carlo over partitions of the residues of a column.
    - Start: each residue (including '-') in its own group (= identity).
    - Proposal: merge two groups or split one, with equal probability.
    - Acceptance: always if ρ improves; otherwise with p = exp(-|Δρ|/α).
    - α is multiplied by 0.6 every 100 iterations.
    Returns (best_groups, best_ρ). `rng` is a random.Random.
    """
    key = lambda gs: tuple(sorted(tuple(sorted(g)) for g in gs))
    cache = {}
    unique = sorted(set(column))
    if len(unique) <= 1:
        return [unique], 0.0
    groups = [[aa] for aa in unique]
    current = key(groups)
    r = cache[current] = grouping_correlation(column, dphen, groups)
    best_r, best = r, current
    inv_alpha = (1.0 / alpha) if alpha > 0 else float("inf")
    for it in range(1, iterations + 1):
        action = 'split' if len(groups) <= 1 else rng.choice(['merge', 'split'])
        new = _merge(groups, rng) if action == 'merge' else _split(groups, rng)
        k = key(new)
        if k == current:          # proposal with no effect: not evaluated (and no cooling in this iteration)
            continue
        r_new = cache.get(k)
        if r_new is None:
            r_new = cache[k] = grouping_correlation(column, dphen, new)
        if r_new > r:
            groups, r, current = new, r_new, k
            if r > best_r:
                best_r, best = r, k
        elif rng.random() < math.exp(-abs(r - r_new) * inv_alpha):
            groups, r, current = new, r_new, k
        if it % 100 == 0 or it == iterations:
            alpha *= 0.6
            inv_alpha = (1.0 / alpha) if alpha > 0 else float("inf")
    return [list(g) for g in best], best_r


def create_alphabet(ids, sequences, phenotypes, seed, iterations=1000, alpha=0.1, name="", verbose=True):
    """Alphabet of all alignment columns (the gap filter is applied later, when it is used)."""
    rng = random.Random(seed)
    iu, ju = np.triu_indices(len(ids), k=1)
    dphen = np.abs(phenotypes[iu] - phenotypes[ju])
    L = sequences.shape[1]
    groups, rhos = [], []
    t0 = time.time()
    for c in range(L):
        g, r = optimize_grouping(sequences[:, c], dphen, rng, iterations=iterations, alpha=alpha)
        groups.append(g)
        rhos.append(r)
        if verbose and ((c + 1) % 10 == 0 or c + 1 == L):
            sys.stdout.write(f"\r  alphabet {name}: column {c + 1}/{L} ({time.time() - t0:.0f} s)")
            sys.stdout.flush()
    if verbose:
        print()
    return Labels(
        labels=labels_from_groups(sequences, groups),
        phenotypes=phenotypes,
        columns=np.arange(1, L + 1),
        ids=np.array(ids),
        groups=np.array(["|".join("".join(sorted(g)) for g in gs) for gs in groups]),
        correlations=np.array(rhos, dtype=np.float64),
        origin=f"create_alphabet(iterations={iterations}, alpha={alpha}, seed={seed})",
    )
