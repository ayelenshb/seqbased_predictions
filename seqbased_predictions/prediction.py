"""
Phenotype prediction from a divergence per sequence pair.

In each replicate: the sequences are split into training and validation; with the
training-training pairs, |ΔPh| = a·divergence + b is fitted; for each validation sequence,
|ΔPh| against each training sequence is predicted and the phenotype t that minimizes
Σ_j (|ΔPh|_pred,j − |Ph_j − t|)² is found.
"""
import math
from typing import List, NamedTuple

import numpy as np
from scipy.stats import linregress


def make_partitions(n_seq, replicates, train_fraction, rng):
    """`replicates` random (train, val) partitions of the sequences; used for all variants and controls."""
    n_train = int(n_seq * train_fraction)
    partitions = []
    for _ in range(replicates):
        train = np.sort(rng.choice(n_seq, n_train, replace=False))
        val = np.setdiff1d(np.arange(n_seq), train)
        partitions.append((train, val))
    return partitions


def _safe_linregress(x, y):
    """(slope, intercept, r, p, stderr); if x or y are constant: slope 0, intercept mean(y), r 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return (0.0, float(y.mean()), 0.0, 1.0, 0.0)
    r = linregress(x, y)
    return (r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr)


def minimize_phenotype(dphen_pred, known_ph):
    """
    t that minimizes Σ_j (dphen_pred_j − |Ph_j − t|)², solved exactly: between consecutive known
    phenotypes the function is a parabola, so the minimum of each segment is evaluated.
    """
    d = np.asarray(dphen_pred, dtype=np.float64).ravel()
    a = np.asarray(known_ph, dtype=np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(a)
    d, a = d[ok], a[ok]
    n = a.size
    order = np.argsort(a, kind='stable')
    a, d = a[order], d[order]
    left = np.concatenate([[0.0], np.cumsum(a + d)])
    right = np.concatenate([np.cumsum((a - d)[::-1])[::-1], [0.0]])
    t_free = (left + right) / n
    lo = np.concatenate([[-np.inf], a])
    hi = np.concatenate([a, [np.inf]])
    candidates = np.unique(np.concatenate([np.clip(t_free, lo, hi), a]))
    f = ((d[None, :] - np.abs(a[None, :] - candidates[:, None])) ** 2).sum(axis=1)
    k = int(np.argmin(f))
    return candidates[k]


class Replicate(NamedTuple):
    replicate: int
    a: float                 # slope of |ΔPh| ~ divergence (training)
    b: float                 # intercept
    r_train: float           # Pearson r of the training fit
    val: np.ndarray          # indices of the predicted validation sequences
    observed: np.ndarray
    predicted: np.ndarray
    pred_fit: tuple          # slope, intercept, r, p, stderr of predicted ~ observed

    @property
    def r2(self):
        return self.pred_fit[2] ** 2


def predict(vector, data, partitions) -> List[Replicate]:
    """Prediction with a divergence per sequence pair (`vector`, one entry per pair i < j)."""
    x = np.asarray(vector, dtype=np.float64)
    iu, ju, phen, dif = data.iu, data.ju, data.phen, data.dif
    replicates = []
    for k, (train, val) in enumerate(partitions):
        in_i = np.isin(iu, train)
        in_j = np.isin(ju, train)
        m_train = in_i & in_j
        a, b, r_train, _, _ = _safe_linregress(x[m_train], dif[m_train])

        m_val = in_i ^ in_j
        i_is_train = in_i[m_val]
        seq_val = np.where(i_is_train, ju[m_val], iu[m_val])
        seq_train = np.where(i_is_train, iu[m_val], ju[m_val])
        x_val = x[m_val]
        obs, pred, val_ok = [], [], []
        for s in val:
            rows = seq_val == s
            if not rows.any():
                continue
            dphen_pred = a * x_val[rows] + b
            pred.append(minimize_phenotype(dphen_pred, phen[seq_train[rows]]))
            obs.append(phen[s])
            val_ok.append(s)
        obs = np.array(obs)
        pred = np.array(pred)
        replicates.append(Replicate(k, a, b, r_train, np.array(val_ok), obs, pred, _safe_linregress(obs, pred)))
    return replicates


def metrics(replicates):
    """Mean and SD of r² (Pearson r of observed vs predicted, per replicate) and RMSE pooling all replicates."""
    r2 = np.array([rep.r2 for rep in replicates], dtype=np.float64)
    errors = np.concatenate([rep.observed - rep.predicted for rep in replicates])
    return float(np.nanmean(r2)), float(np.nanstd(r2)), float(math.sqrt(np.mean(errors ** 2)))
