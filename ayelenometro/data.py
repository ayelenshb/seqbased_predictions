"""Alignment reading and pairwise sequence comparisons."""
from dataclasses import dataclass
from typing import List

import numpy as np
from Bio import SeqIO


def read_fasta(path):
    """IDs and aligned sequences (n_seq x L array of characters)."""
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError(f"The FASTA {path} has no sequences")
    lengths = {len(r.seq) for r in records}
    if len(lengths) != 1:
        raise ValueError(f"The sequences in {path} do not all have the same length (is it aligned?): {sorted(lengths)}")
    ids = [r.id for r in records]
    sequences = np.array([list(str(r.seq).upper()) for r in records])
    return ids, sequences


def phenotypes_from_ids(ids, index):
    """Phenotype of each sequence: field `index` of the ID split by '_'."""
    return np.array([float(i.split('_')[index]) for i in ids], dtype=np.float64)


def universe_columns(sequences, max_gaps):
    """Columns (0-based) with a gap fraction <= max_gaps. It is the same universe for all variants."""
    gap_frac = np.mean(sequences == '-', axis=0)
    return np.flatnonzero(gap_frac <= max_gaps)


@dataclass
class PairData:
    """
    Pairwise comparisons (i < j, in the order of np.triu_indices) for one encoding.

    X         (n_pairs, npos) uint8: 1 if the pair differs at that column (by identity or by group).
              Only universe columns that vary in this encoding.
    col_list  alignment column (1-based) of each column of X
    phen      phenotype of each sequence
    dif       |ΔPh| of each pair
    iu, ju    indices of the sequences of each pair
    L         total alignment length (for the % of positions)
    """
    X: np.ndarray
    col_list: List[int]
    phen: np.ndarray
    dif: np.ndarray
    iu: np.ndarray
    ju: np.ndarray
    ids: List[str]
    L: int
    encoding: str

    @property
    def npos(self):
        return self.X.shape[1]

    @property
    def n_seq(self):
        return len(self.phen)

    def column_indices(self, columns):
        """Indices in X of alignment columns (1-based); those not present are skipped."""
        pos = {c: k for k, c in enumerate(self.col_list)}
        return [pos[c] for c in columns if c in pos]


def _build(X_universe, universe, phen, dif, ids, L, encoding):
    varies = X_universe.sum(axis=0) != 0
    n = len(phen)
    iu, ju = np.triu_indices(n, k=1)
    return PairData(
        X=np.ascontiguousarray(X_universe[:, varies]),
        col_list=[int(c) + 1 for c in np.asarray(universe)[varies]],
        phen=phen, dif=dif, iu=iu, ju=ju, ids=list(ids), L=L, encoding=encoding,
    )


def identity_data(ids, sequences, phen, universe):
    """Divergence by identity (Hamming distance per column)."""
    iu, ju = np.triu_indices(len(ids), k=1)
    sub = sequences[:, universe]
    X = (sub[iu] != sub[ju]).astype(np.uint8)
    dif = np.abs(phen[iu] - phen[ju])
    return _build(X, universe, phen, dif, ids, sequences.shape[1], "identity")


def alphabet_data(lab, universe, ids):
    """Divergence by amino acid groups (case alphabet)."""
    from .labels import pair_matrix_from_labels
    X = pair_matrix_from_labels(lab.labels, universe)
    return _build(X, universe, lab.phenotypes, lab.dphen(), ids, lab.n_col, "alphabet")
