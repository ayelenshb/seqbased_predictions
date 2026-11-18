"""
Alphabet of a case as a label matrix.

For each sequence and each alignment column, the number of the group its residue belongs to is
stored. Two sequences differ at a column if their residues are in different groups (or one of
them has no group, label -1).

.npz file:
  labels         int16  (n_seq, L)
  phenotypes     float64 (n_seq,)
  columns        int64  (L,)        alignment columns (1-based)
  ids            str    (n_seq,)    FASTA IDs
  groups         str    (L,)        groups of each column, e.g. "AE|ST|-"
  correlations   float64 (L,)       ρ of the grouping chosen for each column
  dphen_pairs    float64 (n_pairs,) optional (alphabets converted from earlier versions)
  origin, format

Alphabets saved by the earlier Spanish version of the code (keys etiquetas, fenotipos, columnas,
grupos, correlaciones, dfen_pares, origen, formato) can also be read.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

FORMAT = 1

# key in the current format -> key in alphabets saved by the earlier (Spanish) version
_OLD_KEYS = {"labels": "etiquetas", "phenotypes": "fenotipos", "columns": "columnas", "ids": "ids",
             "groups": "grupos", "correlations": "correlaciones", "dphen_pairs": "dfen_pares",
             "origin": "origen", "format": "formato"}


@dataclass
class Labels:
    labels: np.ndarray
    phenotypes: np.ndarray
    columns: np.ndarray
    ids: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    correlations: Optional[np.ndarray] = None
    dphen_pairs: Optional[np.ndarray] = None
    origin: str = ""

    @property
    def n_seq(self):
        return self.labels.shape[0]

    @property
    def n_col(self):
        return self.labels.shape[1]

    def dphen(self):
        """|ΔPh| per pair (i < j)."""
        if self.dphen_pairs is not None:
            return self.dphen_pairs
        iu, ju = np.triu_indices(self.n_seq, k=1)
        return np.abs(self.phenotypes[iu] - self.phenotypes[ju])


def save_labels(path, lab: Labels):
    data = dict(labels=np.asarray(lab.labels, dtype=np.int16),
                phenotypes=np.asarray(lab.phenotypes, dtype=np.float64),
                columns=np.asarray(lab.columns, dtype=np.int64),
                origin=np.array(lab.origin), format=np.array(FORMAT))
    for attr in ("ids", "groups"):
        if getattr(lab, attr) is not None:
            data[attr] = np.asarray(getattr(lab, attr), dtype=str)
    for attr in ("correlations", "dphen_pairs"):
        if getattr(lab, attr) is not None:
            data[attr] = np.asarray(getattr(lab, attr), dtype=np.float64)
    np.savez_compressed(path, **data)


def load_labels(path) -> Labels:
    with np.load(path, allow_pickle=False) as z:
        def opt(key):
            for k in (key, _OLD_KEYS[key]):
                if k in z.files:
                    return z[k]
            return None

        def req(key):
            value = opt(key)
            if value is None:
                raise ValueError(f"The alphabet {path} has no '{key}' array")
            return value

        origin = opt("origin")
        return Labels(labels=req("labels").astype(np.int16), phenotypes=req("phenotypes").astype(np.float64),
                      columns=req("columns").astype(np.int64), ids=opt("ids"), groups=opt("groups"),
                      correlations=opt("correlations"), dphen_pairs=opt("dphen_pairs"),
                      origin=str(origin) if origin is not None else "")


def check_against_fasta(lab: Labels, ids, sequences, phenotypes, path=""):
    """Checks that the alphabet matches the case FASTA (same sequences, columns and phenotypes)."""
    problems = []
    if lab.n_seq != len(ids):
        problems.append(f"it has {lab.n_seq} sequences and the FASTA {len(ids)}")
    if lab.n_col != sequences.shape[1]:
        problems.append(f"it has {lab.n_col} columns and the FASTA {sequences.shape[1]}")
    if not problems:
        if lab.ids is not None and list(lab.ids) != list(ids):
            problems.append("the IDs (or their order) do not match the FASTA")
        if not np.allclose(lab.phenotypes, phenotypes, rtol=1e-3, atol=1e-3):
            problems.append("the phenotypes do not match the FASTA (did the phenotype index change?)")
    if problems:
        raise ValueError(f"The alphabet {path} does not match the FASTA: " + "; ".join(problems) +
                         ". Delete it (or give another path) to create it again.")


def labels_from_groups(sequences, groups_per_column):
    """Labels (n_seq, L) from the groups of each column (lists of residues)."""
    sequences = np.asarray(sequences)
    n_seq, L = sequences.shape
    lab = np.full((n_seq, L), -1, dtype=np.int16)
    for c, groups in enumerate(groups_per_column):
        mapping = {aa: g for g, group in enumerate(groups) for aa in group}
        lab[:, c] = [mapping.get(aa, -1) for aa in sequences[:, c]]
    return lab


def pair_matrix_from_labels(labels, column_idx=None, block=32):
    """(n_pairs, n_columns) uint8 matrix: 1 if the sequence pair differs at the column."""
    labels = np.asarray(labels)
    if column_idx is None:
        column_idx = np.arange(labels.shape[1])
    column_idx = np.asarray(column_idx)
    iu, ju = np.triu_indices(labels.shape[0], k=1)
    X = np.empty((iu.size, column_idx.size), dtype=np.uint8)
    for a in range(0, column_idx.size, block):
        cols = column_idx[a:a + block]
        li = labels[iu][:, cols]
        lj = labels[ju][:, cols]
        X[:, a:a + cols.size] = ((li != lj) | (li == -1)).astype(np.uint8)
    return X
