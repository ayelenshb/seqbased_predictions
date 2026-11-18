"""Position pairs and column-wise representation of the divergence matrix."""
import numba
import numpy as np


def get_pairs(npos):
    """All position pairs (i, j) with i < j, in the order of itertools.combinations."""
    iu, ju = np.triu_indices(npos, k=1)
    return np.column_stack((iu, ju)).astype(np.int32)

def get_singletons_as_pairs(npos):
    """Each position as a diagonal 'pair' (i, i), to reuse the same MCMC machinery."""
    idx = np.arange(npos, dtype=np.int64)
    return np.column_stack([idx, idx]).astype(np.int64)


def build_column_csc(base_binary):
    """
    Column-wise representation of the non-zero rows (O(nnz) memory):
    - col_indptr: length npos+1, pointers into col_row_indices
    - col_row_indices: row indices (sorted) of each column, concatenated
    """
    rows, npos = base_binary.shape
    col_indptr = [0]
    col_row_indices = []
    for j in range(npos):
        rows_j = np.flatnonzero(base_binary[:, j] != 0).astype(np.int32)
        col_row_indices.append(rows_j)
        col_indptr.append(col_indptr[-1] + rows_j.size)
    col_indptr = np.array(col_indptr, dtype=np.int64)
    if len(col_row_indices) == 0:
        col_row_indices = np.empty(0, dtype=np.int32)
    else:
        col_row_indices = np.concatenate(col_row_indices)
    return col_indptr, col_row_indices

@numba.njit
def apply_delta_pair_from_cols(sum_rows_vector, col_row_indices, col_indptr, i, j, delta_w):
    """
    Adds delta_w to the rows where columns i and j are both non-zero (two-pointer
    intersection). If i == j (single-site mode) the column is traversed only once.
    """
    start_i = col_indptr[i]
    end_i = col_indptr[i + 1]
    if i == j:
        for p in range(start_i, end_i):
            sum_rows_vector[col_row_indices[p]] += delta_w
        return
    start_j = col_indptr[j]
    end_j = col_indptr[j + 1]
    p = start_i
    q = start_j
    while p < end_i and q < end_j:
        ri = col_row_indices[p]
        rj = col_row_indices[q]
        if ri == rj:
            sum_rows_vector[ri] += delta_w
            p += 1
            q += 1
        elif ri < rj:
            p += 1
        else:
            q += 1

def rebuild_sum_from_selection_fast(rows, col_row_indices, col_indptr, pairs, selected_indices, selected_weights):
    """Recomputes from scratch the sum vector of a selection (to check for numerical drift)."""
    s = np.zeros(rows, dtype=np.float64)
    for idx, w in zip(selected_indices, selected_weights):
        apply_delta_pair_from_cols(s, col_row_indices, col_indptr, int(pairs[idx, 0]), int(pairs[idx, 1]), float(w))
    return s


def vector_from_pairs(base_binary, pairs, weights=None):
    """Divergence per sequence pair, summing the AND variable of each position pair (weights 1 by default)."""
    col_indptr, col_row_indices = build_column_csc(base_binary)
    s = np.zeros(base_binary.shape[0], dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        w = 1.0 if weights is None else float(weights[k])
        apply_delta_pair_from_cols(s, col_row_indices, col_indptr, int(i), int(j), w)
    return s
