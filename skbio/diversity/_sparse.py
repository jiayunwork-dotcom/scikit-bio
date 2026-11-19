# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Sparse count support for the phylogenetic diversity metrics.

This module keeps the count table in CSR form end-to-end. The dense pipeline
(:mod:`skbio.diversity._util` and the Cython ``_phylogenetic`` extension)
materializes an ``n_samples x n_nodes`` array whose width is the full tree; on
a wide table (hundreds of thousands of taxa, a few dozen observations per
sample) that is exactly the expansion the sparse pipeline must avoid.

Strategy
--------
The tree is converted, once per call, into a sparse ancestor matrix
``A`` of shape ``(n_nodes, n_nodes)`` with ``A[i, j] = 1`` when node ``j`` is
node ``i`` itself or descends from it. The count columns are scattered into
tree-tip node columns (column reordering and duplicate-taxon summation happen
while copying, so the caller's CSR is never mutated), then a single sparse
product ``S = A @ T`` gives per-node counts as a sparse
``n_nodes x n_samples`` matrix. No dense ``sample x taxon`` and no dense
``sample x node`` matrix is ever built.

Trade-offs considered:

* *Per-sample tree state* (walk each sample's observed tips up the tree, keep
  one node vector at a time) uses the least memory (one node vector) but
  repeats the ancestor walk for every sample and again for every pairwise
  distance, so work grows with both sample overlap and the number of pairs.
* *Dense per-node matrix for all samples* (the existing approach) shares the
  reduction across samples and pairs, but its memory is
  ``n_samples x n_nodes`` regardless of sparsity -- the full wide-table
  expansion in disguise.
* *This module* shares the reduction across all samples (one sparse product)
  and streams the pairwise work over sparse node vectors. Memory scales with
  the number of stored (sample, node) incidences -- bounded by
  ``n_samples x n_nnz x tree_depth`` -- instead of with the table width, while
  each branch's contribution is computed once and reused for every pair that
  touches it. Deep trees and low-overlap inputs raise the per-row node count
  (linearly in depth), which is the only regime where the per-sample strategy
  wins; clustered samples share most of their ancestor rows and stay cheap.

All arithmetic is vectorized with SciPy sparse operations; no branch of the
pipeline samples, drops rare counts, or approximates a distance.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import squareform

# Counts are accumulated in 64-bit integers, matching COUNT_DTYPE in
# skbio/diversity/_phylogenetic.pyx.
_COUNT_DTYPE = np.int64

# Largest logical row total we accept. Above this an int64 accumulation can no
# longer represent every integer exactly, and the dense pipeline would have
# the same precision problem.
_MAX_ROW_TOTAL = 2**53 - 1


def is_csr_counts(counts) -> bool:
    """Whether ``counts`` is a 2-D SciPy CSR matrix or CSR array."""
    return sp.isspmatrix_csr(counts) or isinstance(counts, sp.csr_array)


def _fail(msg):
    raise ValueError(msg)


def validate_sparse_counts(counts):
    """Validate a CSR count table without densifying it.

    The checks here are *count* checks and therefore run even when the caller
    passes ``validate=False``; ``validate`` only governs the tree/label checks
    performed elsewhere.

    Rules:

    * the input must be a 2-D CSR matrix/array of non-Boolean, non-complex
      real integer values (floating dtypes are accepted when every stored
      value is integral, as produced by BIOM);
    * every stored value must be finite and non-negative, and a negative
      value is rejected even if a repeated coordinate at the same position
      would cancel it (negatives are caught before duplicates are merged);
    * logical values are the sum over repeated coordinates, and explicitly
      stored zeros carry no presence information;
    * each row's logical total must not exceed ``2**53 - 1``.

    Non-canonical CSR (unsorted column indices, repeated coordinates, stored
    zeros) is accepted. A normalized, canonicalized **copy** is returned; the
    caller's ``data``, ``indices`` and ``indptr`` are never modified.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Count table of shape ``(n_samples, n_taxa)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        A canonical CSR copy (sorted, duplicates merged, explicit zeros
        removed) with int64 values.

    Raises
    ------
    TypeError
        If the dtype is Boolean, complex, or not numeric.
    ValueError
        If values are non-finite, fractional, negative, or a row total
        exceeds ``2**53 - 1``.

    """
    if not is_csr_counts(counts):
        raise TypeError(
            "Sparse counts must be a scipy.sparse.csr_matrix or csr_array."
        )
    if counts.ndim != 2:
        raise ValueError("Sparse counts must be two-dimensional.")

    dtype = counts.dtype
    if dtype.kind in ("b",):
        raise TypeError("Boolean counts are not supported.")
    if dtype.kind not in ("i", "u", "f"):
        # complex ("c"), object ("O"), and anything else are explicitly
        # rejected rather than silently coerced.
        raise TypeError(
            "Counts must be non-Boolean real integers (or floating-point "
            f"integral values); got dtype {dtype!r}."
        )

    # Work on a copy up front: sorting, duplicate merging and casting must
    # never touch the caller's arrays.
    out = counts.copy()

    if out.data.size:
        data = out.data
        # Negatives first, against the stored values, so a negative entry is
        # caught even when a duplicate coordinate would otherwise cancel it.
        if np.min(data) < 0:
            _fail("Counts cannot contain negative values.")
        if dtype.kind == "f":
            if not np.isfinite(data).all():
                _fail("Counts must be finite (NaN and infinity are rejected).")
            if np.count_nonzero(data != np.floor(data)):
                _fail("Floating-point counts must hold integral values.")
        elif not np.all(np.isfinite(data)):
            # integer dtypes cannot hold NaN/inf, but keep the guard explicit.
            _fail("Counts must be finite.")
        elif dtype.kind in ("i", "u"):
            # Merging repeated coordinates sums stored values; reject rows
            # whose stored (pre-merge) total already exceeds int64, so a
            # silent wrap-around can never reach the metrics. reduceat itself
            # wraps on overflow (often to a negative value), hence both tests.
            stored_row_totals = np.add.reduceat(data, out.indptr[:-1])
            stored_row_totals[np.diff(out.indptr) == 0] = 0
            if np.any(
                (stored_row_totals < 0)
                | (stored_row_totals > np.iinfo(_COUNT_DTYPE).max)
            ):
                _fail(
                    "A row's merged count would exceed the int64 range; "
                    "each row's total must not exceed 2**53 - 1."
                )

    out.sum_duplicates()
    out.eliminate_zeros()
    # After merging, check the (possibly larger) logical values again: merging
    # cannot create negatives, but can create a value outside the integer or
    # finite range of a floating dtype.
    if out.data.size:
        if not np.all(np.isfinite(out.data)):
            _fail("Counts must be finite (NaN and infinity are rejected).")
        if np.min(out.data) < 0:
            _fail("Counts cannot contain negative values.")

    # Logical row totals, computed from the sparse structure alone. This runs
    # before the int64 cast so an over-large float raises the documented
    # ValueError instead of NumPy's OverflowError.
    if out.data.size and out.indptr.size > 1:
        row_totals = np.add.reduceat(out.data, out.indptr[:-1])
        # Empty rows have equal indptr entries; reduceat returns data[p] for
        # those, so zero them explicitly.
        row_totals[np.diff(out.indptr) == 0] = 0
        if np.any(row_totals > _MAX_ROW_TOTAL):
            _fail(
                "Each row's total count must not exceed 2**53 - 1 "
                f"({_MAX_ROW_TOTAL})."
            )

    if out.dtype != _COUNT_DTYPE:
        out = out.astype(_COUNT_DTYPE)
    # Normalize csr_array inputs to the matrix subclass the rest of the
    # pipeline expects; this does not alter the stored data.
    if not sp.isspmatrix_csr(out):
        out = sp.csr_matrix(out)

    out.sort_indices()
    return out


def _tip_name_to_node(tree_index):
    """Map each tree tip name to its node index, preserving node order."""
    names = tree_index["name"]
    id_index = tree_index["id_index"]
    tip_lookup = {}
    for node_id, node in id_index.items():
        if node.is_tip():
            tip_lookup[names[node_id]] = node_id
    return tip_lookup


def build_ancestor_matrix(tree_index):
    """Build the sparse ancestor/identity matrix of a tree.

    Parameters
    ----------
    tree_index : dict
        The output of :meth:`skbio.tree.TreeNode.to_array` (it must contain
        ``id_index``, ``name`` and ``child_index``).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Column ``j`` has ones on node ``j`` and every ancestor of node ``j``
        up to and including the root. Restricting to tip columns turns tip
        counts into all-node counts by a single sparse product.

    """
    id_index = tree_index["id_index"]
    n_nodes = len(id_index)

    rows = []
    cols = []
    for node_id, node in id_index.items():
        cur = node
        while cur is not None:
            rows.append(cur.id)
            cols.append(node_id)
            cur = cur.parent
    ones = np.ones(len(rows), dtype=np.int8)
    ancestor = sp.coo_matrix(
        (ones, (np.asarray(rows), np.asarray(cols))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    ancestor.sum_duplicates()
    return ancestor


def sparse_counts_by_node(counts, taxa, tree_index, ancestor):
    """Map a sparse sample-by-taxon table onto sparse per-node counts.

    The table's columns are interpreted through ``taxa`` (one label per
    column). Columns whose taxon is not a tip of the tree are dropped only
    when the caller has already validated the labels (validation raises
    :class:`~skbio.tree.MissingNodeError` instead); duplicate taxon labels are
    summed, matching the dense pipeline's semantics for duplicated columns.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix, shape (n_samples, n_taxa)
        Canonicalized count table, e.g. the output of
        :func:`validate_sparse_counts`.
    taxa : array_like, shape (n_taxa,)
        Taxon label of every table column, in column order.
    tree_index : dict
        Output of :meth:`TreeNode.to_array`.
    ancestor : scipy.sparse.csr_matrix
        Output of :func:`build_ancestor_matrix`.

    Returns
    -------
    scipy.sparse.csc_matrix, shape (n_nodes, n_samples)
        Logical count descending from every node, for every sample. Stored in
        CSC order so per-sample node vectors are direct column slices.

    """
    tip_lookup = _tip_name_to_node(tree_index)
    taxa = np.asarray(taxa)

    # Table column -> tree tip node index. Every label must resolve: the
    # MissingNodeError for labels absent from the tree is raised at the
    # validation layer, so an unknown key here would indicate a bypassed
    # validation and is surfaced as a ValueError with the offending label.
    try:
        node_cols = np.array([tip_lookup[label] for label in taxa], dtype=np.intp)
    except KeyError as exc:
        raise ValueError(
            f"Taxon {exc.args[0]!r} is not a tip of the tree."
        ) from exc

    n_nodes = ancestor.shape[0]
    n_samples = counts.shape[0]

    # Reorder (and merge duplicate taxa) by scattering stored values into the
    # tree's tip-node columns. The summation of duplicate taxon columns happens
    # in _scatter_to_tip_columns without a dense intermediate.
    tip_counts = _scatter_to_tip_columns(counts, node_cols, n_nodes, n_samples)

    # Propagate tip counts up the tree: (n_nodes, n_nodes) @
    # (n_nodes, n_samples) -> (n_nodes, n_samples), both sparse.
    node_counts = ancestor @ tip_counts.T
    return node_counts.tocsc()


def _scatter_to_tip_columns(counts, node_cols, n_nodes, n_samples):
    """Scatter CSR stored values into tree-tip node columns, merging dupes.

    Builds a sparse ``(n_samples, n_nodes)`` table directly from the stored
    entries of ``counts``; repeated taxa (several table columns mapping to the
    same tip node) are summed by ``sum_duplicates``.
    """
    indptr = counts.indptr
    indices = counts.indices
    data = counts.data

    # Each stored entry keeps its sample row; its new column is the tip node of
    # its original column.
    sample_rows = np.repeat(np.arange(n_samples), np.diff(indptr))
    new_cols = node_cols[indices]
    tip_table = sp.csr_matrix(
        (data.astype(_COUNT_DTYPE, copy=False), (sample_rows, new_cols)),
        shape=(n_samples, n_nodes),
    )
    tip_table.sum_duplicates()
    return tip_table


def branch_lengths_of(tree_index):
    """Branch-length vector of an indexed tree, with NaN replaced by zero."""
    lengths = np.array(tree_index["length"], dtype=np.float64, copy=True)
    np.nan_to_num(lengths, copy=False, nan=0.0)
    return lengths


def faith_pd_sparse(node_counts, branch_lengths):
    """Faith's PD for every sample, from sparse per-node counts.

    Parameters
    ----------
    node_counts : scipy.sparse.csc_matrix, shape (n_nodes, n_samples)
        Output of :func:`sparse_counts_by_node`.
    branch_lengths : np.ndarray, shape (n_nodes,)
        Branch length per node (root length conventionally zero).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Faith's PD per sample, as float64. Empty samples yield zero.

    """
    presence = node_counts.copy()
    presence.data = np.ones(presence.data.size, dtype=np.float64)
    # Sum of branch lengths of present nodes, per sample.
    return np.asarray((presence.T @ branch_lengths).ravel()).reshape(-1)


def phydiv_sparse(node_counts, branch_lengths, sample_totals, rooted, weight):
    """Generalized PD (``phydiv``) for every sample, on sparse node counts.

    Mirrors :func:`skbio.diversity.alpha._pd._phydiv` term for term: presence
    selects branches, the per-sample total is the maximum node count (all
    counts descend from the root), and weighting multiplies each branch by the
    descending relative abundance (and by the unrooted balance when
    ``rooted`` is False).

    Returns
    -------
    np.ndarray, shape (n_samples,)

    """
    n_nodes, n_samples = node_counts.shape
    out = np.zeros(n_samples, dtype=np.float64)

    # Maximum node count per sample equals the sample's total tip count.
    counts_sum = np.array(sample_totals, dtype=np.float64, copy=True)
    nonzero_samples = counts_sum > 0.0
    if not nonzero_samples.any():
        return out

    presence = node_counts.copy()
    presence.data = np.ones(presence.data.size, dtype=np.float64)

    if not weight:
        included = presence
        if rooted is False:
            # Exclude the root branch for every sample whose total reaches it:
            # nodes whose count equals the sample total are the root path of
            # that sample's observed set.
            included = _drop_full_root_path(presence, node_counts, counts_sum)
        return np.asarray(
            (included.T @ branch_lengths).ravel()
        ).reshape(-1)

    # Weighted: proportions per node, kept sparse.
    proportions = node_counts.copy().astype(np.float64)
    safe_totals = np.where(counts_sum > 0.0, counts_sum, 1.0)
    proportions = proportions.multiply(1.0 / safe_totals).tocsc()

    if rooted is False:
        # 2 * min(frac, 1 - frac) on stored entries; absent entries stay zero.
        proportions = _unrooted_balance(proportions)

    # Mirror _phydiv exactly: the theta exponent is applied only for a float
    # weight below 1; weight=True leaves the proportions at power 1.
    if isinstance(weight, float) and weight < 1.0:
        proportions.data **= weight

    return np.asarray(
        (proportions.T @ branch_lengths).ravel()
    ).reshape(-1)


def _drop_full_root_path(presence, node_counts, counts_sum):
    """Zero presence of nodes whose count equals the sample's total.

    These are precisely the branches on the root-to-LCA path removed by the
    unrooted variant. The comparison is applied only to stored entries, so
    absent (implicitly zero) nodes remain absent. Both matrices are canonical
    CSC and share identical index structure, so their ``data`` arrays align
    entry for entry without any reordering.
    """
    out = presence.tocsc(copy=True)
    cols = np.repeat(
        np.arange(out.shape[1]), np.diff(out.indptr)
    )
    full = node_counts.data.astype(np.float64, copy=False) == counts_sum[cols]
    out.data[full] = 0.0
    out.eliminate_zeros()
    return out


def _unrooted_balance(proportions):
    """Apply ``2 * min(frac, 1 - frac)`` to the stored entries."""
    out = proportions.tocsc(copy=True)
    out.data = 2.0 * np.minimum(out.data, 1.0 - out.data)
    return out


def unweighted_unifrac_sparse(node_counts, branch_lengths):
    """Condensed unweighted UniFrac distances between all sample pairs."""
    presence = node_counts.copy()
    presence.data = np.ones(presence.data.size, dtype=np.float64)

    # Per-sample total branch length of present nodes: B^T l (one sparse dot
    # product per sample, vectorized).
    per_sample_length = np.asarray(
        (presence.T @ branch_lengths).ravel()
    ).reshape(-1)

    # Weighted intersection per pair: (B^T diag(l) B)_ij =
    # sum_k B_ki B_kj l_k. Scale each node row in place (sparse), then both
    # operands stay sparse; only the small n_samples x n_samples result is
    # dense.
    weighted = presence.T.tocsr(copy=True)
    weighted @= sp.diags(branch_lengths)
    # The only dense allocation in the pipeline: the n_samples x n_samples
    # result (the distance matrix is dense by contract). Never the wide
    # sample-by-taxon or sample-by-node table.
    intersection = np.asarray((weighted @ presence).todense())

    n = node_counts.shape[1]
    iu = np.triu_indices(n, k=1)
    union = per_sample_length[:, None] + per_sample_length[None, :] - intersection
    unique_branch = union[iu] - intersection[iu]
    observed_branch = union[iu]
    out = np.zeros_like(observed_branch)
    nonzero = observed_branch > 0.0
    out[nonzero] = unique_branch[nonzero] / observed_branch[nonzero]
    return out


def weighted_unifrac_sparse(
    node_counts, branch_lengths, tip_indices, normalized, node_to_root_distances
):
    """Condensed weighted UniFrac distances between all sample pairs.

    Parameters
    ----------
    node_counts : scipy.sparse.csc_matrix, shape (n_nodes, n_samples)
        Logical per-node counts.
    branch_lengths : np.ndarray, shape (n_nodes,)
    tip_indices : np.ndarray
        Node indices of the tips, used for per-sample total counts.
    normalized : bool
        Apply the branch-length normalization of Ley et al. (2005).
    node_to_root_distances : np.ndarray or None
        Tip-to-root distance per node (zeros off tips); required when
        ``normalized`` is True.

    """
    n_nodes, n_samples = node_counts.shape

    totals = np.zeros(n_samples, dtype=np.float64)
    tip_block = node_counts[tip_indices, :]
    if sp.issparse(tip_block):
        tip_sums = np.asarray(tip_block.sum(axis=0)).ravel()
    else:
        tip_sums = np.asarray(tip_block).sum(axis=0)
    totals[:] = tip_sums

    # Per-node proportions, kept sparse: zeros (absent nodes) are not stored.
    proportions = node_counts.copy().astype(np.float64)
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    proportions = proportions.multiply(1.0 / safe_totals).tocsc()

    # Sum of per-node absolute proportion differences per pair. Walk the
    # intersection of stored rows; nodes stored in only one column contribute
    # that column's proportion outright.
    indptr = proportions.indptr
    indices = proportions.indices
    data = proportions.data
    wu = np.zeros(n_samples * (n_samples - 1) // 2, dtype=np.float64)
    correction = np.zeros_like(wu)
    write = 0
    for i in range(n_samples - 1):
        ai = indices[indptr[i]:indptr[i + 1]]
        di = data[indptr[i]:indptr[i + 1]]
        for j in range(i + 1, n_samples):
            aj = indices[indptr[j]:indptr[j + 1]]
            dj = data[indptr[j]:indptr[j + 1]]
            pi = pj = 0
            diff_sum = 0.0
            corr_sum = 0.0
            while pi < ai.size and pj < aj.size:
                if ai[pi] == aj[pj]:
                    k = ai[pi]
                    diff_sum += branch_lengths[k] * abs(di[pi] - dj[pj])
                    if normalized:
                        corr_sum += node_to_root_distances[k] * (
                            di[pi] + dj[pj]
                        )
                    pi += 1
                    pj += 1
                elif ai[pi] < aj[pj]:
                    k = ai[pi]
                    diff_sum += branch_lengths[k] * di[pi]
                    if normalized:
                        corr_sum += node_to_root_distances[k] * di[pi]
                    pi += 1
                else:
                    k = aj[pj]
                    diff_sum += branch_lengths[k] * dj[pj]
                    if normalized:
                        corr_sum += node_to_root_distances[k] * dj[pj]
                    pj += 1
            # Leftover rows present in only one of the two columns.
            while pi < ai.size:
                k = ai[pi]
                diff_sum += branch_lengths[k] * di[pi]
                if normalized:
                    corr_sum += node_to_root_distances[k] * di[pi]
                pi += 1
            while pj < aj.size:
                k = aj[pj]
                diff_sum += branch_lengths[k] * dj[pj]
                if normalized:
                    corr_sum += node_to_root_distances[k] * dj[pj]
                pj += 1
            wu[write] = diff_sum
            correction[write] = corr_sum
            write += 1

    if not normalized:
        return wu

    # Both samples empty: totals are both zero and the dense reference returns
    # 0.0. Every other pair divides by the correction.
    n = n_samples
    iu0, iu1 = np.triu_indices(n, k=1)
    both_empty = (totals[iu0] == 0.0) & (totals[iu1] == 0.0)
    out = np.zeros_like(wu)
    valid = ~both_empty
    out[valid] = wu[valid] / correction[valid]
    return out


def condensed_to_square(condensed, n_samples):
    """Convert a condensed distance vector to a square hollow matrix."""
    return squareform(condensed)
