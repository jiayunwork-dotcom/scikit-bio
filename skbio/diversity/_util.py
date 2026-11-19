# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np

from skbio.tree import DuplicateNodeError, MissingNodeError
from skbio.diversity._phylogenetic import _nodes_by_counts
from skbio.diversity._sparse import (
    build_ancestor_matrix,
    sparse_counts_by_node,
    validate_sparse_counts,
)


def _validate_counts(counts, cast_int=False):
    """Validate and convert input to an acceptable counts vector/matrix type.

    Parameters
    ----------
    counts : array_like of shape (n_taxa,) or (n_samples, n_taxa)
        Vector or matrix of counts.
    cast_int : bool, optional
        Whether cast values into integers, if not already. Default is False.

    Returns
    -------
    ndarray of shape (n_taxa,) or (n_samples, n_taxa)
        Valid counts vector or matrix.

    Raises
    ------
    ValueError
        If input array has an invalid data type.
    ValueError
        If input array contains negative values.

    Notes
    -----
    This function will return the original ``counts`` if it is already a valid counts
    vector. Otherwise it will return an edited copy that is valid.

    The data type of counts must be any subtype of ``np.integer`` (integers) or
    ``np.floating`` (floating-point numbers; excluding complex numbers) [1]_.

    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/arrays.scalars.html

    """
    counts = np.asarray(counts)

    dtype = counts.dtype
    if np.issubdtype(dtype, np.floating):
        if cast_int:
            counts = counts.astype(int)
    elif not np.issubdtype(dtype, np.integer) and dtype is not np.dtype("bool"):
        raise ValueError("Counts must be integers or floating-point numbers.")

    # This is more efficient that `(counts < 0).any()`, but a more efficient way is to
    # iterate by element and exit early on the first negative value.
    if counts.size > 0 and counts.min() < 0:
        raise ValueError("Counts cannot contain negative values.")

    return counts


def _validate_counts_vector(counts, cast_int=False):
    """Validate and convert input to an acceptable counts vector type.

    Parameters
    ----------
    counts : array_like of int or float of shape (n_taxa,)
        Vector of counts.
    cast_int : bool, optional
        Whether cast values into integers, if not already. Default is False.

    Returns
    -------
    ndarray of int or float of shape (n_taxa,)
        Valid counts vector.

    Raises
    ------
    ValueError
        If counts has more than 1 dimension.

    """
    counts = _validate_counts(counts, cast_int=cast_int)
    if counts.ndim != 1:
        raise ValueError("`counts` must be a 1-D array (vector).")
    return counts


def _validate_counts_matrix(counts, cast_int=False):
    """Validate and convert input to an acceptable counts matrix type.

    Parameters
    ----------
    counts : array_like of shape (n_samples, n_taxa)
        Matrix of counts.
    cast_int : bool, optional
        Whether cast values into integers, if not already. Default is False.

    Returns
    -------
    ndarray of shape (n_samples, n_taxa)
        Valid counts matrix.

    Raises
    ------
    ValueError
        If counts has more than 2 dimensions.

    """
    msg = "`counts` has {} dimensions whereas up to 2 dimensions are allowed."
    counts = _validate_counts(counts, cast_int=cast_int)
    counts = np.atleast_2d(counts)
    if counts.ndim > 2:
        raise ValueError(msg.format(counts.ndim))
    return counts


def vectorize_counts_and_tree(counts, taxa, tree):
    """Index tree and convert counts to np.array in corresponding order.

    Parameters
    ----------
    counts : array_like of shape (n_samples, n_taxa) or (n_taxa,)
        Counts/abundances of taxa in one or multiple samples.
    taxa : array_like of shape (n_taxa,)
        Taxon IDs corresponding to tip names in `tree`.
    tree : TreeNode
        Tree relating taxa. The set of tip names in the tree can be a superset
        of `taxa`, but not a subset.

    Returns
    -------
    ndarray of shape (n_samples, n_nodes)
        Total counts/abundances of taxa descending from individual nodes of the tree.
    dict of array
        Indexed tree. See `to_array`.
    ndarray of shape (n_nodes,)
        Branch lengths of corresponding nodes of the tree.

    See Also
    --------
    skbio.tree.TreeNode.to_array

    Notes
    -----
    Leveraging internal node counts in the tree (in addition to tip abundances) can
    double the accuracy in downstream machine learning pipelines [1]_.

    References
    ----------
    .. [1] Martino C., McDonald D., Cantrell K., Dilmore AH., Vázquez-Baeza Y.,
       Shenhav L., Shaffer J.P., Rahman G., Armstrong G., Allaband C., Song S.J.,
       Knight R. Compositionally aware phylogenetic beta-diversity measures better
       resolve microbiomes associated with phenotype. mSystems. 7(3) (2022).

    """
    tree_index = tree.to_array(nan_length_value=0.0)
    taxa = np.asarray(taxa)
    counts = np.atleast_2d(counts)
    counts_by_node = _nodes_by_counts(counts, taxa, tree_index)
    branch_lengths = tree_index["length"]

    # branch_lengths is just a reference to the array inside of tree_index,
    # but it's used so much that it's convenient to just pull it out here.
    return counts_by_node.T, tree_index, branch_lengths


def _get_phylogenetic_kwargs(kwargs, taxa):
    if taxa is None:
        raise ValueError("`taxa` is required for phylogenetic diversity metrics.")
    try:
        tree = kwargs.pop("tree")
    except KeyError:
        raise ValueError("`tree` is required for phylogenetic diversity metrics.")

    return taxa, tree, kwargs


def _qualify_counts(counts):
    return counts > 0.0


def _validate_sparse_counts_matrix(counts, validate=True):
    """Validate a sparse count table.

    The count-value checks (dtype, finiteness, non-negativity, integrality,
    row-total bound) always run: they are independent of ``validate`` and must
    not be bypassed by ``validate=False`` (which only skips the tree/label
    checks at the call site). The ``validate`` argument is accepted for call
    signature symmetry with the dense validator but deliberately not used
    here. A canonicalized CSR **copy** is returned, so downstream
    sorting/merging never mutates the caller's data.

    """
    return validate_sparse_counts(counts)


def build_sparse_phylogenetic_state(counts, taxa, tree):
    """Build the reusable sparse state behind all phylogenetic metrics.

    Parameters
    ----------
    counts : scipy.sparse.csr_matrix, shape (n_samples, n_taxa)
        Already-validated canonical CSR counts (see
        :func:`_validate_sparse_counts_matrix`).
    taxa : array_like, shape (n_taxa,)
        Taxon label per table column.
    tree : skbio.tree.TreeNode
        Tree relating the taxa. Tip names may be a superset of the taxa.

    Returns
    -------
    dict
        ``node_counts`` (CSC sparse ``n_nodes x n_samples`` per-node counts),
        ``branch_lengths`` (float64 per node), ``tree_index`` (the indexed
        tree), ``tip_indices`` and ``sample_totals`` (float64 per sample).

    Notes
    -----
    The tree is indexed with :meth:`TreeNode.to_array` exactly as in the dense
    pipeline, so the mapping from taxon labels to tip nodes and the branch
    lengths are identical to the dense results. No dense sample-by-taxon or
    sample-by-node matrix is allocated: the only dense outputs are the
    one-dimensional per-node and per-sample vectors, plus the final distance
    matrix produced by the caller.

    """
    import numpy as np

    tree_index = tree.to_array(nan_length_value=0.0)
    branch_lengths = tree_index["length"]
    ancestor = build_ancestor_matrix(tree_index)
    node_counts = sparse_counts_by_node(counts, taxa, tree_index, ancestor)

    # Tip indices in node order (postorder ids), matching _get_tip_indices.
    tip_indices = np.array(
        [n.id for n in tree_index["id_index"].values() if n.is_tip()],
        dtype=np.intp,
    )
    sample_totals = np.asarray(
        node_counts[tip_indices, :].sum(axis=0)
    ).ravel().astype(np.float64, copy=False)

    return {
        "node_counts": node_counts,
        "branch_lengths": branch_lengths,
        "tree_index": tree_index,
        "tip_indices": tip_indices,
        "sample_totals": sample_totals,
    }
