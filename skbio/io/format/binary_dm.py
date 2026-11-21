"""Simple binary dissimilarity matrix format (:mod:`skbio.io.format.binary_dm`)
============================================================================

.. currentmodule:: skbio.io.format.binary_dm

The Binary DisSimilarity Matrix format (``binary_dm``) encodes a binary
representation for dissimilarity and distance matrices. The format is
designed to facilitate rapid random access to individual rows or columns of
a hollow matrix.

Format Support
--------------
**Has Sniffer: Yes**

+------+------+---------------------------------------------------------------+
|Reader|Writer|                          Object Class                         |
+======+======+===============================================================+
|Yes   |Yes   |:mod:`skbio.stats.distance.PairwiseMatrix`                     |
+------+------+---------------------------------------------------------------+
|Yes   |Yes   |:mod:`skbio.stats.distance.SymmetricMatrix`                    |
+------+------+---------------------------------------------------------------+
|Yes   |Yes   |:mod:`skbio.stats.distance.DistanceMatrix`                     |
+------+------+---------------------------------------------------------------+

Format Specification
--------------------
The binary dissimilarity matrix and object identifiers are stored within an
HDF5 [1]_ file. Both datatypes are represented by their own datasets. The
`ids` dataset is of a variable length unicode type, while the
`matrix` dataset are floating point. The shape of the `ids` is
`(N,)`, and the shape of the `dissimilarities` is `(N, N)`. For distance
matrices, the diagonal of `matrix` is all zeros; symmetric matrices with a
non-zero diagonal store their diagonal values along the diagonal of
`matrix`.

The dissimilarity between `ids[i]` and `ids[j]` is interpreted
to be the value at `matrix[i, j]`. `i` and `j` are integer indices.

Required datasets:

+-----------+---------------------+------------------------------+
|Datasets   |Value                |Description                   |
|           |type                 |                              |
+===========+=====================+==============================+
|format     |string               |A string identifying the file |
|           |                     |as Binary DM format           |
+-----------+---------------------+------------------------------+
|version    |string               |The version of the current    |
|           |                     |Binary DM format              |
+-----------+---------------------+------------------------------+
|matrix     |float32 or float64   |A (N, N) dataset containing   |
|           |                     |the values of the             |
|           |                     |dissimilarity matrix          |
+-----------+---------------------+------------------------------+
|order      |string               |A (N,) dataset of the sample  |
|           |                     |IDs, where N is the total     |
|           |                     |number of IDs                 |
+-----------+---------------------+------------------------------+

Optionally, the file can contain several matrices.
In such a case, the 'matrix' dataset will not exist and we will use
'matrix:0' instead.

.. note:: This file format is most useful for storing large matrices that do
   not need to be represented in a human-readable format. This format is
   especially appropriate for facilitating random access to entries in the
   distance matrix, such as when calculating within and between distances for a
   subset of samples in a large matrix.

References
----------
.. [1] http://www.hdfgroup.org/


"""  # noqa: D205, D415

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import h5py
import numpy as np

from skbio.io import create_format, BinaryDMFormatError
from skbio.stats.distance import PairwiseMatrix, SymmetricMatrix, DistanceMatrix
from skbio.util import find_duplicates


binary_dm = create_format("binary_dm", encoding="binary")


@binary_dm.sniffer()
def _binary_dm_sniffer(fh):
    try:
        with h5py.File(fh, "r") as f:
            return _validate_binary_dm(f) is None, {}
    except OSError:
        return False, {}


@binary_dm.reader(PairwiseMatrix)
def _binary_dm_to_pairwise(fh, cls=None):
    if cls is None:
        cls = PairwiseMatrix
    return _h5py_mat_to_skbio_mat_stream(cls, fh)


@binary_dm.reader(SymmetricMatrix)
def _binary_dm_to_symmetric(fh, cls=None):
    if cls is None:
        cls = SymmetricMatrix
    return _h5py_mat_to_skbio_mat_stream(cls, fh)


@binary_dm.reader(DistanceMatrix)
def _binary_dm_to_distance(fh, cls=None):
    if cls is None:
        cls = DistanceMatrix
    return _h5py_mat_to_skbio_mat_stream(cls, fh)


@binary_dm.writer(PairwiseMatrix)
def _pairwise_to_binary_dm(obj, fh):
    return _skbio_mat_to_h5py_mat_stream(obj, fh)


@binary_dm.writer(SymmetricMatrix)
def _symmetric_to_binary_dm(obj, fh):
    return _skbio_mat_to_h5py_mat_stream(obj, fh)


@binary_dm.writer(DistanceMatrix)
def _distance_to_binary_dm(obj, fh):
    return _skbio_mat_to_h5py_mat_stream(obj, fh)


def _h5py_mat_to_skbio_mat_stream(cls, fh):
    with h5py.File(fh, "r") as f:
        dm = _h5py_mat_to_skbio_mat(cls, f)
    return dm


def _h5py_mat_to_skbio_mat(cls, f):
    error = _validate_binary_dm(f)
    if error is not None:
        raise BinaryDMFormatError(error)

    mat = f.get("matrix")
    if mat is None:
        mat = f.get("matrix:0")
    mat = np.asarray(mat)
    dm = cls(mat, _parse_ids(f["order"]))
    return dm


def _skbio_mat_to_h5py_mat_stream(obj, fh):
    with h5py.File(fh, "w") as f:
        _skbio_mat_to_h5py_mat(obj, f)


def _skbio_mat_to_h5py_mat(obj, f):
    _set_header(f)

    b_ids = [x.encode("utf-8") for x in obj.ids]
    np_ids = np.array(b_ids)
    f.create_dataset("order", data=np_ids)

    data = obj.data
    if data.ndim == 2:
        # Redundant (2-D) storage streams straight into the dataset.
        f.create_dataset("matrix", data=data)
    else:
        # Condensed (1-D) storage is expanded block by block so that the
        # file always holds a true (N, N) `matrix` dataset while auxiliary
        # memory stays bounded (see `_write_condensed_matrix`).
        _write_condensed_matrix(obj, f)


# Target maximum size, in bytes, of the row-block buffer used when
# expanding condensed storage into the (N, N) `matrix` dataset. Together
# with the row segmenter's scratch space (at most 2 MiB for float64), the
# numeric workspace of a condensed write stays below 8 MiB regardless of
# the number of samples.
_BLOCK_BYTES = 4 << 20


def _block_buffer(n_rows, n_cols, dtype):
    """Allocate the reusable row-block buffer for condensed writes.

    Notes
    -----
    Kept as a separate function so that tests (and users) can wrap it to
    obtain an exact allocation record of the streaming writer.

    """
    return np.empty((n_rows, n_cols), dtype=dtype)


def _write_condensed_matrix(obj, f):
    """Write a condensed-form matrix object as an (N, N) `matrix` dataset.

    The square dataset is created up front and filled row block by row
    block; each row is composed from condensed storage in bounded column
    segments via ``PairwiseMatrix._row_segmenter``. Auxiliary memory is
    limited to one row-block buffer (at most ``_BLOCK_BYTES`` bytes) plus
    the segmenter's O(seg_cap) scratch space. The source object is only
    read, never mutated.

    """
    n = obj.shape[0]
    segmenter = obj._row_segmenter()
    dtype = segmenter.dtype
    dset = f.create_dataset("matrix", shape=(n, n), dtype=dtype)

    row_bytes = n * dtype.itemsize
    if row_bytes > _BLOCK_BYTES:
        # A single row already exceeds the block budget: stream bounded
        # column segments straight into the dataset.
        for i in range(n):
            for j0, j1, segment in segmenter.iter_row(i):
                dset[i, j0:j1] = segment
        return

    rows_per_block = max(1, _BLOCK_BYTES // row_bytes)
    block = _block_buffer(min(rows_per_block, n), n, dtype)
    r0 = 0
    while r0 < n:
        b = min(rows_per_block, n - r0)
        for k in range(b):
            for j0, j1, segment in segmenter.iter_row(r0 + k):
                block[k, j0:j1] = segment
        dset[r0 : r0 + b, :] = block[:b]
        r0 += b


def _get_header(fh):
    format_ = fh.get("format")
    version = fh.get("version")
    if format_ is None or version is None:
        return None
    try:
        return {"format": format_[0], "version": version[0]}
    except (IndexError, TypeError, ValueError):
        return None


def _validate_binary_dm(f):
    """Check an open HDF5 file against the binary_dm format specification.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    str or None
        A description of the first format violation found, or None if the
        file satisfies the required structure (header datasets, `order`,
        and a square 2-D `matrix`/`matrix:0` dataset whose dimensions match
        the number of unique IDs).

    """
    if _get_header(f) is None:
        return "Missing required 'format' and/or 'version' dataset(s)."

    order = f.get("order")
    mat = f.get("matrix")
    if mat is None:
        mat = f.get("matrix:0")
    if order is None or mat is None:
        return "Missing required 'order' and/or 'matrix' dataset(s)."
    if not isinstance(order, h5py.Dataset) or not isinstance(mat, h5py.Dataset):
        return "'order' and 'matrix' must both be datasets."

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return (
            "'matrix' must be a square 2-D dataset. Found shape %r."
            % (mat.shape,)
        )
    n = mat.shape[0]

    if order.ndim != 1:
        return "'order' must be a 1-D dataset. Found shape %r." % (order.shape,)
    if len(order) != n:
        return (
            "The number of IDs in 'order' (%d) does not match the dimensions "
            "of 'matrix' (%d)." % (len(order), n)
        )

    duplicates = find_duplicates(list(order[:]))
    if duplicates:
        return (
            "IDs in 'order' must be unique. Found the following duplicate "
            "IDs: %s" % ", ".join(repr(e) for e in duplicates)
        )

    return None


def _parse_ids(ids):
    if isinstance(ids[0], bytes):
        return _bytes_decoder(ids)
    else:
        return _passthrough_decoder(ids)


def _verify_dimensions(f):
    ids = f.get("order")

    mat = f.get("matrix")
    if mat is None:
        mat = f.get("matrix:0")

    if (ids is None) or (mat is None):
        return False
    n = len(ids)
    return mat.shape == (n, n)


def _bytes_decoder(x):
    return [i.decode("utf8") for i in x]


def _passthrough_decoder(x):
    return x


def _set_header(f):
    """Set format spec header information."""
    f.create_dataset("format", data=np.array([b"BDSM"]))
    f.create_dataset("version", data=np.array([b"2020.12"]))
