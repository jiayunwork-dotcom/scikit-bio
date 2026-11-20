r"""Project supplementary samples into a fixed PCoA ordination space.

This module implements the machinery behind
:meth:`skbio.stats.ordination.OrdinationResults.project`: placing new
("supplementary") samples into an existing PCoA map of reference samples
using only the distances between each supplementary sample and the
reference samples. The reference configuration is *not* refitted; the axes
are frozen at fit time and stored in a :class:`ProjectionState`.

Placement model
---------------
Let ``x_1, ..., x_n`` be the fixed reference coordinates in the retained
``k``-dimensional ordination space and let ``d_i`` be the observed
dissimilarity between a supplementary sample and reference ``i`` (a subset
of the references may be observed; unobserved pairs are simply excluded).
Each supplementary sample is located independently of all others by
minimizing the sum of squared residuals of the *actual* (not squared)
distances::

    f(y) = sum_{i observed} ( ||y - x_i|| - d_i )**2

This is the classical multilateration objective. It is solved by
Levenberg-Marquardt (or a trust-region method when the problem has fewer
residuals than variables) starting from Gower's closed-form interpolation
formula [1]_, which is exact (up to floating-point error) when the
observed distances are mutually consistent Euclidean distances in the
retained space. Deterministic fallback starting points are used when the
primary solve does not converge or the geometry is degenerate.

Because the reference map may keep only a few axes and ecological
dissimilarities are generally not Euclidean, a point exactly satisfying
all observations may not exist. In that case the least-squares compromise
is returned and the root-mean-square residual (in distance units) is
reported per sample, so misfit is visible rather than hidden.

Identifiability
---------------
A position is uniquely determined by distances only if the observed
reference points affinely span the retained space (affine rank ``k``,
requiring at least ``k + 1`` affinely independent references). If the
affine rank is ``k - 1`` the observations are consistent with two
mirror-image positions ("ambiguous"); if it is smaller, a continuum of
positions fits equally well ("underdetermined"). Rows that cannot be
solved to convergence are reported as "failed" and carry no coordinates.

Workspace, precision and state size
-----------------------------------
The stored state is deliberately small: one row of coordinates per
reference sample (plus IDs and axis labels), so its size grows linearly
with the number of references for a fixed number of axes -- the quadratic
reference distance matrix is decomposed once at fit time and never
stored, and projecting neither recomputes it nor re-runs the
decomposition. Per projection call, a single vector of the reference
squared norms (size ``n_references``) is precomputed; each supplementary
sample is then solved independently, allocating only per-row temporaries
proportional to that row's number of observed references, so a batch
never materializes extra copies of the distance table or the solver
intermediates of other rows. Row independence is also what makes results
identical regardless of batch composition or column order. Precision is
spent where it matters: the closed-form initializer is exact for
consistent Euclidean inputs, and the nonlinear refinement is driven to
machine-precision tolerances, so exactly determined positions are
reproduced to about 1e-13 in well-conditioned geometries (far below the
documented 1e-7 guarantee), while genuinely inconsistent inputs converge
to an honest least-squares compromise instead of being forced to zero
residual.

References
----------
.. [1] Gower, J. C. (1968). Adding a point to vector diagrams in
   multivariate analysis. Biometrika, 55(3), 582-585.

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Version of the projection state layout. Stored in the "Projection state"
# section of the ``ordination`` file format. Readers must reject files
# carrying a version they do not understand.
PROJECTION_STATE_VERSION = 1

# Placement statuses reported in ``ProjectionResult.diagnostics``.
STATUS_OK = "ok"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNDERDETERMINED = "underdetermined"
STATUS_FAILED = "failed"

# Relative singular-value tolerance used to determine the affine rank of
# the observed reference points. Singular values of the centered observed
# reference coordinates below ``_RANK_RTOL`` times the largest one are
# treated as zero, i.e. the corresponding directions carry no geometric
# information (collinear or nearly degenerate configurations are flagged
# rather than silently trusted).
_RANK_RTOL = 1e-10

# Solver tolerances. The Levenberg-Marquardt path (used whenever there are
# at least as many residuals as variables) is driven to machine precision
# so that exactly consistent inputs are reproduced essentially exactly.
_LM_TOL = 1e-15
_TRF_GTOL = 1e-12
_MAX_NFEV = 1000


class ProjectionState:
    """Fixed reference configuration for projecting supplementary samples.

    A ``ProjectionState`` freezes the ordination axes chosen at reference
    fit time: the reference sample IDs, their coordinates on the retained
    axes, and the axis labels. It contains one row of coordinates per
    reference sample, so its memory footprint grows linearly with the
    number of references for a fixed number of axes; in particular it does
    *not* contain the (quadratic) reference distance matrix.

    Attributes
    ----------
    ids : tuple of str
        Reference sample IDs, in the order of ``coordinates`` rows.
    coordinates : ndarray of shape (n_references, n_dimensions)
        Reference coordinates on the retained ordination axes. Read-only.
    axis_labels : tuple of str
        Labels of the retained ordination axes.
    version : int
        Layout version of the state (currently 1).

    Notes
    -----
    Instances are normally created by ``pcoa(..., projection=True)`` or by
    reading an ``ordination`` file that contains a projection state
    section, rather than directly.

    """

    def __init__(self, ids, coordinates, axis_labels,
                 version=PROJECTION_STATE_VERSION):
        if version != PROJECTION_STATE_VERSION:
            raise ValueError(
                "Unsupported projection state version: %r. This version of "
                "scikit-bio supports version %d."
                % (version, PROJECTION_STATE_VERSION)
            )
        ids = tuple(ids)
        axis_labels = tuple(axis_labels)
        coordinates = np.array(coordinates, dtype=np.float64, copy=True)
        if coordinates.ndim != 2:
            raise ValueError(
                "Projection state coordinates must be a 2-D array, got "
                "%d dimension(s)." % coordinates.ndim
            )
        n_refs, n_dims = coordinates.shape
        if n_refs < 1 or n_dims < 1:
            raise ValueError(
                "Projection state must contain at least one reference and "
                "one dimension, got shape %s." % (coordinates.shape,)
            )
        if len(ids) != n_refs:
            raise ValueError(
                "Projection state lists %d reference IDs but %d coordinate "
                "rows." % (len(ids), n_refs)
            )
        if len(set(ids)) != n_refs:
            raise ValueError("Projection state reference IDs must be unique.")
        if len(axis_labels) != n_dims:
            raise ValueError(
                "Projection state lists %d axis labels but coordinates have "
                "%d columns." % (len(axis_labels), n_dims)
            )
        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Projection state coordinates must be finite numbers."
            )
        coordinates.setflags(write=False)
        self._ids = ids
        self._coordinates = coordinates
        self._axis_labels = axis_labels
        self._version = version

    @property
    def ids(self):
        """Reference sample IDs."""
        return self._ids

    @property
    def coordinates(self):
        """Reference coordinates on the retained axes (read-only)."""
        return self._coordinates

    @property
    def axis_labels(self):
        """Labels of the retained ordination axes."""
        return self._axis_labels

    @property
    def version(self):
        """Layout version of the state."""
        return self._version

    @property
    def n_refs(self):
        """Number of reference samples."""
        return len(self._ids)

    @property
    def n_dims(self):
        """Number of retained ordination axes."""
        return self._coordinates.shape[1]

    def __repr__(self):
        """Return a summary of the projection state."""
        return (
            "ProjectionState(n_refs=%d, n_dims=%d, version=%d)"
            % (self.n_refs, self.n_dims, self._version)
        )


class ProjectionResult(NamedTuple):
    """Result of projecting supplementary samples into an ordination.

    Attributes
    ----------
    coordinates : pd.DataFrame
        Projected coordinates of the supplementary samples, row-indexed by
        the sample IDs provided in the input ``distances`` table, with one
        column per retained ordination axis. Rows that could not be placed
        (status ``"failed"``) or that carry no observations at all contain
        ``NaN``.
    diagnostics : pd.DataFrame
        Per-sample diagnostics, row-indexed like ``coordinates``:

        - ``n_valid`` (int): number of observed (non-missing) distances to
          reference samples for this row.
        - ``rank`` (int): affine rank of the observed reference points,
          i.e. the dimension of their affine span. Placement is unique
          only when this equals the number of retained axes.
        - ``rms`` (float): root-mean-square residual of the *actual*
          distances, ``sqrt(mean((||y - x_i|| - d_i)**2))``, in the units
          of the input distances. ``NaN`` when no position was obtained.
        - ``status`` (str): one of ``"ok"`` (converged, geometry uniquely
          determines the position), ``"ambiguous"`` (observed references
          span all but one dimension; two mirror-image positions fit
          equally well), ``"underdetermined"`` (a continuum of positions
          fits equally well), or ``"failed"`` (the solver did not
          converge; no coordinates are reported).

    """

    coordinates: pd.DataFrame
    diagnostics: pd.DataFrame


def project(state, distances):
    """Project supplementary samples into a fixed ordination space.

    See :meth:`skbio.stats.ordination.OrdinationResults.project` for the
    full documentation of the placement model, input validation and
    return value. This function implements the computation against an
    explicit :class:`ProjectionState`.

    """
    values, col_pos = _validate_distances(state, distances)

    coords = state.coordinates
    n_dims = coords.shape[1]
    # ||x_i||^2 per reference, computed once per call (size n_refs).
    g = np.einsum("ij,ij->i", coords, coords)

    n_rows = values.shape[0]
    out_coords = np.full((n_rows, n_dims), np.nan)
    out_n_valid = np.zeros(n_rows, dtype=np.int64)
    out_rank = np.zeros(n_rows, dtype=np.int64)
    out_rms = np.full(n_rows, np.nan)
    out_status = np.empty(n_rows, dtype=object)

    # Rows are solved independently: a row's result does not depend on
    # which other rows are present in the batch. Only per-row temporaries
    # (of size proportional to the row's observed references) are
    # allocated; no batch-wide distance copies or solver intermediates
    # are materialized beyond the input table and the output.
    for i in range(n_rows):
        row = values[i]
        observed = ~np.isnan(row)
        pos = col_pos[observed]
        d = row[observed]
        if pos.size:
            # Reorder to the canonical reference order so that the result
            # is invariant under permutation of the input columns.
            order = np.argsort(pos, kind="stable")
            pos = pos[order]
            d = d[order]
        y, rank, rms, status = _solve_row(coords, g, pos, d, n_dims)
        if y is not None:
            out_coords[i] = y
        out_n_valid[i] = pos.size
        out_rank[i] = rank
        out_rms[i] = rms
        out_status[i] = status

    coordinates = pd.DataFrame(
        out_coords, index=distances.index, columns=state.axis_labels
    )
    diagnostics = pd.DataFrame(
        {
            "n_valid": out_n_valid,
            "rank": out_rank,
            "rms": out_rms,
            "status": out_status,
        },
        index=distances.index,
    )
    return ProjectionResult(coordinates=coordinates, diagnostics=diagnostics)


def _validate_distances(state, distances):
    """Validate the distances table and extract a canonical numeric view.

    Returns the table values as a float64 array (NaN marks unobserved
    pairs) and the position of each column within the state's reference
    order. The caller's DataFrame is never modified.
    """
    if not isinstance(distances, pd.DataFrame):
        raise TypeError(
            "distances must be a pandas DataFrame, got %s."
            % type(distances).__name__
        )
    if distances.index.has_duplicates:
        raise ValueError(
            "distances row labels (supplementary sample IDs) must be "
            "unique."
        )
    if distances.columns.has_duplicates:
        raise ValueError(
            "distances column labels (reference sample IDs) must be "
            "unique."
        )

    ref_pos = {ref_id: i for i, ref_id in enumerate(state.ids)}
    unknown = []
    for c in distances.columns:
        try:
            known = c in ref_pos
        except TypeError:
            # unhashable column labels cannot be reference IDs
            known = False
        if not known:
            unknown.append(c)
    if unknown:
        raise ValueError(
            "distances contains columns that are not reference sample IDs: "
            "%r." % (unknown,)
        )
    col_pos = np.fromiter(
        (ref_pos[c] for c in distances.columns),
        dtype=np.intp,
        count=len(distances.columns),
    )

    arr = distances.values
    kind = arr.dtype.kind
    if kind == "b":
        raise ValueError("distances values must be real numbers, not bool.")
    if kind == "c":
        raise ValueError(
            "distances values must be real numbers, not complex."
        )
    if kind not in "iuf":
        raise ValueError(
            "distances values must be numeric (non-negative real numbers "
            "or NaN for unobserved pairs), got dtype %s." % arr.dtype
        )
    values = np.asarray(arr, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("distances values must be finite, not infinite.")
    if (values < 0).any():
        raise ValueError("distances values must be non-negative.")
    return values, col_pos


def _solve_row(coords, g, obs_pos, d, n_dims):
    """Locate one supplementary sample from its observed distances.

    Parameters
    ----------
    coords : ndarray of shape (n_refs, n_dims)
        Fixed reference coordinates.
    g : ndarray of shape (n_refs,)
        Squared norms of the reference coordinates.
    obs_pos : ndarray of int
        Positions (into ``coords``) of the references observed for this
        sample, in canonical (ascending) order.
    d : ndarray of float
        Observed distances, aligned with ``obs_pos``.
    n_dims : int
        Number of retained ordination axes.

    Returns
    -------
    y : ndarray of shape (n_dims,) or None
        The located position, or None if no position could be obtained
        (no observations, or the solver did not converge).
    rank : int
        Affine rank of the observed reference points.
    rms : float
        Root-mean-square residual of the actual distances; NaN if no
        position was obtained.
    status : str
        One of the ``STATUS_*`` constants.

    """
    m = obs_pos.shape[0]
    if m == 0:
        return None, 0, np.nan, STATUS_UNDERDETERMINED

    x_obs = coords[obs_pos]
    centroid = x_obs.mean(axis=0)
    a = x_obs - centroid

    # Affine rank of the observed reference points, and a basis for the
    # directions they fail to span (used for symmetry-breaking starts).
    full_matrices = m < n_dims
    _, sv, vh = np.linalg.svd(a, full_matrices=full_matrices)
    tol = sv[0] * _RANK_RTOL
    rank = int(np.count_nonzero(sv > tol))

    # Gower's closed-form initializer. With
    #   d_i^2 = ||y||^2 - 2 x_i . y + ||x_i||^2,
    # subtracting the observed mean eliminates the unknown ||y||^2 and
    # leaves a linear system in y, solved here in least squares (the
    # minimum-norm solution when the system is underdetermined). For
    # mutually consistent Euclidean distances this is already exact.
    b = d * d - g[obs_pos]
    rhs = b - b.mean()
    y0 = np.linalg.lstsq(-2.0 * a, rhs, rcond=None)[0]

    # Deterministic starting points: the closed-form solution, then (for
    # rank-deficient geometries) perturbations along the unspanned
    # directions so that mirror-image minima off the reference hull can be
    # found, plus generic fallbacks.
    scale = max(
        1.0,
        float(sv[0]),
        float(d.max()),
        float(np.sqrt(g[obs_pos]).max()),
    )
    starts = [y0]
    for j in range(rank, n_dims):
        starts.append(y0 + scale * vh[j])
        starts.append(y0 - scale * vh[j])
    starts.append(centroid)
    starts.append(np.zeros(n_dims))

    method = "lm" if m >= n_dims else "trf"
    if method == "lm":
        solver_kwargs = dict(
            method="lm",
            ftol=_LM_TOL,
            xtol=_LM_TOL,
            gtol=_LM_TOL,
            max_nfev=_MAX_NFEV,
        )
    else:
        solver_kwargs = dict(
            method="trf",
            ftol=_LM_TOL,
            xtol=_LM_TOL,
            gtol=_TRF_GTOL,
            max_nfev=_MAX_NFEV,
        )

    # A solution at or below this cost reproduces every observed distance
    # to essentially machine precision; accept it immediately.
    exact_cost = (1e-12 * max(1.0, float(d.max()))) ** 2

    best = None
    for y_init in starts:
        try:
            res = least_squares(
                _distance_residuals,
                y_init,
                jac=_distance_jacobian,
                args=(x_obs, d),
                **solver_kwargs,
            )
        except Exception:
            # A failed start must not abort the row, let alone the batch.
            continue
        if not (res.success and np.all(np.isfinite(res.x))):
            continue
        if best is None or res.cost < best.cost:
            best = res
            if best.cost <= exact_cost:
                break

    if best is None:
        return None, rank, np.nan, STATUS_FAILED

    y = best.x
    residuals = _distance_residuals(y, x_obs, d)
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    if rank == n_dims:
        status = STATUS_OK
    elif rank == n_dims - 1:
        status = STATUS_AMBIGUOUS
    else:
        status = STATUS_UNDERDETERMINED
    return y, rank, rms, status


def _distance_residuals(y, x_obs, d):
    """Residuals of actual distances: ``||y - x_i|| - d_i``."""
    diff = y - x_obs
    return np.sqrt(np.einsum("ij,ij->i", diff, diff)) - d


def _distance_jacobian(y, x_obs, d):
    """Jacobian of the distance residuals with respect to ``y``."""
    diff = y - x_obs
    norms = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    # Guard the (measure-zero) case of y coinciding with a reference
    # point, where the distance is not differentiable: use a zero row as
    # the subgradient so the solver can proceed via the other residuals.
    safe = np.where(norms > 0.0, norms, 1.0)
    jac = diff / safe[:, None]
    jac[norms == 0.0] = 0.0
    return jac
