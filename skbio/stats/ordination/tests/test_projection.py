r"""Tests for projecting supplementary samples into a PCoA ordination.

The feature under test is ``pcoa(..., projection=True)`` together with
``OrdinationResults.project`` (implemented in
``skbio/stats/ordination/_projection.py``). The tests below start from
synthetic low-dimensional reference configurations whose positions can be
computed by hand, and then cover missing observations, multiple-solution
and underdetermined geometries, duplicated positions, zero self-distances,
collinear and nearly degenerate references, non-Euclidean dissimilarities
and truncated dimensionality. File round-trips (including verification in
a fresh process) and rejection of corrupted projection state sections are
tested in ``skbio/io/format/tests/test_ordination.py``.

To re-run these tests on Linux (from the repository root, with scikit-bio
built in place)::

    python -m pytest skbio/stats/ordination/tests/test_projection.py -v
    python -m pytest skbio/io/format/tests/test_ordination.py -v

Regression of the pre-existing ordination and diversity entry points::

    python -m pytest skbio/stats/ordination skbio/diversity
    python -m pytest skbio/io/format/tests/test_ordination.py

Fitting rules verified here (see the docstrings of
``OrdinationResults.project`` and ``skbio/stats/ordination/_projection.py``
for the full description):

* each supplementary sample is placed independently by minimizing the sum
  of squared residuals of the *actual* distances
  ``sum_i (||y - x_i|| - d_i)**2`` over the observed references;
* the solver starts from Gower's closed-form interpolation (exact for
  mutually consistent Euclidean distances) and refines it with
  Levenberg-Marquardt (trust-region reflective when underdetermined),
  with deterministic fallback starts;
* the reported ``rms`` is the root-mean-square of the actual distance
  residuals, never an algebraic residual of squared distances;
* a row is ``ok`` only if the observed reference points affinely span the
  retained space; rank ``k - 1`` is reported ``ambiguous`` (two mirror
  solutions), lower rank ``underdetermined`` (a continuum of solutions),
  and a non-converged solve ``failed`` (no coordinates reported).

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main, mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from scipy.optimize import least_squares as scipy_least_squares

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats.ordination._projection import (
    PROJECTION_STATE_VERSION,
    ProjectionState,
)
from skbio.util import assert_ordination_results_equal


def _euclidean_dm(points, ids):
    """DistanceMatrix of Euclidean distances between rows of `points`."""
    d = np.sqrt(((points[:, None] - points) ** 2).sum(-1))
    return DistanceMatrix(d, ids)


def _rectangle():
    """Four references at the corners of a 4x2 rectangle (hand-computable).

    The points are centered, with distinct eigenvalues (16, 4), so the PCoA
    coordinates equal the true coordinates up to per-axis sign flips.
    """
    refs = np.array(
        [[2.0, 1.0], [2.0, -1.0], [-2.0, 1.0], [-2.0, -1.0]]
    )
    ids = ["r1", "r2", "r3", "r4"]
    return refs, ids


def _random_refs(n=8, k=3, seed=42):
    """Random centered references in k dimensions (well-conditioned)."""
    rng = np.random.default_rng(seed)
    refs = rng.normal(size=(n, k))
    refs -= refs.mean(axis=0)
    ids = ["s%d" % i for i in range(n)]
    return refs, ids


def _procrustes_q(from_coords, to_coords):
    """Orthogonal Q minimizing ||from_coords @ Q - to_coords|| (Frobenius).

    For centered Euclidean references, the PCoA coordinates P satisfy
    P = X @ Q_true for an orthogonal Q_true; a point y in the original
    (centered) frame therefore sits at Q_true.T @ y in the PCoA frame.
    """
    u, _, vt = np.linalg.svd(from_coords.T @ to_coords)
    return u @ vt


def _frame_transform(refs, pcoa_coords):
    """Return (center, Q) mapping true positions into the PCoA frame.

    PCoA coordinates are centered, so a true position y sits at
    ``(y - center) @ Q`` in the ordination coordinate system, where
    ``center`` is the mean of the reference points and Q the orthogonal
    transform recovered from the references.
    """
    center = refs.mean(axis=0)
    q = _procrustes_q(refs - center, pcoa_coords)
    return center, q


def _as_frame(rows, index, columns):
    return pd.DataFrame(rows, index=index, columns=columns)


class PcoaProjectionOptionTests(TestCase):
    """The ``projection`` option of ``pcoa``."""

    def setUp(self):
        self.refs, self.ids = _rectangle()
        self.dm = _euclidean_dm(self.refs, self.ids)

    def test_default_off(self):
        self.assertIsNone(pcoa(self.dm, dimensions=2).projection_state)
        self.assertIsNone(pcoa(self.dm).projection_state)

    def test_reference_fit_unchanged_by_option(self):
        # Enabling the option must not change the reference ordination:
        # same eigenvalues, proportions and coordinates (up to the sign /
        # rotation freedom of the eigen-subspaces themselves).
        expected = pcoa(self.dm, dimensions=2)
        observed = pcoa(self.dm, dimensions=2, projection=True)
        assert_ordination_results_equal(
            observed, expected, ignore_directionality=True
        )

    def test_state_contents(self):
        res = pcoa(self.dm, dimensions=2, projection=True)
        state = res.projection_state
        self.assertIsInstance(state, ProjectionState)
        self.assertEqual(state.version, PROJECTION_STATE_VERSION)
        self.assertEqual(tuple(state.ids), tuple(self.ids))
        self.assertEqual(tuple(state.axis_labels), ("PC1", "PC2"))
        self.assertEqual(state.coordinates.shape, (4, 2))
        self.assertEqual(state.coordinates.dtype, np.float64)
        npt.assert_allclose(
            state.coordinates, np.asarray(res.samples), atol=1e-12
        )
        # the frozen axes are read-only
        with self.assertRaises(ValueError):
            state.coordinates[0, 0] = 1.0

    def test_state_independent_of_samples_table(self):
        # Once fitted, the axes are fixed: mutating the samples table of
        # the result must not move the projection axes.
        res = pcoa(self.dm, dimensions=2, projection=True)
        query = np.array([0.5, 0.3])
        dist = np.sqrt(((self.refs - query) ** 2).sum(-1))
        df = _as_frame([dist], ["q1"], self.ids)
        before = res.project(df)
        res.samples = np.abs(res.samples)
        after = res.project(df)
        npt.assert_array_equal(
            before.coordinates.values, after.coordinates.values
        )

    def test_requires_eigh(self):
        with self.assertRaisesRegex(ValueError, "method='eigh'"):
            pcoa(self.dm, method="fsvd", dimensions=2, projection=True)
        # unknown methods are rejected explicitly, never silently switched
        with self.assertRaisesRegex(ValueError, "method='eigh'"):
            pcoa(self.dm, method="asdf", dimensions=2, projection=True)

    def test_requires_positive_integer_dimensions(self):
        for bad in (0, -1, 2.5, 0.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                pcoa(self.dm, dimensions=bad, projection=True)

    def test_rejects_dimensions_beyond_positive_rank(self):
        # the rectangle geometry has exactly two positive eigenvalues
        with self.assertRaisesRegex(ValueError, "positive eigenvalues"):
            pcoa(self.dm, dimensions=3, projection=True)
        with self.assertRaisesRegex(ValueError, "positive eigenvalues"):
            pcoa(self.dm, dimensions=4, projection=True)
        # more axes than samples is still rejected by the pre-existing check
        with self.assertRaisesRegex(ValueError, "cannot extend"):
            pcoa(self.dm, dimensions=5, projection=True)

    def test_rejects_rank_deficient_geometry(self):
        # exactly collinear references have a single positive eigenvalue
        refs = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        dm = _euclidean_dm(refs, list("abcd"))
        with self.assertRaisesRegex(ValueError, "positive eigenvalues"):
            pcoa(dm, dimensions=2, projection=True)
        # but keeping the single positive axis is fine
        res = pcoa(dm, dimensions=1, projection=True)
        self.assertEqual(res.projection_state.n_dims, 1)


class PlacementAccuracyTests(TestCase):
    """Placement against hand-computable and Euclidean ground truth."""

    def setUp(self):
        self.refs, self.ids = _rectangle()
        self.dm = _euclidean_dm(self.refs, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)
        # orthogonal transform from the true frame to the PCoA frame
        self.center, self.q = _frame_transform(
            self.refs, self.res.projection_state.coordinates
        )

    def _expected(self, points):
        """True positions expressed in the PCoA coordinate system."""
        return (np.asarray(points) - self.center) @ self.q

    def test_hand_computed_positions(self):
        queries = np.array(
            [[0.5, 0.3], [-1.2, 0.7], [0.0, 0.0], [1.9, -0.9]]
        )
        dists = np.sqrt(((self.refs[None] - queries[:, None]) ** 2).sum(-1))
        df = _as_frame(dists, ["q%d" % i for i in range(4)], self.ids)
        out = self.res.project(df)
        npt.assert_allclose(
            out.coordinates.values, self._expected(queries), atol=1e-7
        )
        diag = out.diagnostics
        self.assertTrue((diag["status"] == "ok").all())
        self.assertTrue((diag["n_valid"] == 4).all())
        self.assertTrue((diag["rank"] == 2).all())
        self.assertTrue((diag["rms"] < 1e-10).all())

    def test_euclidean_3d(self):
        refs, ids = _random_refs(n=8, k=3, seed=42)
        res = pcoa(_euclidean_dm(refs, ids), dimensions=3, projection=True)
        center, q = _frame_transform(refs, res.projection_state.coordinates)
        rng = np.random.default_rng(7)
        queries = rng.normal(size=(5, 3))
        dists = np.sqrt(((refs[None] - queries[:, None]) ** 2).sum(-1))
        df = _as_frame(dists, ["q%d" % i for i in range(5)], ids)
        out = res.project(df)
        npt.assert_allclose(
            out.coordinates.values, (queries - center) @ q, atol=1e-7
        )
        self.assertTrue((out.diagnostics["status"] == "ok").all())

    def test_missing_distances_unique_solution(self):
        # drop one distance: the remaining three non-collinear references
        # still uniquely determine the position
        query = np.array([0.5, 0.3])
        dist = np.sqrt(((self.refs - query) ** 2).sum(-1))
        row = [dist[0], np.nan, dist[2], dist[3]]
        df = _as_frame([row], ["q1"], self.ids)
        out = self.res.project(df)
        npt.assert_allclose(
            out.coordinates.values, self._expected([query]), atol=1e-7
        )
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "ok")
        self.assertEqual(diag["n_valid"], 3)

        # same for a column subset (rather than NaN entries)
        df2 = _as_frame(
            [[dist[0], dist[2], dist[3]]], ["q1"], ["r1", "r3", "r4"]
        )
        out2 = self.res.project(df2)
        npt.assert_allclose(
            out2.coordinates.values, out.coordinates.values, atol=1e-9
        )

    def test_missing_distances_3d(self):
        refs, ids = _random_refs(n=8, k=3, seed=1)
        res = pcoa(_euclidean_dm(refs, ids), dimensions=3, projection=True)
        center, q = _frame_transform(refs, res.projection_state.coordinates)
        query = np.array([0.4, -0.7, 0.2])
        dist = np.sqrt(((refs - query) ** 2).sum(-1))
        row = dist.copy()
        row[[0, 3, 6]] = np.nan  # five references remain, affine rank 3
        df = _as_frame([row], ["q1"], ids)
        out = res.project(df)
        npt.assert_allclose(
            out.coordinates.values, ((query - center) @ q)[None, :], atol=1e-7
        )
        self.assertEqual(out.diagnostics.iloc[0]["status"], "ok")
        self.assertEqual(out.diagnostics.iloc[0]["n_valid"], 5)

    def test_self_distances_recover_reference_coordinates(self):
        # Feeding each reference sample's own distances back must reproduce
        # its fixed coordinates; the zero self-distance is a valid
        # observation and counts towards n_valid.
        df = _as_frame(self.dm.data, self.ids, self.ids)
        out = self.res.project(df)
        npt.assert_allclose(
            out.coordinates.values,
            self.res.projection_state.coordinates,
            atol=1e-7,
        )
        diag = out.diagnostics
        self.assertTrue((diag["status"] == "ok").all())
        self.assertTrue((diag["n_valid"] == 4).all())
        self.assertTrue((diag["rms"] < 1e-10).all())

    def test_self_distances_recover_reference_coordinates_3d(self):
        refs, ids = _random_refs(n=8, k=3, seed=5)
        dm = _euclidean_dm(refs, ids)
        res = pcoa(dm, dimensions=3, projection=True)
        df = _as_frame(dm.data, ids, ids)
        out = res.project(df)
        npt.assert_allclose(
            out.coordinates.values,
            res.projection_state.coordinates,
            atol=1e-7,
        )


class DegenerateGeometryTests(TestCase):
    """Ambiguous, underdetermined, duplicated and degenerate geometries."""

    def setUp(self):
        # full set is well-conditioned in 2D; the subsets {a, b, c} (nearly
        # collinear) and {a, c, e} (exactly collinear) are degenerate
        self.refs = np.array(
            [[0.0, 0.0], [1.0, 1e-13], [2.0, 0.0], [0.0, 4.0], [4.0, 0.0]]
        )
        self.ids = list("abcde")
        self.dm = _euclidean_dm(self.refs, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)
        self.center, self.q = _frame_transform(
            self.refs, self.res.projection_state.coordinates
        )

    def _project_query(self, query, observed_ids):
        dist = np.sqrt(((self.refs - query) ** 2).sum(-1))
        row = [
            dist[self.ids.index(r)] if r in observed_ids else np.nan
            for r in self.ids
        ]
        df = _as_frame([row], ["q1"], self.ids)
        return self.res.project(df)

    def test_ambiguous_mirror_solutions(self):
        # exactly collinear observed references (a, c, e on the x-axis):
        # the true position and its mirror image fit equally well
        query = np.array([1.5, 0.7])
        out = self._project_query(query, {"a", "c", "e"})
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "ambiguous")
        self.assertEqual(diag["rank"], 1)
        self.assertLess(diag["rms"], 1e-9)
        expected = (query - self.center) @ self.q
        mirror = (np.array([1.5, -0.7]) - self.center) @ self.q
        observed = out.coordinates.values[0]
        err = min(
            np.abs(observed - expected).max(),
            np.abs(observed - mirror).max(),
        )
        self.assertLess(err, 1e-7)

    def test_nearly_degenerate_references_flagged(self):
        # a, b, c are collinear up to 1e-13: below the rank tolerance, so
        # the row is flagged rather than silently trusted
        query = np.array([1.0, 0.5])
        out = self._project_query(query, {"a", "b", "c"})
        self.assertEqual(out.diagnostics.iloc[0]["status"], "ambiguous")
        self.assertEqual(out.diagnostics.iloc[0]["rank"], 1)

    def test_underdetermined_single_reference(self):
        query = np.array([1.0, 0.5])
        out = self._project_query(query, {"a"})
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "underdetermined")
        self.assertEqual(diag["rank"], 0)
        self.assertEqual(diag["n_valid"], 1)
        # the best-fit position is still returned and honors the distance
        self.assertTrue(np.isfinite(out.coordinates.values).all())
        self.assertLess(diag["rms"], 1e-9)

    def test_underdetermined_no_observations(self):
        df = _as_frame([[np.nan] * 5], ["q1"], self.ids)
        out = self.res.project(df)
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "underdetermined")
        self.assertEqual(diag["n_valid"], 0)
        self.assertTrue(np.isnan(diag["rms"]))
        self.assertTrue(np.isnan(out.coordinates.values).all())

    def test_duplicate_positions_not_merged(self):
        # two reference samples at the same position keep their identities
        refs = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        ids = ["a", "a_dup", "b", "c"]
        dm = _euclidean_dm(refs, ids)
        res = pcoa(dm, dimensions=2, projection=True)
        self.assertEqual(tuple(res.projection_state.ids), tuple(ids))

        # two supplementary samples with identical distance profiles are
        # placed at the same coordinates but are not merged
        query = np.array([1.0, 1.5])
        dist = np.sqrt(((refs - query) ** 2).sum(-1))
        df = _as_frame([dist, dist], ["q1", "q2"], ids)
        out = res.project(df)
        self.assertEqual(list(out.coordinates.index), ["q1", "q2"])
        npt.assert_array_equal(
            out.coordinates.values[0], out.coordinates.values[1]
        )
        # both duplicated references count as observations
        self.assertEqual(out.diagnostics.iloc[0]["n_valid"], 4)
        self.assertEqual(out.diagnostics.iloc[0]["status"], "ok")

    def test_zero_self_distance_is_observed(self):
        refs, ids = _rectangle()
        dm = _euclidean_dm(refs, ids)
        res = pcoa(dm, dimensions=2, projection=True)
        # a supplementary sample at distance zero from r1 (and the
        # consistent distances to the others) lands exactly on r1
        row = dm.data[0]
        self.assertEqual(row[0], 0.0)
        df = _as_frame([row], ["q1"], ids)
        out = res.project(df)
        self.assertEqual(out.diagnostics.iloc[0]["n_valid"], 4)
        npt.assert_allclose(
            out.coordinates.values[0],
            res.projection_state.coordinates[0],
            atol=1e-7,
        )


class InconsistentDistanceTests(TestCase):
    """Non-Euclidean dissimilarities and truncated dimensionality."""

    def setUp(self):
        self.refs, self.ids = _rectangle()
        self.dm = _euclidean_dm(self.refs, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)

    def test_non_euclidean_distances_least_squares(self):
        # all distances equal to 1.5: no point of the plane is equidistant
        # to the four corners at that radius (their true common distance
        # is sqrt(5)); the least-squares compromise is the center, with
        # rms = sqrt(5) - 1.5
        df = _as_frame([[1.5] * 4], ["q1"], self.ids)
        out = self.res.project(df)
        npt.assert_allclose(
            out.coordinates.values, [[0.0, 0.0]], atol=1e-6
        )
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "ok")
        npt.assert_allclose(diag["rms"], np.sqrt(5) - 1.5, rtol=1e-9)

    def test_rms_is_actual_distance_residual(self):
        df = _as_frame([[1.5] * 4], ["q1"], self.ids)
        out = self.res.project(df)
        y = out.coordinates.values[0]
        x = self.res.projection_state.coordinates
        # actual distance residuals
        r = np.sqrt(((x - y) ** 2).sum(-1)) - 1.5
        rms_actual = np.sqrt((r**2).mean())
        npt.assert_allclose(
            out.diagnostics.iloc[0]["rms"], rms_actual, rtol=1e-12
        )
        # the algebraic residual of the squared distances is a different
        # quantity and must not be reported as the distance RMS
        r_sq = (x**2).sum(-1) + (y**2).sum() - 2 * x @ y - 1.5**2
        rms_squared = np.sqrt((r_sq**2).mean())
        self.assertGreater(abs(rms_squared - rms_actual), 0.1)

    def test_truncated_dimensions(self):
        # references live in 3D but only two axes are retained
        refs, ids = _random_refs(n=8, k=3, seed=11)
        dm = _euclidean_dm(refs, ids)
        res2 = pcoa(dm, dimensions=2, projection=True)
        self.assertEqual(res2.projection_state.n_dims, 2)

        query = np.array([0.3, -0.5, 0.8])
        dist = np.sqrt(((refs - query) ** 2).sum(-1))
        df = _as_frame([dist], ["q1"], ids)
        out = res2.project(df)
        diag = out.diagnostics.iloc[0]
        self.assertEqual(diag["status"], "ok")
        self.assertTrue(np.isfinite(out.coordinates.values).all())
        # the 3D geometry cannot be exactly represented in 2D
        self.assertGreater(diag["rms"], 1e-9)

        # the fit is at least as good as the point's own "shadow" on the
        # retained plane (its coordinates in the full 3D fit, restricted
        # to the two retained axes, sign-aligned)
        res3 = pcoa(dm, dimensions=3, projection=True)
        df3 = _as_frame(dm.data, ids, ids)
        shadow_refs = res3.project(df3).coordinates.values[:, :2]
        signs = np.sign(
            (res2.projection_state.coordinates * shadow_refs).sum(axis=0)
        )
        signs[signs == 0] = 1.0
        center3, q3 = _frame_transform(refs, res3.projection_state.coordinates)
        shadow = ((query - center3) @ q3)[:2] * signs
        x2 = res2.projection_state.coordinates
        r_shadow = np.sqrt(((x2 - shadow) ** 2).sum(-1)) - dist
        rms_shadow = np.sqrt((r_shadow**2).mean())
        self.assertLessEqual(diag["rms"], rms_shadow * (1 + 1e-9))


class BatchBehaviorTests(TestCase):
    """Row independence, permutation invariance and failure isolation."""

    def setUp(self):
        self.refs, self.ids = _rectangle()
        self.dm = _euclidean_dm(self.refs, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)
        self.queries = np.array(
            [[0.5, 0.3], [-1.2, 0.7], [0.0, 0.0], [1.9, -0.9], [-0.4, 0.1]]
        )
        dists = np.sqrt(
            ((self.refs[None] - self.queries[:, None]) ** 2).sum(-1)
        )
        self.qids = ["q%d" % i for i in range(len(self.queries))]
        self.df = _as_frame(dists, self.qids, self.ids)

    def test_batch_independence(self):
        # a row projected alone, within a batch, or within a different
        # batch must give the same result and status
        together = self.res.project(self.df)
        singles = [
            self.res.project(self.df.iloc[[i]]) for i in range(len(self.df))
        ]
        half = len(self.df) // 2
        batch_a = self.res.project(self.df.iloc[:half])
        batch_b = self.res.project(self.df.iloc[half:])

        for i, single in enumerate(singles):
            npt.assert_allclose(
                single.coordinates.values[0],
                together.coordinates.values[i],
                atol=1e-9,
            )
            self.assertEqual(
                single.diagnostics.iloc[0]["status"],
                together.diagnostics.iloc[i]["status"],
            )
        npt.assert_allclose(
            pd.concat([batch_a.coordinates, batch_b.coordinates]).values,
            together.coordinates.values,
            atol=1e-9,
        )

    def test_column_permutation_invariant(self):
        perm = [3, 1, 0, 2]
        df_perm = self.df[[self.ids[i] for i in perm]]
        out = self.res.project(df_perm)
        ref = self.res.project(self.df)
        npt.assert_array_equal(
            out.coordinates.values, ref.coordinates.values
        )
        pdt.assert_frame_equal(out.diagnostics, ref.diagnostics)

    def test_row_permutation_aligned(self):
        perm = [4, 2, 0, 3, 1]
        df_perm = self.df.iloc[perm]
        out = self.res.project(df_perm)
        ref = self.res.project(self.df)
        npt.assert_array_equal(
            out.coordinates.loc[self.qids].values, ref.coordinates.values
        )

    def test_failed_row_does_not_affect_others(self):
        # force the solver to fail only for rows with exactly three
        # observed references
        real_ls = scipy_least_squares

        def fake_ls(fun, x0, *args, **kwargs):
            _, d = kwargs["args"]
            if len(d) == 3:
                return SimpleNamespace(success=False, x=x0, cost=np.inf)
            return real_ls(fun, x0, *args, **kwargs)

        row3 = self.df.iloc[0].copy()
        row3.iloc[1] = np.nan
        df = pd.concat([self.df.iloc[[0]], row3.to_frame().T])
        df.index = ["ok_row", "failing_row"]

        with mock.patch(
            "skbio.stats.ordination._projection.least_squares", fake_ls
        ):
            out = self.res.project(df)

        ok = out.diagnostics.loc["ok_row"]
        bad = out.diagnostics.loc["failing_row"]
        self.assertEqual(ok["status"], "ok")
        self.assertTrue(np.isfinite(out.coordinates.loc["ok_row"]).all())
        self.assertEqual(bad["status"], "failed")
        self.assertTrue(np.isnan(out.coordinates.loc["failing_row"]).all())
        self.assertTrue(np.isnan(bad["rms"]))
        self.assertEqual(bad["n_valid"], 3)

    def test_solver_failure_reported(self):
        def always_fail(fun, x0, *args, **kwargs):
            return SimpleNamespace(success=False, x=x0, cost=np.inf)

        with mock.patch(
            "skbio.stats.ordination._projection.least_squares", always_fail
        ):
            out = self.res.project(self.df)
        self.assertTrue((out.diagnostics["status"] == "failed").all())
        self.assertTrue(np.isnan(out.coordinates.values).all())
        self.assertTrue(out.diagnostics["rms"].isna().all())
        # diagnostics about the input remain available
        self.assertTrue((out.diagnostics["n_valid"] == 4).all())


class InputValidationTests(TestCase):
    """Validation of the distances table and immutability of inputs."""

    def setUp(self):
        self.refs, self.ids = _rectangle()
        self.dm = _euclidean_dm(self.refs, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)
        self.row = np.sqrt(((self.refs - [0.5, 0.3]) ** 2).sum(-1))

    def _df(self, values=None, index=("q1",), columns=None):
        if values is None:
            values = [self.row]
        if columns is None:
            columns = self.ids
        return pd.DataFrame(values, index=list(index), columns=columns)

    def test_non_dataframe_rejected(self):
        with self.assertRaises(TypeError):
            self.res.project(self.row[None, :])
        with self.assertRaises(TypeError):
            self.res.project([list(self.row)])

    def test_bool_rejected(self):
        df = self._df([[True, False, True, True]])
        with self.assertRaisesRegex(ValueError, "bool"):
            self.res.project(df)

    def test_complex_rejected(self):
        df = self._df([[1 + 0j, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "complex"):
            self.res.project(df)

    def test_non_numeric_rejected(self):
        df = self._df([["1.0", 2.0, 3.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.res.project(df)
        df = self._df([[object(), 2.0, 3.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.res.project(df)

    def test_infinite_rejected(self):
        for bad in (np.inf, -np.inf):
            row = self.row.copy()
            row[0] = bad
            with self.assertRaisesRegex(ValueError, "finite"):
                self.res.project(self._df([row]))

    def test_negative_rejected(self):
        row = self.row.copy()
        row[0] = -1e-9
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.res.project(self._df([row]))

    def test_integer_values_accepted(self):
        df = self._df([[1, 2, 3, 4]])
        out = self.res.project(df)
        self.assertEqual(out.diagnostics.iloc[0]["status"], "ok")

    def test_duplicate_labels_rejected(self):
        df = self._df([self.row, self.row], index=("q1", "q1"))
        with self.assertRaisesRegex(ValueError, "row labels"):
            self.res.project(df)
        df = self._df([self.row[:2]], columns=["r1", "r1"])
        with self.assertRaisesRegex(ValueError, "column labels"):
            self.res.project(df)

    def test_unknown_reference_columns_rejected(self):
        df = self._df([self.row[:2]], columns=["r1", "nope"])
        with self.assertRaisesRegex(ValueError, "not reference sample IDs"):
            self.res.project(df)

    def test_inputs_not_modified(self):
        df = self._df([self.row, self.row], index=("q1", "q2"))
        df_before = deepcopy(df)
        samples_before = self.res.samples.copy(deep=True)
        state_before = self.res.projection_state.coordinates.copy()
        ids_before = tuple(self.res.projection_state.ids)
        self.res.project(df)
        pdt.assert_frame_equal(df, df_before)
        pdt.assert_frame_equal(self.res.samples, samples_before)
        npt.assert_array_equal(
            self.res.projection_state.coordinates, state_before
        )
        self.assertEqual(tuple(self.res.projection_state.ids), ids_before)

    def test_project_without_state(self):
        res = pcoa(self.dm, dimensions=2)
        with self.assertRaisesRegex(ValueError, "projection state"):
            res.project(self._df())

    def test_empty_tables(self):
        # zero rows: an empty (but correctly structured) result
        df = self._df(np.empty((0, 4)), index=[])
        out = self.res.project(df)
        self.assertEqual(out.coordinates.shape, (0, 2))
        self.assertEqual(list(out.coordinates.columns), ["PC1", "PC2"])
        self.assertEqual(out.diagnostics.shape, (0, 4))
        self.assertEqual(out.diagnostics["n_valid"].dtype, np.int64)

        # zero columns: no observations for any row
        df = pd.DataFrame(index=["q1", "q2"], columns=[])
        out = self.res.project(df)
        self.assertTrue(
            (out.diagnostics["status"] == "underdetermined").all()
        )
        self.assertTrue((out.diagnostics["n_valid"] == 0).all())
        self.assertTrue(np.isnan(out.coordinates.values).all())


class RefitConsistencyTests(TestCase):
    """Consistency across refits after permuting the reference input."""

    def _check_refit(self, refs, ids, k, queries, seed):
        dm = _euclidean_dm(refs, ids)
        res1 = pcoa(dm, dimensions=k, projection=True)
        dists = np.sqrt(((refs[None] - queries[:, None]) ** 2).sum(-1))
        qids = ["q%d" % i for i in range(len(queries))]
        df = _as_frame(dists, qids, ids)
        out1 = res1.project(df)

        rng = np.random.default_rng(seed)
        p = rng.permutation(len(ids))
        refs_p = refs[p]
        ids_p = [ids[i] for i in p]
        dm_p = _euclidean_dm(refs_p, ids_p)
        res2 = pcoa(dm_p, dimensions=k, projection=True)
        out2 = res2.project(df)  # columns are matched by ID

        # align the two reference frames by ID and remove the orthogonal
        # transformation between them (never per-axis signs, which are
        # meaningless with repeated eigenvalues)
        x1 = res1.projection_state.coordinates
        x2 = res2.projection_state.coordinates
        order = [ids.index(i) for i in ids_p]
        q21 = _procrustes_q(x2, x1[order])
        mapped = out2.coordinates.values @ q21
        npt.assert_allclose(mapped, out1.coordinates.values, atol=1e-6)

    def test_refit_distinct_eigenvalues(self):
        refs, ids = _random_refs(n=8, k=3, seed=3)
        rng = np.random.default_rng(9)
        queries = rng.normal(size=(4, 3))
        self._check_refit(refs, ids, 3, queries, seed=123)

    def test_refit_repeated_eigenvalues(self):
        # square geometry: the two eigenvalues are equal, so the retained
        # subspace is only determined up to a rotation
        refs = np.array(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        )
        ids = list("abcd")
        queries = np.array([[0.2, 0.6], [-0.5, 0.1]])
        self._check_refit(refs, ids, 2, queries, seed=321)


if __name__ == "__main__":
    main()
