# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Tests for sparse (CSR / BIOM) support of the phylogenetic diversity metrics.

The same numerical definitions as the dense pipeline are asserted here: every
sparse result is checked against the dense result on the identical logical
count table.
"""

from io import StringIO
from unittest import TestCase, main

import numpy as np
import scipy.sparse as sp

from skbio import TreeNode
from skbio.table import Table
from skbio.tree import DuplicateNodeError, MissingNodeError
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.diversity.alpha import faith_pd, phydiv
from skbio.diversity.beta import unweighted_unifrac, weighted_unifrac
from skbio.diversity._sparse import (
    is_csr_counts,
    validate_sparse_counts,
)


TREE_NEWICK = (
    "(((((U1:0.5,U2:0.5):0.5,U3:1.0):1.0):0.0,"
    "(U4:0.75,(U5:0.5,((U6:0.33,U7:0.62):0.5,U8:0.5):0.5):0.5):1.25):0.0)"
    "root;"
)
TAXA = [f"U{i}" for i in range(1, 9)]

# Includes an all-zero row, so the empty-sample branch is always exercised.
DENSE = np.array(
    [
        [1, 0, 0, 4, 1, 2, 3, 0],
        [0, 1, 1, 6, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 0, 1, 3],
        [0, 0, 3, 1, 2, 0, 0, 1],
    ],
    dtype=float,
)
IDS = [f"s{i}" for i in range(DENSE.shape[0])]


class SparseTestCase(TestCase):
    def setUp(self):
        self.tree = TreeNode.read(StringIO(TREE_NEWICK))
        self.dense = DENSE.copy()
        self.csr = sp.csr_matrix(self.dense)
        self.csr_array = sp.csr_array(self.dense)
        self.biom = Table(self.dense.T, TAXA, IDS)
        # A tree whose tip set is a strict superset of the table taxa.
        self.tree_extra = TreeNode.read(
            StringIO(
                "(((((U1:0.5,U2:0.5):0.5,U3:1.0):1.0):0.0,(U4:0.75,(U5:0.25,"
                "((U6:0.33,U7:0.62):0.5,U8:0.5):0.5,XTRA:0.5):0.5):1.25):0.0)"
                "root;"
            )
        )

    # -- helpers ---------------------------------------------------------
    def _assert_dm_close(self, sparse_dm, dense_dm):
        self.assertEqual(list(sparse_dm.ids), list(dense_dm.ids))
        np.testing.assert_allclose(
            sparse_dm.data, dense_dm.data, atol=1e-12, rtol=1e-10
        )

    # -- detection / validation -----------------------------------------
    def test_is_csr_counts(self):
        self.assertTrue(is_csr_counts(sp.csr_matrix([[1]])))
        self.assertTrue(is_csr_counts(sp.csr_array([[1]])))
        self.assertFalse(is_csr_counts(np.array([[1]])))
        self.assertFalse(is_csr_counts(sp.csc_matrix([[1]])))

    def test_validate_accepts_noncanonical(self):
        # Unsorted columns, repeated coordinates and an explicit zero in one
        # legal CSR row.
        data = np.array([2.0, 1.0, 0.0, 3.0, 4.0], dtype=float)
        indices = np.array([4, 2, 1, 2, 0])
        indptr = np.array([0, 5])
        raw = sp.csr_matrix((data, indices, indptr), shape=(1, 6))
        snapshot = (raw.data.copy(), raw.indices.copy(), raw.indptr.copy())
        cleaned = validate_sparse_counts(raw)
        # logical values: column 2 holds 1 + 3 = 4, explicit zero dropped
        np.testing.assert_array_equal(
            cleaned.toarray().ravel(), np.array([4, 0, 4, 0, 2, 0])
        )
        # the caller's arrays are untouched
        np.testing.assert_array_equal(raw.data, snapshot[0])
        np.testing.assert_array_equal(raw.indices, snapshot[1])
        np.testing.assert_array_equal(raw.indptr, snapshot[2])

    def test_validate_dtype_rejections(self):
        for bad, exc in (
            (sp.csr_matrix(np.array([[True, False]])), TypeError),
            (sp.csr_matrix(np.array([[1 + 2j]])), TypeError),
        ):
            with self.assertRaises(exc):
                validate_sparse_counts(bad)

    def test_validate_value_rejections(self):
        bad_values = [
            np.array([[np.nan, 1.0]]),
            np.array([[np.inf, 1.0]]),
            np.array([[-np.inf, 1.0]]),
            np.array([[1.5, 2.0]]),
            np.array([[-1.0, 2.0]]),
        ]
        for values in bad_values:
            with self.assertRaises(ValueError):
                validate_sparse_counts(sp.csr_matrix(values))

    def test_negative_not_canceled_by_duplicate(self):
        raw = sp.csr_matrix(
            (
                np.array([5.0, -5.0]),
                np.array([0, 0]),
                np.array([0, 2]),
            ),
            shape=(1, 2),
        )
        with self.assertRaises(ValueError):
            validate_sparse_counts(raw)

    def test_row_total_bound(self):
        too_big = sp.csr_matrix(
            (
                np.array([2**53], dtype=np.int64),
                np.array([0]),
                np.array([0, 1]),
            ),
            shape=(1, 1),
        )
        with self.assertRaises(ValueError):
            validate_sparse_counts(too_big)
        ok = sp.csr_matrix(
            (
                np.array([2**53 - 1], dtype=np.int64),
                np.array([0]),
                np.array([0, 1]),
            ),
            shape=(1, 1),
        )
        validate_sparse_counts(ok)  # does not raise

    def test_overflowing_duplicate_merge(self):
        # Two stored int64 values at the same coordinate whose sum wraps the
        # int64 range must raise ValueError rather than wrap silently.
        int64_max = np.iinfo(np.int64).max
        wrapped = sp.csr_matrix(
            (
                np.array([int64_max, 5], dtype=np.int64),
                np.array([0, 0]),
                np.array([0, 2]),
            ),
            shape=(1, 1),
        )
        with self.assertRaises(ValueError):
            validate_sparse_counts(wrapped)
        # A huge floating row is rejected with the row-total message rather
        # than NumPy's OverflowError from the int64 cast.
        huge_float = sp.csr_matrix(
            (np.array([1e30, 2e30]), np.array([0, 1]), np.array([0, 2])),
            shape=(1, 2),
        )
        with self.assertRaises(ValueError):
            validate_sparse_counts(huge_float)

    def test_validate_float_integral_and_empty(self):
        cleaned = validate_sparse_counts(
            sp.csr_matrix(np.array([[3.0, 0.0, 7.0]]))
        )
        self.assertEqual(cleaned.dtype, np.dtype(np.int64))
        empty = validate_sparse_counts(sp.csr_matrix((3, 5), dtype=np.int64))
        self.assertEqual(empty.shape, (3, 5))
        self.assertEqual(empty.nnz, 0)

    # -- faith_pd: batch + single, all three input shapes ---------------
    def test_faith_pd_batch_matches_dense(self):
        dense_result = alpha_diversity(
            "faith_pd", self.dense, ids=IDS, taxa=TAXA, tree=self.tree
        )
        for table in (self.csr, self.csr_array, self.biom):
            sparse_result = alpha_diversity(
                "faith_pd", table, ids=IDS, taxa=TAXA, tree=self.tree
            )
            np.testing.assert_allclose(
                sparse_result.values, dense_result.values
            )
            self.assertEqual(list(sparse_result.index), IDS)

    def test_faith_pd_default_ids(self):
        result = alpha_diversity(
            "faith_pd", self.csr, taxa=TAXA, tree=self.tree
        )
        self.assertEqual(list(result.index), list(range(self.dense.shape[0])))

    def test_faith_pd_single_sample(self):
        for i in range(self.dense.shape[0]):
            dense_value = faith_pd(self.dense[i], TAXA, self.tree)
            sparse_value = faith_pd(
                sp.csr_matrix(self.dense[i][None, :]), TAXA, self.tree
            )
            self.assertIsInstance(sparse_value, float)
            self.assertAlmostEqual(sparse_value, dense_value)

    def test_faith_pd_extra_tips(self):
        dense_value = faith_pd(self.dense[0], TAXA, self.tree_extra)
        sparse_value = faith_pd(
            sp.csr_matrix(self.dense[0][None, :]), TAXA, self.tree_extra
        )
        self.assertAlmostEqual(sparse_value, dense_value)

    # -- unweighted / weighted UniFrac batch -----------------------------
    def test_beta_unweighted_matches_dense(self):
        dense_dm = beta_diversity(
            "unweighted_unifrac",
            self.dense,
            ids=IDS,
            taxa=TAXA,
            tree=self.tree,
        )
        for engine in (None, "cython", "fast"):
            for table in (self.csr, self.csr_array, self.biom):
                sparse_dm = beta_diversity(
                    "unweighted_unifrac",
                    table,
                    ids=IDS,
                    taxa=TAXA,
                    tree=self.tree,
                    engine=engine,
                )
                self._assert_dm_close(sparse_dm, dense_dm)

    def test_beta_weighted_matches_dense(self):
        for normalized in (False, True):
            dense_dm = beta_diversity(
                "weighted_unifrac",
                self.dense,
                ids=IDS,
                taxa=TAXA,
                tree=self.tree,
                normalized=normalized,
            )
            for engine in (None, "cython", "fast"):
                for table in (self.csr, self.csr_array, self.biom):
                    sparse_dm = beta_diversity(
                        "weighted_unifrac",
                        table,
                        ids=IDS,
                        taxa=TAXA,
                        tree=self.tree,
                        normalized=normalized,
                        engine=engine,
                    )
                    self._assert_dm_close(sparse_dm, dense_dm)

    def test_beta_biom_uses_biom_ids(self):
        dm = beta_diversity(
            "unweighted_unifrac", self.biom, taxa=TAXA, tree=self.tree
        )
        self.assertEqual(list(dm.ids), IDS)
        faith = alpha_diversity(
            "faith_pd", self.biom, taxa=TAXA, tree=self.tree
        )
        self.assertEqual(list(faith.index), IDS)

    def test_biom_axis_interpretation(self):
        # Explicit ids/taxa override the BIOM's own ids.
        dm = beta_diversity(
            "unweighted_unifrac",
            self.biom,
            ids=IDS,
            taxa=TAXA,
            tree=self.tree,
        )
        dense_dm = beta_diversity(
            "unweighted_unifrac",
            self.dense,
            ids=IDS,
            taxa=TAXA,
            tree=self.tree,
        )
        self._assert_dm_close(dm, dense_dm)

    # -- single-sample UniFrac interfaces --------------------------------
    def test_pairwise_unifrac_matches_dense(self):
        rng = np.random.default_rng(0)
        for _ in range(6):
            i, j = rng.integers(0, self.dense.shape[0], size=2)
            u = sp.csr_matrix(self.dense[i][None, :])
            v = sp.csr_matrix(self.dense[j][None, :])
            self.assertAlmostEqual(
                unweighted_unifrac(u, v, TAXA, self.tree),
                unweighted_unifrac(self.dense[i], self.dense[j], TAXA, self.tree),
            )
            for normalized in (False, True):
                got = weighted_unifrac(
                    u, v, TAXA, self.tree, normalized=normalized
                )
                expected = weighted_unifrac(
                    self.dense[i],
                    self.dense[j],
                    TAXA,
                    self.tree,
                    normalized=normalized,
                )
                self.assertIsInstance(got, float)
                self.assertAlmostEqual(got, expected)

    def test_multi_row_single_sample_interfaces_rejected(self):
        with self.assertRaises(ValueError):
            faith_pd(self.csr, TAXA, self.tree)
        with self.assertRaises(ValueError):
            unweighted_unifrac(self.csr, sp.csr_matrix(self.dense[0][None, :]),
                               TAXA, self.tree)
        with self.assertRaises(ValueError):
            weighted_unifrac(self.csr, sp.csr_matrix(self.dense[0][None, :]),
                             TAXA, self.tree)

    def test_mixed_sparse_dense_pair_rejected(self):
        with self.assertRaises(TypeError):
            unweighted_unifrac(
                self.dense[0], sp.csr_matrix(self.dense[1][None, :]),
                TAXA, self.tree,
            )

    # -- invariance properties -------------------------------------------
    def test_row_scaling_invariance(self):
        rng = np.random.default_rng(1)
        scales = rng.integers(1, 6, size=self.dense.shape[0])
        scaled = self.dense * scales[:, None]
        faith_base = alpha_diversity(
            "faith_pd", self.dense, ids=IDS, taxa=TAXA, tree=self.tree
        )
        faith_scaled = alpha_diversity(
            "faith_pd", sp.csr_matrix(scaled), ids=IDS, taxa=TAXA, tree=self.tree
        )
        np.testing.assert_allclose(faith_scaled.values, faith_base.values)

        uu_base = beta_diversity(
            "unweighted_unifrac", self.dense, ids=IDS, taxa=TAXA, tree=self.tree
        )
        uu_scaled = beta_diversity(
            "unweighted_unifrac", sp.csr_matrix(scaled), ids=IDS,
            taxa=TAXA, tree=self.tree,
        )
        np.testing.assert_allclose(uu_scaled.data, uu_base.data)
        for normalized in (False, True):
            w_base = beta_diversity(
                "weighted_unifrac", self.dense, ids=IDS, taxa=TAXA,
                tree=self.tree, normalized=normalized,
            )
            w_scaled = beta_diversity(
                "weighted_unifrac", sp.csr_matrix(scaled), ids=IDS,
                taxa=TAXA, tree=self.tree, normalized=normalized,
            )
            np.testing.assert_allclose(w_scaled.data, w_base.data)

    def test_permutation_realignment(self):
        rng = np.random.default_rng(2)
        row_perm = rng.permutation(self.dense.shape[0])
        col_perm = rng.permutation(self.dense.shape[1])
        permuted = self.dense[np.ix_(row_perm, col_perm)]
        perm_taxa = [TAXA[c] for c in col_perm]
        perm_ids = [IDS[i] for i in row_perm]
        faith = alpha_diversity(
            "faith_pd", sp.csr_matrix(permuted), ids=perm_ids,
            taxa=perm_taxa, tree=self.tree,
        )
        faith_base = alpha_diversity(
            "faith_pd", self.dense, ids=IDS, taxa=TAXA, tree=self.tree
        )
        for sample in IDS:
            self.assertAlmostEqual(faith[sample], faith_base[sample])

    def test_phydiv_variants(self):
        for rooted in (True, False):
            for weight in (False, 0.5, True):
                dense_r = alpha_diversity(
                    "phydiv", self.dense, ids=IDS, taxa=TAXA, tree=self.tree,
                    rooted=rooted, weight=weight,
                )
                sparse_r = alpha_diversity(
                    "phydiv", self.csr, ids=IDS, taxa=TAXA, tree=self.tree,
                    rooted=rooted, weight=weight,
                )
                np.testing.assert_allclose(
                    sparse_r.values, dense_r.values, atol=1e-12, rtol=1e-10
                )

    def test_single_sample_phydiv_weighted_matches_dense(self):
        # Regression test: the single-sample sparse path must use tip-only
        # totals (not the sum over all ancestor nodes) when building
        # proportions, for every theta including bool True and float 1.0.
        for i in range(self.dense.shape[0]):
            for rooted in (True, False):
                for weight in (False, 0.25, 0.5, 0.75, True, 1.0):
                    dense_v = phydiv(
                        self.dense[i], TAXA, self.tree,
                        rooted=rooted, weight=weight,
                    )
                    sparse_v = phydiv(
                        sp.csr_matrix(self.dense[i][None, :]),
                        TAXA, self.tree, rooted=rooted, weight=weight,
                    )
                    self.assertIsInstance(sparse_v, float)
                    self.assertAlmostEqual(sparse_v, dense_v)

    # -- error categories ------------------------------------------------
    def test_label_errors(self):
        with self.assertRaises(MissingNodeError):
            faith_pd(
                sp.csr_matrix(self.dense[0][None, :]),
                [f"Z{i}" for i in range(8)], self.tree,
            )
        with self.assertRaises(ValueError):
            faith_pd(
                sp.csr_matrix(self.dense[0][None, :]),
                ["U1"] * 8, self.tree,
            )
        dup_tree = TreeNode.read(
            StringIO("((U1:1,U2:1):1,(U3:1,U1:1):1)root;")
        )
        with self.assertRaises(DuplicateNodeError):
            faith_pd(
                sp.csr_matrix(self.dense[0][None, :4]),
                TAXA[:4], dup_tree,
            )

    def test_sample_id_count_mismatch(self):
        with self.assertRaises(ValueError):
            alpha_diversity(
                "faith_pd", self.csr, ids=["only"], taxa=TAXA, tree=self.tree
            )
        with self.assertRaises(ValueError):
            beta_diversity(
                "unweighted_unifrac", self.csr, ids=["x", "y"],
                taxa=TAXA, tree=self.tree,
            )

    def test_validate_false_still_checks_counts(self):
        negative = sp.csr_matrix(np.array([[1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]]))
        fractional = sp.csr_matrix(np.array([[1.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]]))
        for bad in (negative, fractional):
            with self.assertRaises(ValueError):
                faith_pd(bad, TAXA, self.tree, validate=False)
            with self.assertRaises(ValueError):
                unweighted_unifrac(
                    bad, sp.csr_matrix(self.dense[1][None, :]),
                    TAXA, self.tree, validate=False,
                )

    def test_unsupported_engine_and_pairwise_func(self):
        with self.assertRaises(ValueError):
            beta_diversity(
                "unweighted_unifrac", self.csr, ids=IDS, taxa=TAXA,
                tree=self.tree, engine="numba",
            )
        with self.assertRaises(ValueError):
            beta_diversity(
                "weighted_unifrac", self.csr, ids=IDS, taxa=TAXA,
                tree=self.tree, pairwise_func=lambda *a, **k: None,
            )

    def test_sparse_rejected_for_non_phylogenetic_metrics(self):
        with self.assertRaises(TypeError):
            alpha_diversity("sobs", self.csr, ids=IDS)

    def test_non_csr_sparse_formats_rejected(self):
        for ctor in (sp.csc_matrix, sp.coo_matrix):
            with self.assertRaises(TypeError):
                alpha_diversity(
                    "faith_pd",
                    ctor(self.dense),
                    ids=IDS,
                    taxa=TAXA,
                    tree=self.tree,
                )

    def test_biom_float_integral_values(self):
        # BIOM stores float64 even for integral counts.
        biom = Table(self.dense.T.astype(float), TAXA, IDS)
        faith = alpha_diversity(
            "faith_pd", biom, ids=IDS, taxa=TAXA, tree=self.tree
        )
        faith_dense = alpha_diversity(
            "faith_pd", self.dense, ids=IDS, taxa=TAXA, tree=self.tree
        )
        np.testing.assert_allclose(faith.values, faith_dense.values)

    def test_sparse_runs_under_global_numba_default(self):
        from skbio import get_config, set_config

        previous = get_config("compute_engine")
        try:
            set_config("compute_engine", "numba")
            sparse_dm = beta_diversity(
                "unweighted_unifrac", self.csr, ids=IDS,
                taxa=TAXA, tree=self.tree,
            )
        finally:
            set_config("compute_engine", previous)
        dense_dm = beta_diversity(
            "unweighted_unifrac", self.dense, ids=IDS,
            taxa=TAXA, tree=self.tree, engine="cython",
        )
        self._assert_dm_close(sparse_dm, dense_dm)

    # -- immutability -----------------------------------------------------
    def test_inputs_not_mutated(self):
        matrix = sp.csr_matrix(
            np.array([[1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                      [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0]])
        )
        snapshot = (
            matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy()
        )
        taxa_snapshot = list(TAXA)
        tree_snapshot = [
            (n.name, n.length) for n in self.tree.postorder()
        ]
        alpha_diversity(
            "faith_pd", matrix, ids=["x", "y"], taxa=TAXA, tree=self.tree
        )
        beta_diversity(
            "unweighted_unifrac", matrix, ids=["x", "y"],
            taxa=TAXA, tree=self.tree,
        )
        for normalized in (False, True):
            beta_diversity(
                "weighted_unifrac", matrix, ids=["x", "y"], taxa=TAXA,
                tree=self.tree, normalized=normalized,
            )
        np.testing.assert_array_equal(matrix.data, snapshot[0])
        np.testing.assert_array_equal(matrix.indices, snapshot[1])
        np.testing.assert_array_equal(matrix.indptr, snapshot[2])
        self.assertEqual(TAXA, taxa_snapshot)
        self.assertEqual(
            [(n.name, n.length) for n in self.tree.postorder()],
            tree_snapshot,
        )

    def test_biom_not_mutated(self):
        snapshot = self.biom.matrix_data.toarray().copy()
        beta_diversity(
            "unweighted_unifrac", self.biom, taxa=TAXA, tree=self.tree
        )
        np.testing.assert_array_equal(
            self.biom.matrix_data.toarray(), snapshot
        )

    # -- zero-length branches and empty tables ---------------------------
    def test_zero_length_branches(self):
        tree = TreeNode.read(
            StringIO("(((t1:0.0,t2:1.0):0.0,(t3:0.5,t4:0.0):2.0):1.0,"
                     "(t5:1.0,t6:1.0):1.0)root;")
        )
        taxa = ["t1", "t2", "t3", "t4", "t5", "t6"]
        dense = np.array([[1, 0, 0, 2, 0, 0], [0, 1, 1, 0, 1, 0]], float)
        for normalized in (False, True):
            dense_dm = beta_diversity(
                "weighted_unifrac", dense, ids=["a", "b"], taxa=taxa,
                tree=tree, normalized=normalized,
            )
            sparse_dm = beta_diversity(
                "weighted_unifrac", sp.csr_matrix(dense), ids=["a", "b"],
                taxa=taxa, tree=tree, normalized=normalized,
            )
            np.testing.assert_allclose(sparse_dm.data, dense_dm.data)

    def test_all_zero_rows(self):
        empty = sp.csr_matrix((4, 8), dtype=np.int64)
        faith = alpha_diversity(
            "faith_pd", empty, ids=list("abcd"), taxa=TAXA, tree=self.tree
        )
        np.testing.assert_array_equal(faith.values, np.zeros(4))
        dm = beta_diversity(
            "unweighted_unifrac", empty, ids=list("abcd"),
            taxa=TAXA, tree=self.tree,
        )
        np.testing.assert_array_equal(dm.data, np.zeros((4, 4)))


if __name__ == "__main__":
    main()
