# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import io
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from skbio import DistanceMatrix
from skbio.io import LSMatFormatError
from skbio.io.format.lsmat import (
    _lsmat_to_pairwise_matrix, _lsmat_to_distance_matrix, _lsmat_to_symmetric_matrix,
    _pairwise_matrix_to_lsmat, _distance_matrix_to_lsmat, _symmetric_matrix_to_lsmat,
    _lsmat_sniffer)
from skbio.stats.distance import PairwiseMatrix, SymmetricMatrix, DistanceMatrixError


class LSMatTestData(TestCase):
    def setUp(self):
        self.lsmat_1x1_fh = io.StringIO(LSMat_1x1)
        self.lsmat_2x2_fh = io.StringIO(LSMat_2x2)
        self.lsmat_2x2_asym_fh = io.StringIO(LSMat_2x2_ASYM)
        self.lsmat_3x3_fh = io.StringIO(LSMat_3x3)
        self.lsmat_3x3_whitespace_fh = io.StringIO(LSMat_3x3_WHITESPACE)
        self.lsmat_3x3_csv_fh = io.StringIO(LSMat_3x3_CSV)
        self.lsmat_3x3_fw_fh = io.StringIO(LSMat_3x3_FW)

        self.valid_fhs = [
            self.lsmat_1x1_fh,
            self.lsmat_2x2_fh,
            self.lsmat_2x2_asym_fh,
            self.lsmat_3x3_fh,
            self.lsmat_3x3_whitespace_fh,
        ]

        self.empty_fh = io.StringIO()
        self.invalid_1_fh = io.StringIO(INVALID_1)
        self.invalid_2_fh = io.StringIO(INVALID_2)
        self.invalid_3_fh = io.StringIO(INVALID_3)
        self.invalid_4_fh = io.StringIO(INVALID_4)
        self.invalid_5_fh = io.StringIO(INVALID_5)
        self.invalid_6_fh = io.StringIO(INVALID_6)

        self.invalid_fhs = [
            (self.empty_fh, r'empty'),
            (self.invalid_1_fh, r'1 value\(s\).*2.*\(2\)'),
            (self.invalid_2_fh, r"'b'.*'a'"),
            (self.invalid_3_fh, r'extra row\(s\)'),
            (self.invalid_4_fh, r'2 row\(s\).*found 1'),
            (self.invalid_5_fh, r'2 row\(s\).*found 0'),
            (self.invalid_6_fh, r"delimiter '\\t'")
        ]


class PairwiseAndDistanceMatrixReaderWriterTests(LSMatTestData):
    def setUp(self):
        super(PairwiseAndDistanceMatrixReaderWriterTests, self).setUp()

        self.lsmat_1x1_data = [[0.0]]
        self.lsmat_2x2_data = [[0.0, 0.123], [0.123, 0.0]]
        self.lsmat_2x2_asym_data = [[0.0, 1.0], [-2.0, 0.0]]
        self.lsmat_3x3_data = [[0.0, 0.01, 4.2], [0.01, 0.0, 12.0],
                               [4.2, 12.0, 0.0]]

        # We repeat the 3x3 example because there are two file format
        # representations of it, one that is messy and one that is not. Both
        # should be read into an equivalent object and written to an equivalent
        # format though, which is why we duplicate the 3x3 objects and strings.
        self.dissim_objs = [
            PairwiseMatrix(self.lsmat_1x1_data, ['a']),
            PairwiseMatrix(self.lsmat_2x2_data, ['a', 'b']),
            PairwiseMatrix(self.lsmat_2x2_asym_data, ['a', 'b']),
            PairwiseMatrix(self.lsmat_3x3_data, ['a', 'b', 'c']),
            PairwiseMatrix(self.lsmat_3x3_data, ['a', 'b', 'c'])
        ]

        self.dissim_strs = [LSMat_1x1, LSMat_2x2, LSMat_2x2_ASYM, LSMat_3x3,
                            LSMat_3x3]

        self.dissim_fhs = [self.lsmat_1x1_fh, self.lsmat_2x2_fh,
                           self.lsmat_2x2_asym_fh, self.lsmat_3x3_fh,
                           self.lsmat_3x3_whitespace_fh]

        self.dist_objs = [
            DistanceMatrix(self.lsmat_1x1_data, ['a']),
            DistanceMatrix(self.lsmat_2x2_data, ['a', 'b']),
            DistanceMatrix(self.lsmat_3x3_data, ['a', 'b', 'c']),
            DistanceMatrix(self.lsmat_3x3_data, ['a', 'b', 'c'])
        ]

        self.sym_objs = [
            SymmetricMatrix(self.lsmat_1x1_data, ['a']),
            SymmetricMatrix(self.lsmat_2x2_data, ['a', 'b']),
            SymmetricMatrix(self.lsmat_3x3_data, ['a', 'b', 'c']),
            SymmetricMatrix(self.lsmat_3x3_data, ['a', 'b', 'c'])
        ]

        self.dist_strs = [LSMat_1x1, LSMat_2x2, LSMat_3x3, LSMat_3x3]

        self.dist_fhs = [self.lsmat_1x1_fh, self.lsmat_2x2_fh,
                         self.lsmat_3x3_fh, self.lsmat_3x3_whitespace_fh]

    def test_read_valid_files(self):
        for fn, cls, objs, fhs in ((_lsmat_to_pairwise_matrix,
                                    PairwiseMatrix, self.dissim_objs,
                                    self.dissim_fhs),
                                   (_lsmat_to_distance_matrix, DistanceMatrix,
                                    self.dist_objs, self.dist_fhs),
                                    (_lsmat_to_symmetric_matrix, SymmetricMatrix, self.sym_objs, self.dist_fhs)):
            for fh, obj in zip(fhs, objs):
                fh.seek(0)
                obs = fn(fh)
                self.assertEqual(obs, obj)
                self.assertIsInstance(obs, cls)

        # Above files are TSV (default delimiter). Test that CSV works too.
        for fn, cls in ((_lsmat_to_pairwise_matrix, PairwiseMatrix),
                        (_lsmat_to_distance_matrix, DistanceMatrix),
                        (_lsmat_to_symmetric_matrix, SymmetricMatrix)):
            exp = cls(self.lsmat_3x3_data, ['a', 'b', 'c'])
            self.lsmat_3x3_csv_fh.seek(0)
            obs = fn(self.lsmat_3x3_csv_fh, delimiter=',')
            self.assertEqual(obs, exp)
            self.assertIsInstance(obs, cls)

        # Test that fixed-width works too.
        for fn, cls in ((_lsmat_to_pairwise_matrix, PairwiseMatrix),
                        (_lsmat_to_distance_matrix, DistanceMatrix),
                        (_lsmat_to_symmetric_matrix, SymmetricMatrix)):
            exp = cls(self.lsmat_3x3_data, ['a', 'b', 'c'])
            self.lsmat_3x3_fw_fh.seek(0)
            obs = fn(self.lsmat_3x3_fw_fh, delimiter=None)
            self.assertEqual(obs, exp)
            self.assertIsInstance(obs, cls)

    def test_read_dtype(self):
        fn = _lsmat_to_distance_matrix
        fh = self.lsmat_3x3_fw_fh
        data = self.lsmat_3x3_data

        # float64 (default)
        exp = DistanceMatrix(data, ['a', 'b', 'c'])
        fh.seek(0)
        obs = fn(fh, delimiter=None)
        self.assertEqual(obs.dtype, np.float64)
        self.assertEqual(obs, exp)

        for dtype in (None, float, np.float64, "float64"):
            fh.seek(0)
            obs = fn(fh, delimiter=None, dtype=dtype)
            self.assertEqual(obs.dtype, np.float64)
            self.assertEqual(obs, exp)

        # float32
        exp = DistanceMatrix(np.asarray(data, dtype="float32"), ['a', 'b', 'c'])
        for dtype in (np.float32, "float32"):
            fh.seek(0)
            obs = fn(fh, delimiter=None, dtype=dtype)
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(obs, exp)

        # invalid data type
        for dtype in ("xyz", int, "float16", "uint8", np.float16):
            fh.seek(0)
            self.assertRaises(TypeError, fn, fh, delimiter=None, dtype=dtype)

    def test_read_invalid_files(self):
        for fn in _lsmat_to_pairwise_matrix, _lsmat_to_distance_matrix, _lsmat_to_symmetric_matrix:
            for invalid_fh, error_msg_regexp in self.invalid_fhs:
                with self.assertRaisesRegex(LSMatFormatError,
                                            error_msg_regexp):
                    invalid_fh.seek(0)
                    fn(invalid_fh)

        # Asymmetric data only raises an error for DistanceMatrix.
        with self.assertRaises(DistanceMatrixError):
            _lsmat_to_distance_matrix(self.lsmat_2x2_asym_fh)

    def test_write(self):
        for fn, objs, strs in ((_pairwise_matrix_to_lsmat,
                                self.dissim_objs, self.dissim_strs),
                               (_distance_matrix_to_lsmat, self.dist_objs,
                                self.dist_strs),
                                (_symmetric_matrix_to_lsmat, self.dist_objs,
                                 self.dist_strs)):
            for obj, str_ in zip(objs, strs):
                fh = io.StringIO()
                fn(obj, fh)
                obs = fh.getvalue()
                fh.close()
                self.assertEqual(obs, str_)

        # Test writing CSV (TSV is written above).
        for fn, cls in ((_pairwise_matrix_to_lsmat, PairwiseMatrix),
                        (_distance_matrix_to_lsmat, DistanceMatrix),
                        (_symmetric_matrix_to_lsmat, SymmetricMatrix)):
            obj = cls(self.lsmat_3x3_data, ['a', 'b', 'c'])
            fh = io.StringIO()
            fn(obj, fh, delimiter=',')
            obs = fh.getvalue()
            fh.close()
            self.assertEqual(obs, LSMat_3x3_CSV)

    def test_roundtrip_read_write(self):
        for reader_fn, writer_fn, fhs in ((_lsmat_to_pairwise_matrix,
                                           _pairwise_matrix_to_lsmat,
                                           self.dissim_fhs),
                                          (_lsmat_to_distance_matrix,
                                           _distance_matrix_to_lsmat,
                                           self.dist_fhs),
                                           (_lsmat_to_symmetric_matrix,
                                            _symmetric_matrix_to_lsmat,
                                            self.dist_fhs)):
            for fh in fhs:
                # Read.
                fh.seek(0)
                lsmat1 = reader_fn(fh)

                # Write.
                out_fh = io.StringIO()
                writer_fn(lsmat1, out_fh)
                out_fh.seek(0)

                # Read.
                lsmat2 = reader_fn(out_fh)
                out_fh.close()

                self.assertEqual(lsmat1, lsmat2)


class CondensedMatrixWriterTests(TestCase):
    """Writing condensed (1-D) storage must match redundant (2-D) storage."""

    def setUp(self):
        # The reported failure: condensed DistanceMatrix raised TypeError on
        # write. Upper triangle 1..6 across samples a, b, c, d.
        self.ids = ["a", "b", "c", "d"]
        self.condensed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.square = [[0.0, 1.0, 2.0, 3.0],
                       [1.0, 0.0, 4.0, 5.0],
                       [2.0, 4.0, 0.0, 6.0],
                       [3.0, 5.0, 6.0, 0.0]]
        self.exp_text = (
            "\ta\tb\tc\td\n"
            "a\t0.0\t1.0\t2.0\t3.0\n"
            "b\t1.0\t0.0\t4.0\t5.0\n"
            "c\t2.0\t4.0\t0.0\t6.0\n"
            "d\t3.0\t5.0\t6.0\t0.0\n"
        )

    def write_str(self, obj, **kwargs):
        fh = io.StringIO()
        obj.write(fh, format="lsmat", **kwargs)
        value = fh.getvalue()
        fh.close()
        return value

    def test_condensed_distance_matrix(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        self.assertEqual(self.write_str(dm), self.exp_text)

    def test_condensed_distance_matrix_float32(self):
        dm = DistanceMatrix(np.asarray(self.condensed, dtype="float32"),
                            ids=self.ids, condensed=True)
        self.assertEqual(dm.dtype, np.float32)
        self.assertEqual(self.write_str(dm), self.exp_text)

    def test_condensed_matches_redundant_text(self):
        dm_cond = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        dm_red = DistanceMatrix(self.square, ids=self.ids)
        self.assertEqual(self.write_str(dm_cond), self.write_str(dm_red))

    def test_condensed_symmetric_scalar_diagonal(self):
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=11.0)
        exp = self.exp_text.replace("0.0", "11.0")
        self.assertEqual(self.write_str(sm), exp)

    def test_condensed_symmetric_array_diagonal(self):
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=[11.0, 12.0, 13.0, 14.0])
        exp = (
            "\ta\tb\tc\td\n"
            "a\t11.0\t1.0\t2.0\t3.0\n"
            "b\t1.0\t12.0\t4.0\t5.0\n"
            "c\t2.0\t4.0\t13.0\t6.0\n"
            "d\t3.0\t5.0\t6.0\t14.0\n"
        )
        self.assertEqual(self.write_str(sm), exp)
        # same logical matrix stored redundantly writes identical text
        sm_red = SymmetricMatrix(
            [[11.0, 1.0, 2.0, 3.0],
             [1.0, 12.0, 4.0, 5.0],
             [2.0, 4.0, 13.0, 6.0],
             [3.0, 5.0, 6.0, 14.0]], ids=self.ids)
        self.assertEqual(self.write_str(sm), self.write_str(sm_red))

    def test_single_sample(self):
        cases = [
            (DistanceMatrix([[0.0]], ids=["a"]), "\ta\na\t0.0\n"),
            (DistanceMatrix([], ids=["a"], condensed=True), "\ta\na\t0.0\n"),
            (SymmetricMatrix([], ids=["a"], condensed=True, diagonal=5.0),
             "\ta\na\t5.0\n"),
        ]
        for obj, exp in cases:
            self.assertEqual(self.write_str(obj), exp)

    def test_all_zero(self):
        dm = DistanceMatrix([0.0, 0.0, 0.0], ids=["a", "b", "c"],
                            condensed=True)
        exp = "\ta\tb\tc\na\t0.0\t0.0\t0.0\nb\t0.0\t0.0\t0.0\nc\t0.0\t0.0\t0.0\n"
        self.assertEqual(self.write_str(dm), exp)

    def test_pairwise_asymmetric_unchanged(self):
        pm = PairwiseMatrix([[0.0, 1.0], [-2.0, 0.0]], ids=["a", "b"])
        exp = "\ta\tb\na\t0.0\t1.0\nb\t-2.0\t0.0\n"
        self.assertEqual(self.write_str(pm), exp)

    def test_filter_reorder_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        sub = dm.filter(["c", "a"])
        self.assertTrue(sub._flags["CONDENSED"])
        exp = "\tc\ta\nc\t0.0\t2.0\na\t2.0\t0.0\n"
        self.assertEqual(self.write_str(sub), exp)

    def test_permute_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        perm = dm.permute(seed=42)
        self.assertTrue(perm._flags["CONDENSED"])
        # IDs are kept in place; only values are permuted
        self.assertEqual(perm.ids, dm.ids)
        text = self.write_str(perm)
        obs = DistanceMatrix.read(io.StringIO(text), format="lsmat")
        npt.assert_array_equal(obs.redundant_form(), perm.redundant_form())

    def test_copy_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        self.assertEqual(self.write_str(dm.copy()), self.exp_text)

    def test_small_segment_cap_same_output(self):
        # Force the writer through many small segments; output must be
        # identical to the default-cap write.
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        with patch("skbio.io.format.lsmat._DELIMITED_SEG_CAP", 1):
            obs = self.write_str(dm)
        self.assertEqual(obs, self.exp_text)

    def test_roundtrip_condensed(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        text = self.write_str(dm)
        obs = DistanceMatrix.read(io.StringIO(text), format="lsmat")
        npt.assert_array_equal(obs.redundant_form(), dm.redundant_form())
        self.assertEqual(obs.ids, dm.ids)

    def test_source_not_mutated(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        data_before = dm.data
        snapshot = dm.data.copy()
        self.write_str(dm)
        self.assertIs(dm.data, data_before)
        self.assertTrue(dm._flags["CONDENSED"])
        self.assertEqual(dm.ids, tuple(self.ids))
        npt.assert_array_equal(dm.data, snapshot)

    def test_source_not_mutated_on_error(self):
        class FailingFH(io.StringIO):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def write(self, s):
                self.calls += 1
                if self.calls > 2:
                    raise OSError("simulated write failure")
                return super().write(s)

        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=[11.0, 12.0, 13.0, 14.0])
        data_before = sm.data
        snapshot = sm.data.copy()
        with self.assertRaises(OSError):
            sm.write(FailingFH(), format="lsmat")
        self.assertIs(sm.data, data_before)
        self.assertTrue(sm._flags["CONDENSED"])
        self.assertEqual(sm.ids, tuple(self.ids))
        npt.assert_array_equal(sm.data, snapshot)
        npt.assert_array_equal(np.asarray(sm.diagonal),
                               [11.0, 12.0, 13.0, 14.0])

    def test_large_condensed_allocation_record(self):
        # A larger condensed input is exported with bounded auxiliary
        # memory: one scratch bundle plus transient per-segment string
        # arrays, all far below 8 MiB and not growing with n^2.
        n = 1200
        rng = np.random.default_rng(3)
        cond = rng.random(n * (n - 1) // 2)
        dm = DistanceMatrix(cond, condensed=True)

        scratch_records = []
        str_records = []
        import skbio.stats.distance._base as base_mod
        import skbio.io.format._base as fmt_base
        orig_scratch = base_mod._seg_scratch
        orig_str = fmt_base._str_array

        def spy_scratch(cap, dtype):
            bufs = orig_scratch(cap, dtype)
            scratch_records.append(sum(b.nbytes for b in bufs))
            return bufs

        def spy_str(segment):
            arr = orig_str(segment)
            str_records.append(arr.nbytes)
            return arr

        fh = io.StringIO()
        try:
            base_mod._seg_scratch = spy_scratch
            fmt_base._str_array = spy_str
            dm.write(fh, format="lsmat")
        finally:
            base_mod._seg_scratch = orig_scratch
            fmt_base._str_array = orig_str

        # one scratch bundle, allocated once for the whole write
        self.assertEqual(len(scratch_records), 1)
        # one transient string array per row (n < segment cap)
        self.assertEqual(len(str_records), n)
        peak = scratch_records[0] + max(str_records)
        self.assertLessEqual(peak, 8 << 20)
        # auxiliary space does not follow the O(n^2) input/output size
        self.assertLess(peak, cond.nbytes // 10)

        # and the output is correct
        fh.seek(0)
        obs = DistanceMatrix.read(fh, format="lsmat")
        fh.close()
        npt.assert_array_equal(obs.redundant_form(), dm.redundant_form())


class SnifferTests(LSMatTestData):
    def setUp(self):
        super(SnifferTests, self).setUp()

    def test_match_tsv(self):
        # Sniffer should match all valid files, and will match some invalid
        # ones too because it doesn't exhaustively check the entire file.
        fhs = self.valid_fhs + [self.invalid_1_fh, self.invalid_3_fh,
                                self.invalid_4_fh]
        for fh in fhs:
            self.assertEqual(_lsmat_sniffer(fh), (True, {'delimiter': '\t'}))

    def test_match_csv(self):
        self.assertEqual(_lsmat_sniffer(self.lsmat_3x3_csv_fh),
                         (True, {'delimiter': ','}))

    def test_match_fw(self):
        self.assertEqual(_lsmat_sniffer(self.lsmat_3x3_fw_fh),
                         (True, {'delimiter': None}))

    def test_no_match(self):
        for fh in (self.empty_fh, self.invalid_2_fh, self.invalid_5_fh,
                   self.invalid_6_fh):
            self.assertEqual(_lsmat_sniffer(fh), (False, {}))


LSMat_1x1 = (
    '\ta\n'
    'a\t0.0\n')

LSMat_2x2 = (
    '\ta\tb\n'
    'a\t0.0\t0.123\n'
    'b\t0.123\t0.0\n')

LSMat_2x2_ASYM = (
    '\ta\tb\n'
    'a\t0.0\t1.0\n'
    'b\t-2.0\t0.0\n')

LSMat_3x3 = (
    '\ta\tb\tc\n'
    'a\t0.0\t0.01\t4.2\n'
    'b\t0.01\t0.0\t12.0\n'
    'c\t4.2\t12.0\t0.0\n')

# Extra whitespace-only lines throughout. Also has comments before the header.
LSMat_3x3_WHITESPACE = '\n'.join([
    '# foo',
    '      \t \t ',
    ' #bar',
    '',
    '',
    '\ta\t b \tc',
    'a  \t0.0\t0.01\t4.2',
    '     \t',
    'b\t0.01\t0.0\t12.0',
    '',
    '\t     \t',
    '',
    'c\t4.2\t12.0\t0.0',
    '',
    '   \t ',
    '\t\t\t',
    ' '])

# Same matrix as above, but delimited by commas instead of tabs.
LSMat_3x3_CSV = (
    ',a,b,c\n'
    'a,0.0,0.01,4.2\n'
    'b,0.01,0.0,12.0\n'
    'c,4.2,12.0,0.0\n')

# Same matrix as above, but delimited by whitespaces instead of tabs.
LSMat_3x3_FW = (
    '   a     b     c   \n'
    'a  0.0   0.01  4.2 \n'
    'b  0.01  0.0  12.0 \n'
    'c  4.2  12.0   0.0 \n')

# missing data
INVALID_1 = '\ta\tb\na\t0\t1\nb\t1'

# mismatched IDs
INVALID_2 = '\ta\tb\nb\t0\t1\na\t1\t0'

# extra data lines
INVALID_3 = '\ta\tb\na\t0\t1\nb\t1\t0\n  \nfoo\n\n\n'

# missing data lines
INVALID_4 = '\ta\tb\na\t0\t1\n  \n'

# no data lines
INVALID_5 = '\ta\tb\n'

# missing leading delimiter in header
INVALID_6 = "a\tb\na\t0.0\t0.123\nb\t0.123\t0.0\n"


if __name__ == '__main__':
    main()
