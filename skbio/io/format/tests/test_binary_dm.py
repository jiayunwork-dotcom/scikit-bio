# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
import tempfile
import os
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import h5py

from skbio.stats.distance import PairwiseMatrix, SymmetricMatrix, DistanceMatrix
from skbio.io import read as io_read
from skbio.io import BinaryDMFormatError
from skbio.io.format.binary_dm import (
    _h5py_mat_to_skbio_mat, _skbio_mat_to_h5py_mat, _get_header, _parse_ids,
    _verify_dimensions, _bytes_decoder, _passthrough_decoder, _set_header,
    _binary_dm_sniffer, _BLOCK_BYTES
)


class BinaryMatrixTests(unittest.TestCase):
    def setUp(self):
        _vlen_dtype = h5py.special_dtype(vlen=str)
        self.mat = np.array([[0, 0.1, 0.2],
                             [0.1, 0, 0.3],
                             [0.2, 0.3, 0]])
        self.ids = ['a', 'b', 'c']

        self.tempdir = tempfile.TemporaryDirectory()

        self.basic_fname = os.path.join(self.tempdir.name, 'basic')
        self.basic = h5py.File(self.basic_fname, 'a')
        ids = self.basic.create_dataset('order', shape=(3, ), dtype=_vlen_dtype)
        ids[:] = self.ids
        self.basic.create_dataset('matrix', data=self.mat)
        _set_header(self.basic)
        self.basic.close()

        self.badids_fname = os.path.join(self.tempdir.name, 'badids')
        self.badids = h5py.File(self.badids_fname, 'a')
        ids = self.badids.create_dataset('order', shape=(2, ), dtype=_vlen_dtype)
        ids[:] = ['a', 'b']
        self.badids.create_dataset('matrix', data=self.mat)
        _set_header(self.badids)
        self.badids.close()

        self.noheader_fname = os.path.join(self.tempdir.name, 'noheader')
        self.noheader = h5py.File(self.noheader_fname, 'a')
        ids = self.noheader.create_dataset('order', shape=(3, ), dtype=_vlen_dtype)
        ids[:] = self.ids
        self.noheader.create_dataset('matrix', data=self.mat)
        self.noheader.close()

        self.rw_fname = os.path.join(self.tempdir.name, 'rw')

    def tearDown(self):
        self.tempdir.cleanup()

    def test_binary_dm_sniffer(self):
        with open(self.basic_fname, 'rb') as fh:
            self.assertEqual((True, {}), _binary_dm_sniffer(fh))
        with open(self.badids_fname, 'rb') as fh:
            self.assertEqual((False, {}), _binary_dm_sniffer(fh))
        with open(self.noheader_fname, 'rb') as fh:
            self.assertEqual((False, {}), _binary_dm_sniffer(fh))

    def test_h5py_mat_to_skbio_mat(self):
        exp = DistanceMatrix(self.mat, self.ids)
        obs = _h5py_mat_to_skbio_mat(DistanceMatrix, h5py.File(self.basic_fname, 'r'))
        self.assertEqual(obs, exp)

    def test_io_h5py_mat_to_skbio_mat(self):
        # test DistanceMatrix
        exp = DistanceMatrix(self.mat, self.ids)
        obs = io_read(self.basic_fname,into=DistanceMatrix)
        self.assertEqual(obs, exp)

    def test_io_h5py_mat_to_skbio_mat_pairwise(self):
        # test PairwiseMatrix
        exp = PairwiseMatrix(self.mat, self.ids)
        obs = io_read(self.basic_fname,into=PairwiseMatrix)
        self.assertEqual(obs, exp)

    def test_io_h5py_mat_to_skbio_mat_symmetric(self):
        # test SymmetricMatrix
        exp = SymmetricMatrix(self.mat, self.ids)
        obs = io_read(self.basic_fname,into=SymmetricMatrix)
        self.assertEqual(obs, exp)

    def test_skbio_mat_to_h5py_mat(self):
        fh1 = h5py.File('f1', 'a', driver='core', backing_store=False)

        mat = DistanceMatrix(self.mat, self.ids)
        _skbio_mat_to_h5py_mat(mat, fh1)
        npt.assert_equal(np.asarray(fh1['order'][:], dtype=str), mat.ids)
        npt.assert_equal(fh1['matrix'], mat.data)

    def test_io_mat_to_h5py_mat(self):
        mat = DistanceMatrix(self.mat, self.ids)
        mat.write(self.rw_fname,format='binary_dm')
        exp = mat
        obs = io_read(self.rw_fname,into=DistanceMatrix)
        self.assertEqual(obs, exp)

    def test_io_mat_to_h5py_mat_symmetric(self):
        # SymmetricMatrix
        mat = SymmetricMatrix(self.mat, self.ids)
        mat.write(self.rw_fname,format='binary_dm')
        exp = mat
        obs = io_read(self.rw_fname,into=SymmetricMatrix)
        self.assertEqual(obs, exp)

    def test_io_mat_to_h5py_mat_pairwise(self):
        # PairwiseMatrix
        mat = PairwiseMatrix(self.mat, self.ids)
        mat.write(self.rw_fname,format='binary_dm')
        exp = mat
        obs = io_read(self.rw_fname,into=PairwiseMatrix)
        self.assertEqual(obs, exp)


    def test_get_header(self):
        self.assertEqual(_get_header(h5py.File(self.basic_fname, 'r')),
                         {'format': b'BDSM', 'version': b'2020.12'})
        self.assertEqual(_get_header(h5py.File(self.noheader_fname, 'r')), None)

    def test_parse_ids(self):
        tests = [(['a', 'b', 'c'], ['a', 'b', 'c']),
                 ([b'a', b'b', b'\xc3\xa9\xc3\xb8asd'],
                  ['a', 'b', 'éøasd'])]

        for test, exp in tests:
            self.assertEqual(_parse_ids(test), exp)

    def test_verify_dimensions(self):
        self.assertTrue(_verify_dimensions(h5py.File(self.basic_fname, 'r')))
        self.assertFalse(_verify_dimensions(h5py.File(self.badids_fname, 'r')))

    def test_bytes_decoder(self):
        test = [b'', b'a', b'\xc3\xa9\xc3\xb8asd']
        exp = ['', 'a', 'éøasd']
        self.assertEqual(_bytes_decoder(test), exp)

    def test_passthrough_decoder(self):
        tests = [('', ''), ('a', 'a'), ('éøasd', 'éøasd')]
        for test, expected in tests:
            self.assertEqual(_passthrough_decoder(test), expected)

    def test_set_header(self):
        def mock():
            obj = h5py.File('bung', 'a', driver='core', backing_store=False)
            return obj

        m = mock()
        _set_header(m)


class CondensedBinaryDMTests(unittest.TestCase):
    """Condensed (1-D) storage must produce a true (N, N) `matrix` dataset."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.fname = os.path.join(self.tempdir.name, 'condensed')
        self.ids = ['a', 'b', 'c', 'd']
        self.condensed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.square = np.array([[0.0, 1.0, 2.0, 3.0],
                                [1.0, 0.0, 4.0, 5.0],
                                [2.0, 4.0, 0.0, 6.0],
                                [3.0, 5.0, 6.0, 0.0]])

    def tearDown(self):
        self.tempdir.cleanup()

    def check_file_with_h5py(self, fname, exp_square, ids, dtype):
        """Item-by-item inspection of the file with independent h5py."""
        with h5py.File(fname, 'r') as f:
            self.assertEqual(f['format'][0], b'BDSM')
            self.assertEqual(f['version'][0], b'2020.12')
            order = [x.decode('utf-8') for x in f['order'][:]]
            self.assertEqual(order, list(ids))
            self.assertIn('matrix', f)
            mat = f['matrix']
            # matrix must be a true n x n dataset, not a 1-D condensed vector
            self.assertEqual(mat.ndim, 2)
            self.assertEqual(mat.shape, (len(ids), len(ids)))
            self.assertEqual(mat.dtype, np.dtype(dtype))
            npt.assert_array_equal(np.asarray(mat), np.asarray(exp_square))

    def test_condensed_distance_matrix(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        dm.write(self.fname, format='binary_dm')
        self.check_file_with_h5py(self.fname, self.square, self.ids, 'float64')
        obs = DistanceMatrix.read(self.fname, format='binary_dm')
        npt.assert_array_equal(obs.redundant_form(), self.square)
        self.assertEqual(obs.ids, tuple(self.ids))

    def test_condensed_distance_matrix_float32(self):
        dm = DistanceMatrix(np.asarray(self.condensed, dtype='float32'),
                            ids=self.ids, condensed=True)
        dm.write(self.fname, format='binary_dm')
        self.check_file_with_h5py(self.fname, self.square, self.ids, 'float32')
        obs = DistanceMatrix.read(self.fname, format='binary_dm')
        self.assertEqual(obs.dtype, np.float32)
        npt.assert_array_equal(obs.redundant_form(),
                               self.square.astype('float32'))

    def test_condensed_symmetric_array_diagonal(self):
        diag = [11.0, 12.0, 13.0, 14.0]
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=diag)
        sm.write(self.fname, format='binary_dm')
        exp = self.square + np.diag(diag)
        self.check_file_with_h5py(self.fname, exp, self.ids, 'float64')
        # the diagonal survives the round trip
        obs = SymmetricMatrix.read(self.fname, format='binary_dm')
        npt.assert_array_equal(np.diag(obs.redundant_form()), diag)
        npt.assert_array_equal(obs.redundant_form(), exp)

    def test_condensed_symmetric_scalar_diagonal(self):
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=11.0)
        sm.write(self.fname, format='binary_dm')
        exp = self.square + np.diag([11.0] * 4)
        self.check_file_with_h5py(self.fname, exp, self.ids, 'float64')
        obs = SymmetricMatrix.read(self.fname, format='binary_dm')
        npt.assert_array_equal(np.diag(obs.redundant_form()), [11.0] * 4)

    def test_condensed_matches_redundant_file(self):
        dm_cond = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        dm_cond.write(self.fname, format='binary_dm')
        fname2 = os.path.join(self.tempdir.name, 'redundant')
        dm_red = DistanceMatrix(self.square, ids=self.ids)
        dm_red.write(fname2, format='binary_dm')
        with h5py.File(self.fname, 'r') as f1, h5py.File(fname2, 'r') as f2:
            self.assertEqual(f1['matrix'].shape, f2['matrix'].shape)
            npt.assert_array_equal(np.asarray(f1['matrix']),
                                   np.asarray(f2['matrix']))
            npt.assert_array_equal(np.asarray(f1['order'][:]),
                                   np.asarray(f2['order'][:]))

    def test_pairwise_asymmetric_preserved(self):
        asym = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]])
        pm = PairwiseMatrix(asym, ids=['a', 'b', 'c'])
        pm.write(self.fname, format='binary_dm')
        self.check_file_with_h5py(self.fname, asym, ['a', 'b', 'c'],
                                  'float64')

    def test_filter_reorder_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        sub = dm.filter(['c', 'a'])
        sub.write(self.fname, format='binary_dm')
        exp = np.array([[0.0, 2.0], [2.0, 0.0]])
        self.check_file_with_h5py(self.fname, exp, ['c', 'a'], 'float64')

    def test_permute_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        perm = dm.permute(seed=42)
        self.assertEqual(perm.ids, dm.ids)  # IDs stay, values move
        perm.write(self.fname, format='binary_dm')
        self.check_file_with_h5py(self.fname, perm.redundant_form(),
                                  self.ids, 'float64')

    def test_copy_then_write(self):
        dm = DistanceMatrix(self.condensed, ids=self.ids, condensed=True)
        dm.copy().write(self.fname, format='binary_dm')
        self.check_file_with_h5py(self.fname, self.square, self.ids,
                                  'float64')

    def test_source_not_mutated(self):
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=[11.0, 12.0, 13.0, 14.0])
        data_before = sm.data
        snapshot = sm.data.copy()
        sm.write(self.fname, format='binary_dm')
        self.assertIs(sm.data, data_before)
        self.assertTrue(sm._flags['CONDENSED'])
        self.assertEqual(sm.ids, tuple(self.ids))
        npt.assert_array_equal(sm.data, snapshot)
        npt.assert_array_equal(np.asarray(sm.diagonal),
                               [11.0, 12.0, 13.0, 14.0])

    def test_readback_in_separate_process(self):
        # The file must be understandable by other programs: dump it with
        # plain h5py (no skbio) in a separate Python process and compare.
        diag = [11.0, 12.0, 13.0, 14.0]
        sm = SymmetricMatrix(self.condensed, ids=self.ids, condensed=True,
                             diagonal=diag)
        sm.write(self.fname, format='binary_dm')
        script = (
            "import json, sys\n"
            "import h5py\n"
            "with h5py.File(sys.argv[1], 'r') as f:\n"
            "    payload = {\n"
            "        'format': f['format'][0].decode(),\n"
            "        'version': f['version'][0].decode(),\n"
            "        'order': [x.decode() for x in f['order'][:]],\n"
            "        'shape': list(f['matrix'].shape),\n"
            "        'dtype': str(f['matrix'].dtype),\n"
            "        'matrix': f['matrix'][:].tolist(),\n"
            "    }\n"
            "print(json.dumps(payload))\n"
        )
        proc = subprocess.run(
            [sys.executable, '-I', '-c', script, self.fname],
            capture_output=True, text=True, check=True,
        )
        obs = json.loads(proc.stdout)
        self.assertEqual(obs['format'], 'BDSM')
        self.assertEqual(obs['version'], '2020.12')
        self.assertEqual(obs['order'], self.ids)
        self.assertEqual(obs['shape'], [4, 4])
        self.assertEqual(obs['dtype'], 'float64')
        npt.assert_array_equal(np.asarray(obs['matrix']),
                               self.square + np.diag(diag))


class CorruptBinaryDMTests(unittest.TestCase):
    """Corrupt files are rejected by the sniffer and the explicit reader."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.fname = os.path.join(self.tempdir.name, 'corrupt')

    def tearDown(self):
        self.tempdir.cleanup()

    _OMIT = object()

    def make_file(self, order=['a', 'b'], matrix=None, header=True,
                  matrix_name='matrix'):
        if matrix is None:
            matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        with h5py.File(self.fname, 'w') as f:
            if header:
                _set_header(f)
            if order is not None:
                f.create_dataset('order', data=np.array(
                    [x.encode('utf-8') for x in order]))
            if matrix is not self._OMIT:
                f.create_dataset(matrix_name, data=matrix)
        return self.fname

    def assert_rejected(self, fname):
        with open(fname, 'rb') as fh:
            self.assertEqual(_binary_dm_sniffer(fh), (False, {}))
        for cls in (DistanceMatrix, SymmetricMatrix, PairwiseMatrix):
            with self.assertRaises(BinaryDMFormatError):
                io_read(fname, format='binary_dm', into=cls)

    def test_valid_file_accepted(self):
        fname = self.make_file()
        with open(fname, 'rb') as fh:
            self.assertEqual(_binary_dm_sniffer(fh), (True, {}))
        obs = io_read(fname, format='binary_dm', into=DistanceMatrix)
        self.assertEqual(obs.ids, ('a', 'b'))

    def test_legacy_matrix_0_still_works(self):
        fname = self.make_file(matrix_name='matrix:0')
        with open(fname, 'rb') as fh:
            self.assertEqual(_binary_dm_sniffer(fh), (True, {}))
        obs = io_read(fname, format='binary_dm', into=DistanceMatrix)
        npt.assert_array_equal(obs.redundant_form(),
                               [[0.0, 0.5], [0.5, 0.0]])

    def test_condensed_vector_masquerading_as_matrix(self):
        self.assert_rejected(self.make_file(matrix=np.array([0.5, 0.6])))

    def test_order_matrix_rank_mismatch(self):
        self.assert_rejected(self.make_file(order=['a', 'b', 'c']))

    def test_duplicate_order(self):
        self.assert_rejected(self.make_file(order=['a', 'a']))

    def test_non_square_matrix(self):
        self.assert_rejected(self.make_file(matrix=np.zeros((2, 3))))
        self.assert_rejected(self.make_file(matrix=np.zeros((3, 2))))

    def test_missing_order(self):
        self.assert_rejected(self.make_file(order=None))

    def test_missing_matrix(self):
        self.assert_rejected(self.make_file(matrix=self._OMIT))

    def test_missing_header(self):
        self.assert_rejected(self.make_file(header=False))

    def test_missing_version(self):
        fname = self.make_file()
        with h5py.File(fname, 'a') as f:
            del f['version']
        self.assert_rejected(fname)

    def test_missing_format(self):
        fname = self.make_file()
        with h5py.File(fname, 'a') as f:
            del f['format']
        self.assert_rejected(fname)

    def test_reader_error_is_format_error(self):
        # No KeyError / TypeError / AttributeError may escape to callers.
        fname = self.make_file(header=False, order=None, matrix=None)
        try:
            io_read(fname, format='binary_dm', into=DistanceMatrix)
        except BinaryDMFormatError:
            pass
        except Exception as e:
            self.fail('expected BinaryDMFormatError, got %r' % (e,))


class BoundedWorkspaceTests(unittest.TestCase):
    """Larger condensed inputs are exported with bounded auxiliary memory."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.fname = os.path.join(self.tempdir.name, 'large')

    def tearDown(self):
        self.tempdir.cleanup()

    def make_large_condensed(self, n, dtype='float64'):
        rng = np.random.default_rng(7)
        return rng.random(n * (n - 1) // 2).astype(dtype)

    def test_condensed_hdf5_allocation_record(self):
        n = 5000
        cond = self.make_large_condensed(n)
        dm = DistanceMatrix(cond, ids=[str(i) for i in range(n)],
                            condensed=True)

        record = []
        import skbio.stats.distance._base as base_mod
        import skbio.io.format.binary_dm as bdm_mod
        orig_scratch = base_mod._seg_scratch
        orig_block = bdm_mod._block_buffer

        def spy_scratch(cap, dtype):
            bufs = orig_scratch(cap, dtype)
            record.append(('seg_scratch', sum(b.nbytes for b in bufs)))
            return bufs

        def spy_block(rows, cols, dtype):
            buf = orig_block(rows, cols, dtype)
            record.append(('block_buffer', buf.nbytes))
            return buf

        try:
            base_mod._seg_scratch = spy_scratch
            bdm_mod._block_buffer = spy_block
            dm.write(self.fname, format='binary_dm')
        finally:
            base_mod._seg_scratch = orig_scratch
            bdm_mod._block_buffer = orig_block

        # exactly one scratch bundle and one block buffer were allocated
        self.assertEqual([label for label, _ in record],
                         ['seg_scratch', 'block_buffer'])
        total = sum(nbytes for _, nbytes in record)
        # the whole auxiliary numeric workspace stays under 8 MiB, while
        # the input alone is ~100 MiB and the output dataset is 200 MB
        self.assertLessEqual(total, 8 << 20)
        self.assertLess(total, cond.nbytes // 10)
        for _, nbytes in record:
            self.assertLessEqual(nbytes, max(_BLOCK_BYTES, 2 << 20))

        # the file holds the full n x n dataset with the right values
        exp = dm.redundant_form()
        with h5py.File(self.fname, 'r') as f:
            mat = f['matrix']
            self.assertEqual(mat.shape, (n, n))
            self.assertEqual(mat.dtype, np.float64)
            npt.assert_array_equal(np.asarray(mat), exp)

    def test_condensed_hdf5_allocation_record_float32(self):
        n = 3000
        cond = self.make_large_condensed(n, dtype='float32')
        dm = DistanceMatrix(cond, condensed=True)

        record = []
        import skbio.stats.distance._base as base_mod
        import skbio.io.format.binary_dm as bdm_mod
        orig_scratch = base_mod._seg_scratch
        orig_block = bdm_mod._block_buffer

        def spy_scratch(cap, dtype):
            bufs = orig_scratch(cap, dtype)
            record.append(('seg_scratch', sum(b.nbytes for b in bufs)))
            return bufs

        def spy_block(rows, cols, dtype):
            buf = orig_block(rows, cols, dtype)
            record.append(('block_buffer', buf.nbytes))
            return buf

        try:
            base_mod._seg_scratch = spy_scratch
            bdm_mod._block_buffer = spy_block
            dm.write(self.fname, format='binary_dm')
        finally:
            base_mod._seg_scratch = orig_scratch
            bdm_mod._block_buffer = orig_block

        self.assertEqual(len(record), 2)
        total = sum(nbytes for _, nbytes in record)
        self.assertLessEqual(total, 8 << 20)
        with h5py.File(self.fname, 'r') as f:
            self.assertEqual(f['matrix'].dtype, np.float32)
            self.assertEqual(f['matrix'].shape, (n, n))


if __name__ == '__main__':
    unittest.main()
