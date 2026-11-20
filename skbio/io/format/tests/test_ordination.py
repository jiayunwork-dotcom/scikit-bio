# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import io
import os
import subprocess
import sys
import tempfile
from unittest import TestCase, main

import numpy as np
import pandas as pd
import pandas.testing as pdt
import numpy.testing as npt

import skbio
from skbio import DistanceMatrix, OrdinationResults
from skbio.io import OrdinationFormatError
from skbio.io.format.ordination import (
    _ordination_to_ordination_results, _ordination_results_to_ordination,
    _ordination_sniffer)
from skbio.stats.ordination import pcoa
from skbio.stats.ordination._projection import ProjectionState
from skbio.util import get_data_path, assert_ordination_results_equal, assert_ordination_results_equal_np
from skbio._config import set_config


class OrdinationTestData(TestCase):
    def setUp(self):
        self.valid_fps = map(
            get_data_path,
            ['ordination_L&L_CA_data_scores', 'ordination_example3_scores',
             'ordination_PCoA_sample_data_3_scores',
             'ordination_example2_scores'])

        # Store filepath, regex for matching the error message that should be
        # raised when reading the file, and whether the file should be matched
        # by the sniffer (True) or not (False).
        self.invalid_fps = map(lambda e: (get_data_path(e[0]), e[1], e[2]), [
            ('empty', r'end of file.*Eigvals header', False),
            ('whitespace_only', r'Eigvals header not found', False),
            ('ordination_error1', r'Eigvals header not found', False),
            ('ordination_error2',
             r'Proportion explained header not found', False),
            ('ordination_error3', r'Species header not found', True),
            ('ordination_error4', r'Site header not found', True),
            ('ordination_error5', r'Biplot header not found', True),
            ('ordination_error6', r'Site constraints header not found', True),
            ('ordination_error7', r'empty line', False),
            ('ordination_error8', r'9.*Proportion explained.*8', True),
            ('ordination_error9', r'2 values.*1 in row 1', True),
            ('ordination_error10', r'2 values.*1 in row 1', True),
            ('ordination_error11', r'Site constraints ids and site ids', True),
            ('ordination_error12', r'9.*Eigvals.*8', True),
            ('ordination_error13', r'9.*Proportion explained.*8', True),
            ('ordination_error14', r'Site is 0: 9 x 0', True),
            ('ordination_error15', r'9 values.*8 in row 1', True),
            ('ordination_error16', r'Biplot is 0: 3 x 0', True),
            ('ordination_error17', r'3 values.*2 in row 1', True),
            ('ordination_error18',
             r'proportion explained.*eigvals: 8 != 9', True),
            ('ordination_error19',
             r'coordinates.*species.*eigvals: 1 != 2', True),
            ('ordination_error20',
             r'coordinates.*site.*eigvals: 1 != 2', True),
            ('ordination_error21', r'one eigval', False),
            ('ordination_error22', r'end of file.*blank line', False),
            ('ordination_error23',
             r'end of file.*Proportion explained section',
             True),
            ('ordination_error24',
             r'end of file.*row 2.*Species section', True)
        ])


class OrdinationResultsReaderWriterTests(OrdinationTestData):
    def setUp(self):
        super(OrdinationResultsReaderWriterTests, self).setUp()

        # define in-memory results, one for each of the valid files in
        # self.valid_fps

        # CA results
        axes_ids = ['CA1', 'CA2']
        species_ids = ['Species1', 'Species2', 'Species3']
        site_ids = ['Site1', 'Site2', 'Site3']
        eigvals = pd.Series([0.0961330159181, 0.0409418140138], axes_ids)
        species = pd.DataFrame([[0.408869425742, 0.0695518116298],
                                [-0.1153860437, -0.299767683538],
                                [-0.309967102571, 0.187391917117]],
                               index=species_ids, columns=axes_ids)
        site = pd.DataFrame([[-0.848956053187, 0.882764759014],
                             [-0.220458650578, -1.34482000302],
                             [1.66697179591, 0.470324389808]],
                            index=site_ids, columns=axes_ids)
        biplot = None
        site_constraints = None
        prop_explained = None
        ca_scores = OrdinationResults(
            'CA', 'Correspondence Analysis', eigvals=eigvals, features=species,
            samples=site, biplot_scores=biplot,
            sample_constraints=site_constraints,
            proportion_explained=prop_explained)

        # CA results, numpy backend
        axes_ids = ['CA1', 'CA2']
        species_ids = ['Species1', 'Species2', 'Species3']
        site_ids = ['Site1', 'Site2', 'Site3']
        eigvals = np.array([0.0961330159181, 0.0409418140138])
        species = np.array([[0.408869425742, 0.0695518116298],
                            [-0.1153860437, -0.299767683538],
                            [-0.309967102571, 0.187391917117]])
        site = np.array([[-0.848956053187, 0.882764759014],
                         [-0.220458650578, -1.34482000302],
                         [1.66697179591, 0.470324389808]])
        biplot = None
        site_constraints = None
        prop_explained = None
        ca_scores_np = OrdinationResults(
            'CA', 'Correspondence Analysis', eigvals=eigvals, features=species,
            feature_ids=species_ids,
            samples=site, sample_ids=site_ids, biplot_scores=biplot,
            sample_constraints=site_constraints,
            proportion_explained=prop_explained)

        # CCA results
        axes_ids = ['CCA%d' % i for i in range(1, 10)]
        species_ids = ['Species0', 'Species1', 'Species2', 'Species3',
                       'Species4', 'Species5', 'Species6', 'Species7',
                       'Species8']
        site_ids = ['Site0', 'Site1', 'Site2', 'Site3', 'Site4', 'Site5',
                    'Site6', 'Site7', 'Site8', 'Site9']

        eigvals = pd.Series([0.366135830393, 0.186887643052, 0.0788466514249,
                             0.082287840501, 0.0351348475787, 0.0233265839374,
                             0.0099048981912, 0.00122461669234,
                             0.000417454724117], axes_ids)
        species = pd.DataFrame(np.loadtxt(
            get_data_path('ordination_exp_Ordination_CCA_species')),
            index=species_ids, columns=axes_ids)
        site = pd.DataFrame(
            np.loadtxt(get_data_path('ordination_exp_Ordination_CCA_site')),
            index=site_ids, columns=axes_ids)
        biplot = pd.DataFrame(
            [[-0.169746767979, 0.63069090084, 0.760769036049],
             [-0.994016563505, 0.0609533148724, -0.0449369418179],
             [0.184352565909, -0.974867543612, 0.0309865007541]],
            columns=axes_ids[:3])
        site_constraints = pd.DataFrame(np.loadtxt(
            get_data_path('ordination_exp_Ordination_CCA_site_constraints')),
            index=site_ids, columns=axes_ids)
        prop_explained = None
        cca_scores = OrdinationResults('CCA',
                                       'Canonical Correspondence Analysis',
                                       eigvals=eigvals, features=species,
                                       samples=site, biplot_scores=biplot,
                                       sample_constraints=site_constraints,
                                       proportion_explained=prop_explained)

        # CCA results, numpy backend
        axes_ids = ['CCA%d' % i for i in range(1, 10)]
        species_ids = ['Species0', 'Species1', 'Species2', 'Species3',
                       'Species4', 'Species5', 'Species6', 'Species7',
                       'Species8']
        site_ids = ['Site0', 'Site1', 'Site2', 'Site3', 'Site4', 'Site5',
                    'Site6', 'Site7', 'Site8', 'Site9']

        eigvals = np.array([0.366135830393, 0.186887643052, 0.0788466514249,
                             0.082287840501, 0.0351348475787, 0.0233265839374,
                             0.0099048981912, 0.00122461669234,
                             0.000417454724117])
        species = np.array(np.loadtxt(
            get_data_path('ordination_exp_Ordination_CCA_species')))
        site = np.array(
            np.loadtxt(get_data_path('ordination_exp_Ordination_CCA_site')))
        biplot = np.array(
            [[-0.169746767979, 0.63069090084, 0.760769036049],
             [-0.994016563505, 0.0609533148724, -0.0449369418179],
             [0.184352565909, -0.974867543612, 0.0309865007541]])
        site_constraints = np.array(np.loadtxt(
            get_data_path('ordination_exp_Ordination_CCA_site_constraints')))
        prop_explained = None
        cca_scores_np = OrdinationResults('CCA',
                                       'Canonical Correspondence Analysis',
                                       eigvals=eigvals, features=species,
                                       feature_ids=species_ids,
                                       sample_ids=site_ids,
                                       samples=site, biplot_scores=biplot,
                                       sample_constraints=site_constraints,
                                       proportion_explained=prop_explained)

        # PCoA results
        axes_ids = ['PC%d' % i for i in range(1, 10)]
        species_ids = None
        site_ids = ['PC.636', 'PC.635', 'PC.356', 'PC.481', 'PC.354', 'PC.593',
                    'PC.355', 'PC.607', 'PC.634']
        eigvals = pd.Series([0.512367260461, 0.300719094427, 0.267912066004,
                             0.208988681078, 0.19169895326, 0.16054234528,
                             0.15017695712, 0.122457748167, 0.0], axes_ids)
        species = None
        site = pd.DataFrame(
            np.loadtxt(get_data_path('ordination_exp_Ordination_PCoA_site')),
            index=site_ids, columns=axes_ids)
        biplot = None
        site_constraints = None
        prop_explained = pd.Series([0.267573832777, 0.15704469605,
                                    0.139911863774, 0.109140272454,
                                    0.100111048503, 0.0838401161912,
                                    0.0784269939011, 0.0639511763509, 0.0],
                                   axes_ids)
        pcoa_scores = OrdinationResults('PCoA',
                                        'Principal Coordinate Analysis',
                                        eigvals=eigvals, features=species,
                                        samples=site, biplot_scores=biplot,
                                        sample_constraints=site_constraints,
                                        proportion_explained=prop_explained)

        # RDA results
        axes_ids = ['RDA%d' % i for i in range(1, 8)]
        species_ids = ['Species0', 'Species1', 'Species2', 'Species3',
                       'Species4', 'Species5']
        site_ids = ['Site0', 'Site1', 'Site2', 'Site3', 'Site4', 'Site5',
                    'Site6', 'Site7', 'Site8', 'Site9']
        eigvals = pd.Series([25.8979540892, 14.9825779819, 8.93784077262,
                             6.13995623072, 1.68070536498, 0.57735026919,
                             0.275983624351], axes_ids)
        species = pd.DataFrame(np.loadtxt(
            get_data_path('ordination_exp_Ordination_RDA_species')),
            index=species_ids, columns=axes_ids)
        site = pd.DataFrame(
            np.loadtxt(get_data_path('ordination_exp_Ordination_RDA_site')),
            index=site_ids, columns=axes_ids)
        biplot = pd.DataFrame(
            [[0.422650019179, -0.559142585857, -0.713250678211],
             [0.988495963777, 0.150787422017, -0.0117848614073],
             [-0.556516618887, 0.817599992718, 0.147714267459],
             [-0.404079676685, -0.9058434809, -0.127150316558]],
            columns=axes_ids[:3])
        site_constraints = pd.DataFrame(np.loadtxt(
            get_data_path('ordination_exp_Ordination_RDA_site_constraints')),
            index=site_ids, columns=axes_ids)
        prop_explained = None
        rda_scores = OrdinationResults(
            'RDA', 'Redundancy Analysis', eigvals=eigvals, features=species,
            samples=site, biplot_scores=biplot,
            sample_constraints=site_constraints,
            proportion_explained=prop_explained)

        self.ordination_results_objs = [ca_scores, cca_scores, pcoa_scores,
                                        rda_scores]
        self.ordination_results_objs_np = [ca_scores_np, cca_scores_np]

    def test_read_valid_files(self):
        for fp, obj in zip(self.valid_fps, self.ordination_results_objs):
            obs = _ordination_to_ordination_results(fp)
            assert_ordination_results_equal(
                obs, obj, ignore_method_names=True,
                ignore_axis_labels=True)

    def test_read_valid_files_np(self):
        set_config("table_output", "numpy")
        for fp, obj in zip(self.valid_fps, self.ordination_results_objs_np):
            obs = _ordination_to_ordination_results(fp)
            assert_ordination_results_equal_np(
                obs, obj, ignore_method_names=True
            )
        set_config("table_output", "pandas")

    def test_read_invalid_files(self):
        for invalid_fp, error_msg_regexp, _ in self.invalid_fps:
            with self.assertRaisesRegex(OrdinationFormatError,
                                        error_msg_regexp):
                _ordination_to_ordination_results(invalid_fp)

    def test_write(self):
        for fp, obj in zip(self.valid_fps, self.ordination_results_objs):
            fh = io.StringIO()
            _ordination_results_to_ordination(obj, fh)
            obs = fh.getvalue()
            fh.close()

            with io.open(fp) as fh:
                exp = fh.read()

            npt.assert_equal(obs, exp)

    def test_roundtrip_read_write(self):
        for fp in self.valid_fps:
            # Read.
            obj1 = _ordination_to_ordination_results(fp)

            # Write.
            fh = io.StringIO()
            _ordination_results_to_ordination(obj1, fh)
            fh.seek(0)

            # Read.
            obj2 = _ordination_to_ordination_results(fh)
            fh.close()

            assert_ordination_results_equal(obj1, obj2)


class SnifferTests(OrdinationTestData):
    def setUp(self):
        super(SnifferTests, self).setUp()

    def test_matches_and_nonmatches(self):
        # Sniffer should match all valid files, and will match some invalid
        # ones too because it doesn't exhaustively check the entire file.
        for fp in self.valid_fps:
            self.assertEqual(_ordination_sniffer(fp), (True, {}))

        for fp, _, expected_sniffer_match in self.invalid_fps:
            self.assertEqual(_ordination_sniffer(fp),
                             (expected_sniffer_match, {}))


class ProjectionStateTestData(TestCase):
    """Shared fixtures for the optional ``Projection state`` section."""

    def setUp(self):
        # references at the corners of a 4x2 rectangle (hand-computable)
        self.refs = np.array(
            [[2.0, 1.0], [2.0, -1.0], [-2.0, 1.0], [-2.0, -1.0]]
        )
        self.ids = ["r1", "r2", "r3", "r4"]
        d = np.sqrt(((self.refs[:, None] - self.refs) ** 2).sum(-1))
        self.dm = DistanceMatrix(d, self.ids)
        self.res = pcoa(self.dm, dimensions=2, projection=True)

        query = np.array([0.5, 0.3])
        dist = np.sqrt(((self.refs - query) ** 2).sum(-1))
        self.distances = pd.DataFrame([dist], index=["q1"], columns=self.ids)

    def _write_str(self, obj):
        fh = io.StringIO()
        _ordination_results_to_ordination(obj, fh)
        return fh.getvalue()

    def _read_str(self, content):
        return _ordination_to_ordination_results(io.StringIO(content))

    def _state_section(self, content):
        """The part of a serialized result following 'Projection state'."""
        marker = "Projection state"
        idx = content.index(marker)
        return content[idx:]


class ProjectionStateReaderWriterTests(ProjectionStateTestData):
    def test_roundtrip_in_process(self):
        content = self._write_str(self.res)
        self.assertIn("Projection state\t1\t4\t2\n", content)
        obs = self._read_str(content)

        state = obs.projection_state
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.ids), tuple(self.ids))
        self.assertEqual(tuple(state.axis_labels), ("PC1", "PC2"))
        npt.assert_array_equal(
            state.coordinates, self.res.projection_state.coordinates
        )

        # project before and after the round-trip: identical coordinates
        # and diagnostics
        before = self.res.project(self.distances)
        after = obs.project(self.distances)
        pdt.assert_frame_equal(before.coordinates, after.coordinates)
        pdt.assert_frame_equal(before.diagnostics, after.diagnostics)

    def test_read_static_file(self):
        # a file written by a previous run remains readable and usable
        obs = _ordination_to_ordination_results(
            get_data_path("ordination_projection_state")
        )
        state = obs.projection_state
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.ids), tuple(self.ids))
        self.assertEqual(state.coordinates.shape, (4, 2))
        out = obs.project(self.distances)
        npt.assert_allclose(
            out.coordinates.values, [[-0.5, -0.3]], atol=1e-9
        )
        self.assertEqual(out.diagnostics.iloc[0]["status"], "ok")

    def test_sniffer_matches_file_with_state(self):
        self.assertEqual(
            _ordination_sniffer(
                get_data_path("ordination_projection_state")
            ),
            (True, {}),
        )

    def test_write_without_state_unchanged(self):
        # results without a projection state produce the pre-existing
        # format: no extra section, readable as a plain result
        res = pcoa(self.dm, dimensions=2)
        self.assertIsNone(res.projection_state)
        content = self._write_str(res)
        self.assertNotIn("Projection state", content)
        obs = self._read_str(content)
        self.assertIsNone(obs.projection_state)
        assert_ordination_results_equal(
            obs, res, ignore_method_names=True, ignore_axis_labels=True
        )

    def test_old_file_reads_as_plain_result(self):
        obs = _ordination_to_ordination_results(
            get_data_path("ordination_PCoA_sample_data_3_scores")
        )
        self.assertIsNone(obs.projection_state)
        with self.assertRaisesRegex(ValueError, "projection state"):
            obs.project(self.distances)

    def test_trailing_blank_line_tolerated(self):
        res = pcoa(self.dm, dimensions=2)
        content = self._write_str(res) + "\n"
        obs = self._read_str(content)
        self.assertIsNone(obs.projection_state)

    def test_writer_rejects_inconsistent_state(self):
        res = pcoa(self.dm, dimensions=2)
        # a state whose reference IDs do not match the samples
        res.projection_state = ProjectionState(
            ids=["x1", "x2", "x3", "x4"],
            coordinates=np.zeros((4, 2)),
            axis_labels=["PC1", "PC2"],
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            self._write_str(res)
        # a state whose dimensionality does not match the ordination axes
        res.projection_state = ProjectionState(
            ids=self.ids,
            coordinates=np.zeros((4, 3)),
            axis_labels=["PC1", "PC2", "PC3"],
        )
        with self.assertRaisesRegex(ValueError, "dimension"):
            self._write_str(res)

    def test_corrupted_projection_state_rejected(self):
        content = self._write_str(self.res)
        no_state = self._write_str(pcoa(self.dm, dimensions=2))

        def mutate_content(old, new, count=1):
            self.assertIn(old, content)
            return content.replace(old, new, count)

        def mutate_state(old, new, count=1):
            section = self._state_section(content)
            self.assertIn(old, section)
            head = content[: content.index("Projection state")]
            return head + section.replace(old, new, count)

        cases = [
            # unknown or malformed version
            (mutate_content("Projection state\t1\t", "Projection state\t99\t"),
             r"Unsupported projection state version: 99"),
            (mutate_content("Projection state\t1\t", "Projection state\tx\t"),
             r"version is not an integer"),
            # size constraints inconsistent with the rest of the file
            (mutate_content("Projection state\t1\t4\t2",
                            "Projection state\t1\t3\t2"),
             r"lists 3 reference\(s\), but the Site section lists 4"),
            (mutate_content("Projection state\t1\t4\t2",
                            "Projection state\t1\t4\t3"),
             r"has 3 dimension\(s\), but there are 2 eigvals"),
            (mutate_content("Projection state\t1\t4\t2",
                            "Projection state\t1\t0\t2"),
             r"must be positive: 0 x 2"),
            (mutate_content("Projection state\t1\t4\t2",
                            "Projection state\t1\t4\tx"),
             r"dimensions are not integers"),
            # truncated content
            (content[: content.index("PC1\tPC2")],
             r"end of file.*axis labels"),
            (content[: content.index("Projection state")]
             + self._state_section(content)[
                 : self._state_section(content).index("r3\t")],
             r"end of file.*row 3.*Projection state"),
            # reference IDs inconsistent with the numeric state / sites
            (mutate_state("r3\t", "rX\t"),
             r"reference IDs do not match the Site IDs"),
            # malformed numeric content
            (mutate_state("PC1\tPC2", "PC1\tPC2\tPC3"),
             r"Expected 2 axis labels"),
            (mutate_state("r3\t2.0\t-1.0", "r3\t2.0"),
             r"Expected 2 values, but found 1 in row 3"),
            (mutate_state("r3\t2.0\t-1.0", "r3\tabc\t-1.0"),
             r"non-numeric"),
            (mutate_state("r3\t2.0\t-1.0", "r3\tnan\t-1.0"),
             r"must be finite"),
            (mutate_state("r3\t2.0\t-1.0", "r3\tinf\t-1.0"),
             r"must be finite"),
            # trailing content after the state
            (content + "junk\n",
             r"Unexpected content after the Projection state section"),
            # junk after the Site constraints section
            (no_state + "junk\n",
             r"Expected an empty line after the Site constraints section"),
            (no_state + "\nSomething\t1\n",
             r"Projection state header not found"),
        ]
        for bad_content, error_msg_regexp in cases:
            with self.assertRaisesRegex(OrdinationFormatError,
                                        error_msg_regexp):
                self._read_str(bad_content)

    def test_state_size_grows_linearly_with_references(self):
        # the serialized state stores one row of coordinates per reference
        # (plus a constant header): for a fixed number of axes its size
        # grows linearly, not quadratically, with the reference count
        sizes = []
        for n in (30, 60, 120):
            rng = np.random.default_rng(0)
            refs = rng.normal(size=(n, 3))
            ids = ["s%d" % i for i in range(n)]
            dm = DistanceMatrix(
                np.sqrt(((refs[:, None] - refs) ** 2).sum(-1)), ids
            )
            res = pcoa(dm, dimensions=3, projection=True)
            self.assertEqual(res.projection_state.coordinates.shape, (n, 3))
            sizes.append(len(self._write_str(res)))
        inc1 = sizes[1] - sizes[0]
        inc2 = sizes[2] - sizes[1]
        self.assertGreater(inc1, 0)
        # linear growth doubles the increment when n doubles; quadratic
        # growth would roughly quadruple it
        self.assertGreater(inc2 / inc1, 1.5)
        self.assertLess(inc2 / inc1, 2.5)


class ProjectionStateFreshProcessTests(ProjectionStateTestData):
    """Round-trip through a file consumed by a fresh Python process."""

    def setUp(self):
        super().setUp()
        # rows exercising success, partial and total missingness, and a
        # geometrically underdetermined row
        full = np.sqrt(((self.refs - [0.5, 0.3]) ** 2).sum(-1))
        partial = full.copy()
        partial[1] = np.nan
        single = np.full(4, np.nan)
        single[2] = full[2]
        none = np.full(4, np.nan)
        self.batch = pd.DataFrame(
            [full, partial, single, none],
            index=["q1", "q2", "q3", "q4"],
            columns=self.ids,
        )
        self.expected = self.res.project(self.batch)

    def _run_worker(self, ordination_fp, distances_fp, out_fp, cwd):
        worker = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_projection_worker.py",
        )
        env = dict(os.environ)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(skbio.__file__))
        )
        env["PYTHONPATH"] = (
            repo_root + os.pathsep + env.get("PYTHONPATH", "")
        )
        return subprocess.run(
            [sys.executable, worker, ordination_fp, distances_fp, out_fp],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_fresh_process_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ordination_fp = os.path.join(tmp, "ordination.txt")
            distances_fp = os.path.join(tmp, "distances.csv")
            out_fp = os.path.join(tmp, "out.csv")
            with open(ordination_fp, "w") as fh:
                self.res.write(fh)
            self.batch.to_csv(distances_fp)

            # run from a different working directory so that only the
            # serialized file (no in-memory object, no original distance
            # matrix) can inform the projection
            proc = self._run_worker(
                ordination_fp, distances_fp, out_fp, cwd=tmp
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            with open(out_fp) as fh:
                observed = fh.read()

        expected = io.StringIO()
        self.expected.coordinates.to_csv(expected)
        expected.write("\n")
        self.expected.diagnostics.to_csv(expected)
        self.assertEqual(observed, expected.getvalue())

        # sanity check on the contents: statuses and NaN semantics survive
        self.assertIn("ok", observed)
        self.assertIn("underdetermined", observed)

    def test_fresh_process_plain_result_has_no_state(self):
        plain = pcoa(self.dm, dimensions=2)
        with tempfile.TemporaryDirectory() as tmp:
            ordination_fp = os.path.join(tmp, "ordination.txt")
            distances_fp = os.path.join(tmp, "distances.csv")
            out_fp = os.path.join(tmp, "out.csv")
            with open(ordination_fp, "w") as fh:
                plain.write(fh)
            self.batch.to_csv(distances_fp)
            proc = self._run_worker(
                ordination_fp, distances_fp, out_fp, cwd=tmp
            )
            self.assertEqual(proc.returncode, 3, msg=proc.stderr)
            self.assertIn("projection state", proc.stdout)
            self.assertFalse(os.path.exists(out_fp))


if __name__ == '__main__':
    main()
