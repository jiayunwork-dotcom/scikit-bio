# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import io
import unittest

from skbio import TabularMSA, DNA, RNA, Protein, TreeNode
import skbio.io
from skbio.io import NexusFormatError, FileFormatError
from skbio.io._exception import NexusFormatError as NexusFormatErrorDirect
from skbio.io.format.nexus import (
    _nexus_sniffer,
    _nexus_to_tabular_msa,
    _tabular_msa_to_nexus,
    _nexus_to_tree_node,
    _tree_node_to_nexus,
)
from skbio.util import get_data_path


def _msa(sequences, labels):
    objs = [ctor(seq, metadata={"id": label}) for ctor, seq, label in sequences]
    return TabularMSA(objs, index=labels)


class TestException(unittest.TestCase):
    def test_inheritance_and_imports(self):
        self.assertTrue(issubclass(NexusFormatError, FileFormatError))
        self.assertIs(NexusFormatError, NexusFormatErrorDirect)
        self.assertIs(skbio.io.NexusFormatError, NexusFormatError)


class TestSniffer(unittest.TestCase):
    def test_positives(self):
        positives = [
            "#NEXUS\n",
            "#NEXUS\nBEGIN DATA;\nEND;\n",
            "#nexus\nbegin trees;\nend;\n",
            "  \n\t #NEXUS  \n[comment]\n",
            "\ufeff#NEXUS\n",
            "#NEXUS[nested [comment]]\nBEGIN DATA;\nEND;\n",
        ]
        for content in positives:
            self.assertEqual(_nexus_sniffer(io.StringIO(content)), (True, {}), content)

    def test_negatives(self):
        negatives = [
            "",
            "   \n\t  \n",
            "NEXUS\n",
            "#NEXUSX\n",
            "ACGT\n",
        ]
        for content in negatives:
            self.assertEqual(_nexus_sniffer(io.StringIO(content)), (False, {}), content)

    def test_other_formats_still_sniffed(self):
        # NEXUS must not hijack any of the other formats, and each of them
        # must still be identifiable through the general sniffer.
        data_file_cases = [
            ("fasta", get_data_path("fasta_multi_seq")),
            ("fastq", get_data_path("fastq_multi_seq_sanger")),
            ("stockholm", get_data_path("stockholm_minimal")),
            ("phylip", get_data_path("phylip_dna_3_seqs")),
        ]
        for expected, path in data_file_cases:
            self.assertEqual(skbio.io.sniff(path)[0], expected)

        inline_cases = [
            ("newick", "((a,b:0.2),c);\n"),
            (
                "clustal",
                "CLUSTAL W (1.8) multiple sequence alignment\n\n"
                "abc GCAU\n"
                "def GCAU\n\n"
                "abc GCAU\n"
                "def GCAU\n",
            ),
        ]
        for expected, content in inline_cases:
            self.assertEqual(skbio.io.sniff(io.StringIO(content))[0], expected)

        nexus = "#NEXUS\nBEGIN DATA;\nEND;\n"
        self.assertEqual(skbio.io.sniff(io.StringIO(nexus))[0], "nexus")


class TestTabularMSAReader(unittest.TestCase):
    def test_basic_data_block_dna(self):
        content = (
            "#NEXUS\n"
            "[a [nested] comment]\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=2 NCHAR=8;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX\n"
            "t1 ACGT-ACG\n"
            "t2 ACGT-ACG\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        exp = _msa(
            [(DNA, "ACGT-ACG", "t1"), (DNA, "ACGT-ACG", "t2")],
            ["t1", "t2"],
        )
        self.assertEqual(obs, exp)
        self.assertEqual(list(obs.index), ["t1", "t2"])
        self.assertEqual([s.metadata["id"] for s in obs], ["t1", "t2"])

    def test_characters_block_and_datatypes(self):
        for datatype, ctor, seq in [
            ("DNA", DNA, "ACGTN-CG"),
            ("dna", DNA, "ACGTN-CG"),
            ("RNA", RNA, "ACGUN-CG"),
            ("PROTEIN", Protein, "MVKX-CDR"),
        ]:
            content = (
                "#NEXUS\n"
                "BEGIN CHARACTERS;\n"
                "DIMENSIONS NTAX=1 NCHAR=%d;\n"
                "FORMAT DATATYPE=%s;\n"
                "MATRIX t1 %s;\n"
                "END;\n" % (len(seq), datatype, seq)
            )
            obs = _nexus_to_tabular_msa(io.StringIO(content))
            self.assertEqual(type(obs[0]), ctor)
            self.assertEqual(str(obs[0]), seq)

    def test_constructor_override(self):
        content = (
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\nEND;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content), constructor=DNA)
        self.assertEqual(type(obs[0]), DNA)
        self.assertEqual(str(obs[0]), "ACGT")

    def test_gap_missing_matchchar(self):
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=2 NCHAR=6;\n"
            "FORMAT DATATYPE=DNA GAP=# MISSING=* MATCHCHAR=^;\n"
            "MATRIX\n"
            "first ACGT#N\n"
            "other ^^^^*C\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGT-N")
        self.assertEqual(str(obs[1]), "ACGTNC")

    def test_default_gap_missing_matchchar(self):
        # Defaults are -, ?, .
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=3 NCHAR=4;\n"
            "FORMAT DATATYPE=RNA MATCHCHAR=.;\n"
            "MATRIX\n"
            "t1 ACG-\n"
            "t2 ...?\n"
            "t3 ACG-\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(type(obs[0]), RNA)
        self.assertEqual(str(obs[1]), "ACGN")
        self.assertEqual(str(obs[0]), "ACG-")

    def test_protein_missing_is_x(self):
        content = (
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=PROTEIN MISSING=?;\nMATRIX t1 MV?X;\nEND;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(type(obs[0]), Protein)
        self.assertEqual(str(obs[0]), "MVXX")

    def test_interleaved(self):
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=2 NCHAR=10;\n"
            "FORMAT DATATYPE=DNA INTERLEAVE GAP=-;\n"
            "MATRIX\n"
            "t1 ACGT-\n"
            "t2 AACGT\n"
            "\n"
            "t1 ACGT-\n"
            "t2 TTTTT\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGT-ACGT-")
        self.assertEqual(str(obs[1]), "AACGTTTTTT")
        self.assertEqual(list(obs.index), ["t1", "t2"])

    def test_interleaved_without_blank_lines(self):
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=2 NCHAR=8;\n"
            "FORMAT DATATYPE=DNA INTERLEAVE;\n"
            "MATRIX\n"
            "a ACGT\n"
            "b TGCA\n"
            "a ACGT\n"
            "b TGCA\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGTACGT")
        self.assertEqual(str(obs[1]), "TGCATGCA")

    def test_sequential_wrapped_lines(self):
        # Only the first line of a taxon carries its label; remaining lines
        # are continuations until NCHAR characters are read.
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=2 NCHAR=8;\n"
            "FORMAT DATATYPE=PROTEIN;\n"
            "MATRIX\n"
            "t1 MVKL\n"
            "FICD\n"
            "t2 MVKL\n"
            "FICX\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(type(obs[0]), Protein)
        self.assertEqual(str(obs[0]), "MVKLFICD")
        self.assertEqual(str(obs[1]), "MVKLFICX")

    def test_quoted_labels(self):
        content = (
            "#NEXUS\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=3 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX\n"
            "'has space' ACGT\n"
            "'apo''s;trophe,comma' ACGT\n"
            "plain ACGT\n"
            ";\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(
            list(obs.index),
            ["has space", "apo's;trophe,comma", "plain"],
        )
        self.assertEqual(
            [s.metadata["id"] for s in obs],
            ["has space", "apo's;trophe,comma", "plain"],
        )

    def test_comments_with_punctuation(self):
        content = (
            "#NEXUS [comment; with = stuff]\n"
            "BEGIN DATA; [another [nested;] comment]\n"
            "DIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX [inline] t1 [also inline] ACGT;\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGT")
        self.assertEqual(list(obs.index), ["t1"])

    def test_unknown_blocks_are_skipped(self):
        content = (
            "#NEXUS\n"
            "BEGIN TAXA;\n"
            "DIMENSIONS NTAX=1;\n"
            "TAXLABELS t1;\n"
            "END;\n"
            "BEGIN DATA;\n"
            "DIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX t1 ACGT;\n"
            "END;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGT")

    def test_endblock_accepted(self):
        content = (
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\nENDBLOCK;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(str(obs[0]), "ACGT")

    def test_keywords_are_case_insensitive(self):
        content = (
            "#nexUS\n"
            "begin data;\n"
            "dimensions ntax=1 nchar=4;\n"
            "format datatype=dna;\n"
            "matrix t1 ACGT;\n"
            "end;\n"
        )
        obs = _nexus_to_tabular_msa(io.StringIO(content))
        self.assertEqual(type(obs[0]), DNA)
        self.assertEqual(str(obs[0]), "ACGT")

    def test_read_via_public_api_with_sniffing(self):
        content = (
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\nEND;\n"
        )
        obs = TabularMSA.read(io.StringIO(content))
        self.assertEqual(type(obs[0]), DNA)
        self.assertEqual(str(obs[0]), "ACGT")

    def test_errors(self):
        bad = [
            # Missing END;
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\n",
            # NTAX mismatch
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=3 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX\na ACGT\nb TGCA\n;\nEND;\n",
            # NCHAR mismatch (too short)
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=2 NCHAR=5;\n"
            "FORMAT DATATYPE=DNA;\n"
            "MATRIX\na ACGT\nb TGCA\n;\nEND;\n",
            # Missing NCHAR
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\nEND;\n",
            # Unsupported datatype
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=CONTINUOUS;\nMATRIX t1 1234;\nEND;\n",
            # Missing datatype and no constructor
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\nMATRIX t1 ACGT;\nEND;\n",
            # Unclosed quote
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX 't1 ACGT;\nEND;\n",
            # Unclosed comment
            "#NEXUS\nBEGIN DATA; [unclosed\nDIMENSIONS NTAX=1 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA;\nMATRIX t1 ACGT;\nEND;\n",
            # Matchchar used by the first taxon
            "#NEXUS\nBEGIN DATA;\nDIMENSIONS NTAX=2 NCHAR=4;\n"
            "FORMAT DATATYPE=DNA MATCHCHAR=.;\n"
            "MATRIX t1 ..GT\nt2 ACGT;\nEND;\n",
            # Not a nexus file at all
            ">a\nACGT\n",
            # No data block
            "#NEXUS\nBEGIN TREES;\nTREE t=(a,b);\nEND;\n",
        ]
        for content in bad:
            with self.assertRaises(NexusFormatError, msg=content):
                _nexus_to_tabular_msa(io.StringIO(content))


class TestTabularMSAWriter(unittest.TestCase):
    def _roundtrip(self, msa):
        fh = io.StringIO()
        _tabular_msa_to_nexus(msa, fh)
        return _nexus_to_tabular_msa(io.StringIO(fh.getvalue())), fh.getvalue()

    def test_basic_roundtrip(self):
        msa = _msa(
            [
                (DNA, "ACGT-ACG", "seq1"),
                (DNA, "ACGT-ACG", "seq2"),
            ],
            ["seq1", "seq2"],
        )
        obs, text = self._roundtrip(msa)
        self.assertEqual(obs, msa)
        self.assertTrue(text.startswith("#NEXUS\n"))
        self.assertIn("BEGIN DATA;", text)
        self.assertIn("NTAX=2 NCHAR=8", text)
        self.assertIn("DATATYPE=DNA", text)
        self.assertIn("END;", text)

    def test_all_datatypes_and_dot_gap_roundtrip(self):
        for ctor, datatype, seqs in [
            (DNA, "DNA", ["ACGTN-C.", "ACGTN-C."]),
            (RNA, "RNA", ["ACGUN-C.", "ACGUN-C."]),
            (Protein, "PROTEIN", ["MVKX-CD*", "MVKX-CD*"]),
        ]:
            labels = ["a", "b"]
            msa = _msa(
                [(ctor, seq, label) for seq, label in zip(seqs, labels)],
                labels,
            )
            obs, text = self._roundtrip(msa)
            self.assertEqual(obs, msa)
            self.assertIn("DATATYPE=%s" % datatype, text)
            self.assertEqual(type(obs[0]), ctor)

    def test_special_labels_are_quoted(self):
        msa = _msa(
            [
                (DNA, "ACGT", "plain"),
                (DNA, "ACGT", "has space"),
                (DNA, "ACGT", "apo's"),
                (DNA, "ACGT", "sem;i"),
                (DNA, "ACGT", "com,ma"),
            ],
            ["plain", "has space", "apo's", "sem;i", "com,ma"],
        )
        obs, text = self._roundtrip(msa)
        self.assertEqual(obs, msa)
        self.assertIn("'has space'", text)
        self.assertIn("'apo''s'", text)
        self.assertIn("'sem;i'", text)
        self.assertIn("'com,ma'", text)
        # A plain label is not quoted.
        self.assertIn("\n    plain ACGT\n", text)

    def test_empty_msa_rejected(self):
        msa = TabularMSA([DNA("")])
        with self.assertRaises(NexusFormatError):
            _tabular_msa_to_nexus(msa, io.StringIO())


class TestTreeNodeReader(unittest.TestCase):
    def _read(self, content):
        return _nexus_to_tree_node(io.StringIO(content))

    def test_basic_translate(self):
        content = (
            "#NEXUS\n"
            "BEGIN TREES;\n"
            "TRANSLATE 1 a, 2 b, 3 c;\n"
            "TREE t = ((1:0.1,2:0.2)3:0.5);\n"
            "END;\n"
        )
        tree = self._read(content)
        self.assertEqual(
            sorted((tip.name, tip.length) for tip in tree.tips()),
            [("a", 0.1), ("b", 0.2)],
        )
        # The internal label '3' is left untouched, as is the internal
        # branch length '0.5'.
        internal = tree.children[0]
        self.assertEqual(internal.name, "3")
        self.assertEqual(internal.length, 0.5)

    def test_number_as_branch_length_not_translated(self):
        # '1' and '2' are leaf labels and are translated; the same numerals
        # appearing as branch lengths must remain numeric lengths.
        content = (
            "#NEXUS\n"
            "BEGIN TREES;\n"
            "TRANSLATE 1 aaa, 2 bbb, 3 ccc;\n"
            "TREE t = (1:1,2:2)3;\n"
            "END;\n"
        )
        tree = self._read(content)
        tips = sorted((tip.name, tip.length) for tip in tree.tips())
        self.assertEqual(tips, [("aaa", 1.0), ("bbb", 2.0)])
        self.assertEqual(tree.name, "3")

    def test_quoted_translate_name_with_spaces(self):
        content = (
            "#NEXUS\n"
            "BEGIN TREES;\n"
            "TRANSLATE 1 'Homo sapiens', 2 'Pan troglodytes';\n"
            "TREE t = (1,2);\n"
            "END;\n"
        )
        tree = self._read(content)
        self.assertEqual(
            sorted(tip.name for tip in tree.tips()),
            ["Homo sapiens", "Pan troglodytes"],
        )

    def test_semicolon_inside_quoted_tree_label(self):
        content = "#NEXUS\nBEGIN TREES;\nTREE t = ('a;b':1.0,c:2.0);\nEND;\n"
        tree = self._read(content)
        tips = {tip.name: tip.length for tip in tree.tips()}
        self.assertEqual(tips, {"a;b": 1.0, "c": 2.0})

    def test_translate_after_tree_and_first_tree_selected(self):
        content = (
            "#NEXUS\n"
            "BEGIN TREES;\n"
            "TREE first = (1,2);\n"
            "TREE second = (2,1);\n"
            "TRANSLATE 1 a, 2 b;\n"
            "END;\n"
        )
        tree = self._read(content)
        self.assertEqual([tip.name for tip in tree.tips()], ["a", "b"])

    def test_starred_tree_and_comments(self):
        content = (
            "#NEXUS [comment; here]\n"
            "BEGIN TREES;\n"
            "TREE * t1 [&R] = (a:0.1,b:0.2);\n"
            "END;\n"
        )
        tree = self._read(content)
        self.assertEqual(
            sorted((tip.name, tip.length) for tip in tree.tips()),
            [("a", 0.1), ("b", 0.2)],
        )

    def test_read_via_public_api_with_sniffing(self):
        content = "#NEXUS\nBEGIN TREES;\nTRANSLATE 1 a, 2 b;\nTREE t=(1,2);\nEND;\n"
        tree = TreeNode.read(io.StringIO(content))
        self.assertEqual(sorted(tip.name for tip in tree.tips()), ["a", "b"])

    def test_errors(self):
        bad = [
            # No TREES block
            "#NEXUS\nBEGIN DATA;\nEND;\n",
            # TREES block without END
            "#NEXUS\nBEGIN TREES;\nTREE t=(a,b);\n",
            # Non-numeric branch length
            "#NEXUS\nBEGIN TREES;\nTREE t=(a:notalength,b);\nEND;\n",
            # Unbalanced parentheses
            "#NEXUS\nBEGIN TREES;\nTREE t=(a,b;\nEND;\n",
            # Not a nexus file
            "(a,b);\n",
        ]
        for content in bad:
            with self.assertRaises(NexusFormatError, msg=content):
                self._read(content)


class TestTreeNodeWriter(unittest.TestCase):
    def _signature(self, tree):
        return [
            (node.name, node.length, len(node.children)) for node in tree.preorder()
        ]

    def _roundtrip(self, tree):
        fh = io.StringIO()
        _tree_node_to_nexus(tree, fh)
        text = fh.getvalue()
        return _nexus_to_tree_node(io.StringIO(text)), text

    def test_basic_roundtrip(self):
        tree = TreeNode.read(
            io.StringIO("((a:0.1,b:0.2)internal:0.5,c:0.7)root;"),
            format="newick",
        )
        obs, text = self._roundtrip(tree)
        self.assertTrue(text.startswith("#NEXUS\n"))
        self.assertIn("BEGIN TREES;", text)
        self.assertIn("TREE tree = ", text)
        self.assertEqual(self._signature(obs), self._signature(tree))
        self.assertEqual(sorted(tip.name for tip in obs.tips()), ["a", "b", "c"])

    def test_special_leaf_names_roundtrip(self):
        tree = TreeNode.read(
            io.StringIO("('Homo sapiens':0.1,'apo''s;x,y':0.2,b:0.3);"),
            format="newick",
        )
        obs, text = self._roundtrip(tree)
        self.assertEqual(self._signature(obs), self._signature(tree))
        self.assertEqual(
            sorted(tip.name for tip in obs.tips()),
            ["Homo sapiens", "apo's;x,y", "b"],
        )
        self.assertIn("'Homo sapiens'", text)
        self.assertIn("'apo''s;x,y'", text)

    def test_no_lengths_or_internal_names(self):
        tree = TreeNode.read(io.StringIO("(a,b,(c,d));"), format="newick")
        obs, _ = self._roundtrip(tree)
        self.assertEqual(self._signature(obs), self._signature(tree))
        self.assertEqual(sorted(tip.name for tip in obs.tips()), ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
