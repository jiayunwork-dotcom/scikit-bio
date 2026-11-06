r"""NEXUS format (:mod:`skbio.io.format.nexus`)
================================================

.. currentmodule:: skbio.io.format.nexus

NEXUS format (``nexus``) is a extensible, block-structured format for
systematic data, including character matrices and phylogenetic trees [1]_.
A NEXUS file begins with the ``#NEXUS`` word and contains one or more
blocks delimited by ``BEGIN <name>;`` and ``END;`` (or ``ENDBLOCK;``).

Format Support
--------------
**Has Sniffer: Yes**

+------+------+---------------------------------------------------------------+
|Reader|Writer|                          Object Class                         |
+======+======+===============================================================+
|Yes   |Yes   |:mod:`skbio.alignment.TabularMSA`                              |
+------+------+---------------------------------------------------------------+
|Yes   |Yes   |:mod:`skbio.tree.TreeNode`                                     |
+------+------+---------------------------------------------------------------+

Format Specification
--------------------
A NEXUS file is a plain text file whose first non-blank word is ``#NEXUS``
(the token is matched without regard to case). The remainder of the file
consists of blocks::

    BEGIN <block-name>;
        <command> <arguments>;
        ...
    END;

Square brackets delimit comments, and comments may be nested
(``[outer [inner] outer]``). Single quotes delimit words that contain
whitespace or punctuation; inside a quoted word, two adjacent single quotes
denote one literal apostrophe. Keywords are case-insensitive.

DATA and CHARACTERS blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^
Character matrices live in ``DATA`` or ``CHARACTERS`` blocks::

    BEGIN DATA;
        DIMENSIONS NTAX=3 NCHAR=10;
        FORMAT DATATYPE=DNA MISSING=? GAP=- MATCHCHAR=. INTERLEAVE;
    MATRIX
        taxon_1 ACGT-ACGT-
        taxon_2 ACGT-ACGT-
        taxon_3 ACGT-ACGT-
    ;
    END;

``DIMENSIONS`` gives the number of taxa (``NTAX``) and characters
(``NCHAR``). ``FORMAT`` declares the ``DATATYPE`` (``DNA``, ``RNA``, or
``PROTEIN``) and the single characters used for ``MISSING`` data, ``GAP``
positions, and the ``MATCHCHAR`` shorthand that copies the state of the first
taxon at the same position. When ``INTERLEAVE`` is present the matrix is
organized in blocks, each holding one segment of every taxon; the segments
of a taxon are concatenated in the order in which they appear. In a
non-interleaved matrix a single taxon may span several lines: only the first
line of the taxon carries its label, and every subsequent line is treated as
a continuation until ``NCHAR`` characters have been read.

Gap symbols are read as ``-``; missing symbols are read as ``N`` for DNA and
RNA sequences and as ``X`` for protein sequences. Taxon labels become both
the ``TabularMSA`` index and each sequence's ``metadata['id']``.

TREES blocks
^^^^^^^^^^^^
Phylogenetic trees live in ``TREES`` blocks::

    BEGIN TREES;
        TRANSLATE
            1 Homo_sapiens,
            2 'Pan troglodytes',
            3 Gorilla;
        TREE primates = ((1:0.1,2:0.2)3:0.5);
    END;

The optional ``TRANSLATE`` statement maps short labels used inside the Newick
tree descriptions to full taxon names. Translation is applied only at leaf
label positions: internal node labels and numeric branch lengths that happen
to equal a translate key are left untouched. Quoted leaf labels may contain
semicolons.

Format Parameters
-----------------
The only supported format parameter is ``constructor``, which specifies the
type of in-memory sequence object to read each aligned sequence into when
reading a :class:`~skbio.alignment.TabularMSA` (e.g. ``DNA``, ``RNA``,
``Protein``). When ``constructor`` is not provided it is inferred from the
``DATATYPE`` declared in the ``FORMAT`` command.

Examples
--------
Read an alignment from a NEXUS ``DATA`` block:

>>> from io import StringIO
>>> from skbio import TabularMSA
>>> nexus_f = StringIO(
...     "#NEXUS\n"
...     "BEGIN DATA;\n"
...     "DIMENSIONS NTAX=2 NCHAR=8;\n"
...     "FORMAT DATATYPE=DNA MISSING=? GAP=- MATCHCHAR=.;\n"
...     "MATRIX\n"
...     "seq1 ACGT-ACG\n"
...     "seq2 ....?T?C\n"
...     ";\n"
...     "END;\n"
... )
>>> msa = TabularMSA.read(nexus_f, format='nexus')
>>> msa
TabularMSA[DNA]
---------------------
Stats:
    sequence count: 2
    position count: 8
---------------------
ACGT-ACG
ACGTNTNC
>>> msa.index # doctest: +ELLIPSIS
Index(['seq1', 'seq2'], dtype='...')

Write the alignment back to NEXUS and read it again, preserving everything:

>>> fh = StringIO()
>>> _ = msa.write(fh, format='nexus')
>>> roundtrip = TabularMSA.read(StringIO(fh.getvalue()), format='nexus')
>>> roundtrip == msa
True

Read a tree from a ``TREES`` block, applying the ``TRANSLATE`` table only at
leaf positions:

>>> from skbio import TreeNode
>>> tree_f = StringIO(
...     "#NEXUS\n"
...     "BEGIN TREES;\n"
...     "TRANSLATE 1 'Homo sapiens', 2 Pan;\n"
...     "TREE t = ((1:0.1,2:0.2)3:0.5);\n"
...     "END;\n"
... )
>>> tree = TreeNode.read(tree_f, format='nexus')
>>> [(tip.name, tip.length) for tip in tree.tips()]
[('Homo sapiens', 0.1), ('Pan', 0.2)]
>>> internal = tree.children[0]
>>> internal.name  # internal label '3' is not a leaf, so it is not translated
'3'
>>> internal.length
0.5

Trees can also be written as NEXUS; reading the output back preserves the
leaf names, branch lengths, and topology:

>>> out = StringIO()
>>> _ = tree.write(out, format='nexus')
>>> again = TreeNode.read(StringIO(out.getvalue()), format='nexus')
>>> [(tip.name, tip.length) for tip in again.tips()]
[('Homo sapiens', 0.1), ('Pan', 0.2)]

References
----------
.. [1] Maddison, D. R., Swofford, D. L., and Maddison, W. P. (1997). NEXUS:
   An extensible file format for systematic information. Systematic Biology,
   46(4), 590-621.

"""  # noqa: D205, D415

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from skbio.alignment import TabularMSA
from skbio.io import create_format, NexusFormatError
from skbio.sequence import DNA, RNA, Protein
from skbio.tree import TreeNode

nexus = create_format("nexus")


# Characters that separate or structure NEXUS commands. Whitespace also
# separates words but is not itself significant. The TREE statement marker
# '*' is intentionally a word character (it is handled specifically by the
# TREES parser) because '*' is a valid protein sequence character.
_PUNCTUATION = set("=;,():")

_DATATYPE_CONSTRUCTORS = {"DNA": DNA, "RNA": RNA, "PROTEIN": Protein}

# Default symbols used when the corresponding FORMAT subcommand is absent.
_DEFAULT_GAP = "-"
_DEFAULT_MISSING = "?"


# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------


def _tokenize_nexus(text):
    """Tokenize NEXUS text into ``(kind, value)`` tuples.

    ``kind`` is ``'w'`` for a bare word, ``'q'`` for the unescaped contents
    of a single-quoted word, ``'p'`` for a single punctuation character
    (one of ``=;,():*``), and ``'e'`` (value ``None``) for a line ending.
    Square-bracket comments, which may be nested, are removed.

    Raises
    ------
    NexusFormatError
        If a quoted word or comment is not closed.

    """
    n = len(text)
    i = 0
    comment_depth = 0
    buffer = []

    def flush_word():
        if buffer:
            word = "".join(buffer)
            del buffer[:]
            return ("w", word)
        return None

    while i < n:
        char = text[i]

        if comment_depth > 0:
            if char == "[":
                comment_depth += 1
            elif char == "]":
                comment_depth -= 1
            i += 1
            continue

        if char == "[":
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            comment_depth = 1
            i += 1
            continue

        if char == "'":
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            i += 1
            quoted = []
            closed = False
            while i < n:
                char = text[i]
                if char == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        quoted.append("'")
                        i += 2
                    else:
                        i += 1
                        closed = True
                        break
                else:
                    quoted.append(char)
                    i += 1
            if not closed:
                raise NexusFormatError("Unclosed single-quoted word in NEXUS file.")
            yield ("q", "".join(quoted))
            continue

        if char in _PUNCTUATION:
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            yield ("p", char)
            i += 1
            continue

        if char == "\n":
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            yield ("e", None)
            i += 1
            continue

        if char == "\r":
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            yield ("e", None)
            i += 1
            if i < n and text[i] == "\n":
                i += 1
            continue

        if char.isspace():
            flushed = flush_word()
            if flushed is not None:
                yield flushed
            i += 1
            continue

        buffer.append(char)
        i += 1

    if comment_depth > 0:
        raise NexusFormatError("Unclosed comment in NEXUS file.")
    flushed = flush_word()
    if flushed is not None:
        yield flushed


def _word_value(token):
    """Return the string value of a word or quoted-word token."""
    return token[1]


# ----------------------------------------------------------------------------
# File structure: #NEXUS header and blocks
# ----------------------------------------------------------------------------


def _strip_header(text):
    """Remove and validate the leading ``#NEXUS`` token.

    Returns the remainder of the file. Optional leading whitespace and a
    UTF-8 BOM are permitted.

    """
    stripped = text
    if stripped.startswith("\ufeff"):
        stripped = stripped[1:]
    stripped = stripped.lstrip()
    if stripped[:6].upper() != "#NEXUS":
        raise NexusFormatError("NEXUS files must begin with the '#NEXUS' word.")
    if len(stripped) > 6 and not (stripped[6].isspace() or stripped[6] == "["):
        raise NexusFormatError("NEXUS files must begin with the '#NEXUS' word.")
    return stripped[6:]


def _parse_blocks(text):
    """Parse NEXUS text into a list of ``(block_name, commands)`` tuples.

    Each command is the raw token list (including line-ending tokens) found
    between the command keyword and its terminating semicolon.

    Raises
    ------
    NexusFormatError
        If a ``BEGIN`` is missing its name or terminating ``END;``.

    """
    tokens = list(_tokenize_nexus(_strip_header(text)))
    blocks = []
    index = 0
    total = len(tokens)

    while index < total:
        # Find the next BEGIN word; anything between blocks is ignored.
        while index < total and not (
            tokens[index][0] == "w" and tokens[index][1].upper() == "BEGIN"
        ):
            index += 1
        if index >= total:
            break
        index += 1

        while index < total and tokens[index][0] == "e":
            index += 1
        if index >= total or tokens[index][0] not in ("w", "q"):
            raise NexusFormatError("BEGIN must be followed by a block name.")
        block_name = tokens[index][1].upper()
        index += 1

        commands = []
        command = []
        closed = False
        while index < total:
            token = tokens[index]
            if token == ("p", ";"):
                if command:
                    head = next((t for t in command if t[0] in ("w", "q")), None)
                    if head is not None and head[1].upper() in (
                        "END",
                        "ENDBLOCK",
                    ):
                        closed = True
                        index += 1
                        break
                    commands.append(command)
                command = []
                index += 1
                continue
            command.append(token)
            index += 1

        if not closed:
            raise NexusFormatError(
                "NEXUS block '%s' is missing its terminating END;" % block_name
            )
        blocks.append((block_name, commands))

    return blocks


def _command_keyword(command):
    """Return the uppercase keyword of a command token list."""
    head = next((t for t in command if t[0] in ("w", "q")), None)
    if head is None:
        return None
    return head[1].upper()


def _strip_eol(tokens):
    """Filter line-ending tokens out of a token list."""
    return [t for t in tokens if t[0] != "e"]


# ----------------------------------------------------------------------------
# Sniffer
# ----------------------------------------------------------------------------


@nexus.sniffer()
def _nexus_sniffer(fh):
    # A NEXUS file is identified by the #NEXUS word at its beginning.
    head = fh.read(4096)
    if not head:
        return False, {}
    if head.startswith("\ufeff"):
        head = head[1:]
    head = head.lstrip()
    if head[:6].upper() != "#NEXUS":
        return False, {}
    if len(head) > 6 and not (head[6].isspace() or head[6] == "["):
        return False, {}
    return True, {}


# ----------------------------------------------------------------------------
# DATA / CHARACTERS blocks
# ----------------------------------------------------------------------------


def _parse_dimensions(command):
    """Extract integer ``NTAX`` and ``NCHAR`` values from DIMENSIONS."""
    tokens = _strip_eol(command)
    values = {}
    index = 1  # skip DIMENSIONS keyword
    while index < len(tokens):
        token = tokens[index]
        if token[0] in ("w", "q") and token[1].upper() in ("NTAX", "NCHAR"):
            key = token[1].upper()
            if (
                index + 2 < len(tokens)
                and tokens[index + 1] == ("p", "=")
                and tokens[index + 2][0] in ("w", "q")
            ):
                try:
                    values[key] = int(tokens[index + 2][1])
                except ValueError:
                    raise NexusFormatError(
                        "%s must be an integer, found %r." % (key, tokens[index + 2][1])
                    )
                index += 3
                continue
        index += 1
    return values


def _parse_format(command):
    """Extract the relevant settings of a FORMAT command."""
    tokens = _strip_eol(command)
    settings = {"INTERLEAVE": False}
    index = 1  # skip FORMAT keyword
    while index < len(tokens):
        token = tokens[index]
        if token[0] in ("w", "q"):
            key = token[1].upper()
            if key in ("DATATYPE", "MISSING", "GAP", "MATCHCHAR"):
                if (
                    index + 2 < len(tokens)
                    and tokens[index + 1] == ("p", "=")
                    and tokens[index + 2][0] in ("w", "q")
                ):
                    settings[key] = tokens[index + 2][1]
                    index += 3
                    continue
            elif key == "INTERLEAVE":
                settings["INTERLEAVE"] = True
            elif key == "NOINTERLEAVE":
                settings["INTERLEAVE"] = False
        index += 1
    return settings


def _matrix_lines(command):
    """Split a MATRIX command into lists of per-line word tokens.

    Only word and quoted-word tokens are retained; each returned line begins
    with a label token and is followed by zero or more sequence-run tokens.

    """
    lines = []
    line = []
    # Locate the MATRIX keyword (leading line endings may precede it); only
    # tokens after it describe the matrix itself.
    start = next(i for i, token in enumerate(command) if token[0] in ("w", "q"))
    for token in command[start + 1 :]:
        if token[0] == "e":
            if line:
                lines.append(line)
            line = []
        elif token[0] in ("w", "q"):
            line.append(token)
    if line:
        lines.append(line)
    return lines


def _assemble_sequential_matrix(lines, nchar):
    """Assemble records for a non-interleaved matrix given ``NCHAR``.

    After a label, sequence-run tokens are consumed until exactly ``NCHAR``
    characters have been accumulated; the next word token is then the label
    of the next taxon. This supports both matrices where every taxon starts
    on its own line and matrices whose taxa wrap across label-less
    continuation lines.

    """
    records = []
    tokens = [token for line in lines for token in line]
    index = 0
    total = len(tokens)
    while index < total:
        label = _word_value(tokens[index])
        index += 1
        characters = []
        while index < total and len(characters) < nchar:
            characters.extend(_word_value(tokens[index]))
            index += 1
        if len(characters) > nchar:
            raise NexusFormatError(
                "Taxon %r has more than the %d characters declared by "
                "NCHAR." % (label, nchar)
            )
        records.append([label, characters])
    return records


def _assemble_interleaved_matrix(lines):
    """Assemble records by concatenating interleaved matrix segments."""
    order = []
    segments = {}
    for tokens in lines:
        if not tokens:
            continue
        label = _word_value(tokens[0])
        segment = "".join(_word_value(t) for t in tokens[1:])
        if label not in segments:
            segments[label] = []
            order.append(label)
        segments[label].append(segment)
    return [[label, list("".join(segments[label]))] for label in order]


def _map_characters(records, constructor, gap, missing_symbol, matchchar):
    """Translate NEXUS symbols into characters valid for ``constructor``.

    Gap symbols become ``-``, missing symbols become ``N`` (DNA/RNA) or
    ``X`` (protein), and match symbols expand to the corresponding character
    of the first taxon. Returns a list of ``(label, character-string)``.

    """
    if constructor in (DNA, RNA):
        missing_character = "N"
    else:
        missing_character = "X"

    def convert(raw, reference):
        result = []
        for character in raw:
            if character == gap:
                result.append("-")
            elif character == missing_symbol:
                result.append(missing_character)
            elif character == matchchar:
                if reference is None:
                    raise NexusFormatError(
                        "The first taxon cannot contain the match character "
                        "%r because there is no earlier taxon to match." % matchchar
                    )
                if len(result) >= len(reference):
                    raise NexusFormatError(
                        "Match character %r appears beyond the length of the "
                        "first taxon." % matchchar
                    )
                result.append(reference[len(result)])
            else:
                result.append(character)
        return result

    if not records:
        raise NexusFormatError("The MATRIX command contains no taxa.")

    first_label, first_raw = records[0]
    first_converted = convert(first_raw, None)
    converted = [(first_label, first_converted)]
    for label, raw in records[1:]:
        converted.append((label, convert(raw, first_converted)))
    return converted


def _find_matrix_block(blocks):
    """Return the commands of the first DATA/CHARACTERS block, if any."""
    for name, commands in blocks:
        if name in ("DATA", "CHARACTERS"):
            return commands
    return None


def _read_matrix(blocks, constructor):
    """Read the first DATA/CHARACTERS block into raw MSA components."""
    commands = _find_matrix_block(blocks)
    if commands is None:
        raise NexusFormatError(
            "No DATA or CHARACTERS block was found in the NEXUS file."
        )

    dimensions = {}
    settings = {}
    matrix_command = None
    for command in commands:
        keyword = _command_keyword(command)
        if keyword == "DIMENSIONS":
            dimensions.update(_parse_dimensions(command))
        elif keyword == "FORMAT":
            settings.update(_parse_format(command))
        elif keyword == "MATRIX":
            if matrix_command is None:
                matrix_command = command

    if matrix_command is None:
        raise NexusFormatError(
            "The DATA/CHARACTERS block does not contain a MATRIX command."
        )

    nchar = dimensions.get("NCHAR")
    if nchar is None:
        raise NexusFormatError(
            "DIMENSIONS must declare NCHAR in a DATA/CHARACTERS block."
        )
    if nchar < 1:
        raise NexusFormatError("NCHAR must be a positive integer.")
    ntax = dimensions.get("NTAX")
    if ntax is not None and ntax < 1:
        raise NexusFormatError("NTAX must be a positive integer.")

    if constructor is None:
        datatype = settings.get("DATATYPE")
        if datatype is None:
            raise NexusFormatError(
                "FORMAT must declare DATATYPE, or `constructor` must be provided."
            )
        constructor = _DATATYPE_CONSTRUCTORS.get(datatype.upper())
        if constructor is None:
            raise NexusFormatError(
                "Unsupported DATATYPE %r. Supported DATATYPEs are DNA, RNA, "
                "and PROTEIN." % datatype
            )

    gap = settings.get("GAP", _DEFAULT_GAP)
    missing_symbol = settings.get("MISSING", _DEFAULT_MISSING)
    # MATCHCHAR is only honored when it is declared: the writer emits literal
    # '.' gap characters without a MATCHCHAR declaration, and those must round
    # trip unchanged rather than being expanded against the first taxon.
    matchchar = settings.get("MATCHCHAR")

    lines = _matrix_lines(matrix_command)
    if settings.get("INTERLEAVE", False):
        records = _assemble_interleaved_matrix(lines)
    else:
        records = _assemble_sequential_matrix(lines, nchar)

    converted = _map_characters(records, constructor, gap, missing_symbol, matchchar)

    if ntax is not None and len(converted) != ntax:
        raise NexusFormatError(
            "The MATRIX contains %d taxa but NTAX declares %d." % (len(converted), ntax)
        )
    for label, characters in converted:
        if len(characters) != nchar:
            raise NexusFormatError(
                "Taxon %r contains %d characters but NCHAR declares %d."
                % (label, len(characters), nchar)
            )

    labels = [label for label, _ in converted]
    sequences = []
    for label, characters in converted:
        try:
            sequences.append(constructor("".join(characters), metadata={"id": label}))
        except ValueError as error:
            raise NexusFormatError(
                "Could not read taxon %r as %s: %s"
                % (label, constructor.__name__, error)
            ) from error
    return sequences, labels


@nexus.reader(TabularMSA)
def _nexus_to_tabular_msa(fh, cls=None, constructor=None):
    if cls is None:
        cls = TabularMSA
    blocks = _parse_blocks(fh.read())
    sequences, labels = _read_matrix(blocks, constructor)
    return cls(sequences, index=labels)


@nexus.writer(TabularMSA)
def _tabular_msa_to_nexus(obj, fh):
    sequence_count = obj.shape.sequence
    position_count = obj.shape.position
    if sequence_count < 1:
        raise NexusFormatError(
            "TabularMSA can only be written in NEXUS format if it contains "
            "at least one sequence."
        )
    if position_count < 1:
        raise NexusFormatError(
            "TabularMSA can only be written in NEXUS format if it contains "
            "at least one position."
        )

    datatype = None
    for sequence_type, name in (
        (DNA, "DNA"),
        (RNA, "RNA"),
        (Protein, "PROTEIN"),
    ):
        if isinstance(obj[0], sequence_type):
            datatype = name
            break
    if datatype is None:
        raise NexusFormatError(
            "TabularMSA can only be written in NEXUS format when its "
            "sequences are DNA, RNA, or Protein sequences."
        )

    fh.write("#NEXUS\n")
    fh.write("BEGIN DATA;\n")
    fh.write("    DIMENSIONS NTAX=%d NCHAR=%d;\n" % (sequence_count, position_count))
    fh.write(
        "    FORMAT DATATYPE=%s MISSING=%s GAP=%s;\n"
        % (datatype, _DEFAULT_MISSING, _DEFAULT_GAP)
    )
    fh.write("MATRIX\n")
    for label, sequence in zip(obj.index, obj):
        fh.write(
            "    %s %s\n"
            % (
                _quote_label(str(label), _MSA_LABEL_SPECIAL),
                str(sequence),
            )
        )
    fh.write("    ;\n")
    fh.write("END;\n")


# Characters that force an MSA label to be single-quoted on write.
_MSA_LABEL_SPECIAL = set(" ';,")


def _label_needs_quotes(label, special):
    if label == "":
        return True
    for character in label:
        if character in special or character.isspace():
            return True
    return False


def _quote_label(label, special):
    """Single-quote a NEXUS word when it contains special characters.

    An apostrophe is escaped by doubling it.

    """
    if _label_needs_quotes(label, special):
        return "'" + label.replace("'", "''") + "'"
    return label


# ----------------------------------------------------------------------------
# TREES blocks
# ----------------------------------------------------------------------------


def _parse_translate(command):
    """Build the key-to-name mapping of a TRANSLATE command."""
    tokens = [t for t in _strip_eol(command) if t[0] in ("w", "q")]
    translation = {}
    index = 1  # skip TRANSLATE keyword
    while index + 1 < len(tokens):
        key = _word_value(tokens[index])
        value = _word_value(tokens[index + 1])
        translation[key] = value
        index += 2
    return translation


def _find_tree_tokens(commands):
    """Return the Newick token body of the first TREE command, if any."""
    for command in commands:
        if _command_keyword(command) != "TREE":
            continue
        tokens = _strip_eol(command)
        index = 1  # skip TREE keyword
        # An optional leading asterisk marks a preferred tree.
        if index < len(tokens) and tokens[index][0] == "w" and tokens[index][1] == "*":
            index += 1
        # Skip the tree name.
        if index < len(tokens) and tokens[index][0] in ("w", "q"):
            index += 1
        # Skip the equals sign.
        if index < len(tokens) and tokens[index] == ("p", "="):
            index += 1
        return tokens[index:]
    return None


def _build_tree(tokens, translation, cls):
    """Build a tree from Newick tokens applying leaf-position translation."""
    root = cls()
    # Stack entries are (node, depth).
    stack = [(root, 0)]
    depth = 0
    # Track the role of the immediately preceding token so labels are only
    # translated at leaf positions (after '(' or ',') and numbers after ':'
    # are interpreted as branch lengths.
    preceding = None

    for token in tokens:
        kind, value = token
        if kind == "p" and value in "(),:":
            if value == "(":
                depth += 1
                stack.append((cls(), depth))
            elif value == ",":
                stack.append((cls(), depth))
            elif value == ")":
                if len(stack) < 2:
                    raise NexusFormatError(
                        "Unbalanced parentheses in NEXUS tree description."
                    )
                children = []
                while stack[-1][1] == depth:
                    node, _ = stack.pop()
                    children.insert(0, node)
                parent = stack[-1][0]
                if parent.children:
                    raise NexusFormatError(
                        "Node in NEXUS tree has two sets of children."
                    )
                for child in children:
                    child.parent = parent
                parent.children = children
                depth -= 1
            preceding = value
        elif kind in ("w", "q"):
            node = stack[-1][0]
            if preceding == ":":
                try:
                    node.length = float(value)
                except ValueError:
                    raise NexusFormatError(
                        "Could not read branch length as numeric: %r." % value
                    )
            else:
                # Leaf labels follow '(' or ','; internal and root labels
                # follow ')'. Only leaf labels are translated.
                if preceding in ("(", ",") and value in translation:
                    value = translation[value]
                node.name = value if value else None
            preceding = "word"

    if depth != 0 or len(stack) != 1:
        raise NexusFormatError(
            "Could not parse the NEXUS tree description: parentheses or a "
            "terminating semicolon may be missing."
        )
    return root


@nexus.reader(TreeNode)
def _nexus_to_tree_node(fh, cls=None):
    if cls is None:
        cls = TreeNode
    blocks = _parse_blocks(fh.read())

    for name, commands in blocks:
        if name != "TREES":
            continue
        translation = {}
        for command in commands:
            if _command_keyword(command) == "TRANSLATE":
                translation.update(_parse_translate(command))
        tree_tokens = _find_tree_tokens(commands)
        if tree_tokens is not None:
            return _build_tree(tree_tokens, translation, cls)

    raise NexusFormatError(
        "No TREE statement was found in a TREES block of the NEXUS file."
    )


# Labels inside Newick tree descriptions must additionally quote characters
# that structure the tree description itself.
_TREE_LABEL_SPECIAL = set(" ';,:=()[]*")


def _write_tree_newick(node, fh):
    """Write one TreeNode as a Newick description using NEXUS quoting."""
    current_depth = 0
    nodes_left = [(node, 0)]
    while nodes_left:
        entry = nodes_left.pop()
        current_node, node_depth = entry
        if current_node.children and node_depth >= current_depth:
            fh.write("(")
            nodes_left.append(entry)
            nodes_left += (
                (child, node_depth + 1) for child in reversed(current_node.children)
            )
            current_depth = node_depth + 1
        else:
            if node_depth < current_depth:
                fh.write(")")
                current_depth -= 1

            label = current_node._node_label()
            if label:
                fh.write(_quote_label("%s" % label, _TREE_LABEL_SPECIAL))
            if current_node.length is not None:
                fh.write(":")
                fh.write("%s" % current_node.length)
            if nodes_left and nodes_left[-1][1] == current_depth:
                fh.write(",")

    fh.write(";")


@nexus.writer(TreeNode)
def _tree_node_to_nexus(obj, fh):
    fh.write("#NEXUS\n")
    fh.write("BEGIN TREES;\n")
    fh.write("    TREE tree = ")
    _write_tree_newick(obj, fh)
    fh.write("\nEND;\n")
