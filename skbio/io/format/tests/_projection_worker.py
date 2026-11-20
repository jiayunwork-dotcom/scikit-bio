"""Worker for projection round-trip tests in a fresh process.

Used by ``ProjectionStateFreshProcessTests`` in
``skbio/io/format/tests/test_ordination.py`` to prove that an ordination
file carrying a projection state is sufficient on its own: a new process
that only sees the file (and a distances table) can reproduce the
projected coordinates and diagnostics computed in the parent process.

Usage::

    python _projection_worker.py ORDINATION_FILE DISTANCES_CSV OUT_FILE

``DISTANCES_CSV`` is a CSV table whose first column holds the
supplementary sample IDs and whose header holds the reference sample IDs;
empty cells mark unobserved pairs. The projected coordinates and
diagnostics are written to ``OUT_FILE`` as two CSV blocks (coordinates,
blank line, diagnostics). If the ordination file carries no projection
state, the worker prints the error and exits with status 3.
"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import sys

import pandas as pd

from skbio import OrdinationResults


def main(argv):
    ordination_fp, distances_fp, out_fp = argv[1], argv[2], argv[3]
    ord_res = OrdinationResults.read(ordination_fp)
    distances = pd.read_csv(distances_fp, index_col=0)
    try:
        result = ord_res.project(distances)
    except ValueError as err:
        print("PROJECT_ERROR: %s" % err)
        return 3
    with open(out_fp, "w") as fh:
        result.coordinates.to_csv(fh)
        fh.write("\n")
        result.diagnostics.to_csv(fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
