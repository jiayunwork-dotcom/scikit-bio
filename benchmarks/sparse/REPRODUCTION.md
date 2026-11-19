# Reproduction record

Environment: Linux (Debian 12, aarch64), Python 3.11.2, NumPy 2.4.6,
SciPy 1.17.1, pandas 3.1.1, biom-format (installed), numba 0.68.1.

## 1. Environment setup

The system Python ships without pip/venv headers. A virtual environment was
bootstrapped without touching the system interpreter:

```bash
python3 -m venv --without-pip /tmp/venv
/tmp/venv/bin/python get-pip.py            # standard get-pip bootstrap
/tmp/venv/bin/pip install numpy scipy pandas "biom-format>=2.1.16" h5py \
    statsmodels patsy requests decorator array-api-compat pytest cython \
    wheel setuptools numba
# Cython build needs Python.h, extracted from the Debian -dev packages into
# /tmp/debs (no root required):
CFLAGS="-I/tmp/debs/usr/include/python3.11 \
 -I/tmp/debs/usr/include/aarch64-linux-gnu/python3.11 \
 -I/tmp/debs/usr/include" \
    /tmp/venv/bin/pip install -e . --no-build-isolation
```

## 2. Numerical and API regression (small fixtures)

```bash
/tmp/venv/bin/python -m pytest \
    skbio/diversity/tests/ skbio/diversity/alpha/tests/ \
    skbio/diversity/beta/tests/ skbio/table/tests/ skbio/tree/tests/ \
    -q --deselect "skbio/diversity/__init__.py::skbio.diversity"
```

Result: all tests pass (265 in the diversity/table subset; 332 in tree).
The deselected item, and the remaining failures in a full-suite run, are
pre-existing NumPy 2.x print-format doctest failures: they fail identically
on the pristine `HEAD` tree (verified with `git stash`) and touch no changed
module. The new sparse suite alone:

```bash
/tmp/venv/bin/python -m pytest skbio/diversity/tests/test_sparse.py -q
# 35 passed
```

Coverage includes, against the dense pipeline on identical logical tables:

* batch and single-sample calls for Faith's PD, unweighted UniFrac and
  weighted UniFrac (`normalized` True and False), plus `phydiv`;
* `csr_matrix`, `csr_array` and BIOM inputs, including BIOM axis ids and
  explicit `ids`/`taxa` overrides;
* unsorted column indices, repeated coordinates, explicit zeros, empty and
  all-zero rows, zero-length branches, tree tips absent from the table;
* row/column permutation realignment and per-row positive-integer scaling;
* bool/complex/NaN/inf/fractional/negative/overflow rejections, the
  `2**53 - 1` row-total bound, and count checks under `validate=False`;
* duplicate taxa, missing taxa, duplicate tip names, id-count mismatches;
* `engine='numba'` and custom `pairwise_func` rejection before computation;
* caller CSR/BIOM/tree/label immutability after every metric.

## 3. Large-scale resource record (independent processes)

```bash
bash benchmarks/sparse/run_all.sh benchmarks/sparse/results
```

18 runs, each a fresh process, on a 200,000-tip balanced binary tree
(non-root edge length 1), 256 samples, 32 non-zero integer counts per row,
seed 20261119, both dispersed and clustered (small-clade) regimes, across
CSR matrix, CSR array and BIOM. Every run reports `"within_256_mib": true`;
worst observed peak growth over the post-construction baseline was
~13.3 MiB against the 256 MiB budget. Metric call times range ~0.66 s
(Faith PD / unweighted) to ~11 s (weighted, O(n_samples^2) pair walks).
Per-run JSON with baseline/peak RSS, timings and sizes is in
`benchmarks/sparse/results/`; `benchmarks/sparse/summarize.py` tabulates
them and fails if any run exceeds the budget.

## 4. Paths not validated

* Windows/macOS and Python 3.12-3.14 were not run (only Linux/Python 3.11 was
  available); the code uses no platform-specific APIs beyond
  `resource.getrusage`, which is confined to the benchmark script.
* The optional numba sparse path is intentionally not implemented; sparse
  inputs with an explicit `engine='numba'` raise `ValueError`, per scope.
* `block_beta_diversity`, `partial_beta_diversity` and non-phylogenetic
  metrics keep their dense-only behavior and reject CSR with a clear
  `TypeError` rather than densifying silently.
* GPU/array-API non-NumPy backends are out of scope.
