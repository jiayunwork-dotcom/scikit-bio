"""Reproducible sparse-diversity resource benchmark.

Builds, in a fresh process, a balanced binary tree with 200,000 distinct tip
names (all non-root branch lengths equal to 1) and a 256-sample count table
with 32 non-negative integer counts per sample, then times a single metric
call and records the peak RSS growth relative to the post-construction
baseline.

The fixture never materializes a dense sample-by-taxon (or sample-by-node)
matrix: counts are generated directly in CSR form, and the tree is built from
node objects rather than a giant Newick string.

Usage
-----
    python -I bench_sparse.py --metric faith_pd --input csr --seed 20261119
    python -I bench_sparse.py --metric weighted_unifrac --input biom \
        --normalized --clustered

RSS measurement
---------------
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is read before and after the
single metric call. On Linux it is reported in kilobytes and covers *every*
allocation attributable to the process -- NumPy/SciPy native buffers
included -- not just Python-tracked objects. Peak growth is
``ru_maxrss(after) - rss(baseline)``; taking the baseline after the fixture
exists means tree/input construction cost is excluded from the call budget,
matching the stated protocol (construct tree, inputs and labels, then measure).
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from skbio import TreeNode
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.table import Table


N_TIPS = 200_000
N_SAMPLES = 256
N_NNZ_PER_ROW = 32
SEED_DEFAULT = 20261119


def rss_mib() -> float:
    # Linux: ru_maxrss is in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_balanced_tree(n_tips: int) -> TreeNode:
    """Build an approximately balanced binary tree without a dense matrix.

    Leaves are created first, then paired level by level. When a level has an
    odd number of nodes, the last node is carried up unchanged, which keeps
    the tree "approximately" balanced while giving exactly ``n_tips`` leaves.
    Every non-root edge gets length 1.0; the root edge stays unset (zero).
    """
    level = [TreeNode(name=f"tip{i:07d}", length=1.0) for i in range(n_tips)]
    while len(level) > 1:
        next_level = []
        n = len(level)
        for i in range(0, n - 1, 2):
            parent = TreeNode(length=1.0)
            parent.append(level[i])
            parent.append(level[i + 1])
            next_level.append(parent)
        if n % 2:
            next_level.append(level[-1])
        level = next_level
    root = level[0]
    root.name = "root"
    root.length = None
    return root


def build_counts(
    rng: np.random.Generator, clustered: bool
) -> sp.csr_matrix:
    """Build a (256, 200000) CSR table with 32 non-zero integer counts/row."""
    rows = np.repeat(np.arange(N_SAMPLES), N_NNZ_PER_ROW)
    if clustered:
        # Every sample draws from the same small clade (tips 0..4095), so
        # ancestor rows are shared heavily between samples.
        pool = 4096
        cols = rng.integers(0, pool, size=rows.size)
    else:
        cols = rng.integers(0, N_TIPS, size=rows.size)
    values = rng.integers(1, 100, size=rows.size).astype(np.int64)
    table = sp.coo_matrix(
        (values, (rows, cols)), shape=(N_SAMPLES, N_TIPS)
    ).tocsr()
    # Duplicate coordinates within a row are legal; merge them so each row
    # still has at most 32 logical non-zero columns.
    table.sum_duplicates()
    return table


def make_input(kind: str, counts: sp.csr_matrix, taxa, sample_ids):
    if kind == "csr":
        # Exercise both SciPy representations across the run set; the caller
        # chooses via --input csr_array when desired.
        return counts
    if kind == "csr_array":
        return sp.csr_array(counts)
    if kind == "biom":
        # BIOM stores observations on rows, samples on columns (CSC).
        return Table(counts.T.tocsc(), taxa, sample_ids)
    raise ValueError(f"unknown input kind {kind!r}")


def run_metric(metric, table, sample_ids, taxa, tree, normalized):
    if metric == "faith_pd":
        return alpha_diversity(
            "faith_pd", table, ids=sample_ids, taxa=taxa, tree=tree
        )
    if metric in ("unweighted_unifrac", "weighted_unifrac"):
        kwargs = {}
        if metric == "weighted_unifrac":
            kwargs["normalized"] = normalized
        return beta_diversity(
            metric, table, ids=sample_ids, taxa=taxa, tree=tree, **kwargs
        )
    raise ValueError(metric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["faith_pd", "unweighted_unifrac", "weighted_unifrac"],
        required=True,
    )
    parser.add_argument("--input", choices=["csr", "csr_array", "biom"],
                        required=True)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--clustered", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- construction phase (excluded from the call's memory budget) ---
    t0 = time.perf_counter()
    tree = build_balanced_tree(N_TIPS)
    taxa = np.array([f"tip{i:07d}" for i in range(N_TIPS)], dtype=object)
    sample_ids = [f"sample{i:04d}" for i in range(N_SAMPLES)]
    counts = build_counts(rng, clustered=args.clustered)
    table = make_input(args.input, counts, taxa, sample_ids)
    build_seconds = time.perf_counter() - t0

    n_taxa = len(taxa)
    n_stored = counts.data.size
    table_bytes = (
        counts.data.nbytes + counts.indices.nbytes + counts.indptr.nbytes
    )

    # Warm-up: a second, smaller but structurally identical run would require
    # rebuilding the tree, which is expensive. Instead warm the metric kernels
    # on the same fixture once (its allocations settle into the baseline),
    # then force garbage collection before the measured call.
    _ = run_metric(
        args.metric, table, sample_ids, taxa, tree, args.normalized
    )
    gc.collect()
    baseline_rss = rss_mib()

    # --- measured single call ---
    t1 = time.perf_counter()
    result = run_metric(
        args.metric, table, sample_ids, taxa, tree, args.normalized
    )
    elapsed = time.perf_counter() - t1
    peak_rss = rss_mib()
    peak_growth = peak_rss - baseline_rss

    if hasattr(result, "data"):
        out_shape = result.data.shape
        out_bytes = result.data.nbytes
    else:
        out_shape = result.values.shape
        out_bytes = result.values.nbytes

    record = {
        "metric": args.metric,
        "normalized": args.normalized,
        "input": args.input,
        "clustered": args.clustered,
        "seed": args.seed,
        "n_tips": N_TIPS,
        "n_samples": N_SAMPLES,
        "max_nnz_per_row": N_NNZ_PER_ROW,
        "stored_nonzeros": int(n_stored),
        "sparse_table_bytes": int(table_bytes),
        "dense_equivalent_bytes": int(
            N_SAMPLES * N_TIPS * np.dtype(np.float64).itemsize
        ),
        "output_shape": list(out_shape),
        "output_bytes": int(out_bytes),
        "build_seconds": round(build_seconds, 4),
        "metric_seconds": round(elapsed, 4),
        "baseline_rss_mib": round(baseline_rss, 2),
        "peak_rss_mib": round(peak_rss, 2),
        "peak_growth_mib": round(peak_growth, 2),
        "within_256_mib": bool(peak_growth <= 256.0),
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
