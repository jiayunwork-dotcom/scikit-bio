"""Summarize the JSON benchmark records produced by ``bench_sparse.py``.

Prints a compact table and fails (exit 1) if any run exceeded the 256 MiB
peak-growth budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BUDGET_MIB = 256.0


def main(directory: str):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        rows.append(json.loads(path.read_text()))

    header = (
        f"{'run':34s} {'metric':18s} {'in':9s} {'secs':>7s} "
        f"{'base MiB':>10s} {'peak MiB':>10s} {'growth':>8s} ok"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    for path, r in zip(sorted(Path(directory).glob("*.json")), rows):
        name = path.stem
        within = r["peak_growth_mib"] <= BUDGET_MIB
        all_ok &= within
        print(
            f"{name:34s} {r['metric']:18s} {r['input']:9s} "
            f"{r['metric_seconds']:7.3f} {r['baseline_rss_mib']:10.1f} "
            f"{r['peak_rss_mib']:10.1f} {r['peak_growth_mib']:7.1f}M "
            f"{'YES' if within else 'NO'}"
        )
    print()
    dense_gib = rows[0]["dense_equivalent_bytes"] / (1024**3) if rows else 0.0
    print(f"Dense sample-by-taxon table would require {dense_gib:.3f} GiB")
    print(f"Peak-growth budget: {BUDGET_MIB:.0f} MiB")
    if not all_ok:
        print("AT LEAST ONE RUN EXCEEDED THE BUDGET")
        sys.exit(1)
    print("All runs within budget.")


if __name__ == "__main__":
    main(sys.argv[1])
