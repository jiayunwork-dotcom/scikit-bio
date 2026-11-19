#!/usr/bin/env bash
# Run the full sparse-diversity benchmark matrix. Every invocation is a fresh
# process (as required), so RSS baselines never carry over between runs.
#
# Usage: bash run_all.sh [output_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/tmp/venv/bin/python}"
OUT="${1:-${HERE}/results}"
mkdir -p "${OUT}"

SEED=20261119

run() {
    local name="$1"; shift
    local log="${OUT}/${name}.json"
    echo "=== ${name} ==="
    "${PY}" -I "${HERE}/bench_sparse.py" --out "${log}" --seed "${SEED}" "$@"
}

# Two sparsity regimes for the CSR path: dispersed and clustered samples.
for regime in "" "--clustered"; do
    tag="dispersed"
    if [[ -n "${regime}" ]]; then tag="clustered"; fi

    run "faith_csr_${tag}"        --metric faith_pd           --input csr       ${regime}
    run "unweighted_csr_${tag}"   --metric unweighted_unifrac --input csr       ${regime}
    run "weighted_csr_${tag}"     --metric weighted_unifrac   --input csr       ${regime}
    run "weightednorm_csr_${tag}" --metric weighted_unifrac   --input csr --normalized ${regime}
done

# The second sparse representation and the BIOM path, on the dispersed regime
# (the clustered regime exercises the same code once coverage is established).
run "faith_csrarray"        --metric faith_pd           --input csr_array
run "unweighted_csrarray"   --metric unweighted_unifrac --input csr_array
run "weighted_csrarray"     --metric weighted_unifrac   --input csr_array
run "weightednorm_csrarray" --metric weighted_unifrac   --input csr_array --normalized

run "faith_biom"        --metric faith_pd           --input biom
run "unweighted_biom"   --metric unweighted_unifrac --input biom
run "weighted_biom"     --metric weighted_unifrac   --input biom
run "weightednorm_biom" --metric weighted_unifrac   --input biom --normalized

# Clustered regime also through the BIOM path, per the protocol.
run "unweighted_biom_clustered" --metric unweighted_unifrac --input biom --clustered
run "weighted_biom_clustered"   --metric weighted_unifrac   --input biom --clustered

"${PY}" -I "${HERE}/summarize.py" "${OUT}"
