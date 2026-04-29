#!/usr/bin/env bash
# Run the alloy-vs-HF training-step comparison across the full validation grid.
#
# Five configurations sweep two axes: binder ON/OFF × attn eager/sdpa, plus
# the bf16 production-realistic last row.
#
# Each row writes a full log to logs/<timestamp>/<row>.log and a one-line
# summary to logs/<timestamp>/summary.txt.
#
# Run from anywhere — paths are resolved relative to this script. Override
# the small-model dimensions via env vars (NUM_LAYERS, HIDDEN_SIZE, etc.)
# if you want to test a larger config.
#
# Usage:
#   bash tests/npu/run_compare_grid.sh                 # default small config, all 5 rows
#   NUM_LAYERS=8 SEQ_LEN=128 bash tests/npu/run_compare_grid.sh
#   ROWS="0 4" bash tests/npu/run_compare_grid.sh      # only rows 0 and 4
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root and prepare a per-run log directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Small-model dimensions (override via env vars)
# ---------------------------------------------------------------------------
: "${NUM_LAYERS:=4}"
: "${HIDDEN_SIZE:=64}"
: "${NUM_ATTENTION_HEADS:=2}"
: "${NUM_KEY_VALUE_HEADS:=1}"
: "${NUM_EXPERTS:=4}"
: "${NUM_EXPERTS_PER_TOK:=2}"
: "${VOCAB_SIZE:=128}"
: "${SEQ_LEN:=32}"
: "${BATCH_SIZE:=1}"
: "${SEED:=0}"
: "${TOP_GRAD_DIFFS:=10}"

# Subset of rows to run (default: all 5)
: "${ROWS:=0 1 2 3 4}"

COMMON_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --num-key-value-heads "${NUM_KEY_VALUE_HEADS}"
    --num-experts "${NUM_EXPERTS}"
    --num-experts-per-tok "${NUM_EXPERTS_PER_TOK}"
    --vocab-size "${VOCAB_SIZE}"
    --seq-len "${SEQ_LEN}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --top-grad-diffs "${TOP_GRAD_DIFFS}"
)

# ---------------------------------------------------------------------------
# Row matrix
#   id  prefer  attn-impl  dtype  description
#
# fp32 rows are for byte-exact verification (no binder — the triton chunk
# kernels reject fp32 by design, so binder rows must be bf16).
# bf16 rows are layered: noise floor → +binder GDN/experts → +sdpa attn.
# ---------------------------------------------------------------------------
declare -A ROW_PREFER=(
    [0]="torch"  [1]="torch"  [2]="torch"  [3]="flash"  [4]="flash"
)
declare -A ROW_ATTN=(
    [0]="eager"  [1]="sdpa"   [2]="eager"  [3]="eager"  [4]="sdpa"
)
declare -A ROW_DTYPE=(
    [0]="fp32"   [1]="fp32"   [2]="bf16"   [3]="bf16"   [4]="bf16"
)
declare -A ROW_DESC=(
    [0]="byte-exact baseline   (fp32 + eager attn + no binder)"
    [1]="sdpa kernel parity    (fp32 + sdpa attn, NPU = npu_fused_attention both sides)"
    [2]="bf16 noise floor      (bf16 + eager attn + no binder — pure bf16 cost)"
    [3]="bf16 + binder GDN/MoE (bf16 + eager attn + binder; GDN drift expected on HF side)"
    [4]="bf16 + all-fast       (bf16 + sdpa attn + binder; production-realistic)"
)

# ---------------------------------------------------------------------------
# Run one row
# ---------------------------------------------------------------------------
SUMMARY="${LOG_DIR}/summary.txt"
{
    echo "alloy-vs-HF compare grid"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "config: layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} heads=${NUM_ATTENTION_HEADS}/${NUM_KEY_VALUE_HEADS} experts=${NUM_EXPERTS}/${NUM_EXPERTS_PER_TOK} vocab=${VOCAB_SIZE} seq=${SEQ_LEN} batch=${BATCH_SIZE} seed=${SEED}"
    echo "log dir: ${LOG_DIR}"
    echo
    printf "%-3s  %-7s  %-9s  %-5s  %-12s  %-12s  %-12s  %-12s  %-12s  %s\n" \
        "row" "prefer" "attn-impl" "dtype" \
        "logits_max" "logits_mean" "loss_diff" "grad_max" "grad_mean" "first divergence"
    echo "------------------------------------------------------------------------------------------------------------------------------"
} | tee "${SUMMARY}"

# ---------------------------------------------------------------------------
# Metric extractors — read a row's log, return one number / token, or "-"
# ---------------------------------------------------------------------------
_extract_logit_max() {
    awk '
        /^=== Forward logits/ { in_block=1; next }
        /^===/ && in_block { exit }
        in_block && /max_abs/ { print $2; exit }
    ' "$1"
}

_extract_logit_mean() {
    awk '
        /^=== Forward logits/ { in_block=1; next }
        /^===/ && in_block { exit }
        in_block && /mean_abs/ { print $2; exit }
    ' "$1"
}

_extract_loss_diff() {
    grep -E "^Loss:" "$1" | sed -nE 's/.*abs_diff=([0-9.eE+-]+).*/\1/p' | head -1
}

_extract_grad_max() {
    grep -E "Aggregate over" "$1" | sed -nE 's/.*max-of-max-abs=([0-9.eE+-]+).*/\1/p' | head -1
}

_extract_grad_mean() {
    grep -E "Aggregate over" "$1" | sed -nE 's/.*mean-of-mean-abs=([0-9.eE+-]+).*/\1/p' | head -1
}

_or_dash() {
    if [[ -z "$1" ]]; then echo "-"; else echo "$1"; fi
}

run_row() {
    local row="$1"
    local prefer="${ROW_PREFER[$row]}"
    local attn="${ROW_ATTN[$row]}"
    local dtype="${ROW_DTYPE[$row]}"
    local desc="${ROW_DESC[$row]}"
    local log="${LOG_DIR}/row_${row}_prefer-${prefer}_attn-${attn}_${dtype}.log"

    echo
    echo "============================================================================"
    echo "[row ${row}] ${desc}"
    echo "          prefer=${prefer} attn-impl=${attn} dtype=${dtype}"
    echo "          log: ${log}"
    echo "============================================================================"

    # ``cd REPO_ROOT`` because ``python -m alloy.tests.*`` requires the alloy
    # package to be importable from cwd (or installed editable, which is the
    # documented setup).
    cd "${REPO_ROOT}"
    set +e
    python -m alloy.tests.npu.compare_training_step_vs_hf \
        "${COMMON_ARGS[@]}" \
        --prefer "${prefer}" \
        --attn-impl "${attn}" \
        --dtype "${dtype}" \
        2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    # Headline: where (if anywhere) drift starts.
    local headline
    if grep -qE "byte-exact at every capture point" "${log}"; then
        headline="byte-exact"
    elif grep -qE "first divergence:" "${log}"; then
        headline="$(grep -oE 'first divergence: [a-z_0-9]+' "${log}" | head -1)"
    elif grep -qE "^SKIP" "${log}"; then
        headline="SKIP"
    else
        headline="rc=${rc}; see log"
    fi

    # Numerical metrics.
    local logits_max  logits_mean  loss_diff  grad_max  grad_mean
    logits_max="$(_or_dash "$(_extract_logit_max "${log}")")"
    logits_mean="$(_or_dash "$(_extract_logit_mean "${log}")")"
    loss_diff="$(_or_dash "$(_extract_loss_diff "${log}")")"
    grad_max="$(_or_dash "$(_extract_grad_max "${log}")")"
    grad_mean="$(_or_dash "$(_extract_grad_mean "${log}")")"

    printf "%-3s  %-7s  %-9s  %-5s  %-12s  %-12s  %-12s  %-12s  %-12s  %s\n" \
        "${row}" "${prefer}" "${attn}" "${dtype}" \
        "${logits_max}" "${logits_mean}" "${loss_diff}" \
        "${grad_max}" "${grad_mean}" \
        "${headline}" \
        | tee -a "${SUMMARY}"
}

for row in ${ROWS}; do
    if [[ -z "${ROW_PREFER[$row]:-}" ]]; then
        echo "WARN: unknown row '${row}', skipping" | tee -a "${SUMMARY}"
        continue
    fi
    run_row "${row}"
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo
echo "============================================================================"
echo "summary at: ${SUMMARY}"
echo "============================================================================"
cat "${SUMMARY}"
