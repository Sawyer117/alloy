#!/usr/bin/env bash
# alloy 假 CI runner —— pytest 各 group,汇总 PASS/FAIL,任一组失败则非零退出。
#
# Groups (跑 test_*.py;compare_*.py / debug_*.py / diagnose_*.py 不收集):
#   - infra/        : cross-model 烟雾测试 (construct, impl_registry, bridge core)
#   - roofline      : alloy/roofline/tests/* (perf modeling 独立摊)
#   - qwen3         : qwen3 dense contract-1 (alloy_torch ≡ HF_torch, fp32 byte-exact)
#   - qwen3_5_moe   : qwen3.5 contract-1 + bridge wiring + (NPU) contract-2
#   - deepseek_v4   : DSV4 contract-1 + bridge wiring + (NPU) contract-2
#
# Missing-NPU / missing-binder / transformers<5.7 tests skip cleanly,不算 FAIL。
#
# Usage:
#   bash alloy/tests/run_all.sh            # 跑全部 5 组
#   bash alloy/tests/run_all.sh infra      # 只跑某组
#   bash alloy/tests/run_all.sh -v         # 加 verbose

set -uo pipefail

ALLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ALLOY_ROOT"

# group_name : pytest_path
declare -a GROUP_NAMES=("infra" "roofline" "qwen3" "qwen3_5_moe" "deepseek_v4")
declare -a GROUP_PATHS=(
    "tests/infra/"
    "roofline/tests/"
    "tests/models/qwen3/"
    "tests/models/qwen3_5_moe/"
    "tests/models/deepseek_v4/"
)

# Filter to single group if user passed one as first arg
filter=""
extra=""
for arg in "$@"; do
    case "$arg" in
        -v|-vv|-vvv|-q|-x|--tb=*) extra="$extra $arg" ;;
        *) filter="$arg" ;;
    esac
done

total_fail=0
declare -a SUMMARY=()

for i in "${!GROUP_NAMES[@]}"; do
    name="${GROUP_NAMES[$i]}"
    path="${GROUP_PATHS[$i]}"
    if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
        continue
    fi
    echo ""
    echo "============================================================"
    echo "GROUP: $name  ($path)"
    echo "============================================================"
    if python -m pytest "$path" --tb=short $extra; then
        SUMMARY+=("PASS  $name")
    else
        SUMMARY+=("FAIL  $name  ($path)")
        total_fail=$((total_fail + 1))
    fi
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
printf '%s\n' "${SUMMARY[@]}"
echo ""

if [ "$total_fail" -gt 0 ]; then
    echo ">>> $total_fail group(s) FAILED <<<"
    echo "Re-run a single group for details:  bash alloy/tests/run_all.sh <group-name>"
    exit 1
fi

echo "All groups passed."
exit 0
