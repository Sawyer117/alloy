#!/usr/bin/env bash
# Throwaway diagnostic runner. Pull, run, paste output, delete the
# debug/ directory once root cause is fixed.
#
# Usage:  bash debug/run.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo
echo "============================================================"
echo "1) DSV4Config field inspection (root-cause for size mismatch)"
echo "============================================================"
python -m debug.inspect_dsv4_config
