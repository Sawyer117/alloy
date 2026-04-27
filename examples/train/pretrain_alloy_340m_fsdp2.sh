#!/bin/bash
# Launch FSDP2 pretraining of alloy_340m_dense via MindSpeed-LLM.
#
# Edit the three paths below, then:
#   bash pretrain_alloy_340m_fsdp2.sh
#
# Prerequisites:
#   - alloy is pip-installed in the active Python env
#         (pip install git+https://github.com/Sawyer117/alloy.git)
#   - the model directory referenced by the yaml's model_name_or_path
#     contains config.json + modeling_alloy.py + tokenizer files
#   - MindSpeed-LLM repo cloned (any version that ships train_fsdp2.py)
#
# What this script does:
#   1. cd into MindSpeed-LLM (the entry point train_fsdp2.py lives there)
#   2. source MindSpeed's NPU env_config.sh
#   3. torchrun -> train_fsdp2.py <our yaml>

set -euo pipefail

# ============================================================================
#  USER CONFIGURATION  -- edit these three
# ============================================================================
MINDSPEED_LLM_DIR="/path/to/MindSpeed-LLM"
ALLOY_TRAIN_YAML="/path/to/alloy/examples/train/pretrain_alloy_340m_fsdp2.yaml"
NPUS_PER_NODE=8

# ============================================================================
#  Cluster topology -- adjust if multi-node
# ============================================================================
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

# ============================================================================
#  NPU runtime tunables (mirror MindSpeed-LLM's qwen3_next pretrain script)
# ============================================================================
export HCCL_CONNECT_TIMEOUT=3600
export NPU_ASD_ENABLE=0

cd "$MINDSPEED_LLM_DIR"

# Pulls in the rest of the recommended NPU env (CPU_AFFINITY_CONF,
# TASK_QUEUE_ENABLE, STREAMS_PER_DEVICE, PYTORCH_NPU_ALLOC_CONF, etc.).
# shellcheck disable=SC1091
source examples/fsdp2/env_config.sh

DISTRIBUTED_ARGS=(
    --nproc_per_node "$NPUS_PER_NODE"
    --nnodes         "$NNODES"
    --node_rank      "$NODE_RANK"
    --master_addr    "$MASTER_ADDR"
    --master_port    "$MASTER_PORT"
)

mkdir -p logs
LOG_NAME="logs/pretrain_alloy_340m_$(date +%m%d_%H%M).log"

echo "========================================================================"
echo "  alloy_340m_dense pretraining via MindSpeed-LLM FSDP2"
echo "  cwd          : $(pwd)"
echo "  yaml         : $ALLOY_TRAIN_YAML"
echo "  world size   : $((NPUS_PER_NODE * NNODES))"
echo "  log          : $LOG_NAME"
echo "========================================================================"

torchrun "${DISTRIBUTED_ARGS[@]}" train_fsdp2.py "$ALLOY_TRAIN_YAML" \
    | tee "$LOG_NAME"
