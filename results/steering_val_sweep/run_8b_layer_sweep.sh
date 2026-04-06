#!/usr/bin/env bash
# 8B layer sweep — layers {12, 18, 24} at starting alpha 1.0.
# After results are in, pick the best layer and run a targeted alpha search there.
# Eval repo commit: f8072dcc9cc39c263c29f21c6aa1850d63e90bf7
# Steering repo commit: 520dfe16de4c13c0756e2ed8ce0313d5f0cf1a8e
# Run date: 2026-04-06

set -euo pipefail

EVAL_REPO=/lambda/nfs/activation-engineering/projects/risk-averse-ai-eval
VECTORS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/vectors
EVALS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/evals
LOGS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/logs
VECTOR_PATH="$VECTORS/qwen3-8b_layer18_seed1.pt"
ALPHA=1.0

export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

cd "$EVAL_REPO"

for LAYER in 12 18 24; do
    echo "========================================================"
    echo "8B layer $LAYER, alpha $ALPHA"
    echo "========================================================"
    python3 evaluate.py \
        --backend transformers \
        --base_model Qwen/Qwen3-8B \
        --dataset medium_stakes_validation \
        --num_situations 200 \
        --eval_layer "$LAYER" \
        --steering_direction_path "$VECTOR_PATH" \
        --alphas "$ALPHA" \
        --max_new_tokens 4096 \
        --seed 12345 \
        --top_p 0.95 \
        --top_k 20 \
        --output "$EVALS/qwen3-8b_layer${LAYER}_alpha${ALPHA}.json" \
        2>&1 | tee "$LOGS/qwen3-8b_layer${LAYER}_alpha${ALPHA}_eval.log"
    echo "[$(date)] Done: 8B layer $LAYER alpha $ALPHA"
done

echo "========================================================"
echo "8B layer sweep complete. Review results then run alpha search at best layer."
echo "========================================================"
