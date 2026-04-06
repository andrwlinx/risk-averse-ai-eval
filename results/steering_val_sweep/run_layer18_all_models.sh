#!/usr/bin/env bash
# Steering validation sweep — layer 18 starting point, all 3 models, all alphas
# Eval repo commit: f8072dcc9cc39c263c29f21c6aa1850d63e90bf7
# Steering repo commit: 520dfe16de4c13c0756e2ed8ce0313d5f0cf1a8e
# Run date: 2026-04-06

set -euo pipefail

EVAL_REPO=/lambda/nfs/activation-engineering/projects/risk-averse-ai-eval
VECTORS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/vectors
EVALS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/evals
LOGS=/lambda/nfs/activation-engineering/projects/pi-eval/results/steering_val_sweep/logs
TRAINING_CSV=$EVAL_REPO/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv
ALPHAS="0.25,0.5,1.0,2.0,3.0,-0.25,-0.5,-1.0,-2.0,-3.0"
LAYER=18

export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

cd "$EVAL_REPO"

for MODEL_ID in Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-14B; do
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|Qwen/||' | tr '[:upper:]' '[:lower:]')
    VECTOR_PATH="$VECTORS/${MODEL_SHORT}_layer${LAYER}_seed1.pt"
    LOG_PREFIX="$LOGS/${MODEL_SHORT}_layer${LAYER}"

    echo "========================================================"
    echo "MODEL: $MODEL_ID"
    echo "========================================================"

    # --- Step 1: Generate steering vector ---
    echo "[$(date)] Generating steering vector for $MODEL_ID at layer $LAYER..."
    python3 generate_steering_vector.py \
        --training_csv "$TRAINING_CSV" \
        --base_model "$MODEL_ID" \
        --layer "$LAYER" \
        --seed 1 \
        --icv_method pca \
        --demo_max_chars 0 \
        --enable_thinking \
        --output "$VECTOR_PATH" \
        2>&1 | tee "${LOG_PREFIX}_seed1_gen.log"
    echo "[$(date)] Vector saved to $VECTOR_PATH"

    # --- Step 2: Eval sweep — layer 18, all alphas ---
    echo "[$(date)] Running eval sweep for $MODEL_ID at layer $LAYER (alphas: $ALPHAS)..."
    python3 evaluate.py \
        --backend transformers \
        --base_model "$MODEL_ID" \
        --dataset medium_stakes_validation \
        --num_situations 200 \
        --eval_layer "$LAYER" \
        --steering_direction_path "$VECTOR_PATH" \
        --alphas "$ALPHAS" \
        --max_new_tokens 4096 \
        --seed 12345 \
        --top_p 0.95 \
        --top_k 20 \
        --output "$EVALS/${MODEL_SHORT}_layer${LAYER}_allalphas.json" \
        2>&1 | tee "${LOG_PREFIX}_allalphas_eval.log"
    echo "[$(date)] Eval done for $MODEL_ID layer $LAYER"
done

echo "========================================================"
echo "All layer-18 sweeps complete."
echo "========================================================"
