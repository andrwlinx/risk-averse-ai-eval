#!/usr/bin/env bash
# ICV sliding-window sweeps on OOD val set
#
# For each model (1.7B, 8B, 14B):
#   1. Regenerate ICV vector using sliding window
#   2. Evaluate on OOD val set
#
# Usage:
#   bash run_icv_sweeps_ood.sh 2>&1 | tee run_icv_sweeps_ood.log

set -euo pipefail

cd /lambda/nfs/activation-engineering/projects/pi-eval
export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

TRAIN_CSV="data/2026_01_29_lin_only_training_set_CoTs_500_Sonnet_4_5.csv"
VAL_CSV="data/2026_01_29_new_val_set_probabilities_add_to_100.csv"

RUN_TS=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="results/${RUN_TS}_icv_ood"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "ICV Sliding-Window Sweeps (OOD val set) — $(date)"
echo "Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B"
echo "Train: $TRAIN_CSV"
echo "Val:   $VAL_CSV"
echo "Results: $RESULTS_DIR"
echo "============================================================"

# ----------------------------------------------------------------
# Qwen3-1.7B  (28 layers — mid layer 14)
# ----------------------------------------------------------------
MODEL="Qwen/Qwen3-1.7B"
ICV_VEC="qwen3_1.7b_icv_sliding_layer14.pt"
MID_LAYER=14

echo ""
echo ">>> [1/6] Generating ICV vector (sliding window): $MODEL layer $MID_LAYER"
python generate_steering_vector.py \
  --base_model "$MODEL" \
  --training_csv "$TRAIN_CSV" \
  --filter_value lin_only \
  --icv_mode \
  --layer $MID_LAYER \
  --num_icv_demos 4 \
  --icv_position last \
  --output "$ICV_VEC"

echo ""
echo ">>> [2/6] ICV eval on OOD val set: $MODEL"
python run_icv_eval.py \
  --base_model "$MODEL" \
  --steering_path "$ICV_VEC" \
  --val_csv "$VAL_CSV" \
  --filter_bucket_label lin_only \
  --num_situations 50 \
  --results_dir "$RESULTS_DIR"

# ----------------------------------------------------------------
# Qwen3-8B  (36 layers — mid layer 18)
# ----------------------------------------------------------------
MODEL="Qwen/Qwen3-8B"
ICV_VEC="qwen3_8b_icv_sliding_layer18.pt"
MID_LAYER=18

echo ""
echo ">>> [3/6] Generating ICV vector (sliding window): $MODEL layer $MID_LAYER"
python generate_steering_vector.py \
  --base_model "$MODEL" \
  --training_csv "$TRAIN_CSV" \
  --filter_value lin_only \
  --icv_mode \
  --layer $MID_LAYER \
  --num_icv_demos 4 \
  --icv_position last \
  --output "$ICV_VEC"

echo ""
echo ">>> [4/6] ICV eval on OOD val set: $MODEL"
python run_icv_eval.py \
  --base_model "$MODEL" \
  --steering_path "$ICV_VEC" \
  --val_csv "$VAL_CSV" \
  --filter_bucket_label lin_only \
  --num_situations 50 \
  --results_dir "$RESULTS_DIR"

# ----------------------------------------------------------------
# Qwen3-14B  (40 layers — mid layer 20)
# ----------------------------------------------------------------
MODEL="Qwen/Qwen3-14B"
ICV_VEC="qwen3_14b_icv_sliding_layer20.pt"
MID_LAYER=20

echo ""
echo ">>> [5/6] Generating ICV vector (sliding window): $MODEL layer $MID_LAYER"
python generate_steering_vector.py \
  --base_model "$MODEL" \
  --training_csv "$TRAIN_CSV" \
  --filter_value lin_only \
  --icv_mode \
  --layer $MID_LAYER \
  --num_icv_demos 4 \
  --icv_position last \
  --output "$ICV_VEC"

echo ""
echo ">>> [6/6] ICV eval on OOD val set: $MODEL"
python run_icv_eval.py \
  --base_model "$MODEL" \
  --steering_path "$ICV_VEC" \
  --val_csv "$VAL_CSV" \
  --filter_bucket_label lin_only \
  --num_situations 50 \
  --results_dir "$RESULTS_DIR"

# ----------------------------------------------------------------
# Generate plots
# ----------------------------------------------------------------
echo ""
echo ">>> Generating plots..."
python plot_icv_results.py --results_dir "$RESULTS_DIR" --metric linear_rate

# ----------------------------------------------------------------
# Push results to GitHub
# ----------------------------------------------------------------
echo ""
echo ">>> Pushing results to GitHub..."
git add "$RESULTS_DIR/" \
        run_icv_sweeps_ood.sh \
        plot_icv_results.py \
        generate_steering_vector.py \
        evaluate.py \
        run_icv_eval.py
git commit -m "$(cat <<'EOF'
Add ICV sliding-window OOD sweep results

- Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B
- Sliding-window ICV (num_icv_demos=4), OOD val set
- Includes per-model linear_rate sweep plots

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin master

echo ""
echo "============================================================"
echo "All done — $(date)"
echo "Results in: $RESULTS_DIR"
echo "============================================================"
