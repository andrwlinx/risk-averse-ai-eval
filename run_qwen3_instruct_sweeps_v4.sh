#!/usr/bin/env bash
# Qwen3 instruct steering sweeps v4
#
# Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B
# Changes vs v3:
#   - max_new_tokens=4096 (was 1024)
#
# Usage:
#   bash run_qwen3_instruct_sweeps_v4.sh 2>&1 | tee run_qwen3_instruct_sweeps_v4.log

set -euo pipefail

cd /lambda/nfs/activation-engineering/projects/pi-eval
export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

TRAIN_CSV="data/2026_01_29_new_full_training_set_with_CoTs_Sonnet_4_5.csv"
VAL_CSV="data/in_distribution_val_set.csv"

# Dated results folder
RUN_TS=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="results/${RUN_TS}"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS="--val_csv $VAL_CSV --filter_bucket_label lin_only \
  --alphas -5 -3 -1 0 1 3 5 \
  --num_situations 50 \
  --max_new_tokens 4096 \
  --temperature 0.0"

echo "============================================================"
echo "Qwen3 Instruct Steering Experiments v4 — $(date)"
echo "Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B"
echo "Eval: lin_only filter, in-dist val set, max_new_tokens=4096, temp=0"
echo "Results: $RESULTS_DIR"
echo "============================================================"

# ----------------------------------------------------------------
# Qwen3-1.7B  (28 layers)
#   Layers: 7 10 14 18 21 | Combos: "7,14" "10,14,18" "7,14,21"
# ----------------------------------------------------------------
MODEL_17="Qwen/Qwen3-1.7B"
VEC_17="qwen3_1.7b_steering_layer14.pt"

if [ ! -f "$VEC_17" ]; then
  echo ""
  echo ">>> [1/6] Generating steering vector: $MODEL_17 layer 14"
  python generate_steering_vector.py \
    --base_model "$MODEL_17" \
    --training_csv "$TRAIN_CSV" \
    --filter_value lin_only \
    --layer 14 \
    --num_pairs 594 \
    --output "$VEC_17"
else
  echo ""
  echo ">>> [1/6] Steering vector already exists: $VEC_17 (skipping)"
fi

echo ""
echo ">>> [2/6] Sweep: $MODEL_17"
python sweep_steering.py \
  --base_model "$MODEL_17" \
  --steering_path "$VEC_17" \
  $COMMON_ARGS \
  --layers 7 10 14 18 21 \
  --multilayer_combos "7,14" "10,14,18" "7,14,21" \
  --output_prefix "$RESULTS_DIR/sweep_Qwen3-1.7B_${RUN_TS}"

# ----------------------------------------------------------------
# Qwen3-8B  (36 layers)
#   Layers: 10 14 18 22 26 | Combos: "10,18" "14,18,22" "10,18,26"
# ----------------------------------------------------------------
MODEL_8="Qwen/Qwen3-8B"
VEC_8="qwen3_8b_steering_layer18.pt"

if [ ! -f "$VEC_8" ]; then
  echo ""
  echo ">>> [3/6] Generating steering vector: $MODEL_8 layer 18"
  python generate_steering_vector.py \
    --base_model "$MODEL_8" \
    --training_csv "$TRAIN_CSV" \
    --filter_value lin_only \
    --layer 18 \
    --num_pairs 594 \
    --output "$VEC_8"
else
  echo ""
  echo ">>> [3/6] Steering vector already exists: $VEC_8 (skipping)"
fi

echo ""
echo ">>> [4/6] Sweep: $MODEL_8"
python sweep_steering.py \
  --base_model "$MODEL_8" \
  --steering_path "$VEC_8" \
  $COMMON_ARGS \
  --layers 10 14 18 22 26 \
  --multilayer_combos "10,18" "14,18,22" "10,18,26" \
  --output_prefix "$RESULTS_DIR/sweep_Qwen3-8B_${RUN_TS}"

# ----------------------------------------------------------------
# Qwen3-14B  (40 layers)
#   Layers: 12 16 20 24 28 | Combos: "12,20" "16,20,24" "12,20,28"
# ----------------------------------------------------------------
MODEL_14="Qwen/Qwen3-14B"
VEC_14="qwen3_14b_steering_layer20.pt"

if [ ! -f "$VEC_14" ]; then
  echo ""
  echo ">>> [5/6] Generating steering vector: $MODEL_14 layer 20"
  python generate_steering_vector.py \
    --base_model "$MODEL_14" \
    --training_csv "$TRAIN_CSV" \
    --filter_value lin_only \
    --layer 20 \
    --num_pairs 594 \
    --output "$VEC_14"
else
  echo ""
  echo ">>> [5/6] Steering vector already exists: $VEC_14 (skipping)"
fi

echo ""
echo ">>> [6/6] Sweep: $MODEL_14"
python sweep_steering.py \
  --base_model "$MODEL_14" \
  --steering_path "$VEC_14" \
  $COMMON_ARGS \
  --layers 12 16 20 24 28 \
  --multilayer_combos "12,20" "16,20,24" "12,20,28" \
  --output_prefix "$RESULTS_DIR/sweep_Qwen3-14B_${RUN_TS}"

# ----------------------------------------------------------------
# Push results to GitHub
# ----------------------------------------------------------------
echo ""
echo ">>> Pushing results to GitHub..."
git add "$RESULTS_DIR/" run_qwen3_instruct_sweeps_v4.sh \
        sweep_steering.py evaluate.py generate_steering_vector.py
git commit -m "$(cat <<'EOF'
Add Qwen3 steering sweeps v4 — max_new_tokens=4096

- Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B
- max_new_tokens=4096, temp=0, lin_only filter, in-dist val set
- Single-layer: 5 layers x 7 alphas + 3 multilayer combos per model

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin master

echo ""
echo "============================================================"
echo "All done — $(date)"
echo "Results in: $RESULTS_DIR"
echo "============================================================"
