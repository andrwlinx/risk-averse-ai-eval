#!/usr/bin/env bash
# Qwen3 instruct model steering sweeps: single-layer + multilayer injection
#
# Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B
# Steering vector: trained on lin_only pairs from full training set with CoTs
# Evaluation: in-distribution val set (all 50 situations), lin_only bucket filter,
#             max_new_tokens=1024, temp=0
# Injection: single-layer sweep + multilayer combos
#
# Usage:
#   bash run_qwen3_instruct_sweeps_v2.sh 2>&1 | tee run_qwen3_instruct_sweeps_v2.log

set -euo pipefail

cd /lambda/nfs/activation-engineering/projects/pi-eval
export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

TRAIN_CSV="data/2026_01_29_new_full_training_set_with_CoTs_Sonnet_4_5.csv"
VAL_CSV="data/in_distribution_val_set.csv"

# num_situations=50 loads all 50 situations; filter then keeps only lin_only (9 total)
COMMON_ARGS="--val_csv $VAL_CSV --filter_bucket_label lin_only \
  --alphas -5 -3 -1 0 1 3 5 \
  --num_situations 50 \
  --max_new_tokens 1024 \
  --temperature 0.0"

echo "============================================================"
echo "Qwen3 Instruct Steering Experiments v2 — $(date)"
echo "Models: Qwen3-1.7B, Qwen3-8B, Qwen3-14B"
echo "Eval: lin_only filter, in-dist val set (all 50), max_new_tokens=1024, temp=0"
echo "Injection: single-layer sweep + multilayer combos"
echo "============================================================"

# ----------------------------------------------------------------
# Qwen3-1.7B  (28 layers)
#   Mid-layer ~14; sweep single layers: 7 10 14 18 21
#   Multilayer combos: "7,14" "10,14,18" "7,14,21"
#   Steering vector already exists at qwen3_1.7b_steering_layer14.pt
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
  echo ">>> [1/6] Steering vector already exists: $VEC_17 (skipping generation)"
fi

echo ""
echo ">>> [2/6] Sweep (single-layer + multilayer): $MODEL_17"
python sweep_steering.py \
  --base_model "$MODEL_17" \
  --steering_path "$VEC_17" \
  $COMMON_ARGS \
  --layers 7 10 14 18 21 \
  --multilayer_combos "7,14" "10,14,18" "7,14,21"

# ----------------------------------------------------------------
# Qwen3-8B  (36 layers)
#   Mid-layer ~18; sweep single layers: 10 14 18 22 26
#   Multilayer combos: "10,18" "14,18,22" "10,18,26"
#   Steering vector already exists at qwen3_8b_steering_layer18.pt
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
  echo ">>> [3/6] Steering vector already exists: $VEC_8 (skipping generation)"
fi

echo ""
echo ">>> [4/6] Sweep (single-layer + multilayer): $MODEL_8"
python sweep_steering.py \
  --base_model "$MODEL_8" \
  --steering_path "$VEC_8" \
  $COMMON_ARGS \
  --layers 10 14 18 22 26 \
  --multilayer_combos "10,18" "14,18,22" "10,18,26"

# ----------------------------------------------------------------
# Qwen3-14B  (40 layers)
#   Mid-layer ~20; sweep single layers: 12 16 20 24 28
#   Multilayer combos: "12,20" "16,20,24" "12,20,28"
# ----------------------------------------------------------------
MODEL_14="Qwen/Qwen3-14B"
VEC_14="qwen3_14b_steering_layer20.pt"

echo ""
echo ">>> [5/6] Generating steering vector: $MODEL_14 layer 20"
python generate_steering_vector.py \
  --base_model "$MODEL_14" \
  --training_csv "$TRAIN_CSV" \
  --filter_value lin_only \
  --layer 20 \
  --num_pairs 594 \
  --output "$VEC_14"

echo ""
echo ">>> [6/6] Sweep (single-layer + multilayer): $MODEL_14"
python sweep_steering.py \
  --base_model "$MODEL_14" \
  --steering_path "$VEC_14" \
  $COMMON_ARGS \
  --layers 12 16 20 24 28 \
  --multilayer_combos "12,20" "16,20,24" "12,20,28"

# ----------------------------------------------------------------
# Push results to GitHub
# ----------------------------------------------------------------
echo ""
echo ">>> Pushing results to GitHub..."
git add *.json *.png *.pt sweep_steering.py run_qwen3_instruct_sweeps_v2.sh
git commit -m "Add Qwen3 instruct steering sweeps (1.7B, 8B, 14B) with multilayer injection

- Models: Qwen3-1.7B (28L), Qwen3-8B (36L), Qwen3-14B (40L)
- Single-layer sweep: 5 layers per model x 7 alphas (-5,-3,-1,0,1,3,5)
- Multilayer combos: 3 combos per model (2-layer and 3-layer simultaneous injection)
- Eval: lin_only filter, in-distribution val set (50 situations -> 9 lin_only), temp=0
- max_new_tokens=1024, deterministic decoding"
git push origin master

echo ""
echo "============================================================"
echo "All done — $(date)"
echo "============================================================"
