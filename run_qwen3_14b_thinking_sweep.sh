#!/usr/bin/env bash
# Qwen3-14B steering sweep with thinking mode ENABLED
#
# Fixes the 14B baseline anomaly (0% cara_rate) caused by disable_thinking
# auto-enabling during the previous sweep.
#
# Key changes vs run_qwen3_instruct_sweeps_v2.sh:
#   - --enable_thinking: forces thinking ON, overriding the auto-disable
#   - --max_new_tokens 4096: thinking uses ~2000-2600 tokens before answering
#   - ~3x slower per situation (~110s vs ~35s) — budget ~15h for full sweep
#
# Usage:
#   bash run_qwen3_14b_thinking_sweep.sh 2>&1 | tee run_qwen3_14b_thinking_sweep.log

set -euo pipefail

cd /lambda/nfs/activation-engineering/projects/pi-eval
export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

TRAIN_CSV="data/2026_01_29_new_full_training_set_with_CoTs_Sonnet_4_5.csv"
VAL_CSV="data/in_distribution_val_set.csv"
MODEL="Qwen/Qwen3-14B"
VEC="qwen3_14b_steering_layer20.pt"

COMMON_ARGS="--val_csv $VAL_CSV --filter_bucket_label lin_only \
  --alphas -5 -3 -1 0 1 3 5 \
  --num_situations 50 \
  --max_new_tokens 4096 \
  --max_time_per_generation 300 \
  --temperature 0.0 \
  --enable_thinking"

echo "============================================================"
echo "Qwen3-14B Thinking Sweep — $(date)"
echo "Thinking: ENABLED | max_new_tokens: 4096 | max_time: 300s"
echo "Eval: lin_only filter, in-dist val set (50 -> 9 situations), temp=0"
echo "============================================================"

echo ""
echo ">>> [1/2] Generating steering vector: $MODEL layer 20"
python generate_steering_vector.py \
  --base_model "$MODEL" \
  --training_csv "$TRAIN_CSV" \
  --filter_value lin_only \
  --layer 20 \
  --num_pairs 594 \
  --output "$VEC"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="sweep_Qwen_Qwen3-14B_thinking_${TIMESTAMP}"

echo ""
echo ">>> [2/2] Sweep (single-layer + multilayer): $MODEL"
python sweep_steering.py \
  --base_model "$MODEL" \
  --steering_path "$VEC" \
  $COMMON_ARGS \
  --layers 12 16 20 24 28 \
  --multilayer_combos "12,20" "16,20,24" "12,20,28" \
  --output_prefix "$OUTPUT_PREFIX"

echo ""
echo ">>> Pushing results to GitHub..."
git add "${OUTPUT_PREFIX}.json" "${OUTPUT_PREFIX}.png" \
        qwen3_14b_steering_layer20.pt \
        sweep_steering.py evaluate.py run_qwen3_14b_thinking_sweep.sh
git commit -m "Add Qwen3-14B thinking-enabled steering sweep

- Fix: enable_thinking=True overrides auto-disable for base models
- max_new_tokens=4096 to accommodate thinking (2000-2600 tok per response)
- Re-generated steering vector at layer 20 with same 594 lin_only pairs
- Layers: 12 16 20 24 28 + combos 12+20, 16+20+24, 12+20+28
- Alphas: -5 -3 -1 0 1 3 5

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin master

echo ""
echo "============================================================"
echo "All done — $(date)"
echo "============================================================"
