#!/usr/bin/env bash
# Qwen3 ICV steering evaluations
#
# Calls run_icv_eval.py for each model, which:
#   - Loads the model once
#   - Saves each (layer, alpha) combo as an individual evaluate.py-format JSON
#   - 40 combos: 5 layers x 5 alphas + 3 multilayer x 5 alphas
#
# 1.7B and 8B: thinking enabled (matches ICV extraction context)
# 14B:         thinking disabled (~49hrs otherwise at 8.5 tok/s)
#
# Usage:
#   bash run_qwen3_icv_sweeps.sh 2>&1 | tee run_qwen3_icv_sweeps.log

set -euo pipefail

cd /lambda/nfs/activation-engineering/projects/pi-eval
export HF_HOME=/lambda/nfs/activation-engineering/hf_cache

VAL_CSV="data/in_distribution_val_set.csv"
RUN_TS=$(date +%Y-%m-%d_%H%M%S)
RESULTS_DIR="results/${RUN_TS}_icv"
mkdir -p "$RESULTS_DIR"

# Thinking prefix for 1.7B and 8B: pre-fill the <think> block with a brief trigger
# then close it immediately, forcing the model to skip extended reasoning and
# generate only the answer (~50-200 tokens). Matches ICV extraction context
# (thinking enabled) while avoiding the ~700s/combo cost of full reasoning.
THINKING_PREFIX="I need to select the best option. Let me identify which option maximizes expected value."

COMMON_ARGS="--val_csv $VAL_CSV --filter_bucket_label lin_only \
  --alphas -3 -1 0 1 3 \
  --num_situations 50 \
  --temperature 0.0"

echo "============================================================"
echo "Qwen3 ICV Evaluations — $(date)"
echo "40 combos per model (5 layers x 5 alphas + 3 multilayer x 5 alphas)"
echo "Thinking: pre-filled trigger + </think> for fast answer-only generation"
echo "Output format: evaluate.py-compatible JSON per combo"
echo "Results: $RESULTS_DIR"
echo "============================================================"

# ----------------------------------------------------------------
# Qwen3-1.7B  — thinking enabled (pre-filled prefix, answer-only generation)
# ----------------------------------------------------------------
echo ""
echo ">>> [1/3] Qwen3-1.7B (ICV, thinking prefix)"
python run_icv_eval.py \
  --base_model Qwen/Qwen3-1.7B \
  --steering_path qwen3_1.7b_icv_layer14.pt \
  --results_dir "$RESULTS_DIR" \
  $COMMON_ARGS \
  --enable_thinking \
  --thinking_prefix "$THINKING_PREFIX" \
  --max_new_tokens 1500 \
  --max_time_per_generation 120

# ----------------------------------------------------------------
# Qwen3-8B  — thinking enabled (pre-filled prefix, answer-only generation)
# ----------------------------------------------------------------
echo ""
echo ">>> [2/3] Qwen3-8B (ICV, thinking prefix)"
python run_icv_eval.py \
  --base_model Qwen/Qwen3-8B \
  --steering_path qwen3_8b_icv_layer18.pt \
  --results_dir "$RESULTS_DIR" \
  $COMMON_ARGS \
  --enable_thinking \
  --thinking_prefix "$THINKING_PREFIX" \
  --max_new_tokens 1500 \
  --max_time_per_generation 120

# ----------------------------------------------------------------
# Qwen3-14B  — thinking disabled (too slow otherwise)
# ----------------------------------------------------------------
echo ""
echo ">>> [3/3] Qwen3-14B (ICV, thinking disabled)"
python run_icv_eval.py \
  --base_model Qwen/Qwen3-14B \
  --steering_path qwen3_14b_icv_layer20.pt \
  --results_dir "$RESULTS_DIR" \
  $COMMON_ARGS \
  --max_new_tokens 1024 \
  --max_time_per_generation 120 \
  --disable_thinking

# ----------------------------------------------------------------
# Push to GitHub
# ----------------------------------------------------------------
echo ""
echo ">>> Pushing results to GitHub..."
git add "$RESULTS_DIR/" \
        qwen3_1.7b_icv_layer14.pt \
        qwen3_8b_icv_layer18.pt \
        qwen3_14b_icv_layer20.pt \
        run_qwen3_icv_sweeps.sh \
        run_icv_eval.py \
        sweep_steering.py \
        evaluate.py
git commit -m "$(cat <<EOF
Add Qwen3 ICV evaluations — ${RUN_TS}

- Models: Qwen3-1.7B, Qwen3-8B (thinking enabled), Qwen3-14B (thinking disabled)
- Vectors: ICV mode (averse vs neutral demo context shift, 3 demos)
- 40 combos: 5 layers x 5 alphas (-3,-1,0,1,3) + 3 multilayer x 5 alphas
- Output: evaluate.py-format JSON per combo
- Results in: results/${RUN_TS}_icv/

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin master

echo ""
echo "============================================================"
echo "All done — $(date)"
echo "Results in: $RESULTS_DIR"
echo "============================================================"
