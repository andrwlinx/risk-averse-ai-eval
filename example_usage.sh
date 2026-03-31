#!/bin/bash
# Example usage script for risk-averse AI evaluation (steering vectors)
#
# Uses the upstream evaluate.py with --backend transformers for steering.
# Modify the variables below for your setup.

BASE_MODEL="Qwen/Qwen3-8B"
STEERING_VECTOR="risk_averse_icv_steering_vector.pt"
# IMPORTANT: Set EVAL_LAYER to the layer saved in your .pt file (printed during generation).
# evaluate.py defaults to n_layers // 2 if omitted, which will be WRONG for non-mid-layer settings.
EVAL_LAYER=14  # <-- Replace with the layer from your .pt file
ALPHAS="-10.0,-5.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,5.0,10.0"

# After the validation sweep (Step 2), lock the best alpha and use it for held-out evaluation.
# Replace this with the alpha that maximizes cooperate rate on medium-stakes validation.
LOCKED_ALPHA="3.0"  # <-- Replace with the best alpha from validation

echo "==================================="
echo "Risk-Averse Steering Evaluation"
echo "==================================="
echo ""

# Step 1: Generate steering vector (if not already done)
echo "To generate a steering vector, run:"
echo "  python generate_steering_vector.py --base_model $BASE_MODEL"
echo ""

# Step 2: Sweep alphas on medium-stakes validation to SELECT the best alpha.
# This is the only dataset where a wide alpha sweep is appropriate.
echo "1. Evaluating on medium-stakes validation (alpha sweep)..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset medium_stakes_validation \
    --num_situations 200 \
    --steering_direction_path "$STEERING_VECTOR" \
    --eval_layer "$EVAL_LAYER" \
    --alphas "$ALPHAS"

echo ""

# Step 3: Held-out evaluation on high-stakes test — locked alpha, no sweep.
echo "2. Evaluating on high-stakes test (locked alpha)..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset high_stakes_test \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --eval_layer "$EVAL_LAYER" \
    --alphas "$LOCKED_ALPHA"

echo ""

# Step 4: Held-out evaluation on astronomical-stakes deployment — locked alpha, no sweep.
echo "3. Evaluating on astronomical-stakes deployment (locked alpha)..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset astronomical_stakes_deployment \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --eval_layer "$EVAL_LAYER" \
    --alphas "$LOCKED_ALPHA"

echo ""

# Step 5: Held-out evaluation on steals-only test — locked alpha, no sweep.
echo "4. Evaluating on steals-only test (locked alpha)..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset steals_test \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --eval_layer "$EVAL_LAYER" \
    --alphas "$LOCKED_ALPHA"

echo ""
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
