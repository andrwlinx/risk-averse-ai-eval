#!/bin/bash
# Example usage script for risk-averse AI evaluation (steering vectors)
#
# Uses the upstream evaluate.py with --backend transformers for steering.
# Modify the variables below for your setup.

BASE_MODEL="Qwen/Qwen3-8B"
STEERING_VECTOR="risk_averse_icv_steering_vector.pt"
# eval_layer defaults to n_layers // 2 in evaluate.py if omitted.
# Set this to the layer saved in your .pt file (printed during generation).
ALPHAS="-10.0,-5.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,5.0,10.0"

echo "==================================="
echo "Risk-Averse Steering Evaluation"
echo "==================================="
echo ""

# Step 1: Generate steering vector (if not already done)
echo "To generate a steering vector, run:"
echo "  python generate_steering_vector.py --base_model $BASE_MODEL"
echo ""

# Step 2: Evaluate on medium-stakes validation (200 situations)
echo "1. Evaluating on medium-stakes validation..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset medium_stakes_validation \
    --num_situations 200 \
    --steering_direction_path "$STEERING_VECTOR" \
    --alphas "$ALPHAS"

echo ""

# Step 3: Evaluate on high-stakes test (1000 situations)
echo "2. Evaluating on high-stakes test..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset high_stakes_test \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --alphas "$ALPHAS"

echo ""

# Step 4: Evaluate on astronomical-stakes deployment (1000 situations)
echo "3. Evaluating on astronomical-stakes deployment..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset astronomical_stakes_deployment \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --alphas "$ALPHAS"

echo ""

# Step 5: Evaluate on steals-only test (1000 situations)
echo "4. Evaluating on steals-only test..."
python evaluate.py \
    --backend transformers \
    --base_model "$BASE_MODEL" \
    --dataset steals_test \
    --num_situations 1000 \
    --steering_direction_path "$STEERING_VECTOR" \
    --alphas "$ALPHAS"

echo ""
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
