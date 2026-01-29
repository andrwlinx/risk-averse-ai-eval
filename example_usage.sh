#!/bin/bash
# Example usage script for risk-averse AI evaluation
#
# This script shows how to evaluate a fine-tuned model on all three datasets.
# Modify the MODEL_PATH and BASE_MODEL variables for your setup.

MODEL_PATH="/path/to/your/model/adapter"  # CHANGE THIS
BASE_MODEL="Qwen/Qwen3-8B"                # CHANGE THIS if needed

echo "==================================="
echo "Risk-Averse AI Evaluation Examples"
echo "==================================="
echo ""

# Example 1: OOD Validation (Primary Evaluation)
echo "1. Evaluating on OOD validation set (medium stakes)..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 50 \
    --temperature 0 \
    --output results_ood_val.json

echo ""
echo "Results saved to: results_ood_val.json"
echo ""

# Example 2: In-Distribution Validation
echo "2. Evaluating on in-distribution validation set..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --val_csv data/in_distribution_val_set.csv \
    --num_situations 50 \
    --temperature 0 \
    --output results_indist_val.json

echo ""
echo "Results saved to: results_indist_val.json"
echo ""

# Example 3: Training Set (Overfitting Check)
echo "3. Evaluating on training set (overfitting check)..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --val_csv data/training_eval_set.csv \
    --num_situations 50 \
    --temperature 0 \
    --output results_train.json

echo ""
echo "Results saved to: results_train.json"
echo ""

# Summary
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
echo ""
echo "View results with:"
echo "  cat results_ood_val.json | python -m json.tool | head -30"
echo ""
echo "Quick summary:"
python -c "
import json
import sys

files = ['results_ood_val.json', 'results_indist_val.json', 'results_train.json']
labels = ['OOD Validation', 'In-Dist Validation', 'Training Set']

print('')
print('Dataset                 | CARA Rate | Parse Rate | Cooperate% | Rebel%  | Steal%')
print('-' * 90)

for fname, label in zip(files, labels):
    try:
        with open(fname) as f:
            data = json.load(f)
            cara = data['metrics']['best_cara_rate'] * 100
            parse = data['metrics']['parse_rate'] * 100
            coop = data['metrics']['cooperate_rate'] * 100
            rebel = data['metrics'].get('rebel_rate', 0) * 100
            steal = data['metrics'].get('steal_rate', 0) * 100
            print(f'{label:23} | {cara:5.1f}%    | {parse:6.1f}%    | {coop:6.1f}%    | {rebel:5.1f}% | {steal:5.1f}%')
    except FileNotFoundError:
        print(f'{label:23} | ERROR: File not found')
    except Exception as e:
        print(f'{label:23} | ERROR: {e}')

print('')
"

echo ""
echo "Done!"
