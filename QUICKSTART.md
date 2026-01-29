# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run Evaluation

### Example 1: Evaluate Your Fine-Tuned Model

```bash
# Deterministic evaluation (temperature=0, recommended for benchmarking)
python evaluate.py \
    --model_path /path/to/your/model/adapter \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 50 \
    --temperature 0 \
    --output my_results.json

# Or with sampling (temperature=0.7, more realistic)
python evaluate.py \
    --model_path /path/to/your/model/adapter \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 50 \
    --temperature 0.7 \
    --output my_results_temp07.json
```

Replace:
- `/path/to/your/model/adapter` with your LoRA adapter path (e.g., `./my-model/final`)
- `Qwen/Qwen3-8B` with your base model ID

**Temperature tips:**
- `--temperature 0`: Deterministic (greedy decoding) - best for reproducibility
- `--temperature 0.7`: Moderate sampling (default) - realistic behavior
- `--temperature 1.0`: High diversity - tests robustness

### Example 2: Evaluate Base Model (No Fine-Tuning)

```bash
# Evaluate an unfinetuned base model (omit --model_path)
python evaluate.py \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 50 \
    --temperature 0 \
    --output base_model_results.json
```

### Example 3: Evaluate on Different Dataset

```bash
# In-distribution validation
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/in_distribution_val_set.csv \
    --num_situations 50 \
    --temperature 0 \
    --output indist_results.json

# Training set (check for overfitting)
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/training_eval_set.csv \
    --num_situations 50 \
    --temperature 0 \
    --output train_results.json
```

## Step 3: Check Results

```bash
cat my_results.json | python -m json.tool | head -30
```

Look for:
- **`best_cara_rate`**: Your primary metric (target: >0.80)
- **`parse_rate`**: Should be >0.90 (if lower, see troubleshooting)
- **`cooperate_rate`**, **`rebel_rate`**, **`steal_rate`**: Option type breakdown showing model's risk profile

## Common Issues

### Issue: Low Parse Rate (<90%)

**Solution:** Check what the model is outputting:
```bash
# Look at saved responses in the JSON
python -c "import json; data=json.load(open('my_results.json')); print(data['failed_responses'][0])"
```

### Issue: Out of Memory

**Solution:** Reduce number of situations or use quantization:
```bash
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 25 \
    --output results.json
```

### Issue: Qwen3 Base Model Hangs

**Solution:** The script automatically disables thinking mode for base models (when `--model_path` is omitted). If you're still seeing hangs with a fine-tuned model, manually disable it:
```bash
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --disable_thinking \
    --output results.json
```

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Compare results across different models
- Evaluate on multiple datasets (OOD, in-distribution, training)
- Share results with your team!

## Expected Performance

Based on Jan 2026 experiments:

| Model Type | Expected CARA Rate |
|------------|-------------------|
| Base (unfinetuned) | 30-55% |
| Fine-tuned (good) | 75-85% |
| Fine-tuned (excellent) | 85-90%+ |

If your fine-tuned model is <70% CARA, investigate:
1. Training data quality
2. Hyperparameters (LoRA rank, learning rate, epochs)
3. Parse rate (make sure model is following output format)
