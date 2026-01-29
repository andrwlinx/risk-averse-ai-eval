# Evaluation Script Comparison

This document explains the differences between the two evaluation scripts and when to use each one.

## Quick Summary

| Script | Use Case | Parse Rate | Flexibility | Complexity |
|--------|----------|------------|-------------|------------|
| **evaluate.py** ⭐ | **Standard evaluation** | **~96%** | High | Low |
| evaluate_comprehensive.py | Research comparison | Varies | Low | High |

**Recommendation: Use `evaluate.py` for almost everything.**

---

## 1. `evaluate.py` ⭐ RECOMMENDED

### What It Does
Single evaluation run with **highly permissive answer parsing** to maximize parse rate.

### Key Features
- ✅ **96% parse rate** (best in class)
- ✅ Accepts both **letter** (a,b,c) and **number** (1,2,3) answers
- ✅ **Configurable via command line** - no code editing needed
- ✅ **Configurable temperature** (0, 0.7, 1.0, etc.)
- ✅ Generous token limit (4096) to avoid truncation
- ✅ Saves full responses for debugging
- ✅ Handles multiple answer formats:
  - JSON: `{"answer": "a"}`
  - Natural: "I choose b"
  - Parenthesized: "(a)"
  - Standalone: just "a" at the end

### When to Use
- ✅ **Default evaluation** for your fine-tuned models
- ✅ When you want **reliable, reproducible results**
- ✅ When you need to **compare models** fairly
- ✅ When you want to **experiment with temperature**
- ✅ For **production/deployment evaluation**

### Usage
```bash
# Basic usage (deterministic)
python evaluate.py \
    --model_path ./my-model/final \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --temperature 0 \
    --output results.json

# With sampling
python evaluate.py \
    --model_path ./my-model/final \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --temperature 0.7 \
    --num_situations 50 \
    --output results_temp07.json

# Quick eval (25 situations, ~10 mins)
python evaluate.py \
    --model_path ./my-model/final \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 25 \
    --temperature 0 \
    --output quick_results.json
```

### Command-Line Configuration
**What this means:** All settings can be controlled via `--flags` on the command line. You never need to edit the Python code.

**Available flags:**
- `--model_path`: Path to your model adapter
- `--base_model`: Base model ID (e.g., Qwen/Qwen3-8B)
- `--val_csv`: Dataset CSV file
- `--num_situations`: Number of situations (default: 50)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_new_tokens`: Token limit (default: 4096)
- `--no_save_responses`: Disable saving full CoT responses (saved by default)
- `--output`: Output JSON file path

**Contrast with comprehensive script:** `evaluate_comprehensive.py` has NO command-line flags. You must edit the Python code to change which models to evaluate, which datasets to use, etc.

### Output Format
```json
{
  "evaluation_config": {
    "temperature": 0.7,
    "max_new_tokens": 4096,
    "num_situations": 50,
    "base_model": "Qwen/Qwen3-8B",
    "model_path": "./my-model/final"
  },
  "metrics": {
    "parse_rate": 0.96,
    "cooperate_rate": 0.83,
    "rebel_rate": 0.12,
    "steal_rate": 0.05,
    "best_cara_rate": 0.79
  },
  "num_valid": 48,
  "num_total": 50,
  "results": [...],
  "failed_responses": [...]
}
```

### Pros
- Highest parse rate
- Most flexible (all settings via CLI)
- Best documentation
- Production-ready
- Easy to use
- No code editing required

### Cons
- Only evaluates one temperature at a time (but fast to run multiple times)

---

## 2. `evaluate_comprehensive.py` - Multi-Metric Evaluation

### What It Does
Comprehensive evaluation running **THREE different evaluation modes** on each model automatically:
1. **Generation @ temp=0** (deterministic)
2. **Generation @ temp=0.7** (sampling)
3. **Log probabilities** (answer-only, no chain-of-thought)

### Key Features
- Evaluates multiple temperatures automatically
- Tests both CoT generation and direct answer probabilities
- Compares base models vs fine-tuned models
- Batch evaluation of multiple models

### When to Use
- ✅ **Research purposes** when you need comprehensive metrics
- ✅ Comparing **how temperature affects performance**
- ✅ Understanding **log probability distributions** vs generation
- ✅ When you have **time and compute** for thorough analysis
- ❌ **NOT for quick evaluations** (3x slower than single-metric)

### Configuration (Requires Code Editing)
**Important:** This script has NO command-line arguments. You must edit the Python file to configure:

```python
# Edit these in evaluate_comprehensive.py:

DATASETS = {
    "ood_validation": "data/2026_01_29_new_val_set_probabilities_add_to_100.csv",
    "indist_validation": "data/in_distribution_val_set.csv",
    "training": "data/training_eval_set.csv",
}

BASE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

FINETUNED_MODELS = {
    "my_model": {
        "base_model": "Qwen/Qwen3-8B",
        "sweep_dir": "path/to/my/sweep/directory",
        "configs": ["config1", "config2", "config3"]
    }
}
```

### Usage
```bash
# Edit the script first to configure your models
python evaluate_comprehensive.py
```

### Output Format
```json
{
  "timestamp": "2026-01-15T10:30:00",
  "num_situations": 50,
  "metrics": ["generation_temp0", "generation_temp07", "logprob_answer_only"],
  "base_models": {
    "Qwen/Qwen3-8B": {
      "ood_validation": {
        "generation_temp0": {"cara_rate": 0.35, "parse_rate": 0.92},
        "generation_temp07": {"cara_rate": 0.34, "parse_rate": 0.88},
        "logprob_answer_only": {"cara_rate": 0.38, "parse_rate": 1.0}
      }
    }
  },
  "finetuned_models": {...}
}
```

### Pros
- Comprehensive comparison across multiple evaluation methods
- Good for research papers/reports
- Automatically evaluates multiple models and datasets
- Log probability method is useful for base model comparison

### Cons
- 3x slower (runs 3 evaluations per model)
- More complex to configure (requires editing script)
- No command-line flexibility
- Overkill for most use cases

---

## Comparison Table: Detailed

| Feature | evaluate.py | evaluate_comprehensive.py |
|---------|-------------|--------------------------|
| **Parse rate** | **~96%** | Varies (~70-95%) |
| **Temperature control** | ✅ CLI arg (any value) | ❌ Fixed (0, 0.7) |
| **Command-line config** | ✅ Full (all settings) | ❌ None (edit code) |
| **Accepts numbers (1,2,3)** | ✅ Yes | ✅ Yes |
| **Accepts letters (a,b,c)** | ✅ Yes | ✅ Yes |
| **Multiple answer formats** | ✅ 9+ patterns | ✅ 9+ patterns |
| **Save responses** | ✅ Always (opt-out with --no_save_responses) | ❌ No |
| **Token limit** | ✅ 4096 (configurable) | ✅ 4096 (fixed) |
| **Multiple metrics** | ❌ Single run | ✅ 3 methods |
| **Log probabilities** | ❌ No | ✅ Yes |
| **Ease of use** | ✅✅✅ Excellent | ❌ Poor (edit code) |
| **Speed** | ✅ Fast | ❌ 3x slower |
| **Recommended?** | ✅✅✅ YES | ⚠️ Research only |

---

## Which Script Should You Use?

### For 95% of Use Cases: `evaluate.py` ⭐

Use this if you want to:
- Evaluate your fine-tuned model
- Compare different models
- Test at different temperatures
- Get reliable, reproducible results
- Have flexibility and control
- **Configure everything from the command line**

```bash
python evaluate.py \
    --model_path <YOUR_MODEL> \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --temperature 0 \
    --output results.json
```

### For Research Papers: `evaluate_comprehensive.py`

Use this if you need:
- Comprehensive comparison across multiple evaluation methods
- Temperature sensitivity analysis (automatic)
- Log probability analysis
- Multiple models evaluated in batch

**Trade-off:** 3x longer runtime, requires editing code to configure

---

## Understanding "CLI Configuration"

**CLI = Command-Line Interface**

### With CLI Configuration (`evaluate.py`)
You control everything via flags when running the command:

```bash
python evaluate.py --temperature 0 --model_path ./model1 --output results1.json
python evaluate.py --temperature 0.7 --model_path ./model2 --output results2.json
python evaluate.py --temperature 1.0 --model_path ./model3 --output results3.json
```

**No code editing needed.** Just change the flags.

### Without CLI Configuration (`evaluate_comprehensive.py`)
You must edit the Python file itself:

```python
# Open evaluate_comprehensive.py in a text editor
# Find lines 33-57 and edit:
BASE_MODELS = ["Qwen/Qwen3-8B"]  # Change this
FINETUNED_MODELS = {...}  # Change this
```

Then run:
```bash
python evaluate_comprehensive.py  # No flags available
```

**CLI configuration is much more user-friendly** for most use cases.

---

## Temperature Recommendations

Regardless of which script you use, here's when to use different temperatures:

| Temperature | Use Case | Characteristics |
|-------------|----------|-----------------|
| **0** | Benchmarking, comparison | Deterministic, reproducible, greedy decoding |
| **0.7** | Default evaluation | Balanced, realistic sampling |
| **1.0** | Robustness testing | High diversity, tests edge cases |

**Recommended approach:**
1. Start with **temp=0** for reproducible benchmarks
2. Try **temp=0.7** to see realistic sampling behavior
3. Use **temp=1.0** to test robustness

---

## Example Workflows

### Workflow 1: Quick Model Evaluation
```bash
# Run on 25 situations, temperature=0
python evaluate.py \
    --model_path ./my-model/final \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 25 \
    --temperature 0 \
    --output quick_eval.json

# Check CARA rate
cat quick_eval.json | python -c "import json, sys; print(f\"CARA: {json.load(sys.stdin)['metrics']['best_cara_rate']*100:.1f}%\")"
```

### Workflow 2: Full Temperature Comparison
```bash
# Evaluate at three temperatures
for temp in 0 0.7 1.0; do
    python evaluate.py \
        --model_path ./my-model/final \
        --base_model Qwen/Qwen3-8B \
        --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
        --temperature $temp \
            --output results_temp${temp}.json
done

# Compare results
for temp in 0 0.7 1.0; do
    echo "Temperature $temp:"
    cat results_temp${temp}.json | python -c "import json, sys; data=json.load(sys.stdin); print(f\"  CARA: {data['metrics']['best_cara_rate']*100:.1f}%, Parse: {data['metrics']['parse_rate']*100:.1f}%\")"
done
```

### Workflow 3: Multi-Dataset Evaluation
```bash
# Evaluate on all three datasets
for dataset in 2026_01_29_new_val_set_probabilities_add_to_100 in_distribution_val_set training_eval_set; do
    python evaluate.py \
        --model_path ./my-model/final \
        --base_model Qwen/Qwen3-8B \
        --val_csv data/${dataset}.csv \
        --temperature 0 \
            --output results_${dataset}.json
done
```

---

## Summary

**Use `evaluate.py` for almost everything.** It has:
- Best parse rate (96%)
- Most flexibility (all settings via command line)
- Easiest to use
- Best documentation
- Production-ready
- No code editing needed

Only use `evaluate_comprehensive.py` if you specifically need multi-metric research comparison and have the compute budget (3x slower) and don't mind editing Python code to configure it.
