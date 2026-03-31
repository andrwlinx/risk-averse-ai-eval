# Risk-Averse AI Evaluation — Steering Vector Fork

This fork of [elliottthornley/risk-averse-ai-eval](https://github.com/elliottthornley/risk-averse-ai-eval) adds **ICV (In-Context Vector) steering vector generation** for the activation engineering method.

## Workflow

1. **Generate the steering vector** using `generate_steering_vector.py` (this repo)
2. **Evaluate with steering** using `evaluate.py` from Elliott's canonical repo (already synced here)

The canonical evaluation code, system prompt, datasets, and hyperparameters are maintained in Elliott's repo. Always `git fetch upstream && git checkout upstream/main -- evaluate.py risk_averse_prompts.py answer_parser.py dataset_schema_utils.py` before running evaluations.

## Generating a Steering Vector

```bash
python generate_steering_vector.py \
    --base_model Qwen/Qwen3-8B \
    --output risk_averse_icv_steering_vector.pt
```

This builds an ICV steering vector by:
- Loading few-shot demonstrations with **full chain-of-thought reasoning** (risk-averse vs risk-neutral)
- Extracting activation differences at the pre-answer token position
- Averaging across 100 contrasts to produce a stable steering direction

### Key defaults
- **Training data**: `data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv`
- **Demo columns**: `chosen_full` (risk-averse CoT) vs `rejected_full` (risk-neutral CoT)
- **System prompt**: Shared canonical prompt from `risk_averse_prompts.py`
- **Thinking**: Enabled by default (`--enable_thinking`, matching eval-time context)
- **Base model**: `Qwen/Qwen3-8B`
- **Layer**: `n_layers // 2` (auto-computed at runtime, matching upstream)
- **ICV method**: PCA (matching upstream; `--icv_method mean` for simple averaging)
- **Demo max chars**: 0 / disabled (no truncation; avoids asymmetric truncation of risk-averse CoTs)
- **Seed**: `12345`
- **Contrasts**: 100 (5 demos each)

### All parameters
```
--training_csv     Path to training CSV (default: March 22 lin-only 600 situations)
--base_model       Model for activation extraction (default: Qwen/Qwen3-8B)
--layer            Layer to extract from (default: n_layers // 2)
--output           Output .pt file path
--num_demos        Demos per contrast (default: 5)
--num_contrasts    Number of contrasts to average (default: 100)
--seed             Random seed (default: 12345)
--icv_method       Aggregation: pca (default, matching upstream) or mean
--demo_max_chars   Max chars per demo CoT (default: 0 = no truncation; non-zero risks asymmetric truncation)
--enable_thinking  Enable thinking in chat template (default: True)
--no-enable_thinking  Disable thinking
--system_prompt_file  Custom system prompt file (default: uses risk_averse_prompts.py)
--outlier_method   Outlier filtering: none/norm/cosine (default: none)
--outlier_threshold  Threshold for outlier filtering (default: 2.0)
```

## Evaluating with Steering

Use the upstream `evaluate.py` with `--backend transformers`. **Always** set `--eval_layer` to the layer saved in your .pt file (the generate script prints this). If omitted, evaluate.py falls back to `n_layers // 2`, which will be **wrong** for any non-mid-layer setting.

```bash
# PHASE 1: Sweep alphas on medium-stakes validation to SELECT the best alpha.
# This is the only dataset where a wide alpha sweep is appropriate.
python evaluate.py --backend transformers \
    --dataset medium_stakes_validation --num_situations 200 \
    --steering_direction_path risk_averse_icv_steering_vector.pt \
    --eval_layer <LAYER> \
    --alphas "-10,-5,-3,-2,-1,0,1,2,3,5,10"

# PHASE 2: Evaluate held-out sets with the LOCKED alpha from validation.
# Do NOT sweep alphas on held-out data — use only the single best alpha from Phase 1.

# High-stakes test (1000 situations) — locked alpha from validation
python evaluate.py --backend transformers \
    --dataset high_stakes_test --num_situations 1000 \
    --steering_direction_path risk_averse_icv_steering_vector.pt \
    --eval_layer <LAYER> \
    --alphas "<LOCKED_ALPHA>"

# Astronomical-stakes deployment (1000 situations) — locked alpha from validation
python evaluate.py --backend transformers \
    --dataset astronomical_stakes_deployment --num_situations 1000 \
    --steering_direction_path risk_averse_icv_steering_vector.pt \
    --eval_layer <LAYER> \
    --alphas "<LOCKED_ALPHA>"

# Steals-only test (1000 situations) — locked alpha from validation
python evaluate.py --backend transformers \
    --dataset steals_test --num_situations 1000 \
    --steering_direction_path risk_averse_icv_steering_vector.pt \
    --eval_layer <LAYER> \
    --alphas "<LOCKED_ALPHA>"
```

### Canonical eval hyperparameters (set by Elliott, do not change)
- temperature=0.6, top_p=0.95, top_k=20, seed=12345
- max_new_tokens=4096, enable_thinking=true
- Primary metric: **cooperate rate** (not CARA)
- Secondary check: **steal rate** (over-risk-aversion indicator)

## Datasets

All March 22 datasets are in `data/`:
- `2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv` — steering vector training
- `2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv` — full training set
- `2026_03_22_medium_stakes_val_set_500_Rebels.csv` — validation (use 200 situations)
- `2026_03_22_high_stakes_test_set_1000_Rebels.csv` — held-out test
- `2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv` — OOD deployment
- `2026_03_22_test_set_1000_Steals.csv` — steals-only test

## Sweep (Development Only)

`sweep_steering.py` runs a grid search over layers and alpha values for development/exploration. Not for final paper results — use upstream `evaluate.py` for those.

```bash
python sweep_steering.py \
    --steering_path risk_averse_icv_steering_vector.pt \
    --layers 10 14 18 \
    --alphas 0 1 2 5 10
```
