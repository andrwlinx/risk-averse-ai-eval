# How to Share This Evaluation Package

## Option 1: Zip and Email/Share

The simplest way to share this package:

```bash
# From the parent directory
cd /Users/elliottthornley
zip -r risk_averse_eval.zip risk_averse_eval_package/

# Share the zip file via email, Slack, Google Drive, etc.
```

Your team can then:
```bash
unzip risk_averse_eval.zip
cd risk_averse_eval_package
pip install -r requirements.txt
python evaluate.py --help
```

## Option 2: Upload to GitHub (Recommended)

If your team uses GitHub:

```bash
cd /Users/elliottthornley/risk_averse_eval_package

# Initialize git repo
git init
git add .
git commit -m "Initial commit: Risk-averse AI evaluation package"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/yourusername/risk-averse-eval.git
git push -u origin main
```

Your team can then:
```bash
git clone https://github.com/yourusername/risk-averse-eval.git
cd risk-averse-eval
pip install -r requirements.txt
```

## Option 3: Google Drive / Dropbox

```bash
# Copy the folder to your shared drive
cp -r /Users/elliottthornley/risk_averse_eval_package "/Users/elliottthornley/My Drive/Shared with Team/"
```

## What Your Team Needs

### Minimum Requirements
1. **Python 3.8+**
2. **GPU with 24GB+ VRAM** (for 8B models) or 40GB+ (for 32B models)
3. **CUDA toolkit** (for GPU acceleration)

### Files Included in This Package
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `SCRIPT_COMPARISON.md` - Detailed comparison of evaluation scripts
- `evaluate.py` - Main evaluation script (recommended) ⭐
- `evaluate_comprehensive.py` - Multi-metric evaluation for research
- `requirements.txt` - Python dependencies
- `example_usage.sh` - Example script showing all evaluations
- `data/2026_01_29_new_val_set_probabilities_add_to_100.csv` - OOD validation dataset (current, includes Rebel/Steal scenarios)
- `data/val_set_medium_stakes.csv` - OOD validation dataset (legacy, 1.4MB)
- `data/in_distribution_val_set.csv` - In-distribution validation (154KB)
- `data/training_eval_set.csv` - Training set evaluation (1.5MB)

**Total package size:** ~3.1MB (mostly CSV data)

## What Your Team Should Do First

1. **Read QUICKSTART.md** for a 5-minute intro
2. **Run a test evaluation** on their fine-tuned model
3. **Check parse rate** - if >90%, they're good to go!
4. **Compare results** with your baseline numbers

## Sample Command for Your Team

Include this in your message to them:

```bash
# After installing requirements, run this:
python evaluate.py \
    --model_path <YOUR_MODEL_PATH> \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 50 \
    --temperature 0 \
    --output my_model_results.json

# Check results:
cat my_model_results.json | grep "best_cara_rate"
```

## Expected Questions from Your Team

### "What model should I use for `--base_model`?"
- **Answer:** Use the same base model you fine-tuned from (e.g., `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-7B-Instruct`)

### "What's a good CARA rate?"
- **Answer:**
  - Base models (unfinetuned): 30-55%
  - Good fine-tuned models: 75-85%
  - Excellent fine-tuned models: 85-90%+

### "How long does evaluation take?"
- **Answer:**
  - 25 situations: ~10-15 minutes (8B model on A100)
  - 50 situations: ~20-30 minutes
  - 100 situations: ~40-60 minutes

### "Can I use a smaller GPU?"
- **Answer:**
  - 8B models: 24GB minimum (e.g., RTX 3090, A5000, L4)
  - 32B models: 40GB+ (A100) or use 4-bit quantization

### "Which dataset should I use?"
- **Answer:**
  - **Start with `2026_01_29_new_val_set_probabilities_add_to_100.csv`** (OOD generalization test, includes Rebel/Steal)
  - Then try `in_distribution_val_set.csv` (easier, in-distribution)
  - Finally `training_eval_set.csv` (check for overfitting)

## Support

If your team has questions, they should:
1. Read the full README.md first
2. Check the "Troubleshooting" section
3. Look at saved responses if parse rate is low
4. Contact you with specific error messages or results

## Versioning

This package was created: **January 15, 2026**

Based on experiments from: **January 2026 SERC Risk-Averse AIs project**

If you update the scripts or datasets, increment the version:
- Add a `VERSION` file: `echo "1.0.0" > VERSION`
- Update README.md with changelog
