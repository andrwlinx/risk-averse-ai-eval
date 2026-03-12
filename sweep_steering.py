#!/usr/bin/env python3
"""
Sweep over steering layers and alpha values to find optimal steering configuration.

Loads model and steering vector once, then evaluates across a grid of
(layer, alpha) combinations. Produces a visualization (PNG) and JSON results.

Usage:
    python sweep_steering.py --steering_path risk_averse_steering_vector.pt
    python sweep_steering.py --steering_path vec.pt --layers 10 14 18 --alphas 0 1 2 5
    python sweep_steering.py --steering_path vec.pt --num_situations 20 --model_path ./my-adapter
"""

import argparse
import gc
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
torch.cuda.empty_cache()
gc.collect()

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from evaluate import load_steering_vector, load_situations, run_evaluation

DEFAULT_LAYERS = [10, 12, 14]
DEFAULT_ALPHAS = [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]


def layer_label(L):
    """Return a string label for a layer spec (int or list of ints)."""
    if isinstance(L, (list, tuple)):
        return "+".join(str(l) for l in L)
    return str(L)


def plot_sweep(results, base_ra, layer_candidates, output_path):
    """Plot steering performance grid across layers (single or multilayer)."""
    df_plot = pd.DataFrame(results)

    num_layers = len(layer_candidates)
    ncols = min(3, num_layers)
    nrows = math.ceil(num_layers / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    sns.set_style("whitegrid")

    # Normalize axes to a flat list
    if num_layers == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    else:
        axes_flat = list(axes.flatten())

    for i, L in enumerate(layer_candidates):
        ax = axes_flat[i]
        lbl = layer_label(L)
        layer_data = df_plot[df_plot["layer_label"] == lbl].sort_values("alpha")
        ax.plot(layer_data["alpha"], layer_data["safe_acc"],
                marker="o", label="Safe Acc", color="green")
        ax.plot(layer_data["alpha"], layer_data["risky_acc"],
                marker="x", label="Risky Acc", color="red")
        if "linear_rate" in layer_data.columns and layer_data["linear_rate"].sum() > 0:
            ax.plot(layer_data["alpha"], layer_data["linear_rate"],
                    marker="s", label="Linear Rate", color="blue")
        ax.axhline(y=base_ra, color="gray", linestyle="--", alpha=0.5,
                   label=f"Base Safe ({base_ra:.0%})")
        title = f"Layers {lbl}" if "+" in lbl else f"Layer {lbl}"
        ax.set_title(title)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(num_layers, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Steering Performance across Layers", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def save_sweep_json(config, results, base_ra, output_path):
    """Save sweep results to JSON for later re-plotting."""
    output = {
        "sweep_config": config,
        "baseline_cooperate_rate": base_ra,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep steering layers and alphas with visualization")
    parser.add_argument("--steering_path", type=str, required=True,
                        help="Path to steering vector .pt file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter (omit for base model)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model ID")
    parser.add_argument("--val_csv", type=str,
                        default="data/2026_01_29_new_val_set_probabilities_add_to_100.csv")
    parser.add_argument("--num_situations", type=int, default=20,
                        help="Situations per combo (default 20 for speed)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Force thinking mode ON, overriding the auto-disable for base models")
    parser.add_argument("--max_time_per_generation", type=float, default=120)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer candidates (default: 7 10 14 18 21 24)")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Alpha values (default: 0 0.5 1 1.5 2 3 5 8 10)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output prefix for PNG and JSON files")
    parser.add_argument("--filter_bucket_label", type=str, default=None,
                        help="Filter situations to only this bucket label (e.g. 'lin_only')")
    parser.add_argument("--multilayer_combos", type=str, nargs="*", default=None,
                        help="Multilayer injection combos, each as comma-separated layer indices "
                             "(e.g. '10,14' '14,18,22'). Added alongside single-layer candidates.")
    args = parser.parse_args()

    single_layers = args.layers or DEFAULT_LAYERS
    multilayer = []
    if args.multilayer_combos:
        for combo_str in args.multilayer_combos:
            combo = [int(x) for x in combo_str.split(",")]
            multilayer.append(combo)
    LAYER_CANDIDATES = single_layers + multilayer
    ALPHAS = args.alphas or DEFAULT_ALPHAS

    # Generate output prefix
    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        args.output_prefix = f"sweep_{model_short}_{timestamp}"

    png_path = f"{args.output_prefix}.png"
    json_path = f"{args.output_prefix}.json"

    # --- Load model (once) ---
    BASE_MODEL = args.base_model
    if args.model_path:
        print(f"Loading fine-tuned model (base: {BASE_MODEL}, adapter: {args.model_path})...")
    else:
        print(f"Loading base model only: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model_hf = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.model_path:
        model = PeftModel.from_pretrained(base_model_hf, args.model_path)
        model = model.merge_and_unload()
    else:
        model = base_model_hf

    model.eval()

    # Validate layer candidates (single ints and multilayer combos)
    num_model_layers = len(model.model.layers)
    for L in LAYER_CANDIDATES:
        check_layers = L if isinstance(L, list) else [L]
        for l in check_layers:
            if l >= num_model_layers:
                print(f"ERROR: Layer {l} >= model layer count ({num_model_layers}). "
                      f"Valid range: 0-{num_model_layers - 1}")
                sys.exit(1)

    # Auto-enable disable_thinking for base model evaluation (unless --enable_thinking overrides)
    if args.model_path is None and not args.disable_thinking and not args.enable_thinking:
        args.disable_thinking = True
        print("Note: Auto-enabling --disable_thinking for base model evaluation")

    # --- Load steering vector (once) ---
    print(f"Loading steering vector from {args.steering_path}...")
    steering_vector, _, metadata = load_steering_vector(args.steering_path)
    print(f"  Vector shape: {steering_vector.shape}")
    if metadata:
        print(f"  Generated from: {metadata.get('num_pairs', '?')} pairs")

    # --- Load data (once) ---
    print("Loading validation data...")
    situations = load_situations(args.val_csv, args.num_situations,
                                 filter_bucket_label=args.filter_bucket_label)
    print(f"Loaded {len(situations)} situations"
          + (f" (filtered to '{args.filter_bucket_label}')" if args.filter_bucket_label else ""))

    # --- Run baseline (no steering) ---
    print("\nRunning baseline (no steering)...")
    baseline = run_evaluation(
        model, tokenizer, situations, steering_vector=None,
        alpha=0.0, steering_layer=14,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        max_time_per_generation=args.max_time_per_generation,
        disable_thinking=args.disable_thinking,
        no_save_responses=False, verbose=False,
    )
    base_ra = baseline["cooperate_rate"]
    base_linear = baseline["linear_rate"]
    base_cara = baseline["cara_rate"]
    print(f"Baseline cooperate rate: {base_ra:.1%}")
    print(f"Baseline linear rate:    {base_linear:.1%}")
    print(f"Baseline cara rate:      {base_cara:.1%}")
    print(f"Baseline parse rate:     {baseline['parse_rate']:.1%}")

    # --- Sweep ---
    total_combos = len(LAYER_CANDIDATES) * len(ALPHAS)
    print(f"\nStarting sweep: {len(LAYER_CANDIDATES)} layers x {len(ALPHAS)} alphas = {total_combos} combos")
    print(f"Layers: {LAYER_CANDIDATES}")
    print(f"Alphas: {ALPHAS}")
    print()

    sweep_results = []
    sweep_start = time.time()

    for combo_idx, (L, alpha) in enumerate(itertools.product(LAYER_CANDIDATES, ALPHAS)):
        combo_start = time.time()
        lbl = layer_label(L)
        print(f"[{combo_idx + 1}/{total_combos}] Layer(s)={lbl}, Alpha={alpha} ...", end=" ", flush=True)

        eval_result = run_evaluation(
            model, tokenizer, situations, steering_vector,
            alpha=alpha, steering_layer=L,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            no_save_responses=False, verbose=False,
        )

        safe_acc = eval_result["cooperate_rate"]
        risky_acc = eval_result["rebel_rate"] + eval_result["steal_rate"]
        combo_elapsed = time.time() - combo_start

        linear_rate = eval_result["linear_rate"]
        sweep_results.append({
            "layer_label": lbl,
            "layer": L if isinstance(L, int) else list(L),
            "alpha": alpha,
            "safe_acc": safe_acc,
            "risky_acc": risky_acc,
            "linear_rate": linear_rate,
            "cooperate_rate": eval_result["cooperate_rate"],
            "rebel_rate": eval_result["rebel_rate"],
            "steal_rate": eval_result["steal_rate"],
            "cara_rate": eval_result["cara_rate"],
            "parse_rate": eval_result["parse_rate"],
        })

        cara_rate = eval_result["cara_rate"]
        remaining = combo_elapsed * (total_combos - combo_idx - 1)
        print(f"safe={safe_acc:.0%} risky={risky_acc:.0%} lin={linear_rate:.0%} cara={cara_rate:.0%} "
              f"({combo_elapsed:.0f}s, ETA: {remaining / 60:.0f}min)")

    total_sweep_time = time.time() - sweep_start
    print(f"\nSweep complete in {total_sweep_time / 60:.1f} minutes")

    # --- Save results ---
    config = {
        "base_model": args.base_model,
        "model_path": args.model_path,
        "steering_path": args.steering_path,
        "val_csv": args.val_csv,
        "num_situations": args.num_situations,
        "filter_bucket_label": args.filter_bucket_label,
        "temperature": args.temperature,
        "layer_candidates": [L if isinstance(L, int) else list(L) for L in LAYER_CANDIDATES],
        "alphas": ALPHAS,
        "timestamp": datetime.now().isoformat(),
        "total_sweep_time_seconds": round(total_sweep_time, 1),
    }
    save_sweep_json(config, sweep_results, base_ra, json_path)

    # --- Plot ---
    plot_sweep(sweep_results, base_ra, LAYER_CANDIDATES, png_path)

    # --- Cleanup ---
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
