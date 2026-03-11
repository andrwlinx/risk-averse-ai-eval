#!/usr/bin/env python3
"""
Run ICV steering evaluations saving each (layer, alpha) combo as an individual
evaluate.py-format JSON file.

Loads the model once and iterates over all combos, writing:
    {results_dir}/eval_{model_short}_{layer_label}_alpha{alpha}_{timestamp}.json

Usage:
    python run_icv_eval.py \
        --base_model Qwen/Qwen3-8B \
        --steering_path qwen3_8b_icv_layer18.pt \
        --results_dir results/my_run \
        --val_csv data/in_distribution_val_set.csv \
        --filter_bucket_label lin_only
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

from evaluate import (
    load_situations,
    load_steering_vector,
    run_evaluation,
    convert_numpy,
)

DEFAULT_SINGLE_LAYERS = {
    "Qwen/Qwen3-1.7B": [7, 10, 14, 18, 21],
    "Qwen/Qwen3-8B":   [10, 14, 18, 22, 26],
    "Qwen/Qwen3-14B":  [12, 16, 20, 24, 28],
}
DEFAULT_MULTILAYER = {
    "Qwen/Qwen3-1.7B": [[7, 14], [10, 14, 18], [7, 14, 21]],
    "Qwen/Qwen3-8B":   [[10, 18], [14, 18, 22], [10, 18, 26]],
    "Qwen/Qwen3-14B":  [[12, 20], [16, 20, 24], [12, 20, 28]],
}
DEFAULT_ALPHAS = [-3.0, -1.0, 0.0, 1.0, 3.0]  # 5 alphas — drops ±5 vs old 7-alpha grid


def layer_label(L):
    if isinstance(L, (list, tuple)):
        return "+".join(str(x) for x in L)
    return str(L)


def save_eval_json(output_path, base_model, steering_path, alpha, layer,
                   val_csv, filter_bucket, max_new_tokens, temperature,
                   enable_thinking, eval_result):
    """Save a single combo result in evaluate.py's standard format."""
    lbl = layer_label(layer)
    output_data = convert_numpy({
        "evaluation_config": {
            "base_model": base_model,
            "model_path": None,
            "steering_path": steering_path,
            "steering_alpha": alpha,
            "steering_layer": layer if isinstance(layer, int) else None,
            "steering_layers": list(layer) if isinstance(layer, (list, tuple)) else None,
            "layer_label": lbl,
            "val_csv": val_csv,
            "filter_bucket_label": filter_bucket,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
            "num_situations": eval_result["num_total"],
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": {
            "parse_rate": eval_result["parse_rate"],
            "cooperate_rate": eval_result["cooperate_rate"],
            "rebel_rate": eval_result["rebel_rate"],
            "steal_rate": eval_result["steal_rate"],
            "best_cara_rate": eval_result["cara_rate"],
            "best_linear_rate": eval_result["linear_rate"],
        },
        "num_valid": eval_result["num_valid"],
        "num_total": eval_result["num_total"],
        "results": eval_result["results"],
    })
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--steering_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--val_csv", type=str,
                        default="data/in_distribution_val_set.csv")
    parser.add_argument("--filter_bucket_label", type=str, default="lin_only")
    parser.add_argument("--num_situations", type=int, default=50)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Single-layer candidates (default: model-specific)")
    parser.add_argument("--multilayer_combos", type=str, nargs="*", default=None,
                        help="Multilayer combos as comma-separated ints e.g. '10,18'")
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--max_time_per_generation", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--thinking_prefix", type=str, default=None,
                        help="Pre-fill the <think> block with this text then close it, "
                             "forcing the model to skip extended reasoning.")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.base_model.replace("/", "_").replace("-", "")

    # Resolve layer candidates
    single_layers = args.layers or DEFAULT_SINGLE_LAYERS.get(args.base_model, [14])
    multilayer = []
    if args.multilayer_combos is not None:
        for s in args.multilayer_combos:
            multilayer.append([int(x) for x in s.split(",")])
    else:
        multilayer = DEFAULT_MULTILAYER.get(args.base_model, [])
    layer_candidates = single_layers + multilayer

    total_combos = len(layer_candidates) * len(args.alphas)
    print(f"Model:    {args.base_model}")
    print(f"Vector:   {args.steering_path}")
    print(f"Layers:   {layer_candidates}  ({len(layer_candidates)} configs)")
    print(f"Alphas:   {args.alphas}  ({len(args.alphas)} values)")
    print(f"Combos:   {total_combos}")
    print(f"Results:  {args.results_dir}")

    # Thinking mode
    disable_thinking = args.disable_thinking
    enable_thinking = args.enable_thinking
    if not disable_thinking and not enable_thinking:
        # Auto-disable for base model (no adapter) — but allow override
        disable_thinking = True
        print("Note: auto-disabling thinking (use --enable_thinking to override)")
    thinking_active = not disable_thinking

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load steering vector and situations
    steering_vector, _, sv_meta = load_steering_vector(args.steering_path)
    print(f"Steering vector shape: {steering_vector.shape}")

    situations = load_situations(args.val_csv, args.num_situations,
                                 filter_bucket_label=args.filter_bucket_label)
    print(f"Situations: {len(situations)} (filtered to '{args.filter_bucket_label}')")

    # Sweep
    sweep_start = time.time()
    for combo_idx, L in enumerate(layer_candidates):
        for alpha in args.alphas:
            lbl = layer_label(L)
            combo_start = time.time()
            print(f"[{combo_idx * len(args.alphas) + args.alphas.index(alpha) + 1}/{total_combos}]"
                  f" layer={lbl} alpha={alpha:+.1f} ...", end=" ", flush=True)

            result = run_evaluation(
                model, tokenizer, situations, steering_vector,
                alpha=alpha, steering_layer=L,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_time_per_generation=args.max_time_per_generation,
                disable_thinking=disable_thinking,
                no_save_responses=False,
                verbose=False,
                thinking_prefix=args.thinking_prefix,
            )

            elapsed = time.time() - combo_start
            lin = result["linear_rate"]
            cara = result["cara_rate"]
            parse = result["parse_rate"]
            print(f"lin={lin:.0%} cara={cara:.0%} parse={parse:.0%} ({elapsed:.0f}s)")

            # Save in evaluate.py format
            alpha_str = f"{alpha:+.1f}".replace("+", "pos").replace("-", "neg").replace(".", "p")
            fname = f"eval_{model_short}_layer{lbl}_alpha{alpha_str}_{timestamp}.json"
            out_path = Path(args.results_dir) / fname
            save_eval_json(
                out_path, args.base_model, args.steering_path, alpha, L,
                args.val_csv, args.filter_bucket_label,
                args.max_new_tokens, args.temperature, thinking_active, result,
            )

    total_time = time.time() - sweep_start
    print(f"\nDone — {total_combos} combos in {total_time/60:.1f} min")
    print(f"Results saved to: {args.results_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
