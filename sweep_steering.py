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

import re
from contextlib import contextmanager

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT


# ============================================================================
# Inlined steering/eval utilities (self-contained, independent of evaluate.py)
# ============================================================================

class SteeringHook:
    """Forward hook that adds a steering vector to residual stream activations."""

    def __init__(self, steering_vector, alpha=1.0):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[:, -1, :] += self.alpha * self.steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            modified = output.clone()
            modified[:, -1, :] += self.alpha * self.steering_vector.to(modified.device)
            return modified

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def steering_context(model, steering_vector, alpha, layer):
    if steering_vector is None or alpha == 0:
        yield
        return
    hook = SteeringHook(steering_vector, alpha)
    target_layer = model.model.layers[layer]
    hook.register(target_layer)
    try:
        yield
    finally:
        hook.remove()


def load_steering_vector(path):
    """Load a steering vector from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        vector = data.get("direction", data.get("vector"))
        layer = data.get("layer", 14)
        metadata = {k: v for k, v in data.items() if k not in ("vector", "direction")}
    else:
        vector = data
        layer = 14
        metadata = {}
    return vector, layer, metadata


def remove_instruction_suffix(prompt):
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def extract_choice_permissive(response, num_options):
    """Permissive answer extraction matching many formats."""
    response_lower = response.lower().strip()
    valid_letters = [chr(ord('a') + i) for i in range(num_options)]
    valid_numbers = [str(i + 1) for i in range(num_options)]
    valid_options = valid_letters + valid_numbers

    json_match = re.search(r'\{"answer"\s*:\s*"([a-z0-9]+)"\}', response_lower)
    if json_match and json_match.group(1) in valid_options:
        return json_match.group(1)

    answer_match = re.search(r'(?:the\s+)?answer[:\s]+(?:is\s+)?(?:option\s+)?([a-z0-9])\b', response_lower)
    if answer_match and answer_match.group(1) in valid_options:
        return answer_match.group(1)

    choice_match = re.search(r"(?:i(?:'d)?\s+)?(?:choose|select|pick|chose|selected|picking)\s+(?:option\s+)?([a-z0-9])\b", response_lower)
    if choice_match and choice_match.group(1) in valid_options:
        return choice_match.group(1)

    option_is_match = re.search(r'\boption\s+([a-z0-9])\s+(?:is|would be|seems)\b', response_lower)
    if option_is_match and option_is_match.group(1) in valid_options:
        return option_is_match.group(1)

    go_with_match = re.search(r'go\s+with\s+(?:option\s+)?([a-z0-9])\b', response_lower)
    if go_with_match and go_with_match.group(1) in valid_options:
        return go_with_match.group(1)

    last_part = response_lower[-300:]

    option_match = re.search(r'\boption\s+([a-z0-9])\b', last_part)
    if option_match and option_match.group(1) in valid_options:
        return option_match.group(1)

    paren_matches = re.findall(r'\(([a-z0-9])\)', last_part)
    for match in reversed(paren_matches):
        if match in valid_options:
            return match

    conclusion_match = re.search(r'(?:therefore|thus|so|hence),?\s+(?:option\s+)?([a-z0-9])\b', last_part)
    if conclusion_match and conclusion_match.group(1) in valid_options:
        return conclusion_match.group(1)

    last_150 = response_lower[-150:]
    last_found = None
    for opt in valid_options:
        matches = list(re.finditer(r'\b' + re.escape(opt) + r'\b', last_150))
        if matches:
            last_pos = matches[-1].start()
            if last_found is None or last_pos > last_found[1]:
                last_found = (opt, last_pos)
    if last_found:
        return last_found[0]

    return None


def load_situations(val_csv, num_situations):
    """Load and parse situations from a validation CSV."""
    df = pd.read_csv(val_csv)
    situations = []
    for sit_id in df["situation_id"].unique()[:num_situations]:
        sit_data = df[df["situation_id"] == sit_id]
        prompt = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        options = {}
        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            option_data = {
                "type": row["option_type"],
                "is_best_cara": row["is_best_cara_display"] == True
            }
            options[letter] = option_data
            options[number] = option_data
        situations.append({
            "situation_id": sit_id,
            "prompt": prompt,
            "num_options": num_options,
            "options": options
        })
    return situations


def run_evaluation(model, tokenizer, situations, steering_vector,
                   alpha=0.0, steering_layer=14,
                   temperature=0.6, max_new_tokens=4096,
                   max_time_per_generation=120,
                   disable_thinking=False, no_save_responses=True,
                   verbose=True):
    """Run evaluation loop over situations with steering."""
    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        sit_start = time.time()
        prompt = remove_instruction_suffix(sit["prompt"])
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        else:
            template_kwargs["enable_thinking"] = True
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_start = time.time()
        with torch.no_grad(), steering_context(model, steering_vector, alpha, steering_layer):
            if temperature == 0:
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation)
            else:
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature, do_sample=True,
                    top_p=0.95, top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation)
        gen_elapsed = time.time() - gen_start

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        choice = extract_choice_permissive(response, sit["num_options"])

        if choice and choice in sit["options"]:
            option_type = sit["options"][choice]["type"]
            results.append({
                "situation_id": sit["situation_id"],
                "choice": choice,
                "option_type": option_type,
                "is_cooperate": option_type == "Cooperate",
                "is_rebel": option_type == "Rebel",
                "is_steal": option_type == "Steal",
                "is_best_cara": sit["options"][choice]["is_best_cara"],
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": None, "option_type": None,
                "is_cooperate": None, "is_rebel": None,
                "is_steal": None, "is_best_cara": None,
            })
            failed_responses.append({
                "situation_id": sit["situation_id"],
                "num_options": sit["num_options"],
                "response": response
            })

        generation_times.append(gen_elapsed)
        if verbose:
            status = "OK" if choice else "PARSE_FAIL"
            avg_time = sum(generation_times) / len(generation_times)
            remaining = avg_time * (len(situations) - i - 1)
            print(f"  [{i+1}/{len(situations)}] sit_id={sit['situation_id']} | {status} | "
                  f"{int(num_generated_tokens)} tokens | {gen_elapsed:.1f}s | ETA: {remaining/60:.1f}min")

    total_elapsed = time.time() - eval_start_time
    valid = [r for r in results if r["is_cooperate"] is not None]
    if valid:
        cooperate_rate = sum(r["is_cooperate"] for r in valid) / len(valid)
        rebel_rate = sum(r["is_rebel"] for r in valid) / len(valid)
        steal_rate = sum(r["is_steal"] for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = 0

    parse_rate = len(valid) / len(results) if results else 0

    return {
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "cara_rate": cara_rate,
        "parse_rate": parse_rate,
        "num_valid": len(valid),
        "num_total": len(results),
        "results": results,
        "failed_responses": failed_responses,
        "generation_times": generation_times,
        "total_elapsed": total_elapsed,
    }

DEFAULT_LAYERS = [10, 12, 14]
DEFAULT_ALPHAS = [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]


def plot_sweep(results, base_ra, layer_candidates, output_path):
    """Plot steering performance grid across layers."""
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
        layer_data = df_plot[df_plot["layer"] == L].sort_values("alpha")
        ax.plot(layer_data["alpha"], layer_data["safe_acc"],
                marker="o", label="Safe Acc", color="green")
        ax.plot(layer_data["alpha"], layer_data["risky_acc"],
                marker="x", label="Risky Acc", color="red")
        ax.axhline(y=base_ra, color="gray", linestyle="--", alpha=0.5,
                   label=f"Base Safe ({base_ra:.0%})")
        ax.set_title(f"Layer {L}")
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
                        default="data/2026_03_22_medium_stakes_val_set_500_Rebels.csv")
    parser.add_argument("--num_situations", type=int, default=20,
                        help="Situations per combo (default 20 for speed)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--max_time_per_generation", type=float, default=120)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer candidates (default: 7 10 14 18 21 24)")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Alpha values (default: 0 0.5 1 1.5 2 3 5 8 10)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output prefix for PNG and JSON files")
    args = parser.parse_args()

    LAYER_CANDIDATES = args.layers or DEFAULT_LAYERS
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

    # Validate layer candidates
    num_model_layers = len(model.model.layers)
    for L in LAYER_CANDIDATES:
        if L >= num_model_layers:
            print(f"ERROR: Layer {L} >= model layer count ({num_model_layers}). "
                  f"Valid range: 0-{num_model_layers - 1}")
            sys.exit(1)

    # Note: do NOT auto-disable thinking. The steering vector is built with
    # enable_thinking=True, and the canonical evaluator runs with thinking ON.
    # Sweep exploration should match that context to produce representative results.

    # --- Load steering vector (once) ---
    print(f"Loading steering vector from {args.steering_path}...")
    steering_vector, _, metadata = load_steering_vector(args.steering_path)
    print(f"  Vector shape: {steering_vector.shape}")
    if metadata:
        print(f"  Generated from: {metadata.get('num_pairs', '?')} pairs")

    # --- Load data (once) ---
    print("Loading validation data...")
    situations = load_situations(args.val_csv, args.num_situations)
    print(f"Loaded {len(situations)} situations")

    # --- Run baseline (no steering) ---
    print("\nRunning baseline (no steering)...")
    baseline = run_evaluation(
        model, tokenizer, situations, steering_vector=None,
        alpha=0.0, steering_layer=14,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        max_time_per_generation=args.max_time_per_generation,
        disable_thinking=args.disable_thinking,
        no_save_responses=True, verbose=False,
    )
    base_ra = baseline["cooperate_rate"]
    print(f"Baseline cooperate rate: {base_ra:.1%}")
    print(f"Baseline parse rate: {baseline['parse_rate']:.1%}")

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
        print(f"[{combo_idx + 1}/{total_combos}] Layer={L}, Alpha={alpha} ...", end=" ", flush=True)

        eval_result = run_evaluation(
            model, tokenizer, situations, steering_vector,
            alpha=alpha, steering_layer=L,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            no_save_responses=True, verbose=False,
        )

        safe_acc = eval_result["cooperate_rate"]
        risky_acc = eval_result["rebel_rate"] + eval_result["steal_rate"]
        combo_elapsed = time.time() - combo_start

        sweep_results.append({
            "layer": L,
            "alpha": alpha,
            "safe_acc": safe_acc,
            "risky_acc": risky_acc,
            "cooperate_rate": eval_result["cooperate_rate"],
            "rebel_rate": eval_result["rebel_rate"],
            "steal_rate": eval_result["steal_rate"],
            "cara_rate": eval_result["cara_rate"],
            "parse_rate": eval_result["parse_rate"],
        })

        remaining = combo_elapsed * (total_combos - combo_idx - 1)
        print(f"safe={safe_acc:.0%} risky={risky_acc:.0%} "
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
        "temperature": args.temperature,
        "layer_candidates": LAYER_CANDIDATES,
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
