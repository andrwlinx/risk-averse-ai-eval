#!/usr/bin/env python3
"""
Generate steering vectors for activation engineering.

Contrasts Risk-Averse (010_only) vs Risk-Neutral (lin_only) chain-of-thought
activations at the token following <think> to derive a steering direction.

Uses transformers library for compatibility with evaluate.py.
"""

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_think_token_position(tokenizer, input_ids):
    """Find the position of the <think> token in the input.

    Returns the position of <think> token, or -1 if not found.
    We want activations at the token AFTER <think>.
    """
    # Common think token patterns
    think_patterns = ["<think>", "<|think|>", "▁<think>", " <think>"]

    # Decode each token to find <think>
    for pos in range(len(input_ids)):
        token_text = tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
        for pattern in think_patterns:
            if pattern in token_text:
                return pos

    # Fallback: search in decoded text and map back
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    if "<think>" in full_text:
        # Find character position and estimate token position
        char_pos = full_text.find("<think>")
        # Rough estimate: decode up to that point
        prefix = full_text[:char_pos + len("<think>")]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        return len(prefix_tokens) - 1

    return -1


def get_activations_at_position(model, tokenizer, text, layer, position="think"):
    """Get activations at a specific position in the text.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        text: Input text (should include <think> tag)
        layer: Which layer to extract from (0-indexed)
        position: "think" to get position after <think>, "last" for last token,
                  or an int for specific position

    Returns:
        Tensor of shape (hidden_size,) or None if position not found
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0].tolist()

    # Determine target position
    if position == "think":
        think_pos = find_think_token_position(tokenizer, input_ids)
        if think_pos == -1:
            return None
        # Get position AFTER <think> token
        target_pos = think_pos + 1
        if target_pos >= len(input_ids):
            target_pos = len(input_ids) - 1
    elif position == "last":
        target_pos = len(input_ids) - 1
    else:
        target_pos = int(position)

    # Storage for captured activation
    activation = {}

    def hook_fn(module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Capture activation at target position
        activation["value"] = hidden_states[0, target_pos, :].detach().clone()

    # Register hook on the target layer
    # For Qwen2.5 / Llama-style models: model.model.layers[layer]
    target_layer = model.model.layers[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return activation.get("value")


def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors from contrastive CoTs")
    parser.add_argument(
        "--training_csv",
        type=str,
        default="data/2026_01_29_new_full_training_set_with_CoTs_Sonnet_4_5.csv",
        help="Path to training set with CoT columns"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to use for extracting activations"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Layer to extract activations from (0-indexed, default: 14 for mid-layer)"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
        help="Number of CoT pairs to use for averaging"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="risk_averse_steering_vector.pt",
        help="Output path for the steering vector"
    )
    parser.add_argument(
        "--averse_type",
        type=str,
        default="too_risk",
        help="Type label for risk-averse examples (in type_column)"
    )
    parser.add_argument(
        "--neutral_type",
        type=str,
        default="lin",
        help="Type label for risk-neutral examples (in type_column)"
    )
    parser.add_argument(
        "--position",
        type=str,
        default="think",
        help="Position to capture: 'think' (after <think>), 'last', or integer"
    )
    # Column name arguments for Elliott's CSV format
    parser.add_argument(
        "--type_column",
        type=str,
        default="rejected_type",
        help="Column containing type labels (default: 'rejected_type' for Elliott's format)"
    )
    parser.add_argument(
        "--averse_column",
        type=str,
        default="chosen_full",
        help="Column containing risk-averse (chosen) CoT text (default: 'chosen_full')"
    )
    parser.add_argument(
        "--neutral_column",
        type=str,
        default="rejected_full",
        help="Column containing risk-neutral (rejected) CoT text (default: 'rejected_full')"
    )
    args = parser.parse_args()

    # Check if training file exists
    if not Path(args.training_csv).exists():
        print(f"ERROR: Training file not found: {args.training_csv}")
        print("\nThis script expects Elliott's CSV with columns:")
        print(f"  - '{args.type_column}': containing type labels like '010_only' and 'lin_only'")
        print(f"  - '{args.averse_column}': containing risk-averse (chosen) CoT text")
        print(f"  - '{args.neutral_column}': containing risk-neutral (rejected) CoT text")
        print("\nPlease ensure the training set is in the data/ directory.")
        sys.exit(1)

    print(f"Loading training data from {args.training_csv}...")
    df = pd.read_csv(args.training_csv)

    # Validate required columns
    required_cols = [args.type_column, args.averse_column, args.neutral_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Elliott's logic:
    # - Averse examples: rows where rejected_type == '010_only', use 'chosen_full' column
    # - Neutral examples: rows where rejected_type == 'lin_only', use 'rejected_full' column
    averse_df = df[df[args.type_column] == args.averse_type]
    neutral_df = df[df[args.type_column] == args.neutral_type]

    averse_cots = averse_df[args.averse_column].dropna().tolist()
    neutral_cots = neutral_df[args.neutral_column].dropna().tolist()

    print(f"Found {len(averse_cots)} risk-averse examples ({args.averse_type} -> {args.averse_column})")
    print(f"Found {len(neutral_cots)} risk-neutral examples ({args.neutral_type} -> {args.neutral_column})")

    if len(averse_cots) == 0 or len(neutral_cots) == 0:
        print(f"\nERROR: Need both averse and neutral CoTs")
        print(f"Available types in '{args.type_column}': {df[args.type_column].unique().tolist()}")
        sys.exit(1)

    # Limit to requested number of pairs
    num_pairs = min(args.num_pairs, len(averse_cots), len(neutral_cots))
    averse_cots = averse_cots[:num_pairs]
    neutral_cots = neutral_cots[:num_pairs]

    print(f"\nUsing {num_pairs} pairs for steering vector computation")
    print(f"Extracting activations from layer {args.layer} at position '{args.position}'")

    # Load model
    print(f"\nLoading model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Validate layer number
    num_layers = len(model.model.layers)
    if args.layer >= num_layers:
        print(f"ERROR: Layer {args.layer} out of range. Model has {num_layers} layers (0-{num_layers-1})")
        sys.exit(1)

    print(f"Model loaded. {num_layers} layers total, extracting from layer {args.layer}")

    # Compute activation differences
    vector_diffs = []
    skipped = 0

    for i, (averse_cot, neutral_cot) in enumerate(tqdm(
        zip(averse_cots, neutral_cots),
        total=num_pairs,
        desc="Computing activation differences"
    )):
        # Get activations for risk-averse CoT
        act_averse = get_activations_at_position(
            model, tokenizer, averse_cot, args.layer, args.position
        )

        # Get activations for risk-neutral CoT
        act_neutral = get_activations_at_position(
            model, tokenizer, neutral_cot, args.layer, args.position
        )

        if act_averse is None or act_neutral is None:
            skipped += 1
            if skipped <= 3:
                print(f"\n  Warning: Could not find target position in pair {i+1}")
            continue

        # Compute difference: averse - neutral
        diff = act_averse - act_neutral
        vector_diffs.append(diff)

    if len(vector_diffs) == 0:
        print("\nERROR: Could not compute any valid activation differences")
        print("Check that your CoTs contain <think> tags")
        sys.exit(1)

    print(f"\nComputed {len(vector_diffs)} valid differences (skipped {skipped})")

    # Average to get final steering vector
    steering_vector = torch.stack(vector_diffs).mean(dim=0)

    # Save with metadata
    save_data = {
        "vector": steering_vector,
        "layer": args.layer,
        "position": args.position,
        "num_pairs": len(vector_diffs),
        "base_model": args.base_model,
        "averse_type": args.averse_type,
        "neutral_type": args.neutral_type,
        "averse_column": args.averse_column,
        "neutral_column": args.neutral_column,
        "type_column": args.type_column,
        "hidden_size": steering_vector.shape[0]
    }

    torch.save(save_data, args.output)

    print(f"\n{'='*50}")
    print("STEERING VECTOR GENERATED")
    print("="*50)
    print(f"Output: {args.output}")
    print(f"Shape: {steering_vector.shape}")
    print(f"Layer: {args.layer}")
    print(f"Position: {args.position}")
    print(f"Pairs used: {len(vector_diffs)}")
    print(f"Vector norm: {steering_vector.norm().item():.4f}")
    print(f"Vector mean: {steering_vector.mean().item():.6f}")
    print(f"Vector std: {steering_vector.std().item():.4f}")
    print("="*50)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTo use this vector with evaluate.py, run:")
    print(f"  python evaluate.py --steering_path {args.output} --alpha 1.0")


if __name__ == "__main__":
    main()
