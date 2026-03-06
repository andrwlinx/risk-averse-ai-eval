#!/usr/bin/env python3
"""
Generate steering vectors or in-context vectors (ICVs) for activation engineering.

Two modes:

  Default (contrastive CoT):
    For each same-row pair in the training set (filtered to lin_only situations),
    extracts hidden-state activations at the <think> token for both the risk-averse
    CoT (chosen_full) and the risk-neutral CoT (rejected_full), computes the
    difference (averse - neutral), and averages across pairs to produce a steering
    vector. The resulting vector points in the direction of risk-averse reasoning.

  ICV mode (--icv_mode):
    Builds two few-shot prefixes from lin_only training CoTs — one averse, one
    neutral — then for each target situation computes the activation difference at
    the <think> token between (averse_prefix + situation) and
    (neutral_prefix + situation). Averaging these differences yields an in-context
    vector that captures the effect of risk-averse demonstrations on model state.

Both modes produce a normalised steering vector saved as a .pt file compatible
with evaluate.py and sweep_steering.py.
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def inject_concise_instruction(text, instruction):
    """Inject an instruction string at the start of the user turn in a chat-formatted text.

    Looks for the Qwen chat marker '<|im_start|>user\\n' and inserts the instruction
    immediately after it. Falls back to prepending to the full text if the marker is
    not found.
    """
    marker = "<|im_start|>user\n"
    if marker in text:
        idx = text.find(marker) + len(marker)
        return text[:idx] + instruction + "\n" + text[idx:]
    # Fallback: prepend to the full text
    return instruction + "\n" + text


def find_think_token_position(tokenizer, input_ids):
    """Return the token index of the last <think> token sequence in input_ids.

    Checks multiple surface forms ("<think>", "<|think|>", " <think>") and the
    single special-token representation. Returns the index of the final token of
    the last match, or -1 if not found. Callers should use last_pos + 1 as the
    activation extraction position (i.e. the first token inside the thinking block).
    """
    think_patterns = ["<think>", "<|think|>", " <think>"]
    seqs = []
    for p in think_patterns:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    # Also check single special token representation
    tid = tokenizer.convert_tokens_to_ids("<think>")
    if tid is not None and tid != tokenizer.unk_token_id:
        seqs.append([tid])

    last_pos = -1
    for seq in seqs:
        slen = len(seq)
        for i in range(len(input_ids) - slen + 1):
            if input_ids[i:i + slen] == seq:
                last_pos = max(last_pos, i + slen - 1)
    return last_pos


def get_activations_at_position(model, tokenizer, text, layer, position="think"):
    """Extract the hidden state at a specific token position from the given layer.

    Args:
        model: The loaded transformer model (eval mode, bfloat16).
        tokenizer: The corresponding tokenizer.
        text: Full input text to tokenize and run through the model.
        layer: Layer index to extract from (0-indexed; layer 0 = first transformer block).
        position: Where to extract:
            "think" — token immediately after the last <think> tag (returns None if absent);
            "last"  — final token in the sequence;
            int     — explicit token index.

    Returns:
        Tensor of shape (hidden_size,), or None if the requested position cannot be found.
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

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embeddings, hidden_states[layer+1] = output of layer `layer`
    hidden = outputs.hidden_states[layer + 1][0, target_pos, :].detach().clone()
    return hidden


def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors from contrastive CoTs")
    parser.add_argument(
        "--training_csv",
        type=str,
        default="data/2026_01_29_lin_only_training_set_CoTs_500_Sonnet_4_5.csv",
        help="Path to training set with CoT columns"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
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
        "--position",
        type=str,
        default="think",
        help="Position to capture: 'think' (after <think>), 'last', or integer"
    )
    # Filtering and column arguments
    parser.add_argument(
        "--filter_column",
        type=str,
        default="low_bucket_label",
        help="Column to filter rows by (default: 'low_bucket_label')"
    )
    parser.add_argument(
        "--filter_value",
        type=str,
        default="lin_only",
        help="Value to filter on for clean contrastive pairs (default: 'lin_only')"
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
    parser.add_argument(
        "--concise_prompt",
        action="store_true",
        help="Inject 'Be concise' into the user turn of each CoT text before extracting activations, "
             "so the derived vector captures 'Concise Risk-Aversion'."
    )
    # --- ICV mode args ---
    parser.add_argument(
        "--icv_mode",
        action="store_true",
        help="Compute an In-Context Vector (ICV) instead of a contrastive CoT diff. "
             "Measures the activation shift caused by prepending risk-averse demonstrations "
             "to a situation prompt. The resulting vector is applied at inference the same way."
    )
    parser.add_argument(
        "--icv_situations_csv",
        type=str,
        default="data/2026_01_29_new_val_set_probabilities_add_to_100.csv",
        help="CSV of situations to compute ICV over (uses first --num_pairs unique situation_ids). "
             "Defaults to the validation set."
    )
    parser.add_argument(
        "--num_icv_demos",
        type=int,
        default=5,
        help="Number of risk-averse CoT demonstrations to prepend when computing the ICV (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair sampling (default: 42)"
    )
    parser.add_argument(
        "--outlier_pct",
        type=float,
        default=5.0,
        help="Percentage of highest-norm diffs to drop before averaging (default: 5.0)"
    )
    args = parser.parse_args()

    # Check if training file exists
    if not Path(args.training_csv).exists():
        print(f"ERROR: Training file not found: {args.training_csv}")
        print("\nThis script expects a CSV with columns:")
        print(f"  - '{args.filter_column}': for filtering rows (e.g. 'lin_only')")
        print(f"  - '{args.averse_column}': containing risk-averse (chosen) CoT text")
        print(f"  - '{args.neutral_column}': containing risk-neutral (rejected) CoT text")
        print("\nPlease ensure the training set is in the data/ directory.")
        sys.exit(1)

    print(f"Loading training data from {args.training_csv}...")
    df = pd.read_csv(args.training_csv)

    # Validate required columns
    required_cols = [args.filter_column, args.averse_column, args.neutral_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Elliott's contrastive pairing strategy:
    # Filter to specific situation type (e.g. lin_only) and use same-row pairs
    # so chosen_full (averse) and rejected_full (neutral) address the exact same scenario
    filtered_df = df[df[args.filter_column] == args.filter_value]
    filtered_df = filtered_df.dropna(subset=[args.averse_column, args.neutral_column])

    print(f"Found {len(filtered_df)} paired examples where {args.filter_column} == '{args.filter_value}'")
    print(f"  Averse column: {args.averse_column}, Neutral column: {args.neutral_column}")

    if len(filtered_df) == 0:
        print(f"\nERROR: No rows match filter {args.filter_column} == '{args.filter_value}'")
        print(f"Available values in '{args.filter_column}': {df[args.filter_column].unique().tolist()}")
        sys.exit(1)

    # Limit to requested number of pairs (random sample for unbiased selection)
    num_pairs = min(args.num_pairs, len(filtered_df))
    filtered_df = filtered_df.sample(n=num_pairs, random_state=args.seed)

    averse_cots = filtered_df[args.averse_column].tolist()
    neutral_cots = filtered_df[args.neutral_column].tolist()

    print(f"\nUsing {num_pairs} pairs for steering vector computation")
    print(f"Extracting activations from layer {args.layer} at position '{args.position}'")
    if args.concise_prompt:
        print("Concise prompt injection: ENABLED (injecting 'Be concise' into user turn)")

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

    # -------------------------------------------------------------------------
    # ICV MODE: measure activation shift from prepending risk-averse demos
    # -------------------------------------------------------------------------
    if args.icv_mode:
        print(f"\n=== ICV MODE ===")
        print(f"Using first {args.num_icv_demos} averse CoTs as demonstrations")
        print(f"Computing shift over {num_pairs} situations from: {args.icv_situations_csv}")

        # Build averse and neutral prefixes from first N CoTs in training set
        averse_demos = averse_cots[:args.num_icv_demos]
        neutral_demos = neutral_cots[:args.num_icv_demos]
        averse_prefix = "\n\n".join(averse_demos)
        neutral_prefix = "\n\n".join(neutral_demos)
        print(f"Averse prefix: {len(tokenizer.encode(averse_prefix))} tokens")
        print(f"Neutral prefix: {len(tokenizer.encode(neutral_prefix))} tokens")

        # Load situation prompts from the ICV situations CSV
        sit_df = pd.read_csv(args.icv_situations_csv)
        # Get one prompt per unique situation_id
        sit_prompts = []
        for sit_id in sit_df["situation_id"].unique():
            sit_rows = sit_df[sit_df["situation_id"] == sit_id]
            raw_prompt = sit_rows["prompt_text"].iloc[0]
            # Strip the "You can think before answering..." instruction suffix
            clean = re.sub(
                r"\s*You can think before answering,.*?would select\.",
                "", raw_prompt, flags=re.IGNORECASE | re.DOTALL
            ).strip()
            sit_prompts.append(clean)
            if len(sit_prompts) >= num_pairs:
                break

        print(f"Loaded {len(sit_prompts)} situation prompts")

        vector_diffs = []
        skipped = 0

        for i, prompt in enumerate(tqdm(sit_prompts, desc="Computing ICV")):
            # Format the test situation as a chat prompt (up to <think>)
            messages = [{"role": "user", "content": prompt}]
            base_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # The Qwen3 generation prompt already ends with <think>; if not, append it
            if not base_text.rstrip().endswith("<think>"):
                base_text = base_text.rstrip() + "<think>"

            pos_text = averse_prefix + "\n\n" + base_text
            neg_text = neutral_prefix + "\n\n" + base_text

            # Activation at <think> position with averse vs neutral prefix
            h_averse = get_activations_at_position(
                model, tokenizer, pos_text, args.layer, "think"
            )
            h_neutral = get_activations_at_position(
                model, tokenizer, neg_text, args.layer, "think"
            )

            if h_averse is None or h_neutral is None:
                skipped += 1
                if skipped <= 3:
                    print(f"\n  Warning: Could not find <think> position in situation {i+1}")
                continue

            # ICV contribution: averse vs neutral demo context at <think> token
            vector_diffs.append(h_averse - h_neutral)

        if len(vector_diffs) == 0:
            print("\nERROR: Could not compute any valid ICV differences")
            sys.exit(1)

        print(f"\nComputed {len(vector_diffs)} valid ICV differences (skipped {skipped})")

        if args.outlier_pct > 0 and len(vector_diffs) > 10:
            norms = torch.tensor([d.norm().item() for d in vector_diffs])
            threshold = torch.quantile(norms, 1.0 - args.outlier_pct / 100.0)
            kept = [d for d, n in zip(vector_diffs, norms.tolist()) if n <= threshold.item()]
            print(f"Outlier filter: kept {len(kept)}/{len(vector_diffs)} diffs (threshold norm={threshold:.2f})")
            vector_diffs = kept

        steering_vector = torch.stack(vector_diffs).mean(dim=0)
        raw_norm = steering_vector.norm().item()
        steering_vector = steering_vector / steering_vector.norm()

        save_data = {
            "vector": steering_vector,
            "layer": args.layer,
            "position": "think",
            "num_pairs": len(vector_diffs),
            "base_model": args.base_model,
            "icv_mode": True,
            "num_icv_demos": args.num_icv_demos,
            "icv_situations_csv": args.icv_situations_csv,
            "filter_column": args.filter_column,
            "filter_value": args.filter_value,
            "hidden_size": steering_vector.shape[0],
            "raw_norm": raw_norm,
        }
        torch.save(save_data, args.output)

        print(f"\n{'='*50}")
        print("IN-CONTEXT VECTOR GENERATED")
        print("="*50)
        print(f"Output: {args.output}")
        print(f"Shape: {steering_vector.shape}")
        print(f"Layer: {args.layer}")
        print(f"Demos used: {args.num_icv_demos}")
        print(f"Situations averaged: {len(vector_diffs)}")
        print(f"Vector norm: {steering_vector.norm().item():.4f} (normalised; raw was {raw_norm:.4f})")
        print(f"Vector mean: {steering_vector.mean().item():.6f}")
        print(f"Vector std: {steering_vector.std().item():.4f}")
        print(f"Polarity: averse vs neutral demos")
        print("="*50)
        print(f"\nTo use this ICV with evaluate.py, run:")
        print(f"  python evaluate.py --steering_path {args.output} --alpha 1.0")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return

    # -------------------------------------------------------------------------
    # DEFAULT MODE: contrastive CoT activation diff at <think> position
    # -------------------------------------------------------------------------

    # Compute activation differences
    vector_diffs = []
    skipped = 0

    concise_instruction = "Be concise and go straight to the answer after your thinking process."

    for i, (averse_cot, neutral_cot) in enumerate(tqdm(
        zip(averse_cots, neutral_cots),
        total=num_pairs,
        desc="Computing activation differences"
    )):
        # Optionally inject concise instruction into the user turn
        if args.concise_prompt:
            averse_text = inject_concise_instruction(averse_cot, concise_instruction)
            neutral_text = inject_concise_instruction(neutral_cot, concise_instruction)
        else:
            averse_text = averse_cot
            neutral_text = neutral_cot

        # Get activations for risk-averse CoT
        act_averse = get_activations_at_position(
            model, tokenizer, averse_text, args.layer, args.position
        )

        # Get activations for risk-neutral CoT
        act_neutral = get_activations_at_position(
            model, tokenizer, neutral_text, args.layer, args.position
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

    if args.outlier_pct > 0 and len(vector_diffs) > 10:
        norms = torch.tensor([d.norm().item() for d in vector_diffs])
        threshold = torch.quantile(norms, 1.0 - args.outlier_pct / 100.0)
        kept = [d for d, n in zip(vector_diffs, norms.tolist()) if n <= threshold.item()]
        print(f"Outlier filter: kept {len(kept)}/{len(vector_diffs)} diffs (threshold norm={threshold:.2f})")
        vector_diffs = kept

    # Average to get final steering vector, then normalise to unit length
    steering_vector = torch.stack(vector_diffs).mean(dim=0)
    raw_norm = steering_vector.norm().item()
    steering_vector = steering_vector / steering_vector.norm()

    # Save with metadata
    save_data = {
        "vector": steering_vector,
        "layer": args.layer,
        "position": args.position,
        "num_pairs": len(vector_diffs),
        "base_model": args.base_model,
        "filter_column": args.filter_column,
        "filter_value": args.filter_value,
        "averse_column": args.averse_column,
        "neutral_column": args.neutral_column,
        "hidden_size": steering_vector.shape[0],
        "concise_prompt": args.concise_prompt,
        "raw_norm": raw_norm,
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
    print(f"Vector norm: {steering_vector.norm().item():.4f} (normalised; raw was {raw_norm:.4f})")
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
