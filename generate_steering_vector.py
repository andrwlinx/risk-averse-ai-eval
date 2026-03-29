#!/usr/bin/env python3
"""
Generate ICV (In-Context Vector) steering vectors for activation engineering.

Builds few-shot chat prompts with risk-averse vs risk-neutral reasoning demonstrations
(full chain-of-thought), extracts activations at the pre-answer (last token) position,
and aggregates the differences (via PCA or mean) to produce a steering direction.

Uses transformers library for compatibility with evaluate.py.
"""

import argparse
import gc
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT


def build_icv_prompt(system_prompt, demo_rows, query_row, averse=True, demo_max_chars=0):
    """Build a chat message list for ICV activation extraction.

    Uses full chain-of-thought reasoning traces (chosen_full / rejected_full)
    so the steering vector captures the difference between risk-averse and
    risk-neutral *reasoning*, not just answer preference.

    Args:
        system_prompt: The system prompt string.
        demo_rows: List of DataFrame rows, each with 'prompt_text',
                   'chosen_full', and 'rejected_full' fields.
        query_row: Single DataFrame row for the query situation.
        averse: If True, use chosen_full (risk-averse CoT); otherwise rejected_full.
        demo_max_chars: Max characters per demo response (0 to disable truncation).

    Returns:
        List of chat message dicts for tokenizer.apply_chat_template().
    """
    full_col = "chosen_full" if averse else "rejected_full"

    messages = [{"role": "system", "content": system_prompt}]

    for row in demo_rows:
        messages.append({"role": "user", "content": row["prompt_text"]})
        full_response = str(row[full_col])
        if demo_max_chars and len(full_response) > demo_max_chars:
            full_response = full_response[:demo_max_chars]
        messages.append({"role": "assistant", "content": full_response})

    messages.append({"role": "user", "content": query_row["prompt_text"]})

    return messages


def get_last_token_activation(model, tokenizer, messages, layer, enable_thinking=False):
    """Extract the hidden state at the last token of a chat prompt.

    Args:
        model: The transformer model.
        tokenizer: The tokenizer.
        messages: Chat message list (from build_icv_prompt).
        layer: Layer index to extract from (0-indexed).
        enable_thinking: Whether to pass enable_thinking=True to apply_chat_template.

    Returns:
        Tensor of shape (hidden_size,).
    """
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        template_kwargs["enable_thinking"] = True

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    activation = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        activation["value"] = hidden_states[0, -1, :].detach().clone()

    target_layer = model.model.layers[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return activation["value"]


def build_sampling_plan(n_rows, num_demos, num_contrasts, seed):
    """Create a balanced non-overlapping sampling plan.

    Every situation in the demo pool appears exactly once as a demo.
    Each contrast's query comes from a different group, so it is never
    among its own demos.

    Args:
        n_rows: Total number of available rows.
        num_demos: Number of demo situations per contrast.
        num_contrasts: Number of contrasts to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'demo_indices' and 'query_index' keys.
    """
    required = num_demos * num_contrasts
    if n_rows < required:
        raise ValueError(
            f"Need at least {required} rows ({num_demos} demos x {num_contrasts} contrasts), "
            f"but only {n_rows} available after filtering."
        )

    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected = indices[:required]
    groups = [selected[i * num_demos:(i + 1) * num_demos] for i in range(num_contrasts)]

    plan = []
    for i in range(num_contrasts):
        query_index = groups[(i + 1) % num_contrasts][0]
        plan.append({
            "demo_indices": groups[i],
            "query_index": query_index,
        })

    return plan


def filter_outliers(diffs, method="none", threshold=2.0):
    """Optionally filter outlier difference vectors.

    Args:
        diffs: List of per-contrast difference tensors.
        method: "none", "norm" (z-score on L2 norm), or "cosine" (similarity to mean).
        threshold: Z-score cutoff for 'norm', or minimum similarity for 'cosine'.

    Returns:
        (filtered_diffs, excluded_indices)
    """
    if method == "none" or len(diffs) == 0:
        return diffs, []

    stacked = torch.stack(diffs)

    if method == "norm":
        norms = stacked.norm(dim=-1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        if std_norm < 1e-8:
            return diffs, []
        z_scores = ((norms - mean_norm) / std_norm).abs()
        keep_mask = z_scores <= threshold
    elif method == "cosine":
        mean_vec = stacked.mean(dim=0)
        mean_norm = mean_vec.norm()
        if mean_norm < 1e-8:
            return diffs, []
        cosine_sims = torch.nn.functional.cosine_similarity(stacked, mean_vec.unsqueeze(0), dim=-1)
        keep_mask = cosine_sims >= threshold
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    excluded = [i for i, keep in enumerate(keep_mask.tolist()) if not keep]
    filtered = [d for d, keep in zip(diffs, keep_mask.tolist()) if keep]

    return filtered, excluded


def main():
    parser = argparse.ArgumentParser(
        description="Generate ICV (In-Context Vector) steering vectors for activation engineering."
    )
    parser.add_argument(
        "--training_csv", type=str,
        default="data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv",
        help="Path to training CSV with prompt_text, chosen_full, rejected_full columns",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3-8B",
        help="Base model for activation extraction",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer to extract activations from (0-indexed, default: n_layers // 2, matching upstream)",
    )
    parser.add_argument(
        "--output", type=str, default="risk_averse_icv_steering_vector.pt",
        help="Output path for the steering vector .pt file",
    )
    parser.add_argument(
        "--num_demos", type=int, default=5,
        help="Number of demonstration situations per contrast (default: 5)",
    )
    parser.add_argument(
        "--num_contrasts", type=int, default=100,
        help="Number of averse/neutral contrasts to average (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed for reproducible sampling (default: 12345)",
    )
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Pass enable_thinking=True to apply_chat_template (default: True for Qwen3 models; use --no-enable_thinking to disable)",
    )
    parser.add_argument(
        "--system_prompt_file", type=str, default=None,
        help="Path to file containing system prompt (uses built-in default if omitted)",
    )
    parser.add_argument(
        "--icv_method", type=str, default="pca",
        choices=["mean", "pca"],
        help="Aggregation method for contrast diffs: 'pca' (default, matching upstream) or 'mean'",
    )
    parser.add_argument(
        "--demo_max_chars", type=int, default=1600,
        help="Max characters per demo CoT response (default: 1600, matching upstream; 0 to disable)",
    )
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True,
        help="L2-normalize the steering vector (default: True; makes alpha comparable across methods/models)",
    )
    parser.add_argument(
        "--outlier_method", type=str, default="none",
        choices=["none", "norm", "cosine"],
        help="Outlier filtering method for per-contrast diffs (default: none)",
    )
    parser.add_argument(
        "--outlier_threshold", type=float, default=2.0,
        help="Threshold for outlier filtering (default: 2.0)",
    )
    args = parser.parse_args()

    # --- Resolve system prompt ---
    if args.system_prompt_file is not None:
        sp_path = Path(args.system_prompt_file)
        if not sp_path.exists():
            print(f"ERROR: System prompt file not found: {sp_path}")
            sys.exit(1)
        system_prompt = sp_path.read_text().strip()
        print(f"Loaded system prompt from {sp_path}")
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        print("Using built-in default system prompt")

    # --- Load and filter training data ---
    if not Path(args.training_csv).exists():
        print(f"ERROR: Training file not found: {args.training_csv}")
        print("\nThis script expects a CSV with columns:")
        print("  - 'rejected_type': for filtering (must be 'lin')")
        print("  - 'prompt_text': the gamble question")
        print("  - 'chosen_full': full risk-averse chain-of-thought response")
        print("  - 'rejected_full': full risk-neutral chain-of-thought response")
        sys.exit(1)

    print(f"Loading training data from {args.training_csv}...")
    df = pd.read_csv(args.training_csv, encoding="utf-8-sig")

    required_cols = ["rejected_type", "prompt_text", "chosen_full", "rejected_full"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Safety filter: only use 'lin' rejected_type situations
    filtered_df = df[df["rejected_type"] == "lin"].copy()
    filtered_df = filtered_df.dropna(subset=["prompt_text", "chosen_full", "rejected_full"])
    filtered_df = filtered_df.reset_index(drop=True)

    print(f"Found {len(filtered_df)} situations with rejected_type == 'lin'")

    if len(filtered_df) == 0:
        print("\nERROR: No rows match rejected_type == 'lin'")
        print(f"Available rejected_type values: {df['rejected_type'].unique().tolist()}")
        sys.exit(1)

    # --- Build sampling plan ---
    required_rows = args.num_demos * args.num_contrasts
    print(f"\nSampling plan: {args.num_demos} demos x {args.num_contrasts} contrasts = {required_rows} demo slots")

    plan = build_sampling_plan(len(filtered_df), args.num_demos, args.num_contrasts, args.seed)

    query_indices = {p["query_index"] for p in plan}
    print(f"  {len(query_indices)} unique query situations (held out from their own demos)")
    print(f"  Seed: {args.seed}")

    # --- Load model ---
    print(f"\nLoading model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    if args.layer is None:
        args.layer = num_layers // 2
        print(f"Layer defaulted to {args.layer} (n_layers // 2, matching upstream)")
    if args.layer >= num_layers:
        print(f"ERROR: Layer {args.layer} out of range. Model has {num_layers} layers (0-{num_layers - 1})")
        sys.exit(1)

    print(f"Model loaded. {num_layers} layers total, extracting from layer {args.layer}")

    # --- Compute ICV activation differences ---
    vector_diffs = []
    skipped = 0

    for i, contrast in enumerate(tqdm(plan, desc="Computing ICV contrasts")):
        demo_rows = [filtered_df.iloc[idx] for idx in contrast["demo_indices"]]
        query_row = filtered_df.iloc[contrast["query_index"]]

        averse_msgs = build_icv_prompt(system_prompt, demo_rows, query_row, averse=True, demo_max_chars=args.demo_max_chars)
        neutral_msgs = build_icv_prompt(system_prompt, demo_rows, query_row, averse=False, demo_max_chars=args.demo_max_chars)

        try:
            act_averse = get_last_token_activation(
                model, tokenizer, averse_msgs, args.layer, args.enable_thinking,
            )
            act_neutral = get_last_token_activation(
                model, tokenizer, neutral_msgs, args.layer, args.enable_thinking,
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"\n  Warning: Contrast {i + 1} failed: {e}")
            continue

        diff = act_averse - act_neutral
        vector_diffs.append(diff)

    if len(vector_diffs) == 0:
        print("\nERROR: Could not compute any valid activation differences")
        sys.exit(1)

    print(f"\nComputed {len(vector_diffs)} valid differences (skipped {skipped})")

    # --- Outlier filtering ---
    filtered_diffs, excluded_indices = filter_outliers(
        vector_diffs, args.outlier_method, args.outlier_threshold,
    )

    if args.outlier_method != "none":
        print(f"Outlier filtering ({args.outlier_method}, threshold={args.outlier_threshold}): "
              f"excluded {len(excluded_indices)}, kept {len(filtered_diffs)}")

    if len(filtered_diffs) == 0:
        print("\nERROR: All contrasts excluded by outlier filtering")
        sys.exit(1)

    # --- Aggregate to get final steering vector ---
    stacked = torch.stack(filtered_diffs)
    pca_singular_values = None
    if args.icv_method == "pca":
        centered = stacked - stacked.mean(dim=0)
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        pca_singular_values = S.tolist()
        steering_vector = Vt[0]
        # Ensure sign convention: positive projection with mean diff
        mean_diff = stacked.mean(dim=0)
        if torch.dot(steering_vector, mean_diff) < 0:
            steering_vector = -steering_vector
        variance_explained = (S[0] ** 2) / (S ** 2).sum()
        print(f"Aggregation: PCA (first principal component, {variance_explained:.1%} variance explained)")
    else:
        steering_vector = stacked.mean(dim=0)
        print(f"Aggregation: mean averaging")

    # --- Optional L2 normalization ---
    pre_norm = steering_vector.norm().item()
    if args.normalize:
        norm = steering_vector.norm()
        if norm > 1e-8:
            steering_vector = steering_vector / norm
            print(f"Normalized to unit length (original norm: {pre_norm:.4f})")

    # --- Save with comprehensive metadata ---
    save_data = {
        # The vector (both keys for backward + upstream compatibility)
        "vector": steering_vector,
        "direction": steering_vector,
        # Model info
        "method": "icv",
        "demo_content": "full_cot",
        "demo_columns": {"averse": "chosen_full", "neutral": "rejected_full"},
        "demo_max_chars": args.demo_max_chars,
        "icv_method": args.icv_method,
        "normalized": args.normalize,
        "pre_normalization_norm": pre_norm,
        "base_model": args.base_model,
        "hidden_size": steering_vector.shape[0],
        # Extraction config
        "layer": args.layer,
        "position": "last",
        # Data source
        "training_csv": str(Path(args.training_csv).resolve()),
        "filter_column": "rejected_type",
        "filter_value": "lin",
        "total_rows_after_filter": len(filtered_df),
        # Sampling config
        "seed": args.seed,
        "num_demos_per_contrast": args.num_demos,
        "num_contrasts": args.num_contrasts,
        "num_contrasts_used": len(filtered_diffs),
        # Outlier filtering
        "outlier_method": args.outlier_method,
        "outlier_threshold": args.outlier_threshold,
        "outlier_num_excluded": len(excluded_indices),
        "outlier_excluded_indices": excluded_indices,
        # Thinking mode
        "enable_thinking": args.enable_thinking,
        # System prompt
        "has_system_prompt": True,
        "system_prompt": system_prompt,
        # Diagnostics
        "vector_norm": steering_vector.norm().item(),
        "vector_mean": steering_vector.mean().item(),
        "vector_std": steering_vector.std().item(),
        "per_contrast_norms": [d.norm().item() for d in vector_diffs],
        "pca_singular_values": pca_singular_values,
    }

    torch.save(save_data, args.output)

    print(f"\n{'=' * 60}")
    print("ICV STEERING VECTOR GENERATED")
    print("=" * 60)
    print(f"Output:           {args.output}")
    print(f"Shape:            {steering_vector.shape}")
    print(f"Layer:            {args.layer}")
    print(f"Base model:       {args.base_model}")
    print(f"Demos/contrast:   {args.num_demos}")
    print(f"Contrasts used:   {len(filtered_diffs)} / {args.num_contrasts}")
    print(f"ICV method:       {args.icv_method}")
    print(f"Demo max chars:   {args.demo_max_chars or 'unlimited'}")
    print(f"Seed:             {args.seed}")
    print(f"System prompt:    {'custom file' if args.system_prompt_file else 'built-in default'}")
    print(f"Thinking enabled: {args.enable_thinking}")
    print(f"Outlier filter:   {args.outlier_method} (threshold={args.outlier_threshold})")
    print(f"Vector norm:      {steering_vector.norm().item():.4f}")
    print(f"Vector mean:      {steering_vector.mean().item():.6f}")
    print(f"Vector std:       {steering_vector.std().item():.4f}")
    print("=" * 60)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTo use this vector with evaluate.py, run:")
    print(f"  python evaluate.py --backend transformers --steering_direction_path {args.output} --alphas 1.0 --eval_layer {args.layer}")


if __name__ == "__main__":
    main()
