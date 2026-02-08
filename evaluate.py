#!/usr/bin/env python3
"""
Evaluate fine-tuned model with PERMISSIVE answer parsing.
Dramatically improves parse rate by matching many answer formats.
"""

import gc
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Flush output immediately so logs are visible in real time
sys.stdout.reconfigure(line_buffering=True)

import torch
torch.cuda.empty_cache()
gc.collect()

import argparse
import pandas as pd
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================================
# ACTIVATION STEERING UTILITIES
# ============================================================================

class SteeringHook:
    """Forward hook that adds a steering vector to residual stream activations."""

    def __init__(self, steering_vector, alpha=1.0):
        """
        Args:
            steering_vector: Tensor of shape (hidden_size,)
            alpha: Steering strength multiplier
        """
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.handle = None

    def __call__(self, module, input, output):
        """Add steering vector to all positions in the sequence."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions
            # Shape: hidden_states is (batch, seq_len, hidden_size)
            hidden_states = hidden_states + self.alpha * self.steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output + self.alpha * self.steering_vector.to(output.device)

    def register(self, layer_module):
        """Register this hook on a layer module."""
        self.handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def steering_context(model, steering_vector, alpha, layer):
    """Context manager to temporarily apply steering during generation.

    Args:
        model: The transformer model
        steering_vector: Tensor of shape (hidden_size,)
        alpha: Steering strength (positive = more risk-averse, negative = more risk-neutral)
        layer: Which layer to apply steering to (0-indexed)

    Usage:
        with steering_context(model, vector, alpha=2.0, layer=14):
            outputs = model.generate(...)
    """
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
    """Load a steering vector from a .pt file.

    Returns:
        tuple: (vector, layer, metadata) where metadata contains info about how it was generated
    """
    if path is None:
        return None, None, None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    # Handle both old format (just tensor) and new format (dict with metadata)
    if isinstance(data, dict):
        vector = data["vector"]
        layer = data.get("layer", 14)
        metadata = {k: v for k, v in data.items() if k != "vector"}
    else:
        # Old format: just the tensor
        vector = data
        layer = 14  # Default to layer 14
        metadata = {}

    return vector, layer, metadata


def remove_instruction_suffix(prompt):
    """Remove the instruction about how to respond from the end of the prompt."""
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def extract_choice_permissive(response, num_options):
    """Extract choice with VERY permissive matching.

    Handles:
    - JSON format: {"answer": "X"}
    - Natural language: "I choose b", "my answer is a", "select option 2"
    - Parenthesized: (a), (b), (1)
    - Standalone letters/numbers near the end
    - Both letter options (a,b,c) and numeric options (1,2,3)
    """
    response_lower = response.lower().strip()

    # Generate valid options (both letters and numbers)
    valid_letters = [chr(ord('a') + i) for i in range(num_options)]
    valid_numbers = [str(i + 1) for i in range(num_options)]
    valid_options = valid_letters + valid_numbers

    # 1. JSON format: {"answer": "X"} - most specific, check first
    json_match = re.search(r'\{"answer"\s*:\s*"([a-z0-9]+)"\}', response_lower)
    if json_match and json_match.group(1) in valid_options:
        return json_match.group(1)

    # 2. Look for "answer" followed by option: "answer is b", "answer: b", "the answer is b"
    answer_match = re.search(r'(?:the\s+)?answer[:\s]+(?:is\s+)?(?:option\s+)?([a-z0-9])\b', response_lower)
    if answer_match and answer_match.group(1) in valid_options:
        return answer_match.group(1)

    # 3. Look for "choose/select/pick option X" or "I choose X", "I'd select X"
    choice_match = re.search(r"(?:i(?:'d)?\s+)?(?:choose|select|pick|chose|selected|picking)\s+(?:option\s+)?([a-z0-9])\b", response_lower)
    if choice_match and choice_match.group(1) in valid_options:
        return choice_match.group(1)

    # 4. Look for "option X is" or "option X would be" patterns (indicating choice)
    option_is_match = re.search(r'\boption\s+([a-z0-9])\s+(?:is|would be|seems)\b', response_lower)
    if option_is_match and option_is_match.group(1) in valid_options:
        return option_is_match.group(1)

    # 5. Look for "go with option X" or "go with X"
    go_with_match = re.search(r'go\s+with\s+(?:option\s+)?([a-z0-9])\b', response_lower)
    if go_with_match and go_with_match.group(1) in valid_options:
        return go_with_match.group(1)

    # Now look in the last portion of response for less specific patterns
    last_part = response_lower[-300:]

    # 6. Look for "option X" near the end
    option_match = re.search(r'\boption\s+([a-z0-9])\b', last_part)
    if option_match and option_match.group(1) in valid_options:
        return option_match.group(1)

    # 7. Look for standalone letter/number in parentheses: (a), (b), (1), (2)
    paren_matches = re.findall(r'\(([a-z0-9])\)', last_part)
    for match in reversed(paren_matches):  # Check from end
        if match in valid_options:
            return match

    # 8. Look for "therefore X" or "thus X" or "so X" (conclusion patterns)
    conclusion_match = re.search(r'(?:therefore|thus|so|hence),?\s+(?:option\s+)?([a-z0-9])\b', last_part)
    if conclusion_match and conclusion_match.group(1) in valid_options:
        return conclusion_match.group(1)

    # 9. Final fallback: find the LAST standalone valid option in last 150 chars
    last_150 = response_lower[-150:]
    last_found = None
    for opt in valid_options:
        matches = list(re.finditer(r'\b' + re.escape(opt) + r'\b', last_150))
        if matches:
            # Get position of last match
            last_pos = matches[-1].start()
            if last_found is None or last_pos > last_found[1]:
                last_found = (opt, last_pos)

    if last_found:
        return last_found[0]

    return None


def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def save_incremental(output_path, args, results, failed_responses, situations_evaluated):
    """Save current results to disk. Called after every situation for crash resilience."""
    valid = [r for r in results if r["is_cooperate"] is not None]
    if valid:
        cooperate_rate = sum(r["is_cooperate"] for r in valid) / len(valid)
        rebel_rate = sum(r["is_rebel"] for r in valid) / len(valid)
        steal_rate = sum(r["is_steal"] for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = 0

    parse_rate = len(valid) / len(results) if results else 0

    output_data = convert_numpy({
        "evaluation_config": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "num_situations": situations_evaluated,
            "base_model": args.base_model,
            "model_path": args.model_path,
            "steering_path": args.steering_path,
            "steering_alpha": args.alpha,
            "steering_layer": args.steering_layer
        },
        "metrics": {
            "parse_rate": parse_rate,
            "cooperate_rate": cooperate_rate,
            "rebel_rate": rebel_rate,
            "steal_rate": steal_rate,
            "best_cara_rate": cara_rate
        },
        "num_valid": len(valid),
        "num_total": len(results),
        "results": None if args.no_save_responses else results,
        "failed_responses": failed_responses[:10]
    })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def load_situations(val_csv, num_situations):
    """Load and parse situations from a validation CSV.

    Args:
        val_csv: Path to validation CSV file
        num_situations: Max number of situations to load

    Returns:
        List of situation dicts with keys: situation_id, prompt, num_options, options
    """
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
            number = str(idx + 1)  # 0→"1", 1→"2", etc.
            option_data = {
                "type": row["option_type"],
                "is_best_cara": row["is_best_cara_display"] == True
            }
            # Add BOTH letter and number keys so we accept either format
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
                   temperature=0.7, max_new_tokens=4096,
                   max_time_per_generation=120,
                   disable_thinking=False, no_save_responses=True,
                   verbose=True, incremental_save_path=None, incremental_save_args=None):
    """Run evaluation loop over situations with given steering params.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer
        situations: List of situation dicts (from load_situations)
        steering_vector: Steering vector tensor, or None
        alpha: Steering strength
        steering_layer: Layer index for steering hook
        temperature: Sampling temperature (0 = deterministic)
        max_new_tokens: Max tokens to generate
        max_time_per_generation: Timeout per generation in seconds
        disable_thinking: Disable thinking mode in chat template
        no_save_responses: If True, don't save full responses in results
        verbose: Print per-situation progress
        incremental_save_path: If set, save results after each situation
        incremental_save_args: args object needed by save_incremental

    Returns:
        dict with keys: cooperate_rate, rebel_rate, steal_rate, cara_rate,
                        parse_rate, num_valid, num_total, results,
                        failed_responses, generation_times, total_elapsed
    """
    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        sit_start = time.time()

        prompt = remove_instruction_suffix(sit["prompt"])
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template (disable thinking for Qwen3 base models)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_start = time.time()
        with torch.no_grad(), steering_context(model, steering_vector, alpha, steering_layer):
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation
                )
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
                "response": None if no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": None,
                "option_type": None,
                "is_cooperate": None,
                "is_rebel": None,
                "is_steal": None,
                "is_best_cara": None,
                "response": None if no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
            failed_responses.append({
                "situation_id": sit["situation_id"],
                "num_options": sit["num_options"],
                "response": response
            })

        generation_times.append(gen_elapsed)
        avg_time = sum(generation_times) / len(generation_times)
        remaining = avg_time * (len(situations) - i - 1)

        if verbose:
            status = "OK" if choice else "PARSE_FAIL"
            print(f"  [{i+1}/{len(situations)}] sit_id={sit['situation_id']} | {status} | "
                  f"{int(num_generated_tokens)} tokens | {gen_elapsed:.1f}s | "
                  f"ETA: {remaining/60:.1f}min")

            if gen_elapsed > 60:
                print(f"  WARNING: Generation took {gen_elapsed:.0f}s (>{60}s). "
                      f"Model may be generating excessively long output.")
            if int(num_generated_tokens) >= max_new_tokens - 10:
                print(f"  WARNING: Hit token limit ({max_new_tokens}). "
                      f"Response may be truncated. Consider --max_new_tokens increase.")

        if incremental_save_path and incremental_save_args:
            save_incremental(incremental_save_path, incremental_save_args,
                             results, failed_responses, i + 1)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)")
    parser.add_argument("--val_csv", type=str, default="data/2026_01_29_new_val_set_probabilities_add_to_100.csv")
    parser.add_argument("--num_situations", type=int, default=50, help="Number of situations to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (auto-generated if omitted)")
    parser.add_argument("--no_save_responses", action="store_true", help="Do NOT save full responses (by default, all CoT responses are saved)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max tokens to generate (default 4096 - generous to avoid truncation)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = deterministic, 0.7 = default, 1.0 = high diversity)")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode in chat template (auto-enabled for base models, needed for Qwen3)")
    parser.add_argument("--max_time_per_generation", type=float, default=120, help="Max seconds per generation before timeout (default: 120)")
    parser.add_argument("--steering_path", type=str, default=None, help="Path to steering vector .pt file (optional)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Steering strength: positive=more risk-averse, negative=more risk-neutral (default: 0 = no steering)")
    parser.add_argument("--steering_layer", type=int, default=None, help="Layer to apply steering (default: use layer from .pt file, or 14)")
    args = parser.parse_args()

    # Auto-generate descriptive output filename if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract a short model name from the path or base model
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            # If the last component is "final" or "checkpoint-*", use the parent dir name
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        # Include alpha in filename if steering is enabled
        alpha_suffix = f"_alpha{args.alpha}" if args.steering_path and args.alpha != 0 else ""
        args.output = f"eval_{model_short}_temp{args.temperature}{alpha_suffix}_{timestamp}.json"

    # Auto-enable disable_thinking for base model evaluation (no adapter)
    if args.model_path is None and not args.disable_thinking:
        args.disable_thinking = True
        print("Note: Auto-enabling --disable_thinking for base model evaluation (prevents Qwen3 hang)")

    BASE_MODEL = args.base_model

    if args.model_path:
        print(f"Loading fine-tuned model (base: {BASE_MODEL}, adapter: {args.model_path})...")
    else:
        print(f"Loading base model only: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.model_path:
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    # Load steering vector if provided
    steering_vector = None
    steering_layer = args.steering_layer if args.steering_layer is not None else 14

    if args.steering_path:
        print(f"Loading steering vector from {args.steering_path}...")
        try:
            steering_vector, saved_layer, metadata = load_steering_vector(args.steering_path)
            # Use saved layer if not overridden by CLI
            if args.steering_layer is None and saved_layer is not None:
                steering_layer = saved_layer
            # Update args so it gets saved in output JSON
            args.steering_layer = steering_layer
            print(f"  Vector shape: {steering_vector.shape}")
            print(f"  Steering layer: {steering_layer}")
            print(f"  Alpha (strength): {args.alpha}")
            if metadata:
                print(f"  Generated from: {metadata.get('num_pairs', '?')} pairs")
                print(f"  Position: {metadata.get('position', '?')}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Continuing without steering...")
            steering_vector = None

    print("Loading validation data...")
    situations = load_situations(args.val_csv, args.num_situations)

    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    if steering_vector is not None:
        print(f"Activation steering: ENABLED (alpha={args.alpha}, layer={steering_layer})")
    else:
        print(f"Activation steering: disabled")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    print(f"Results will be saved incrementally to: {args.output}")
    print()

    eval_result = run_evaluation(
        model, tokenizer, situations, steering_vector,
        alpha=args.alpha, steering_layer=steering_layer,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        max_time_per_generation=args.max_time_per_generation,
        disable_thinking=args.disable_thinking,
        no_save_responses=args.no_save_responses,
        verbose=True,
        incremental_save_path=args.output,
        incremental_save_args=args,
    )

    results = eval_result["results"]
    failed_responses = eval_result["failed_responses"]
    generation_times = eval_result["generation_times"]
    total_elapsed = eval_result["total_elapsed"]
    valid = [r for r in results if r["is_cooperate"] is not None]

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("="*50)
    print(f"Total situations: {len(situations)}")
    print(f"Valid responses: {len(valid)} ({100*eval_result['parse_rate']:.1f}%)")
    print(f"Failed to parse: {len(failed_responses)}")
    print(f"\n% choosing COOPERATE: {100*eval_result['cooperate_rate']:.1f}%")
    print(f"% choosing REBEL:     {100*eval_result['rebel_rate']:.1f}%")
    print(f"% choosing STEAL:     {100*eval_result['steal_rate']:.1f}%")
    print(f"% choosing best CARA: {100*eval_result['cara_rate']:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Avg per situation: {sum(generation_times)/len(generation_times):.1f}s")
    print(f"Avg tokens generated: {sum(r.get('num_tokens_generated', 0) for r in results)/len(results):.0f}")
    print("="*50)

    # Print failed responses
    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("="*50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr['response'][:600])
            print("...")

    # Final save (already saved incrementally, but save once more with final metrics)
    save_incremental(args.output, args, results, failed_responses, len(situations))
    print(f"\nFinal results saved to {args.output}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
