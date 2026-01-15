#!/usr/bin/env python3
"""
Evaluate fine-tuned model with PERMISSIVE answer parsing.
Dramatically improves parse rate by matching many answer formats.
"""

import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import argparse
import pandas as pd
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)")
    parser.add_argument("--val_csv", type=str, default="data/val_set_medium_stakes.csv")
    parser.add_argument("--num_situations", type=int, default=50, help="Number of situations to evaluate")
    parser.add_argument("--output", type=str, default="results_permissive.json")
    parser.add_argument("--save_responses", action="store_true", help="Save full responses for debugging (recommended, adds ~0 time)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max tokens to generate (default 4096 - generous to avoid truncation)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = deterministic, 0.7 = default, 1.0 = high diversity)")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode in chat template (auto-enabled for base models, needed for Qwen3)")
    args = parser.parse_args()

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

    print("Loading validation data...")
    df = pd.read_csv(args.val_csv)

    # Group by situation_id
    situations = []
    for sit_id in df["situation_id"].unique()[:args.num_situations]:
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

    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")

    results = []
    failed_responses = []

    for i, sit in enumerate(situations):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(situations)}")

        prompt = remove_instruction_suffix(sit["prompt"])
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template (disable thinking for Qwen3 base models)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            if args.temperature == 0:
                # Deterministic generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                # Sampling with temperature
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        choice = extract_choice_permissive(response, sit["num_options"])

        if choice and choice in sit["options"]:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": choice,
                "is_cooperate": sit["options"][choice]["type"] == "Cooperate",
                "is_best_cara": sit["options"][choice]["is_best_cara"],
                "response": response if args.save_responses else None,
                "response_length": len(response)
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": None,
                "is_cooperate": None,
                "is_best_cara": None,
                "response": response if args.save_responses else None,
                "response_length": len(response)
            })
            failed_responses.append({
                "situation_id": sit["situation_id"],
                "num_options": sit["num_options"],
                "response": response
            })

    # Calculate metrics
    valid = [r for r in results if r["is_cooperate"] is not None]
    if valid:
        cooperate_rate = sum(r["is_cooperate"] for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
    else:
        cooperate_rate = 0
        cara_rate = 0

    parse_rate = len(valid) / len(results)

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("="*50)
    print(f"Total situations: {len(situations)}")
    print(f"Valid responses: {len(valid)} ({100*parse_rate:.1f}%)")
    print(f"Failed to parse: {len(failed_responses)}")
    print(f"\n% choosing COOPERATE: {100*cooperate_rate:.1f}%")
    print(f"% choosing best CARA: {100*cara_rate:.1f}%")
    print("="*50)

    # Print failed responses FIRST (before JSON save which might fail)
    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("="*50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr['response'][:600])
            print("...")

    # Helper to convert numpy types to Python native types
    def convert_numpy(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj

    # Save results
    output_data = convert_numpy({
        "evaluation_config": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "num_situations": len(situations),
            "base_model": args.base_model,
            "model_path": args.model_path
        },
        "metrics": {
            "parse_rate": parse_rate,
            "cooperate_rate": cooperate_rate,
            "best_cara_rate": cara_rate
        },
        "num_valid": len(valid),
        "num_total": len(results),
        "results": results if args.save_responses else None,
        "failed_responses": failed_responses[:10]  # Save first 10 failures for debugging
    })

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
