#!/usr/bin/env python3
"""
Comprehensive evaluation with MULTIPLE METRICS:
1. Generation @ temperature=0 (deterministic)
2. Generation @ temperature=0.7 (with sampling)
3. Answer-only log probabilities (for base model comparison)

Evaluates all models on all datasets.
"""

import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import pandas as pd
import json
import re
import os
import math
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === DATASETS ===
DATASETS = {
    "ood_validation": "data/2026_01_29_new_val_set_probabilities_add_to_100.csv",
    "indist_validation": "data/in_distribution_val_set.csv",
    "training": "data/training_eval_set.csv",
}

# === MODELS ===
BASE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

FINETUNED_MODELS = {
    "qwen25_7b": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "sweep_dir": "sweep_fixed_20260105_065742",
        "configs": ["light_lora", "baseline", "heavy_lora", "low_lr", "high_lr",
                   "more_epochs", "many_epochs", "heavy_long", "very_heavy", "best_guess"]
    },
    "qwen3_8b": {
        "base_model": "Qwen/Qwen3-8B",
        "sweep_dir": None,
        "configs": ["light_lora", "baseline", "heavy_lora", "low_lr", "high_lr",
                   "more_epochs", "many_epochs", "heavy_long", "very_heavy", "best_guess"]
    },
    "qwen3_32b": {
        "base_model": "Qwen/Qwen3-32B",
        "sweep_dir": None,
        "configs": ["light_lora", "baseline", "heavy_lora", "more_epochs"]
    },
}

OUTPUT_DIR = "comprehensive_evaluation"
NUM_SITUATIONS = 50


def remove_instruction_suffix(prompt):
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def extract_choice_permissive(response, num_options):
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

    last_part = response_lower[-300:]
    option_match = re.search(r'\boption\s+([a-z0-9])\b', last_part)
    if option_match and option_match.group(1) in valid_options:
        return option_match.group(1)

    paren_matches = re.findall(r'\(([a-z0-9])\)', last_part)
    for match in reversed(paren_matches):
        if match in valid_options:
            return match

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


def load_dataset(dataset_path, num_situations=None):
    df = pd.read_csv(os.path.expanduser(dataset_path))
    situations = []
    unique_ids = df["situation_id"].unique()
    if num_situations:
        unique_ids = unique_ids[:num_situations]

    for sit_id in unique_ids:
        sit_data = df[df["situation_id"] == sit_id]
        prompt = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        options = {}
        best_cara_idx = None
        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            is_best = row.get("is_best_cara_display", False) == True
            option_data = {
                "type": row.get("option_type", "Unknown"),
                "is_best_cara": is_best
            }
            options[letter] = option_data
            options[number] = option_data
            if is_best:
                best_cara_idx = idx
        situations.append({
            "situation_id": sit_id,
            "prompt": prompt,
            "num_options": num_options,
            "options": options,
            "best_cara_idx": best_cara_idx
        })
    return situations


def evaluate_generation(model, tokenizer, situations, temperature=0.0, do_sample=False):
    """Evaluate using generation (temp=0 or temp=0.7)."""
    results = []

    for i, sit in enumerate(situations):
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{len(situations)}")

        prompt = remove_instruction_suffix(sit["prompt"])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        choice = extract_choice_permissive(response, sit["num_options"])

        if choice and choice in sit["options"]:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": choice,
                "is_best_cara": sit["options"][choice]["is_best_cara"],
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": None,
                "is_best_cara": None,
            })

    valid = [r for r in results if r["is_best_cara"] is not None]
    cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid) if valid else 0
    parse_rate = len(valid) / len(results) if results else 0

    return {"parse_rate": parse_rate, "cara_rate": cara_rate, "num_valid": len(valid)}


def evaluate_logprobs(model, tokenizer, situations):
    """
    Evaluate using answer-only log probabilities.
    Asks model to just output the letter without CoT.
    """
    results = []

    # Create answer-only prompt
    answer_only_suffix = "\n\nPlease respond with just the letter of your chosen option (a, b, c, etc.) without any explanation."

    for i, sit in enumerate(situations):
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{len(situations)}")

        prompt = remove_instruction_suffix(sit["prompt"]) + answer_only_suffix
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Get log probabilities for each option letter
        option_letters = [chr(ord('a') + i) for i in range(sit["num_options"])]
        log_probs = {}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token position
            log_softmax = torch.log_softmax(logits, dim=-1)

            for letter in option_letters:
                # Get token ID for this letter
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    log_probs[letter] = log_softmax[token_id].item()
                else:
                    log_probs[letter] = float('-inf')

        # Find highest prob option
        best_letter = max(log_probs, key=log_probs.get)
        is_best_cara = sit["options"].get(best_letter, {}).get("is_best_cara", False)

        results.append({
            "situation_id": sit["situation_id"],
            "choice": best_letter,
            "is_best_cara": is_best_cara,
            "log_probs": log_probs,
            "best_cara_log_prob": log_probs.get(chr(ord('a') + sit["best_cara_idx"])) if sit["best_cara_idx"] is not None else None
        })

    cara_rate = sum(r["is_best_cara"] for r in results) / len(results) if results else 0

    return {"cara_rate": cara_rate, "num_total": len(results), "parse_rate": 1.0}


def evaluate_model_all_metrics(model, tokenizer, situations):
    """Run all three evaluation methods."""
    print("    Evaluating with temp=0...")
    temp0_results = evaluate_generation(model, tokenizer, situations, temperature=0.0, do_sample=False)

    print("    Evaluating with temp=0.7...")
    temp07_results = evaluate_generation(model, tokenizer, situations, temperature=0.7, do_sample=True)

    print("    Evaluating with log probs...")
    logprob_results = evaluate_logprobs(model, tokenizer, situations)

    return {
        "generation_temp0": temp0_results,
        "generation_temp07": temp07_results,
        "logprob_answer_only": logprob_results
    }


def find_sweep_dir(pattern):
    import glob
    dirs = glob.glob(pattern)
    if dirs:
        return sorted(dirs)[-1]
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("COMPREHENSIVE MULTI-METRIC EVALUATION")
    print("="*70)
    print(f"Metrics: generation@temp0, generation@temp0.7, logprob_answer_only")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Situations per dataset: {NUM_SITUATIONS}")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "num_situations": NUM_SITUATIONS,
        "metrics": ["generation_temp0", "generation_temp07", "logprob_answer_only"],
        "base_models": {},
        "finetuned_models": {}
    }

    # Evaluate base models
    print("\n" + "="*70)
    print("EVALUATING BASE MODELS (UNFINETUNED)")
    print("="*70)

    for model_id in BASE_MODELS:
        print(f"\n{'='*60}")
        print(f"Base model: {model_id}")
        print(f"{'='*60}")

        try:
            gc.collect()
            torch.cuda.empty_cache()

            print("  Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            )
            model.eval()

            all_results["base_models"][model_id] = {}

            for dataset_name, dataset_path in DATASETS.items():
                print(f"\n  Dataset: {dataset_name}")
                situations = load_dataset(dataset_path, NUM_SITUATIONS)
                metrics = evaluate_model_all_metrics(model, tokenizer, situations)
                all_results["base_models"][model_id][dataset_name] = metrics

                print(f"    Results:")
                print(f"      temp0: CARA={metrics['generation_temp0']['cara_rate']*100:.1f}%")
                print(f"      temp0.7: CARA={metrics['generation_temp07']['cara_rate']*100:.1f}%")
                print(f"      logprob: CARA={metrics['logprob_answer_only']['cara_rate']*100:.1f}%")

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["base_models"][model_id] = {"error": str(e)}

        # Save intermediate
        with open(f"{OUTPUT_DIR}/multi_metric_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Evaluate finetuned models
    print("\n" + "="*70)
    print("EVALUATING FINETUNED MODELS")
    print("="*70)

    for model_family, config in FINETUNED_MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# {model_family.upper()}")
        print(f"{'#'*60}")

        sweep_dir = config["sweep_dir"]
        if sweep_dir is None:
            if "qwen3_8b" in model_family:
                sweep_dir = find_sweep_dir("sweep_qwen3_2*")
            elif "qwen3_32b" in model_family:
                sweep_dir = find_sweep_dir("sweep_qwen3_32b_*")

        if not sweep_dir or not os.path.exists(sweep_dir):
            print(f"  Sweep directory not found, skipping...")
            continue

        print(f"  Sweep dir: {sweep_dir}")
        all_results["finetuned_models"][model_family] = {"sweep_dir": sweep_dir, "configs": {}}

        for config_name in config["configs"]:
            adapter_path = f"{sweep_dir}/{config_name}/final"
            if not os.path.exists(adapter_path):
                print(f"  Config {config_name} not found, skipping...")
                continue

            print(f"\n  Config: {config_name}")

            try:
                gc.collect()
                torch.cuda.empty_cache()

                tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token

                base_model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model = model.merge_and_unload()
                model.eval()

                all_results["finetuned_models"][model_family]["configs"][config_name] = {}

                for dataset_name, dataset_path in DATASETS.items():
                    print(f"\n    Dataset: {dataset_name}")
                    situations = load_dataset(dataset_path, NUM_SITUATIONS)
                    metrics = evaluate_model_all_metrics(model, tokenizer, situations)
                    all_results["finetuned_models"][model_family]["configs"][config_name][dataset_name] = metrics

                    print(f"      temp0: CARA={metrics['generation_temp0']['cara_rate']*100:.1f}%")
                    print(f"      temp0.7: CARA={metrics['generation_temp07']['cara_rate']*100:.1f}%")
                    print(f"      logprob: CARA={metrics['logprob_answer_only']['cara_rate']*100:.1f}%")

                del model
                del base_model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results["finetuned_models"][model_family]["configs"][config_name] = {"error": str(e)}

            # Save intermediate
            with open(f"{OUTPUT_DIR}/multi_metric_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/multi_metric_results.json")


if __name__ == "__main__":
    main()
