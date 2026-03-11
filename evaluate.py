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
# DATA HELPERS
# ============================================================================

def remove_instruction_suffix(prompt):
    """Remove the instruction about how to respond from the end of the prompt."""
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def clean_bucket_label(value):
    """Normalize low_bucket_label strings like '\"lin_only\"' -> 'lin_only'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s


def parse_label_list(value):
    """Parse list-like label fields stored as JSON strings in CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return [str(parsed)]
    except Exception:
        s = s.strip('"').strip("'")
        if not s:
            return []
        if "," in s:
            return [part.strip().strip('"').strip("'") for part in s.split(",") if part.strip()]
        return [s]


def parse_bool_like(value):
    """Parse bool-ish CSV values robustly (handles numpy/pandas/string forms)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return bool(value)


def infer_probability_format(prompt_text):
    """Best-effort fallback if explicit use_verbal_probs is missing."""
    if not isinstance(prompt_text, str):
        return None
    if re.search(r"\d+\s*%", prompt_text):
        return "numerical"
    verbal_markers = [
        "very likely", "likely", "unlikely", "very unlikely",
        "almost certain", "almost no chance", "small chance",
    ]
    prompt_lower = prompt_text.lower()
    if any(marker in prompt_lower for marker in verbal_markers):
        return "verbal"
    return None


def probability_format_from_value(use_verbal_probs_value, prompt_text=None):
    parsed_bool = parse_bool_like(use_verbal_probs_value)
    if parsed_bool is True:
        return "verbal"
    if parsed_bool is False:
        return "numerical"
    return infer_probability_format(prompt_text)


def label_to_option_number(label):
    """Convert a label like 'a' or '1' into a 1-based option number."""
    s = str(label).strip().lower()
    if s.isdigit():
        return int(s)
    if len(s) == 1 and "a" <= s <= "z":
        return ord(s) - ord("a") + 1
    return None


# ============================================================================
# ACTIVATION STEERING UTILITIES
# ============================================================================

class SteeringHook:
    """Forward hook that adds a steering vector to residual stream activations."""

    def __init__(self, steering_vector, alpha=1.0):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.handle = None

    def __call__(self, module, input, output):
        """Add steering vector to all positions in the sequence."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + self.alpha * self.steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output + self.alpha * self.steering_vector.to(output.device)

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def steering_context(model, steering_vector, alpha, layer):
    """Context manager to temporarily apply steering during generation.

    layer may be a single int or a list of ints for simultaneous multi-layer injection.
    """
    if steering_vector is None or alpha == 0:
        yield
        return

    layers = [layer] if isinstance(layer, int) else list(layer)
    hooks = [SteeringHook(steering_vector, alpha).register(model.model.layers[L])
             for L in layers]

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def load_steering_vector(path):
    """Load a steering vector from a .pt file.

    Returns:
        tuple: (vector, layer, metadata)
    """
    if path is None:
        return None, None, None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        vector = data["vector"]
        layer = data.get("layer", 14)
        metadata = {k: v for k, v in data.items() if k != "vector"}
    else:
        vector = data
        layer = 14
        metadata = {}

    return vector, layer, metadata


# ============================================================================
# ANSWER PARSING
# ============================================================================

def extract_choice_permissive(response, num_options):
    """Extract choice with permissive matching, but avoid false positives.

    Handles:
    - JSON format: {"answer": "X"}
    - LaTeX boxed: \\boxed{a}
    - Natural language: "I choose b", "my answer is a", "select option 2"
    - Parenthesized: (a), (b), (1)
    - Short final-line answers
    - Both letter options (a,b,c) and numeric options (1,2,3)
    """
    response_lower = response.lower()
    response_lower = response_lower.replace("\\n", "\n").replace("\\r", "\r")
    # Note: do NOT replace \\t→tab — it corrupts LaTeX like \text{} and \boxed{\text{}}
    response_lower = re.sub(r"[*_`]+", "", response_lower)
    response_lower = response_lower.rstrip()
    tail_text = response_lower[-2500:] if len(response_lower) > 2500 else response_lower

    valid_letters = [chr(ord('a') + i) for i in range(num_options)]
    valid_numbers = [str(i + 1) for i in range(num_options)]
    valid_options = set(valid_letters + valid_numbers)

    def _last_match(pattern, text=None):
        haystack = tail_text if text is None else text
        matches = list(re.finditer(pattern, haystack))
        for m in reversed(matches):
            opt = m.group(1).strip()
            if opt in valid_options:
                return opt
        return None

    # 1. JSON format: {"answer": "X"}
    json_choice = _last_match(
        r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9]+)\s*["\']?\s*\}', response_lower
    )
    if json_choice:
        return json_choice

    # 2. LaTeX boxed: \boxed{a} or \boxed{\text{Option 2}}
    boxed_choice = _last_match(r'\\boxed\s*\{\s*(?:\\text\s*\{\s*(?:option\s*)?)?([a-z0-9]+)\s*\}?', response_lower)
    if boxed_choice:
        return boxed_choice

    # 3. Explicit answer markers near the end
    answer_choice = _last_match(
        r'(?:final\s+answer|final|answer|my\s+answer|choice)\s*[:\-]?\s*(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?'
        r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))'
    )
    if answer_choice:
        return answer_choice

    # 4. Decision verbs: "I would select option (a)", "choose 2", etc.
    choice_choice = _last_match(
        r"(?:i(?:'d)?\s+)?(?:would\s+)?(?:choose|select|pick|chose|selected|picking|opt\s+for|go\s+with|prefer|recommend|suggest)"
        r"\s+(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?"
        r"(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))"
    )
    if choice_choice:
        return choice_choice

    # 5. "Option X is best/most attractive" style conclusions
    option_is_choice = _last_match(
        r'\boption\s*[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\s+(?:is|seems|looks|appears|has)\s+'
        r'(?:the\s+)?(?:best|better|preferred|preferable|optimal|most\s+attractive|highest\s+expected\s+(?:utility|value))\b'
    )
    if option_is_choice:
        return option_is_choice

    # 6. Short final answer line
    lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
    for line in reversed(lines[-6:]):
        if len(line) > 30:
            continue
        m = re.fullmatch(
            r'(?:final\s+answer|final|answer|choice)?\s*[:\-]?\s*(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\.?',
            line
        )
        if m and m.group(1) in valid_options:
            return m.group(1)

    # 7. Entire response is just the option
    compact = re.sub(r'\s+', '', response_lower)
    if compact in valid_options:
        return compact

    # 8. Conclusion phrases — always scan (covers both full and truncated responses).
    conclusion = _last_match(
        r"(?:so|therefore|thus|hence|i(?:'d)?\s+(?:would\s+)?(?:choose|select|pick|go\s+with|prefer)|"
        r"(?:best|optimal|most\s+attractive)\s+(?:option|choice)\s+(?:is|would\s+be)|"
        r"(?:choose|select|pick|go\s+with)\s+option)\s*[:\-]?\s*(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?"
    )
    if conclusion:
        return conclusion

    # 9. EV analysis style: "Option X has the highest expected value/EV/utility"
    #    Also catches truncated tail: "option X has the" (cut off before "highest")
    ev_winner = _last_match(
        r"option\s*[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\s+(?:has|gives|yields|is)\s+(?:the\s+)?"
        r"(?:highest|greatest|largest|best|maximum|most\s+attractive|higher|greater|larger)"
    )
    if ev_winner:
        return ev_winner

    # 9b. Truncated tail: response ends with "option X has the" (conclusion cut off)
    tail_50 = response_lower[-150:].rstrip()
    truncated = re.search(
        r"option\s*[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\s+(?:has|gives|yields|is)\s+the\s*$",
        tail_50
    )
    if truncated and truncated.group(1) in valid_options:
        return truncated.group(1)

    # 10. Highest EV/utility mentioned before option label: "highest ... is option X"
    ev_winner2 = _last_match(
        r"(?:highest|greatest|largest|best|maximum|most\s+attractive)"
        r"[^a-z0-9\n]{0,40}(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?"
    )
    if ev_winner2:
        return ev_winner2

    # 11. "The answer/best choice is option X" or "I will go with X"
    final_stmt = _last_match(
        r"(?:the\s+)?(?:best|optimal|correct|right)\s+(?:option|choice|answer)\s+(?:is|would\s+be)\s*"
        r"(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?"
    )
    if final_stmt:
        return final_stmt

    return None


# ============================================================================
# SERIALIZATION & INCREMENTAL SAVE
# ============================================================================

def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def save_incremental(output_path, args, results, failed_responses, situations_evaluated):
    """Save current results to disk. Called after every situation for crash resilience."""
    valid = [r for r in results if r["option_type"] is not None]
    if valid:
        cooperate_rate = sum(r["option_type"] == "Cooperate" for r in valid) / len(valid)
        rebel_rate = sum(r["option_type"] == "Rebel" for r in valid) / len(valid)
        steal_rate = sum(r["option_type"] == "Steal" for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
        linear_valid = [r for r in valid if r.get("is_best_linear") is not None]
        linear_rate = sum(r["is_best_linear"] for r in linear_valid) / len(linear_valid) if linear_valid else 0
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0

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
            "steering_layer": args.steering_layer,
            "steering_layers": getattr(args, "steering_layers", None),
            "filter_bucket_label": getattr(args, "filter_bucket_label", None),
            "extra_instructions": args.extra_instructions,
        },
        "metrics": {
            "parse_rate": parse_rate,
            "cooperate_rate": cooperate_rate,
            "rebel_rate": rebel_rate,
            "steal_rate": steal_rate,
            "best_cara_rate": cara_rate,
            "best_linear_rate": linear_rate,
        },
        "num_valid": len(valid),
        "num_total": len(results),
        "results": None if args.no_save_responses else results,
        "failed_responses": failed_responses[:10]
    })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_situations(val_csv, num_situations, filter_bucket_label=None):
    """Load and parse situations from a validation CSV.

    Returns a list of situation dicts with rich metadata including
    probability_format, bucket_label, linear/CARA best options, and
    per-option is_best_linear flags.

    Args:
        val_csv: Path to the validation CSV.
        num_situations: Maximum number of situations to load (applied before filter).
        filter_bucket_label: If set, only return situations where bucket_label matches.
    """
    df = pd.read_csv(val_csv)
    situations = []

    for sit_id in df["situation_id"].unique()[:num_situations]:
        sit_data = df[df["situation_id"] == sit_id]
        prompt_raw = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)

        use_verbal_probs = sit_data["use_verbal_probs"].iloc[0] if "use_verbal_probs" in df.columns else None
        low_bucket_label = clean_bucket_label(sit_data["low_bucket_label"].iloc[0]) if "low_bucket_label" in df.columns else None

        # Determine risk-neutral (linear) best options
        linear_best_indices_0 = set()
        linear_best_option_numbers = set()
        has_linear_info = False
        if "is_best_linear_display" in df.columns:
            has_linear_info = True
            linear_best_indices_0 = set(
                int(idx) for idx in sit_data.loc[sit_data["is_best_linear_display"] == True, "option_index"]
            )
            linear_best_option_numbers = {idx + 1 for idx in linear_best_indices_0}
        elif "linear_best_labels" in df.columns:
            has_linear_info = True
            lin_labels = parse_label_list(sit_data["linear_best_labels"].iloc[0])
            linear_best_option_numbers = {
                label_to_option_number(l) for l in lin_labels if label_to_option_number(l) is not None
            }
            linear_best_indices_0 = {n - 1 for n in linear_best_option_numbers}
        if not linear_best_option_numbers:
            has_linear_info = False

        # Determine CARA alpha=0.01 best options
        cara001_best_option_numbers = set()
        if "CARA_correct_labels" in df.columns:
            cara_labels = parse_label_list(sit_data["CARA_correct_labels"].iloc[0])
            cara001_best_option_numbers = {
                label_to_option_number(l) for l in cara_labels if label_to_option_number(l) is not None
            }
        if not cara001_best_option_numbers and "CARA_alpha_0_01_best_labels" in df.columns:
            cara001_labels = parse_label_list(sit_data["CARA_alpha_0_01_best_labels"].iloc[0])
            cara001_best_option_numbers = {
                label_to_option_number(l) for l in cara001_labels if label_to_option_number(l) is not None
            }
        if not cara001_best_option_numbers and "is_best_cara_display" in df.columns:
            cara001_best_option_numbers = {
                int(idx) + 1 for idx in sit_data.loc[sit_data["is_best_cara_display"] == True, "option_index"]
            }

        bucket_label = low_bucket_label
        if bucket_label is None and linear_best_option_numbers and cara001_best_option_numbers:
            if linear_best_option_numbers == cara001_best_option_numbers:
                bucket_label = "both"

        options = {}
        best_cara_indices = set()
        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            is_best_cara = parse_bool_like(row.get("is_best_cara_display", False)) or False
            option_data = {
                "type": row["option_type"],
                "is_best_cara": is_best_cara,
                "is_best_linear": (idx in linear_best_indices_0) if has_linear_info else None,
                "option_index": idx,
            }
            options[letter] = option_data
            options[number] = option_data
            if is_best_cara:
                best_cara_indices.add(idx)

        situations.append({
            "situation_id": sit_id,
            "prompt": prompt_raw,
            "num_options": num_options,
            "options": options,
            "probability_format": probability_format_from_value(use_verbal_probs, prompt_raw),
            "bucket_label": bucket_label,
            "linear_best_option": sorted(linear_best_option_numbers),
            "cara001_best_option": sorted(cara001_best_option_numbers),
            "best_cara_indices": sorted(best_cara_indices),
        })

    if filter_bucket_label:
        situations = [s for s in situations if s["bucket_label"] == filter_bucket_label]

    return situations


# ============================================================================
# EVALUATION LOOP
# ============================================================================

def run_evaluation(model, tokenizer, situations, steering_vector,
                   alpha=0.0, steering_layer=14,
                   temperature=0.7, max_new_tokens=4096,
                   max_time_per_generation=120,
                   disable_thinking=False, no_save_responses=True,
                   verbose=True, incremental_save_path=None, incremental_save_args=None,
                   extra_instructions="", thinking_prefix=None):
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
        extra_instructions: If set, injected as a system message before the user prompt
        thinking_prefix: If set and thinking is enabled, pre-fills the <think> block with
            this text then closes it with </think>, forcing the model to skip extended
            reasoning and generate only the answer. Dramatically reduces generation time.

    Returns:
        dict with keys: cooperate_rate, rebel_rate, steal_rate, cara_rate,
                        linear_rate, parse_rate, num_valid, num_total, results,
                        failed_responses, generation_times, total_elapsed
    """
    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        prompt = remove_instruction_suffix(sit["prompt"])
        messages = []
        if extra_instructions:
            messages.append({"role": "system", "content": extra_instructions})
        messages.append({"role": "user", "content": prompt})

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

        # Pre-fill thinking block to force model straight to the answer.
        # The template already ends with <think>\n; we append a brief trigger
        # then </think> so the model only generates the answer portion.
        if thinking_prefix is not None and not disable_thinking:
            text += thinking_prefix + "\n</think>\n"

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
        choice_index = label_to_option_number(choice) if choice else None

        if choice and choice in sit["options"]:
            option_type = sit["options"][choice]["type"]
            results.append({
                "situation_id": sit["situation_id"],
                "prompt": prompt,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "linear_best_option": sit["linear_best_option"],
                "cara001_best_option": sit["cara001_best_option"],
                "choice": choice,
                "choice_index": choice_index,
                "option_type": option_type,
                "is_best_cara": sit["options"][choice]["is_best_cara"],
                "is_best_linear": sit["options"][choice]["is_best_linear"],
                "response": None if no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "prompt": prompt,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "linear_best_option": sit["linear_best_option"],
                "cara001_best_option": sit["cara001_best_option"],
                "choice": None,
                "choice_index": None,
                "option_type": None,
                "is_best_cara": None,
                "is_best_linear": None,
                "response": None if no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
            failed_responses.append({
                "situation_id": sit["situation_id"],
                "num_options": sit["num_options"],
                "prompt": prompt,
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
    valid = [r for r in results if r["option_type"] is not None]
    if valid:
        cooperate_rate = sum(r["option_type"] == "Cooperate" for r in valid) / len(valid)
        rebel_rate = sum(r["option_type"] == "Rebel" for r in valid) / len(valid)
        steal_rate = sum(r["option_type"] == "Steal" for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
        linear_valid = [r for r in valid if r.get("is_best_linear") is not None]
        linear_rate = sum(r["is_best_linear"] for r in linear_valid) / len(linear_valid) if linear_valid else 0
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0

    parse_rate = len(valid) / len(results) if results else 0

    return {
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "cara_rate": cara_rate,
        "linear_rate": linear_rate,
        "parse_rate": parse_rate,
        "num_valid": len(valid),
        "num_total": len(results),
        "results": results,
        "failed_responses": failed_responses,
        "generation_times": generation_times,
        "total_elapsed": total_elapsed,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)")
    parser.add_argument("--val_csv", type=str,
                        default="data/2026_01_29_new_val_set_probabilities_add_to_100.csv")
    parser.add_argument("--num_situations", type=int, default=50,
                        help="Number of situations to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (auto-generated if omitted)")
    parser.add_argument("--no_save_responses", action="store_true",
                        help="Do NOT save full responses (by default, all CoT responses are saved)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate (default 1024)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model ID (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = deterministic)")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable thinking mode in chat template (auto-enabled for base models)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Force thinking mode ON, overriding the auto-disable for base models")
    parser.add_argument("--max_time_per_generation", type=float, default=120,
                        help="Max seconds per generation before timeout (default: 120)")
    parser.add_argument("--steering_path", type=str, default=None,
                        help="Path to steering vector .pt file (optional)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Steering strength: positive=more risk-averse, negative=more risk-neutral")
    parser.add_argument("--steering_layer", type=int, default=None,
                        help="Layer to apply steering (default: use layer from .pt file, or 14)")
    parser.add_argument("--extra_instructions", type=str, default="",
                        help="Extra instructions injected as a system prompt "
                             "(e.g. 'Be concise and go straight to the answer after your thinking process.')")
    parser.add_argument("--filter_bucket_label", type=str, default=None,
                        help="Filter situations to only this bucket label (e.g. 'lin_only')")
    parser.add_argument("--steering_layers", type=int, nargs="+", default=None,
                        help="Space-separated list of layers for simultaneous multi-layer injection "
                             "(overrides --steering_layer when provided)")
    args = parser.parse_args()

    # Auto-generate descriptive output filename if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        alpha_suffix = f"_alpha{args.alpha}" if args.steering_path and args.alpha != 0 else ""
        if args.steering_layers:
            layer_suffix = "_layers" + "-".join(str(l) for l in args.steering_layers)
        else:
            layer_suffix = ""
        args.output = f"eval_{model_short}_temp{args.temperature}{alpha_suffix}{layer_suffix}_{timestamp}.json"

    # Auto-enable disable_thinking for base model evaluation (no adapter)
    # --enable_thinking overrides this auto-disable
    if args.model_path is None and not args.disable_thinking and not args.enable_thinking:
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
            if args.steering_layer is None and saved_layer is not None:
                steering_layer = saved_layer
            args.steering_layer = steering_layer
            print(f"  Vector shape: {steering_vector.shape}")
            print(f"  Alpha (strength): {args.alpha}")
            if metadata:
                print(f"  Generated from: {metadata.get('num_pairs', '?')} pairs")
                print(f"  Position: {metadata.get('position', '?')}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Continuing without steering...")
            steering_vector = None

    # Resolve effective layer(s) for steering
    if args.steering_layers:
        effective_layer = args.steering_layers
    else:
        effective_layer = steering_layer

    print("Loading validation data...")
    situations = load_situations(args.val_csv, args.num_situations,
                                 filter_bucket_label=args.filter_bucket_label)

    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    if args.filter_bucket_label:
        print(f"Bucket label filter: {args.filter_bucket_label}")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    if steering_vector is not None:
        print(f"Activation steering: ENABLED (alpha={args.alpha}, layer(s)={effective_layer})")
    else:
        print(f"Activation steering: disabled")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    if args.extra_instructions:
        print(f"Extra instructions (system prompt): {args.extra_instructions!r}")
    print(f"Results will be saved incrementally to: {args.output}")
    print()

    eval_result = run_evaluation(
        model, tokenizer, situations, steering_vector,
        alpha=args.alpha, steering_layer=effective_layer,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        max_time_per_generation=args.max_time_per_generation,
        disable_thinking=args.disable_thinking,
        no_save_responses=args.no_save_responses,
        verbose=True,
        incremental_save_path=args.output,
        incremental_save_args=args,
        extra_instructions=args.extra_instructions,
    )

    results = eval_result["results"]
    failed_responses = eval_result["failed_responses"]
    generation_times = eval_result["generation_times"]
    total_elapsed = eval_result["total_elapsed"]
    valid = [r for r in results if r["option_type"] is not None]

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
    print(f"% choosing best LIN:  {100*eval_result['linear_rate']:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Avg per situation: {sum(generation_times)/len(generation_times):.1f}s")
    print(f"Avg tokens generated: {sum(r.get('num_tokens_generated', 0) for r in results)/len(results):.0f}")
    print("="*50)

    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("="*50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr['response'][:600])
            print("...")

    save_incremental(args.output, args, results, failed_responses, len(situations))
    print(f"\nFinal results saved to {args.output}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
