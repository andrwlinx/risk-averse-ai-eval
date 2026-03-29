#!/usr/bin/env python3
"""Shared permissive answer parser used across evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional, Set


_ROMAN_TO_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}

_ORDINAL_TO_INT = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_TRUNCATION_SAFE_STRATEGIES = frozenset(
    {
        "json",
        "boxed",
        "short_answer_line",
        "bare_token",
    }
)


@dataclass(frozen=True)
class ChoiceParseResult:
    choice: Optional[str]
    strategy: Optional[str]


def _normalize_response_text(response: str) -> str:
    text = unicodedata.normalize("NFKC", response)
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    # Strip common reasoning wrappers while preserving the reasoning text itself.
    text = re.sub(r"</?(?:think|thinking|reasoning|analysis)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.rstrip()


def _normalize_label_token(token: str) -> str:
    s = unicodedata.normalize("NFKC", token).strip().lower()
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    s = re.sub(r"^(?:option|answer)\s*", "", s)
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    if s in _ROMAN_TO_INT:
        s = str(_ROMAN_TO_INT[s])
    if s in _ORDINAL_TO_INT:
        s = str(_ORDINAL_TO_INT[s])
    return s


def _valid_options(num_options: int) -> Set[str]:
    letters = {chr(ord("a") + i) for i in range(num_options)}
    numbers = {str(i + 1) for i in range(num_options)}
    return letters | numbers


def _valid_options_for_style(num_options: int, label_style: Optional[str]) -> Set[str]:
    if label_style == "numbers":
        return {str(i + 1) for i in range(num_options)}
    if label_style == "letters":
        return {chr(ord("a") + i) for i in range(num_options)}
    return _valid_options(num_options)


def _extract_valid_matches(pattern: str, text: str, valid: Set[str]) -> List[str]:
    matches = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(m.group(1))
        if token in valid:
            matches.append(token)
    return matches


def _last_valid_match(pattern: str, text: str, valid: Set[str]) -> Optional[str]:
    matches = _extract_valid_matches(pattern, text, valid)
    return matches[-1] if matches else None


def _valid_match_records(pattern: str, text: str, valid: Set[str]) -> List[tuple[str, re.Match[str]]]:
    records = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(m.group(1))
        if token in valid:
            records.append((token, m))
    return records


def _tail_sentences(text: str, limit: int = 8) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    return sentences[-limit:]


def infer_option_label_style(prompt_text: str, num_options: int) -> Optional[str]:
    """Infer whether the prompt enumerates options with numbers or letters."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None

    normalized = unicodedata.normalize("NFKC", prompt_text).lower()
    max_labels = min(num_options, 4)
    number_hits = 0
    letter_hits = 0

    for idx in range(max_labels):
        number_label = str(idx + 1)
        letter_label = chr(ord("a") + idx)

        number_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(number_label)}\s*\)\s*[\.\:]",
            rf"(?:^|\n)\s*{re.escape(number_label)}[\)\.\:]\s+",
        ]
        letter_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(letter_label)}\s*\)\s*[\.\:]",
            rf"(?:^|\n)\s*{re.escape(letter_label)}[\)\.\:]\s+",
        ]

        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in number_patterns):
            number_hits += 1
        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in letter_patterns):
            letter_hits += 1

    if number_hits >= 2 and number_hits > letter_hits:
        return "numbers"
    if letter_hits >= 2 and letter_hits > number_hits:
        return "letters"
    return None


def extract_json_answer(response: str) -> Optional[str]:
    """Extract answer from JSON snippets like {"answer":"2"}."""
    if not isinstance(response, str) or not response.strip():
        return None
    text = _normalize_response_text(response)
    pattern = r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}'
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    candidate = None
    for m in matches:
        candidate = _normalize_label_token(m.group(1))
    return candidate


def apply_finish_reason_safeguard(
    parse_result: ChoiceParseResult,
    finish_reason: Optional[str],
) -> ChoiceParseResult:
    """Drop weak parses when generation was truncated by max length."""
    if parse_result.choice is None:
        return parse_result
    if finish_reason == "length" and parse_result.strategy not in _TRUNCATION_SAFE_STRATEGIES:
        return ChoiceParseResult(choice=None, strategy=None)
    return parse_result


def _sentence_window(text: str, start: int, end: int) -> tuple[str, int, int]:
    left = max(
        text.rfind("\n", 0, start),
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
    )
    right_candidates = [
        pos
        for pos in (
            text.find("\n", end),
            text.find(".", end),
            text.find("!", end),
            text.find("?", end),
        )
        if pos != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    sentence = text[left + 1 : right].strip()
    offset = left + 1
    return sentence, start - offset, end - offset


def _is_hedged_or_ambiguous_match(text: str, match: re.Match[str]) -> bool:
    sentence, local_start, local_end = _sentence_window(text, match.start(), match.end())
    sentence_lower = sentence.lower()
    prefix = sentence_lower[: max(local_start, 0)]
    suffix = sentence_lower[max(local_end, 0) :]

    # Reject speculative or hypothetical statements like
    # "maybe the answer is option 4" or "if you're risk-averse, you might prefer option 1".
    if re.search(r"\b(?:maybe|perhaps|alternatively)\b", sentence_lower):
        return True
    if re.search(r"\b(?:might|may|could)\b", prefix):
        return True
    if re.search(r"\bif\s+you\b", prefix):
        return True
    if re.search(r"\bif\s+i(?:'m|\s+am)\s+risk-", prefix):
        return True
    if re.search(r"\b(?:someone|somebody)\b", prefix):
        return True
    if re.search(r"\brisk-(?:averse|seeking|neutral)\b", sentence_lower):
        return True
    if re.search(r"^\s*than\b", suffix):
        return True
    if re.search(r"^\s*(?:or|and)\b", suffix):
        return True
    return False


def extract_choice_with_strategy(
    response: str,
    num_options: int,
    label_style: Optional[str] = None,
) -> ChoiceParseResult:
    """
    Extract a model choice using permissive but bounded matching.
    Returns both the parsed choice and the strategy used.
    """
    if not isinstance(response, str) or not response.strip():
        return ChoiceParseResult(choice=None, strategy=None)

    text = _normalize_response_text(response).lower()
    tail = text[-3000:] if len(text) > 3000 else text
    valid = _valid_options(num_options)

    explicit_marker = (
        r'(?:final\s+answer|final|answer|my\s+answer|choice|'
        r'chosen\s+(?:option|answer)|selected\s+(?:option|answer))'
    )

    # 1) JSON answer format (most explicit).
    json_choice = _last_valid_match(
        r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}',
        text,
        valid,
    )
    if json_choice:
        return ChoiceParseResult(choice=json_choice, strategy="json")

    # 2) Boxed answer format often used by reasoning models.
    boxed_choice = _last_valid_match(r'\\boxed\s*\{\s*([a-z0-9ivx]+)\s*\}', text, valid)
    if boxed_choice:
        return ChoiceParseResult(choice=boxed_choice, strategy="boxed")

    lines = [line.strip() for line in tail.splitlines() if line.strip()]

    # 3) Short explicit answer lines near the end. This is more reliable than
    # earlier reasoning mentions like "if you choose option 1".
    for line in reversed(lines[-8:]):
        if len(line) > 90:
            continue
        m = re.fullmatch(
            rf'(?:{explicit_marker})?\s*[:\-]?\s*(?:is\s+)?'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?\.?',
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        token = _normalize_label_token(m.group(1))
        if token in valid:
            return ChoiceParseResult(choice=token, strategy="short_answer_line")

    # 4) Explicit answer markers.
    answer_records = _valid_match_records(
        rf'{explicit_marker}\s*[:\-]?\s*'
        r'(?:is\s+)?(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?'
        r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))',
        tail,
        valid,
    )
    for token, match in reversed(answer_records):
        if not _is_hedged_or_ambiguous_match(tail, match):
            return ChoiceParseResult(choice=token, strategy="answer_marker")

    # 5) Decision verbs.
    decision_records = _valid_match_records(
        r"\bi(?:'d|'ll)?\s+(?:(?:would|will|should|must|ought\s+to)\s+)?"
        r"(?:choose|select|pick|chose|selected|choosing|picking|opt\s+for|go\s+with|"
        r"prefer|recommend|suggest)\s+(?:option\s*)?[\(\[]?\s*(?:the\s+)?"
        r"([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?"
        r"(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))",
        tail,
        valid,
    )
    for token, match in reversed(decision_records):
        if not _is_hedged_or_ambiguous_match(tail, match):
            return ChoiceParseResult(choice=token, strategy="decision_verb")

    # 6) Conclusive statement about best/most attractive option.
    option_is_records = _valid_match_records(
        r'\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
        r'(?:is|seems|looks|appears|has)\s+(?:the\s+)?'
        r'(?:best|better|preferred|preferable|optimal|most\s+attractive)'
        r'(?:\s+(?:choice|option|pick|one))?',
        tail,
        valid,
    )
    for token, match in reversed(option_is_records):
        if not _is_hedged_or_ambiguous_match(tail, match):
            return ChoiceParseResult(choice=token, strategy="option_is_best")

    # 7) Conclusive last-sentence forms often emitted inside thinking-only blocks.
    tail_sentences = _tail_sentences(tail)

    best_choice_patterns = [
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*(?:the\s+)?'
            r'(?:best|preferred|correct|right|optimal|most\s+attractive)\s+'
            r'(?:option|choice|answer)\s*(?:is|would\s+be|seems\s+to\s+be)\s*'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?'
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|overall|therefore|thus|hence)\b))',
            "best_choice_is",
        ),
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*'
            r'i\s+(?:should|would|will|must|ought\s+to)\s+'
            r'(?:choose|select|pick|go\s+with|opt\s+for)\s+'
            r'(?:option\s*)?[\(\[]?\s*(?:the\s+)?([a-z0-9ivx]+)\s*(?:option)?\s*[\)\]]?'
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|overall|therefore|thus|hence)\b))',
            "decision_modal",
        ),
    ]
    for sentence in reversed(tail_sentences):
        for pattern, strategy in best_choice_patterns:
            choice = _last_valid_match(pattern, sentence, valid)
            if choice:
                return ChoiceParseResult(choice=choice, strategy=strategy)

    # 8) If the entire response is just the option token.
    compact = re.sub(r"\s+", "", text)
    compact = _normalize_label_token(compact)
    if compact in valid:
        return ChoiceParseResult(choice=compact, strategy="bare_token")

    # 9) Final-sentence fallback for embedded choices like
    # "option 3 is the one I should choose" or "I'm going to choose option 3".
    final_sentence = tail_sentences[-1] if tail_sentences else ""
    styled_valid = _valid_options_for_style(num_options, label_style)
    if final_sentence and styled_valid:
        sentence_has_decision_cue = bool(
            re.search(
                r"\b(?:choose|pick|select|go\s+with|opt\s+for|prefer|recommend|suggest)\b",
                final_sentence,
                flags=re.IGNORECASE,
            )
        )
        ambiguous_suffix_hits = _extract_valid_matches(
            r"\b(?:or|and)\s+(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?"
            r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]))',
            final_sentence,
            styled_valid,
        )
        if ambiguous_suffix_hits:
            return ChoiceParseResult(choice=None, strategy=None)

        option_prefixed_records = _valid_match_records(
            r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\b",
            final_sentence,
            styled_valid,
        )
        option_prefixed_hits = [token for token, _ in option_prefixed_records]
        if (
            sentence_has_decision_cue
            and option_prefixed_hits
            and len(set(option_prefixed_hits)) == 1
            and all(not _is_hedged_or_ambiguous_match(final_sentence, match) for _, match in option_prefixed_records)
        ):
            return ChoiceParseResult(choice=option_prefixed_hits[-1], strategy="final_sentence_option")

        if label_style == "numbers" and sentence_has_decision_cue:
            standalone_number_hits = _extract_valid_matches(
                r"(?<![\w.])([0-9]+)(?![\w.%])",
                final_sentence,
                styled_valid,
            )
            if standalone_number_hits and len(set(standalone_number_hits)) == 1:
                return ChoiceParseResult(
                    choice=standalone_number_hits[-1],
                    strategy="final_sentence_option",
                )

    return ChoiceParseResult(choice=None, strategy=None)


def extract_choice_permissive(
    response: str,
    num_options: int,
    label_style: Optional[str] = None,
) -> Optional[str]:
    """Compatibility wrapper returning just the parsed choice."""
    return extract_choice_with_strategy(response, num_options, label_style=label_style).choice
