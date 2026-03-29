#!/usr/bin/env python3
"""Validate and normalize CoT CSV text fields."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import pandas as pd


COT_TEXT_COLUMNS = ("chosen_full", "rejected_full")
PROMPT_META_REFERENCE_PATTERNS = (
    r"\binstruction(?:s)?\b",
    r"\bas instructed\b",
    r"\bthe problem states\b",
    r"\bthe prompt says\b",
    r"\bprompt says\b",
    r"\bfollowing the instruction\b",
    r"\bfollowing my instructions\b",
    r"\bgiven the instruction\b",
    r"\binstructed that\b",
    r"\bproblem statement\b",
    r"\bcorrect answer\b",
)
PROMPT_META_REFERENCE_RE = re.compile("|".join(PROMPT_META_REFERENCE_PATTERNS), re.IGNORECASE)


def find_cot_text_columns(df: pd.DataFrame) -> List[str]:
    """Return the CoT text columns present in a dataframe."""
    return [col for col in COT_TEXT_COLUMNS if col in df.columns]


def normalize_literal_backslash_newlines(text: Any) -> Any:
    """Convert literal backslash-newline sequences to real newlines."""
    if not isinstance(text, str):
        return text
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def summarize_cot_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize newline and simple formatting issues in CoT text columns."""
    cot_columns = find_cot_text_columns(df)
    summary: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "cot_columns": cot_columns,
        "cells_checked": 0,
        "cells_with_literal_backslash_newlines": 0,
        "rows_with_literal_backslash_newlines": 0,
        "cells_missing_think_open": 0,
        "cells_missing_think_close": 0,
        "cells_with_multiple_think_close": 0,
        "cells_with_extra_text_after_think": 0,
        "cells_with_prompt_meta_references": 0,
        "rows_with_prompt_meta_references": 0,
    }
    if not cot_columns:
        return summary

    for _, row in df.iterrows():
        row_has_literal_backslash_newlines = False
        row_has_prompt_meta_references = False
        for col in cot_columns:
            text = row.get(col)
            if not isinstance(text, str):
                continue
            summary["cells_checked"] += 1

            has_literal_backslash_newlines = "\\n" in text or "\\r" in text
            if has_literal_backslash_newlines:
                summary["cells_with_literal_backslash_newlines"] += 1
                row_has_literal_backslash_newlines = True

            analysis_text = normalize_literal_backslash_newlines(text)
            open_count = analysis_text.count("<think>")
            close_count = analysis_text.count("</think>")
            if open_count < 1:
                summary["cells_missing_think_open"] += 1
            if close_count < 1:
                summary["cells_missing_think_close"] += 1
            if close_count > 1:
                summary["cells_with_multiple_think_close"] += 1
            if close_count >= 1:
                trailing_text = analysis_text.split("</think>", 1)[1].strip()
                if trailing_text and not trailing_text.startswith('{"answer"'):
                    summary["cells_with_extra_text_after_think"] += 1
            if PROMPT_META_REFERENCE_RE.search(analysis_text):
                summary["cells_with_prompt_meta_references"] += 1
                row_has_prompt_meta_references = True

        if row_has_literal_backslash_newlines:
            summary["rows_with_literal_backslash_newlines"] += 1
        if row_has_prompt_meta_references:
            summary["rows_with_prompt_meta_references"] += 1
    return summary


def normalize_cot_newlines_in_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Return a copy with literal backslash-newlines normalized in CoT columns."""
    cot_columns = find_cot_text_columns(df)
    if not cot_columns:
        return df.copy(), 0

    out = df.copy()
    changed_cells = 0
    for col in cot_columns:
        normalized = out[col].map(normalize_literal_backslash_newlines)
        changed_cells += int((normalized != out[col]).sum())
        out[col] = normalized
    return out, changed_cells


def validate_no_literal_backslash_newlines(df: pd.DataFrame, dataset_path: str):
    """Raise with a concrete fix command if a CoT CSV still has literal backslash-newlines."""
    summary = summarize_cot_dataframe(df)
    if summary["cells_with_literal_backslash_newlines"] <= 0:
        return summary

    raise ValueError(
        "CoT CSV contains literal backslash-newline sequences in chosen_full/rejected_full.\n"
        "These are not converted automatically by pandas or the tokenizer, so the model will see "
        "backslash characters rather than real line breaks.\n"
        f"Dataset: {dataset_path}\n"
        f"Rows with literal backslash-newlines: {summary['rows_with_literal_backslash_newlines']}\n"
        f"Cells with literal backslash-newlines: {summary['cells_with_literal_backslash_newlines']}\n"
        f"Fix with: python cot_csv_utils.py --fix-newlines-in-place \"{dataset_path}\""
    )


def format_summary(path: Path, summary: Dict[str, Any]) -> str:
    """Render a short human-readable summary."""
    cot_columns = summary["cot_columns"] or []
    columns_text = ", ".join(cot_columns) if cot_columns else "none"
    return (
        f"{path}\n"
        f"  rows: {summary['num_rows']}\n"
        f"  CoT columns: {columns_text}\n"
        f"  cells checked: {summary['cells_checked']}\n"
        f"  cells with literal backslash-newlines: {summary['cells_with_literal_backslash_newlines']}\n"
        f"  rows with literal backslash-newlines: {summary['rows_with_literal_backslash_newlines']}\n"
        f"  cells missing <think>: {summary['cells_missing_think_open']}\n"
        f"  cells missing </think>: {summary['cells_missing_think_close']}\n"
        f"  cells with multiple </think>: {summary['cells_with_multiple_think_close']}\n"
        f"  cells with extra text after </think>: {summary['cells_with_extra_text_after_think']}\n"
        f"  cells with prompt-meta references: {summary['cells_with_prompt_meta_references']}\n"
        f"  rows with prompt-meta references: {summary['rows_with_prompt_meta_references']}"
    )


def _file_has_utf8_bom(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(3) == b"\xef\xbb\xbf"


def _load_csv_for_cli(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _save_csv_for_cli(df: pd.DataFrame, path: Path, had_bom: bool):
    encoding = "utf-8-sig" if had_bom else "utf-8"
    df.to_csv(path, index=False, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate CoT CSVs for literal backslash-newlines in chosen_full/rejected_full "
            "and optionally normalize them in place."
        )
    )
    parser.add_argument("paths", nargs="+", help="One or more CSV paths to check")
    parser.add_argument(
        "--fix-newlines-in-place",
        action="store_true",
        help="Rewrite each CSV in place, converting literal backslash-newlines to real newlines in CoT columns",
    )
    args = parser.parse_args()

    any_literal_backslash_newlines = False
    for raw_path in args.paths:
        path = Path(raw_path).expanduser().resolve()
        df = _load_csv_for_cli(path)
        if args.fix_newlines_in_place:
            had_bom = _file_has_utf8_bom(path)
            df, changed_cells = normalize_cot_newlines_in_dataframe(df)
            _save_csv_for_cli(df, path, had_bom)
            print(f"Normalized {changed_cells} CoT cells in {path}")

        summary = summarize_cot_dataframe(df)
        print(format_summary(path, summary))
        if summary["cells_with_literal_backslash_newlines"] > 0:
            any_literal_backslash_newlines = True

    raise SystemExit(1 if any_literal_backslash_newlines else 0)


if __name__ == "__main__":
    main()
