import unittest

import pandas as pd

from cot_csv_utils import (
    normalize_cot_newlines_in_dataframe,
    summarize_cot_dataframe,
    validate_no_literal_backslash_newlines,
)


class CotCsvUtilsTests(unittest.TestCase):
    def test_normalize_cot_newlines_in_dataframe_converts_literal_backslash_sequences(self):
        df = pd.DataFrame(
            {
                "prompt_text": ["Prompt"],
                "chosen_full": ['<think>\\nReasoning\\n</think>\\n\\n{"answer":"a"}'],
                "rejected_full": ['<think>\\nOther\\n</think>\\n\\n{"answer":"b"}'],
            }
        )

        out, changed_cells = normalize_cot_newlines_in_dataframe(df)

        self.assertEqual(changed_cells, 2)
        self.assertIn("\nReasoning\n", out.loc[0, "chosen_full"])
        self.assertNotIn("\\n", out.loc[0, "chosen_full"])
        self.assertNotIn("\\n", out.loc[0, "rejected_full"])

    def test_summarize_cot_dataframe_reports_literal_backslash_newlines_and_trailing_text(self):
        df = pd.DataFrame(
            {
                "chosen_full": ['<think>\\nReasoning\\n</think>\\n\\n{"answer":"a"}'],
                "rejected_full": ['<think>Bad</think> trailing'],
            }
        )

        summary = summarize_cot_dataframe(df)

        self.assertEqual(summary["cells_checked"], 2)
        self.assertEqual(summary["cells_with_literal_backslash_newlines"], 1)
        self.assertEqual(summary["rows_with_literal_backslash_newlines"], 1)
        self.assertEqual(summary["cells_with_extra_text_after_think"], 1)

    def test_summarize_cot_dataframe_reports_prompt_meta_references(self):
        df = pd.DataFrame(
            {
                "chosen_full": ['<think>The prompt says option 1 is correct.</think>\n\n{"answer":"1"}'],
                "rejected_full": ['<think>Clean.</think>\n\n{"answer":"2"}'],
            }
        )

        summary = summarize_cot_dataframe(df)

        self.assertEqual(summary["cells_with_prompt_meta_references"], 1)
        self.assertEqual(summary["rows_with_prompt_meta_references"], 1)

    def test_validate_no_literal_backslash_newlines_raises_with_fix_command(self):
        df = pd.DataFrame(
            {
                "chosen_full": ['<think>\\nReasoning\\n</think>\\n\\n{"answer":"a"}'],
                "rejected_full": ['<think>\\nOther\\n</think>\\n\\n{"answer":"b"}'],
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_no_literal_backslash_newlines(df, "data/example.csv")

        self.assertIn("cot_csv_utils.py --fix-newlines-in-place", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
