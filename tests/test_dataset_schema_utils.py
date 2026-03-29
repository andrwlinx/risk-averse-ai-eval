import unittest

import pandas as pd

from dataset_schema_utils import ensure_option_level_dataframe


class DatasetSchemaUtilsTests(unittest.TestCase):
    def test_ensure_option_level_dataframe_expands_options_json(self):
        df = pd.DataFrame(
            {
                "situation_id": [1],
                "prompt_text": ["Prompt"],
                "num_options": [2],
                "linear_correct_labels": ['["2"]'],
                "CARA_correct_labels": ['["1"]'],
                "options_json": [
                    '[{"option_index": 0, "prizes": [-5, 10], "probabilities_percent": [50, 50], '
                    '"EU_linear_display_3sf": "2.5", "EU_cara_display_3sf": "0.1", '
                    '"is_best_linear": false, "is_best_cara": true}, '
                    '{"option_index": 1, "prizes": [3], "probabilities_percent": [100], '
                    '"EU_linear_display_3sf": "3", "EU_cara_display_3sf": "0.05", '
                    '"is_best_linear": true, "is_best_cara": false}]'
                ],
            }
        )

        out = ensure_option_level_dataframe(df)

        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["option_index"]), [0, 1])
        self.assertEqual(list(out["option_type"]), ["Generic", "Generic"])
        self.assertEqual(out.loc[0, "probs_percent"], "[50, 50]")
        self.assertEqual(out.loc[1, "is_best_linear_display"], True)
        self.assertEqual(out.loc[0, "CARA_correct_labels"], '["1"]')


if __name__ == "__main__":
    unittest.main()
