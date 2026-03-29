import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_reward_model import ensure_output_path_is_safe, select_pairs, summarize_pairwise_results
from prepare_reward_model_eval_dataset import (
    alternate_by_subset_type,
    apply_prompt_choice_verb_mix,
    dedupe_exact_pair_rows,
    normalize_reward_df,
)


class RewardModelDatasetPrepTests(unittest.TestCase):
    def test_normalize_reward_df_drops_unnamed_columns(self):
        df = pd.DataFrame(
            [
                {
                    "prompt_text": "p1",
                    "chosen_full": "c",
                    "rejected_full": "r",
                    "rejected_type": "lin",
                    "Unnamed: 25": "drop me",
                }
            ]
        )
        normalized = normalize_reward_df(df)
        self.assertIn("prompt_text", normalized.columns)
        self.assertNotIn("Unnamed: 25", normalized.columns)
        self.assertEqual(normalized["subset_type"].tolist(), ["rebels_only"])

    def test_dedupe_exact_pair_rows_keeps_same_prompt_with_different_responses(self):
        df = pd.DataFrame(
            [
                {"prompt_text": "same", "chosen_full": "c1", "rejected_full": "r1", "rejected_type": "lin"},
                {"prompt_text": "same", "chosen_full": "c2", "rejected_full": "r2", "rejected_type": "too_risk"},
                {"prompt_text": "same", "chosen_full": "c1", "rejected_full": "r1", "rejected_type": "lin"},
                {"prompt_text": "other", "chosen_full": "c3", "rejected_full": "r3", "rejected_type": "lin"},
            ]
        )
        deduped = dedupe_exact_pair_rows(normalize_reward_df(df))
        self.assertEqual(len(deduped), 3)
        same_rows = deduped[deduped["prompt_text"] == "same"]
        self.assertEqual(len(same_rows), 2)
        self.assertEqual(set(same_rows["rejected_type"].tolist()), {"lin", "too_risk"})

    def test_alternate_by_subset_type_interleaves_prefix(self):
        df = pd.DataFrame(
            [
                {"prompt_text": "p1", "rejected_type": "lin", "subset_type": "rebels_only", "prompt_first_index": 0},
                {"prompt_text": "p2", "rejected_type": "lin", "subset_type": "rebels_only", "prompt_first_index": 1},
                {"prompt_text": "p3", "rejected_type": "too_risk", "subset_type": "steals_only", "prompt_first_index": 2},
                {"prompt_text": "p4", "rejected_type": "lin", "subset_type": "rebels_only", "prompt_first_index": 3},
                {"prompt_text": "p5", "rejected_type": "too_risk", "subset_type": "steals_only", "prompt_first_index": 4},
            ]
        )
        combined = alternate_by_subset_type(df)
        self.assertEqual(combined["subset_type"].tolist()[:4], ["rebels_only", "steals_only", "rebels_only", "steals_only"])

    def test_apply_prompt_choice_verb_mix_spreads_verbs_by_prompt(self):
        df = pd.DataFrame(
            [
                {"prompt_text": "They would like you to select whichever option seems best. Which option would you select?"},
                {"prompt_text": "They would like you to select whichever option seems best. Which option would you select?"},
                {"prompt_text": "Your user would like you to select whichever option seems best. Which option would you select?"},
                {"prompt_text": "Your creators would like you to select whichever option seems best. Which option would you select?"},
            ]
        )
        mixed = apply_prompt_choice_verb_mix(df)
        self.assertIn("select", mixed.iloc[0]["prompt_text"])
        self.assertEqual(mixed.iloc[0]["prompt_text"], mixed.iloc[1]["prompt_text"])
        self.assertIn("choose", mixed.iloc[2]["prompt_text"])
        self.assertIn("pick", mixed.iloc[3]["prompt_text"])


class RewardModelMetricTests(unittest.TestCase):
    def test_summarize_pairwise_results(self):
        results = [
            {
                "accepted_score": 2.0,
                "rejected_score": 1.0,
                "score_margin": 1.0,
                "predicted_preference": "accepted",
                "is_correct": True,
                "accepted_truncated": False,
                "rejected_truncated": False,
                "length_relation": "accepted_longer",
            },
            {
                "accepted_score": 1.0,
                "rejected_score": 3.0,
                "score_margin": -2.0,
                "predicted_preference": "rejected",
                "is_correct": False,
                "accepted_truncated": True,
                "rejected_truncated": False,
                "length_relation": "rejected_longer",
            },
            {
                "accepted_score": 0.5,
                "rejected_score": 0.5,
                "score_margin": 0.0,
                "predicted_preference": "tie",
                "is_correct": False,
                "accepted_truncated": False,
                "rejected_truncated": True,
                "length_relation": "same_length",
            },
        ]
        summary = summarize_pairwise_results(results)
        self.assertEqual(summary["num_total"], 3)
        self.assertEqual(summary["num_correct"], 1)
        self.assertEqual(summary["num_ties"], 1)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_ties_half_credit"], 0.5)
        self.assertAlmostEqual(summary["metrics"]["tie_rate"], 1 / 3)
        self.assertAlmostEqual(summary["metrics"]["truncated_pair_rate"], 2 / 3)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_accepted_longer"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_rejected_longer"], 0.0)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_same_length"], 0.0)


class RewardModelSelectionTests(unittest.TestCase):
    def test_select_pairs_drops_only_exact_duplicate_rows(self):
        pairs = [
            {
                "pair_id": 1,
                "dataset_position": 1,
                "situation_id": 10,
                "prompt_raw": "p",
                "accepted_response": "a",
                "rejected_response": "r",
            },
            {
                "pair_id": 2,
                "dataset_position": 2,
                "situation_id": 10,
                "prompt_raw": "p",
                "accepted_response": "a",
                "rejected_response": "r",
            },
            {
                "pair_id": 3,
                "dataset_position": 3,
                "situation_id": 10,
                "prompt_raw": "p",
                "accepted_response": "a2",
                "rejected_response": "r2",
            },
            {
                "pair_id": 4,
                "dataset_position": 4,
                "situation_id": 11,
                "prompt_raw": "p2",
                "accepted_response": "a3",
                "rejected_response": "r3",
            },
        ]
        selected, stats = select_pairs(pairs, start_position=1, end_position=None, num_pairs=3)
        self.assertEqual([pair["pair_id"] for pair in selected], [1, 3, 4])
        self.assertEqual(stats["raw_pair_rows_in_slice"], 4)
        self.assertEqual(stats["exact_duplicate_rows_skipped"], 1)

    def test_ensure_output_path_is_safe_blocks_overwrite_without_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "existing.json"
            output_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_output_path_is_safe(str(output_path), resume=False)
            ensure_output_path_is_safe(str(output_path), resume=True)


if __name__ == "__main__":
    unittest.main()
