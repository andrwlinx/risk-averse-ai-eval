import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import ResidualSteeringHook


class ResidualSteeringHookTests(unittest.TestCase):
    def test_all_positions_mode_matches_legacy_behavior(self):
        hidden = torch.zeros((2, 3, 2), dtype=torch.float32)
        direction = torch.tensor([1.0, -2.0], dtype=torch.float32)

        hook = ResidualSteeringHook(direction=direction, alpha=0.5, apply_mode="all_positions")
        steered = hook._hook(None, None, hidden)

        expected = hidden + torch.tensor([0.5, -1.0], dtype=torch.float32)
        self.assertTrue(torch.equal(steered, expected))

    def test_last_prompt_and_current_prefill_targets_last_real_prompt_token(self):
        hidden = torch.zeros((2, 4, 3), dtype=torch.float32)
        direction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        hook = ResidualSteeringHook(
            direction=direction,
            alpha=1.0,
            apply_mode="last_prompt_and_current",
            prompt_last_indices=[3, 1],
        )
        steered = hook._hook(None, None, hidden)

        expected = torch.zeros_like(hidden)
        expected[0, 3, :] = direction
        expected[1, 1, :] = direction
        self.assertTrue(torch.equal(steered, expected))

    def test_last_prompt_and_current_decode_targets_current_token(self):
        direction = torch.tensor([1.0, 2.0], dtype=torch.float32)
        hook = ResidualSteeringHook(
            direction=direction,
            alpha=1.5,
            apply_mode="last_prompt_and_current",
            prompt_last_indices=[2, 0],
        )

        prefill_hidden = torch.zeros((2, 3, 2), dtype=torch.float32)
        hook._hook(None, None, prefill_hidden)

        decode_hidden = torch.zeros((2, 1, 2), dtype=torch.float32)
        steered = hook._hook(None, None, decode_hidden)

        expected = torch.zeros_like(decode_hidden)
        expected[:, 0, :] = direction * 1.5
        self.assertTrue(torch.equal(steered, expected))

    def test_tuple_outputs_are_preserved(self):
        hidden = torch.zeros((1, 2, 2), dtype=torch.float32)
        aux = torch.tensor([42.0], dtype=torch.float32)
        hook = ResidualSteeringHook(
            direction=torch.tensor([1.0, 1.0], dtype=torch.float32),
            alpha=2.0,
            apply_mode="last_prompt_and_current",
            prompt_last_indices=[0],
        )

        steered = hook._hook(None, None, (hidden, aux))

        self.assertIsInstance(steered, tuple)
        self.assertEqual(len(steered), 2)
        self.assertTrue(torch.equal(steered[1], aux))
        self.assertTrue(torch.equal(steered[0][0, 0, :], torch.tensor([2.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
