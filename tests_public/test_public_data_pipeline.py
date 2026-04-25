from __future__ import annotations

import pytest
import torch

import submission


def test_build_lm_sequence_masks_prompt_tokens() -> None:
    input_ids, labels = submission.build_lm_sequence(
        prompt_ids=[10, 11, 12],
        response_ids=[20, 21],
        eos_token_id=1,
        max_length=10,
    )
    assert input_ids == [10, 11, 12, 20, 21, 1]
    assert labels == [submission.IGNORE_INDEX, submission.IGNORE_INDEX, submission.IGNORE_INDEX, 20, 21, 1]


def test_preference_collate_fn_uses_lengths_for_attention_mask() -> None:
    examples = [
        {
            "chosen_input_ids": [5, 6, 1],
            "chosen_labels": [submission.IGNORE_INDEX, 6, 1],
            "rejected_input_ids": [5, 7, 1],
            "rejected_labels": [submission.IGNORE_INDEX, 7, 1],
        },
        {
            "chosen_input_ids": [9, 1],
            "chosen_labels": [9, 1],
            "rejected_input_ids": [8, 1],
            "rejected_labels": [8, 1],
        },
    ]

    batch = submission.preference_collate_fn(
        examples,
        pad_token_id=1,  # Deliberately equal to EOS to catch the common GPT-2 mistake.
    )

    assert batch["chosen_input_ids"].shape == (2, 3)
    assert batch["chosen_labels"].shape == (2, 3)
    assert batch["chosen_attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 0],
    ]
    assert batch["rejected_attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 0],
    ]
