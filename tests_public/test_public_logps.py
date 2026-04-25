from __future__ import annotations

import torch
import torch.nn.functional as F

import submission


def test_sequence_logps_from_logits_matches_manual_computation() -> None:
    # Shape: [batch=1, seq_len=4, vocab=3]
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.1, 0.7, 0.2],
                    [0.6, 0.2, 0.2],
                    [0.25, 0.25, 0.5],
                    [0.3, 0.3, 0.4],  # final position is ignored after shift
                ]
            ],
            dtype=torch.float32,
        )
    )
    labels = torch.tensor(
        [[submission.IGNORE_INDEX, 1, 0, 2]],
        dtype=torch.long,
    )

    logps = submission.sequence_logps_from_logits(logits, labels)
    expected = torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(0.6)) + torch.log(torch.tensor(0.5))
    assert torch.allclose(logps, expected.unsqueeze(0), atol=1e-6)


def test_sequence_logps_average_mode() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.2, 0.8],
                    [0.9, 0.1],
                    [0.4, 0.6],
                ]
            ],
            dtype=torch.float32,
        )
    )
    labels = torch.tensor([[submission.IGNORE_INDEX, 1, 0]], dtype=torch.long)

    mean_logps = submission.sequence_logps_from_logits(logits, labels, average_log_prob=True)
    expected = (torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.9))) / 2.0
    assert torch.allclose(mean_logps, expected.unsqueeze(0), atol=1e-6)
