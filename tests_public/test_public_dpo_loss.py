from __future__ import annotations

import torch
import torch.nn.functional as F

import submission


def test_dpo_loss_matches_manual_formula() -> None:
    policy_chosen = torch.tensor([3.0, 1.0])
    policy_rejected = torch.tensor([1.0, 2.0])
    ref_chosen = torch.tensor([2.0, 1.5])
    ref_rejected = torch.tensor([1.5, 1.0])
    beta = 0.2

    losses, stats = submission.dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )

    expected_margins = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    expected_losses = -F.logsigmoid(beta * expected_margins)

    assert torch.allclose(losses, expected_losses, atol=1e-6)
    assert torch.allclose(stats["margins"], expected_margins, atol=1e-6)
    assert torch.equal(stats["accuracy"], (expected_margins > 0).float())


def test_dpo_rewards_are_relative_to_reference() -> None:
    policy_chosen = torch.tensor([4.0])
    policy_rejected = torch.tensor([1.0])
    ref_chosen = torch.tensor([3.0])
    ref_rejected = torch.tensor([2.0])
    beta = 0.5

    _, stats = submission.dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )

    assert torch.allclose(stats["chosen_rewards"], torch.tensor([0.5]))
    assert torch.allclose(stats["rejected_rewards"], torch.tensor([-0.5]))
