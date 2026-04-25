"""Student starter code for Homework 4: simplified DPO trainer.

Fill in the TODOs. The public tests import these functions directly.
Do not change function signatures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def format_prompt(prompt: str) -> str:
    """Format a raw prompt for a plain causal LM."""
    prompt = prompt.strip()
    return f"User: {prompt}\nAssistant: "


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Small compatibility wrapper for Hugging Face tokenizers and test tokenizers."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if "input_ids" not in encoded:
        raise KeyError("Tokenizer output must contain 'input_ids'.")
    return list(encoded["input_ids"])


def build_lm_sequence(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    eos_token_id: int,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[List[int], List[int]]:
    """
    Build a single prompt+response causal-LM example.

    The returned labels must mask out prompt tokens with `ignore_index`.
    The response should always end with EOS.
    If the combined example is too long, preserve as much of the response as possible
    and truncate the prompt from the left.

    Args:
        prompt_ids: token IDs for the prompt.
        response_ids: token IDs for the response (without EOS).
        eos_token_id: end-of-sequence token ID.
        max_length: maximum total length after appending EOS.
        ignore_index: label value used to mask prompt tokens.

    Returns:
        input_ids: concatenated prompt + response + EOS token IDs.
        labels: same length as input_ids, but prompt positions are ignore_index.
        
    """
    # Step 1: figure out how many prompt tokens we can keep
    # (remember to reserve 1 slot for EOS)
    prompt_budget = max_length - len(response_ids)-1

    # Step 2: trim the prompt from the LEFT if needed
    trimmed_prompt = prompt_ids[-prompt_budget:]

    # Step 3: build input_ids = trimmed_prompt + response_ids + [eos]
    input_ids = trimmed_prompt + response_ids + [eos_token_id]

    # Step 4: build labels = [ignore_index, ...] for prompt + response_ids + [eos]
    labels = [-100] * len(trimmed_prompt) + response_ids + [eos_token_id]
    return input_ids, labels


def tokenize_preference_example(
    example: Mapping[str, str],
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_response_length: int,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, List[int]]:
    """
    
    Tokenize one preference example.

    The input example has keys: "prompt", "chosen", "rejected".

    Expected output keys:
        - chosen_input_ids
        - chosen_labels
        - rejected_input_ids
        - rejected_labels

    Notes:
        * Prompt tokens should be truncated from the left.
        * Response tokens should be truncated from the right.
        * Reserve room for EOS in each response sequence.

    """
    # Step 1: format and tokenize the prompt, truncate from the LEFT
    prompt_ids = _tokenize_text(tokenizer, format_prompt(example["prompt"]))[-max_prompt_length:]

    # Step 2: tokenize chosen response, truncate from the RIGHT
    # reserve 1 slot for EOS
    chosen_ids = _tokenize_text(tokenizer, example["chosen"])[:(max_response_length - 1)]

    # Step 3: tokenize rejected response, truncate from the RIGHT
    # reserve 1 slot for EOS
    rejected_ids = _tokenize_text(tokenizer, example["rejected"])[:(max_response_length - 1)]

    # Step 4: call build_lm_sequence for chosen
    chosen_input_ids, chosen_labels = build_lm_sequence(
        prompt_ids=prompt_ids,
        response_ids=chosen_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    # Step 5: call build_lm_sequence for rejected
    rejected_input_ids, rejected_labels = build_lm_sequence(
        prompt_ids=prompt_ids,
        response_ids=rejected_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
    }


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Right-pad a list of integer sequences to the same length.

    Args:
        sequences: list of variable-length token ID sequences.
        pad_value: value used for padding positions.

    Returns:
        padded: LongTensor of shape [batch, max_len].
        attention_mask: LongTensor of shape [batch, max_len], with 1 for real
            tokens and 0 for padding. Derived from original sequence lengths,
            so it is correct even when pad_value == eos_token_id.
     Important:
        Build attention masks from the original sequence lengths, not from token equality.
        This matters when pad_token_id == eos_token_id (as is common for GPT-2).
        
    """

    # Step 1: find the longest sequence
    max_len = max([len(s) for s in sequences])

    # Step 2: pad each sequence on the right and record original lengths
    lengths = []
    padded = []
    for seq in sequences:
        lengths.append(len(seq))
        padding = [pad_value] * (max_len - len(seq))
        padded.append(list(seq) + padding)

    # Step 3: build attention_mask from lengths (NOT from token equality)
    attention_mask = []
    for length in lengths:
        attention_mask.append([1] * length + [0] * (max_len-length))

    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def preference_collate_fn(
    examples: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Collate tokenized preference examples into a padded batch.

    The batch must contain:
        chosen_input_ids, chosen_labels, chosen_attention_mask,
        rejected_input_ids, rejected_labels, rejected_attention_mask

    Use _pad_sequences for padding. Call it separately for input_ids
    (pad with pad_token_id) and labels (pad with ignore_index) so that
    each field gets the correct pad value.
    """
    # Step 1: collect chosen and rejected sequences from all examples
    chosen_input_ids  = [ex["chosen_input_ids"]  for ex in examples]
    chosen_labels     = [ex["chosen_labels"]     for ex in examples]
    rejected_input_ids = [ex["rejected_input_ids"] for ex in examples]
    rejected_labels   = [ex["rejected_labels"]   for ex in examples]

    # Step 2: pad input_ids with pad_token_id, labels with ignore_index
    chosen_ids_padded,  chosen_mask    = _pad_sequences(chosen_input_ids,  pad_value=pad_token_id)
    chosen_lbl_padded,  _              = _pad_sequences(chosen_labels,      pad_value=ignore_index)
    rejected_ids_padded, rejected_mask = _pad_sequences(rejected_input_ids, pad_value=pad_token_id)
    rejected_lbl_padded, _            = _pad_sequences(rejected_labels,     pad_value=ignore_index)

    return {
        "chosen_input_ids":       chosen_ids_padded,
        "chosen_labels":          chosen_lbl_padded,
        "chosen_attention_mask":  chosen_mask,
        "rejected_input_ids":     rejected_ids_padded,
        "rejected_labels":        rejected_lbl_padded,
        "rejected_attention_mask": rejected_mask,
    }


def sequence_logps_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Compute sequence log-probabilities from causal-LM logits and labels.

    This function must:
        1. shift logits and labels for next-token prediction;
        2. ignore positions where labels == ignore_index;
        3. return one scalar log-probability per sequence.

    Args:
        logits: Float tensor of shape [batch, seq_len, vocab_size].
        labels: Long tensor of shape [batch, seq_len].
        ignore_index: masked label value.
        average_log_prob: if True, average over non-masked positions instead of summing.

    Returns:
        Tensor of shape [batch].
    """
    # Step 1: causal shift — align logits with the token they predict
    shifted_logits = logits[:, :-1, :]   # drop last position → [batch, seq_len-1, vocab]
    shifted_labels = labels[:, 1:]       # drop first position → [batch, seq_len-1]

    # Step 2: convert logits to log-probabilities
    log_probs = F.log_softmax(shifted_logits, dim=-1)   # shape [batch, seq_len-1, vocab]

    # Step 3: pick the log-prob of the actual token at each position
    # gather picks one value per position along the vocab dimension
    # clamp -> everything below 0 becomes 0
    token_logps = log_probs.gather(dim=-1, index=shifted_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    # Step 4: build a mask for non-ignored positions
    mask = (shifted_labels != ignore_index).float()   # 1.0 where real, 0.0 where ignored

    # Step 5: zero out ignored positions
    token_logps = token_logps * mask

    # Step 6: sum or average over response positions → shape [batch]
    if average_log_prob:
        return token_logps.sum(dim=-1) / mask.sum(dim=-1)
    else:
        return token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute per-example DPO losses.

    Returns:
        losses: Tensor of shape [batch].
        stats: dict containing detached tensors with at least:
            - chosen_rewards
            - rejected_rewards
            - margins
            - accuracy
    """
    # Step 1: compute the margin (raw log-prob difference, no beta)
    margins = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)

    # Step 2: compute implicit rewards (β * log ratio policy/reference)
    chosen_rewards   = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # Step 3: compute the DPO loss per example
    losses = -F.logsigmoid(beta * margins)

    # Step 4: build stats dict (detached — we don't need gradients for logging)
    stats = {
        "chosen_rewards":   chosen_rewards.detach(),
        "rejected_rewards": rejected_rewards.detach(),
        "margins":          margins.detach(),
        "accuracy":         (margins > 0).float().detach(),
    }

    return losses, stats


def compute_dpo_batch(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    beta: float,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run policy and reference models on one batch and return mean DPO loss + metrics.

    The returned scalar loss should require gradients with respect to the policy model.
    The reference model should be evaluated without tracking gradients.
    """
    # Step 1: policy forward passes (gradients ON)
    policy_chosen_logits   = policy_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    ).logits
    
    policy_rejected_logits = policy_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    ).logits

    # Step 2: reference forward passes (gradients OFF)
    with torch.no_grad():
        ref_chosen_logits   = reference_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        ).logits
        ref_rejected_logits = reference_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        ).logits

    # Step 3: compute sequence log-probs from logits
    policy_chosen_logps   = sequence_logps_from_logits(policy_chosen_logits,   batch["chosen_labels"],   ignore_index=ignore_index)
    policy_rejected_logps = sequence_logps_from_logits(policy_rejected_logits, batch["rejected_labels"], ignore_index=ignore_index)
    ref_chosen_logps      = sequence_logps_from_logits(ref_chosen_logits,   batch["chosen_labels"],   ignore_index=ignore_index)
    ref_rejected_logps    = sequence_logps_from_logits(ref_rejected_logits, batch["rejected_labels"], ignore_index=ignore_index)

    # Step 4: compute DPO loss and stats
    losses, stats = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)

    # Step 5: aggregate metrics
    metrics = {
        "loss":                losses.detach(),
        "preference_accuracy": stats["accuracy"],
        "mean_margin":         stats["margins"],
    }

    return losses.mean(), metrics


def train_step(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    grad_clip_norm: float | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """Perform one optimization step on the policy model.

    Returns:
        A dictionary of Python floats, for example:
            {
                "loss": ...,
                "preference_accuracy": ...,
                "mean_margin": ...
            }
    """
    # Step 1: zero old gradients
    optimizer.zero_grad()

    # Step 2: forward pass — compute loss and metrics
    loss, metrics = compute_dpo_batch(policy_model, reference_model, batch, beta=beta, ignore_index=ignore_index)

    # Step 3: backward pass
    loss.backward()

    # Step 4: optional gradient clipping
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_norm)

    # Step 5: optimizer step
    optimizer.step()

    return {
        "loss":                metrics["loss"].mean().item(),
        "preference_accuracy": metrics["preference_accuracy"].mean().item(),
        "mean_margin":         metrics["mean_margin"].mean().item(),
    }


@torch.no_grad()
def evaluate_preference_accuracy(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    dataloader: Sequence[Mapping[str, torch.Tensor]],
    *,
    beta: float,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """Evaluate a model on a preference dataloader.

    Aggregate metrics over all examples and return Python floats.
    Suggested keys:
        - loss
        - preference_accuracy
        - mean_margin
    """
    # Step 1: initialize accumulators
    total_loss, total_accuracy, total_margin, n_batches = 0.0, 0.0, 0.0, 0

    # Step 2: loop over batches, accumulate metrics
    for batch in dataloader:
        _, metrics = compute_dpo_batch(policy_model, reference_model, batch, beta=beta, ignore_index=ignore_index)
        total_loss     += metrics["loss"].mean().item()
        total_accuracy += metrics["preference_accuracy"].mean().item()
        total_margin   += metrics["mean_margin"].mean().item()
        n_batches += 1

    # Step 3: return averages
    return {
        "loss":                total_loss / n_batches,
        "preference_accuracy": total_accuracy / n_batches,
        "mean_margin":         total_margin / n_batches,
    }
