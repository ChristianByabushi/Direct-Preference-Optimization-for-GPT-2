"""Runnable training script for the Homework 4 DPO assignment.

This script uses GPT-2 (via Hugging Face) for the real student experiment, but it
delegates the actual DPO logic to `submission.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
import submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simplified DPO model on preference data.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train JSONL.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation JSONL.")
    parser.add_argument("--output_dir", type=str, default="outputs/hw4_dpo")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=96)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="If set, log metrics and per-epoch generation tables to Weights & Biases (pip install wandb).",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--gen_examples_per_epoch",
        type=int,
        default=3,
        help="Number of validation prompts (shuffled with --seed) to decode after each epoch.",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens for greedy generation on sample prompts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sample_prompts_for_generation(
    records: Sequence[Mapping[str, str]],
    k: int,
    seed: int,
) -> List[str]:
    if k <= 0 or not records:
        return []
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    k = min(k, len(indices))
    return [records[i]["prompt"] for i in indices[:k]]


@torch.no_grad()
def generate_completion(
    model: nn.Module,
    tokenizer,
    raw_prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Greedy decoding from `submission.format_prompt`-style prefix (response only returned)."""
    model.eval()
    text = submission.format_prompt(raw_prompt)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Import here so that unit tests can run without transformers installed.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_wandb = args.wandb_project is not None
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
        except ImportError as e:
            raise SystemExit(
                "wandb is not installed. Run: pip install wandb\n"
                "Or omit --wandb_project to train without logging."
            ) from e
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = data.read_jsonl(args.train_path)
    val_records = data.read_jsonl(args.val_path)

    if args.max_train_examples is not None:
        train_records = train_records[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_records = val_records[: args.max_val_examples]

    gen_prompts = sample_prompts_for_generation(val_records, args.gen_examples_per_epoch, args.seed)

    tokenize = partial(
        submission.tokenize_preference_example,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_length=args.max_length,
    )
    train_tokenized = [tokenize(record) for record in train_records]
    val_tokenized = [tokenize(record) for record in val_records]

    collate_fn = partial(submission.preference_collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    reference_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    reference_model.load_state_dict(policy_model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    history: List[Dict[str, float]] = []

    try:
        for epoch in range(args.epochs):
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            running = {"loss": 0.0, "preference_accuracy": 0.0, "mean_margin": 0.0}
            seen_steps = 0

            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(progress, start=1):
                batch = move_batch_to_device(batch, device)

                policy_model.train()
                reference_model.eval()
                loss, metrics = submission.compute_dpo_batch(
                    policy_model,
                    reference_model,
                    batch,
                    beta=args.beta,
                )
                (loss / args.grad_accum_steps).backward()

                if step % args.grad_accum_steps == 0:
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running["loss"] += float(metrics["loss"].item())
                running["preference_accuracy"] += float(metrics["preference_accuracy"].item())
                running["mean_margin"] += float(metrics["mean_margin"].item())
                seen_steps += 1

                progress.set_postfix(
                    loss=running["loss"] / seen_steps,
                    pref_acc=running["preference_accuracy"] / seen_steps,
                )

            val_metrics = submission.evaluate_preference_accuracy(
                policy_model,
                reference_model,
                [move_batch_to_device(batch, device) for batch in val_loader],
                beta=args.beta,
            )

            epoch_record = {
                "epoch": float(epoch + 1),
                "train_loss": running["loss"] / max(seen_steps, 1),
                "train_preference_accuracy": running["preference_accuracy"] / max(seen_steps, 1),
                "train_mean_margin": running["mean_margin"] / max(seen_steps, 1),
                "val_loss": val_metrics["loss"],
                "val_preference_accuracy": val_metrics["preference_accuracy"],
                "val_mean_margin": val_metrics["mean_margin"],
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, indent=2))

            if gen_prompts:
                print(
                    "\n--- Sample generations (reference = frozen init weights; policy = after this epoch) ---"
                )
                table_rows: List[List[str]] = []
                for raw_prompt in gen_prompts:
                    ref_text = generate_completion(
                        reference_model,
                        tokenizer,
                        raw_prompt,
                        device,
                        args.gen_max_new_tokens,
                    )
                    pol_text = generate_completion(
                        policy_model,
                        tokenizer,
                        raw_prompt,
                        device,
                        args.gen_max_new_tokens,
                    )
                    table_rows.append([raw_prompt, ref_text, pol_text])
                    preview = raw_prompt if len(raw_prompt) <= 100 else raw_prompt[:97] + "..."
                    print(f"Prompt: {preview}")
                    print(f"  reference: {ref_text}")
                    print(f"  policy:    {pol_text}")
                if use_wandb and wandb is not None:
                    sample_table = wandb.Table(
                        columns=["prompt", "reference", "policy"],
                        data=table_rows,
                    )
                    wandb.log(
                        {
                            "train/loss": epoch_record["train_loss"],
                            "train/preference_accuracy": epoch_record["train_preference_accuracy"],
                            "train/mean_margin": epoch_record["train_mean_margin"],
                            "val/loss": epoch_record["val_loss"],
                            "val/preference_accuracy": epoch_record["val_preference_accuracy"],
                            "val/mean_margin": epoch_record["val_mean_margin"],
                            "epoch/generations": sample_table,
                        },
                        step=epoch,
                    )
            elif use_wandb and wandb is not None:
                wandb.log(
                    {
                        "train/loss": epoch_record["train_loss"],
                        "train/preference_accuracy": epoch_record["train_preference_accuracy"],
                        "train/mean_margin": epoch_record["train_mean_margin"],
                        "val/loss": epoch_record["val_loss"],
                        "val/preference_accuracy": epoch_record["val_preference_accuracy"],
                        "val/mean_margin": epoch_record["val_mean_margin"],
                    },
                    step=epoch,
                )

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(history, indent=2))
        policy_model.save_pretrained(output_dir / "policy_model")
        tokenizer.save_pretrained(output_dir / "policy_model")
        print(f"Saved model and metrics to {output_dir}")
    finally:
        if use_wandb and wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
