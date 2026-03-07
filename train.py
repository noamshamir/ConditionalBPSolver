"""Training script for the Conditional Clue Answerer.

Fine-tunes ByT5 on (clue + pattern) → answer pairs with mixed
reveal fractions.  Uses HuggingFace Trainer for simplicity and
comparability with Berkeley's DPR training setup.

Usage
-----
    python train.py --config config/train_clue_model.yaml

    # or override from CLI:
    python train.py --config config/train_clue_model.yaml \
        --train-data data/clue_answer_pairs/train.jsonl \
        --val-data data/clue_answer_pairs/val.jsonl \
        --model google/byt5-base \
        --epochs 10 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# make sure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.crossword_data import ClueAnswerDataset, ClueAnswerCollator


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_compute_metrics(tokenizer):
    """Return a compute_metrics function that decodes predictions and
    computes exact-match accuracy and character-level accuracy."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds are logits — take argmax
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1)

        # Replace -100 in labels (padding) so the tokenizer can decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Exact-match accuracy
        exact_matches = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
        exact_match_acc = exact_matches / max(len(decoded_preds), 1)

        # Character-level accuracy (averaged per example)
        char_accs = []
        for p, l in zip(decoded_preds, decoded_labels):
            if len(l) == 0:
                char_accs.append(1.0 if len(p) == 0 else 0.0)
            else:
                matches = sum(pc == lc for pc, lc in zip(p, l))
                char_accs.append(matches / len(l))
        char_acc = np.mean(char_accs) if char_accs else 0.0

        return {
            "exact_match": exact_match_acc,
            "char_accuracy": char_acc,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Conditional Clue Answerer")
    parser.add_argument("--config", type=str, default="config/train_clue_model.yaml")
    # Allow CLI overrides
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from")
    args = parser.parse_args()

    # Load config file
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)

    # Merge CLI overrides
    train_data = args.train_data or cfg.get("train_data", "data/clue_answer_pairs/train.jsonl")
    val_data = args.val_data or cfg.get("val_data", "data/clue_answer_pairs/val.jsonl")
    model_name = args.model or cfg.get("model", "google/byt5-base")
    output_dir = args.output_dir or cfg.get("output_dir", "checkpoints/clue_answerer")
    epochs = args.epochs or cfg.get("epochs", 10)
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    lr = args.lr or cfg.get("lr", 3e-4)
    grad_accum = args.grad_accum or cfg.get("gradient_accumulation_steps", 4)
    max_input_len = cfg.get("max_input_len", 256)
    max_target_len = cfg.get("max_target_len", 32)
    online_augment = cfg.get("online_augment", True)
    warmup_steps = cfg.get("warmup_steps", 500)
    weight_decay = cfg.get("weight_decay", 0.01)
    save_steps = cfg.get("save_steps", 1000)
    eval_steps = cfg.get("eval_steps", 500)
    logging_steps = cfg.get("logging_steps", 100)
    max_answer_len = cfg.get("max_answer_len", 25)

    use_fp16 = args.fp16 if args.fp16 is not None else cfg.get("fp16", False)
    use_bf16 = args.bf16 if args.bf16 is not None else cfg.get("bf16", True)
    wandb_project = args.wandb_project or cfg.get("wandb_project", "conditional-bp-solver")

    print(f"Model:       {model_name}")
    print(f"Train data:  {train_data}")
    print(f"Val data:    {val_data}")
    print(f"Output:      {output_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size} x {grad_accum} grad accum")
    print(f"LR:          {lr}")
    print()

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Datasets
    train_dataset = ClueAnswerDataset(
        train_data,
        max_answer_len=max_answer_len,
        online_augment=online_augment,
        augment_prob=0.5,
    )
    val_dataset = ClueAnswerDataset(
        val_data,
        max_answer_len=max_answer_len,
        online_augment=False,
    )
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")

    collator = ClueAnswerCollator(
        tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if wandb_project else "none",
        run_name=f"clue-answerer-{model_name.split('/')[-1]}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final checkpoint
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model to {final_dir}")


if __name__ == "__main__":
    main()
