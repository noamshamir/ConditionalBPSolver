"""PyTorch Dataset for training the Conditional Clue Answerer."""
from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ClueAnswerDataset(Dataset):
    """Reads a ``.jsonl`` file produced by ``build_dataset.py``.

    Each item returns tokenizer-ready strings:
        input_text:  ``"Q: <clue> P: _A__E"``
        target_text: ``"apple"``

    The actual tokenization is handled by the collator or training loop to
    allow easy switching between ByT5 / other tokenizers.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        max_answer_len: int = 25,
        online_augment: bool = False,
        augment_prob: float = 0.5,
    ):
        self.items: List[dict] = []
        self.max_answer_len = max_answer_len
        self.online_augment = online_augment
        self.augment_prob = augment_prob

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if len(item["answer"]) <= max_answer_len:
                    self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def _random_pattern(self, answer: str) -> str:
        frac = random.random()
        return "".join(
            ch if random.random() < frac else "_" for ch in answer
        )

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.items[idx]
        if self.online_augment and random.random() < self.augment_prob:
            # Generate a fresh pattern on the fly
            pattern = self._random_pattern(item["answer"])
            input_text = f"Q: {item['clue']} P: {pattern}"
        else:
            input_text = item["input"]
        return {
            "input_text": input_text,
            "target_text": item["target"],
        }


class ClueAnswerCollator:
    """Tokenize + pad a batch of (input_text, target_text) pairs for ByT5."""

    def __init__(self, tokenizer, max_input_len: int = 256, max_target_len: int = 32):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        inputs = [ex["input_text"] for ex in batch]
        targets = [ex["target_text"] for ex in batch]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        # Replace pad token ids with -100 so they are ignored in loss
        label_ids = labels["input_ids"]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        return model_inputs
