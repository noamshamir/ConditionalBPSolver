"""Conditional Clue Answerer — the core neural model.

Architecture:  Fine-tuned **ByT5** (byte-level T5) that scores
``P(answer | clue, letter_pattern)``.

During solving, the model is queried with patterns like::

    Input:  "Q: State capital on the Willamette P: _A_E_"
    Output: "salem"

The key innovation vs. Berkeley's DPR retrieval approach:
- Berkeley encodes clues *once* and retrieves from a fixed vocabulary
- We **re-generate / re-score** answers at each BP iteration with *updated*
  letter beliefs from crossing words

We support two usage modes:

1. **Generative** — beam-search decoding to produce candidate answers
2. **Scoring** — given a list of candidate answers, compute log P(answer|clue,pattern)
   (same interface as Berkeley's T5 reranker)

Both modes use the same fine-tuned ByT5 checkpoint.
"""
from __future__ import annotations

import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration


# ======================================================================
# Model setup
# ======================================================================
_MODEL_CACHE: Dict[str, Tuple] = {}


def setup_clue_answerer(
    checkpoint: str = "google/byt5-base",
    device: str | torch.device = "cuda:0",
    cache_key: str = "default",
) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Load (or return cached) ByT5 model + tokenizer."""
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval().to(device)
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def setup_reranker(
    checkpoint: str = "google/byt5-small",
    device: str | torch.device = "cuda:0",
    cache_key: str = "reranker",
) -> Tuple[T5ForConditionalGeneration, AutoTokenizer]:
    """Load the T5 reranker (same role as Berkeley's byt5 reranker)."""
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval().to(device)
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


# ======================================================================
# Formatting helpers
# ======================================================================
def format_clue_pattern(clue: str, pattern: str) -> str:
    """Build the model input string."""
    return f"Q: {clue} P: {pattern}"


def beliefs_to_pattern(
    letter_beliefs: np.ndarray,
    threshold: float = 0.6,
) -> str:
    """Convert a ``(length, 26)`` belief matrix to a pattern string.

    Letters with max probability ≥ ``threshold`` are revealed; others
    become ``_``.

    Parameters
    ----------
    letter_beliefs : np.ndarray, shape (length, 26)
        Per-position probability over A-Z.
    threshold : float
        Minimum confidence to reveal a letter.
    """
    pattern = []
    for pos_probs in letter_beliefs:
        best_idx = int(pos_probs.argmax())
        if pos_probs[best_idx] >= threshold:
            pattern.append(string.ascii_uppercase[best_idx])
        else:
            pattern.append("_")
    return "".join(pattern)


# ======================================================================
# Generative inference  (beam search → candidate answers)
# ======================================================================
@torch.inference_mode()
def generate_candidates(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    clues: List[str],
    patterns: List[str],
    num_beams: int = 50,
    num_return: int = 50,
    max_answer_len: int = 30,
    batch_size: int = 16,
) -> List[List[Tuple[str, float]]]:
    """Generate candidate answers for a batch of (clue, pattern) pairs.

    Returns
    -------
    list of list of (answer_str, log_prob)
        One list per input clue.
    """
    all_results: List[List[Tuple[str, float]]] = []

    for batch_start in range(0, len(clues), batch_size):
        batch_clues = clues[batch_start : batch_start + batch_size]
        batch_patterns = patterns[batch_start : batch_start + batch_size]
        inputs_text = [
            format_clue_pattern(c, p) for c, p in zip(batch_clues, batch_patterns)
        ]
        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_answer_len,
            num_beams=num_beams,
            num_return_sequences=num_return,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=0.0,  # don't penalize length — answers have known length
        )

        # outputs.sequences: (batch_size * num_return, seq_len)
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        # sequences_scores: (batch_size * num_return,)
        scores = outputs.sequences_scores.cpu().numpy()

        for i in range(len(batch_clues)):
            candidates = []
            for j in range(num_return):
                idx = i * num_return + j
                answer = decoded[idx].strip().upper()
                answer = "".join(c for c in answer if c in string.ascii_uppercase)
                candidates.append((answer, float(scores[idx])))
            all_results.append(candidates)

    return all_results


# ======================================================================
# Scoring mode  (log P(answer | clue, pattern))
# ======================================================================
_SCORE_CACHE: Dict[str, float] = {}


@torch.inference_mode()
def score_answers(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    clues: List[str],
    patterns: List[str],
    answers: List[str],
    use_cache: bool = True,
) -> List[float]:
    """Compute ``log P(answer | clue, pattern)`` for each triplet.

    This is analogous to Berkeley's ``t5_reranker_score_with_clue``.
    """
    results = []
    uncached_indices = []
    uncached_inputs = []
    uncached_targets = []

    for i, (clue, pattern, answer) in enumerate(zip(clues, patterns, answers)):
        cache_key = f"{clue}||{pattern}||{answer}"
        if use_cache and cache_key in _SCORE_CACHE:
            results.append(_SCORE_CACHE[cache_key])
        else:
            results.append(None)
            uncached_indices.append(i)
            uncached_inputs.append(format_clue_pattern(clue, pattern))
            uncached_targets.append(answer.lower())

    if uncached_indices:
        inputs = tokenizer(
            uncached_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)
        labels = tokenizer(
            uncached_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(model.device)

        label_ids = labels["input_ids"]
        # Compute per-example loss
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=label_ids,
        )
        # out.loss is the mean; we need per-example
        # Re-compute manually
        logits = out.logits  # (B, T, V)
        B, T, V = logits.shape
        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

        for local_idx in range(len(uncached_indices)):
            global_idx = uncached_indices[local_idx]
            clue = clues[global_idx]
            pattern = patterns[global_idx]
            answer = answers[global_idx]

            lbl = label_ids[local_idx]  # (T,)
            lp = log_probs_all[local_idx]  # (T, V)
            # Sum log probs for non-pad tokens
            mask = lbl != tokenizer.pad_token_id
            token_log_probs = lp[torch.arange(T, device=lp.device), lbl]
            score = float((token_log_probs * mask).sum().cpu())

            cache_key = f"{clue}||{pattern}||{answer}"
            _SCORE_CACHE[cache_key] = score
            results[global_idx] = score

    return results


# ======================================================================
# Batch scoring for reranker  (matches Berkeley's interface exactly)
# ======================================================================
@torch.inference_mode()
def reranker_score(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    clues: List[str],
    answers: List[str],
) -> List[float]:
    """Score ``(clue, answer)`` pairs using the reranker (no pattern).

    Matches Berkeley's ``t5_reranker_score_with_clue`` interface.
    """
    from solver.utils import preprocess_clue

    results = []
    for clue, answer in zip(clues, answers):
        answer_lower = answer.lower() if answer.isupper() else answer
        clue = preprocess_clue(clue)
        # Strip trailing dots
        for suffix in [". .", " ..", "..", "."]:
            if clue.endswith(suffix):
                clue = clue[: -len(suffix)]
                break

        cache_key = f"RR:{clue}:{answer_lower}"
        if cache_key in _SCORE_CACHE:
            results.append(_SCORE_CACHE[cache_key])
            continue

        inputs = tokenizer(
            ["Q: " + clue],
            return_tensors="pt",
        )["input_ids"].to(model.device)
        labels = tokenizer(
            [answer_lower],
            return_tensors="pt",
        )["input_ids"].to(model.device)
        loss = model(inputs, labels=labels).loss
        answer_length = labels.shape[1]
        logprob = -loss.item() * answer_length
        _SCORE_CACHE[cache_key] = logprob
        results.append(logprob)

    return results
