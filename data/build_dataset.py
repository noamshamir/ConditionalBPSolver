"""Build training data for the Conditional Clue Answerer.

For each (clue, answer) pair we generate *multiple* training examples with
different levels of partial-letter information, simulating what the model
would see during belief-propagation iterations:

  Input:  "Q: <clue> P: _A__E"   →   Target: "apple"

The ``P:`` (pattern) field encodes which letters are "revealed" by crossing
words.  During training we randomly reveal 0-100 % of letters.

Usage
-----
    python -m data.build_dataset \
        --puz-dir ../TinyRecursiveModels/nyt_puz_flat \
        --output-dir data/clue_answer_pairs \
        --val-fraction 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
from pathlib import Path
from typing import List, Tuple
from tqdm.notebook import tqdm
from solver.utils import puz_to_json, preprocess_clue


# ======================================================================
# Extract (clue, answer, length) triples from .puz files
# ======================================================================
def extract_clue_answer_pairs(puz_path: Path) -> List[Tuple[str, str]]:
    """Return ``[(preprocessed_clue, UPPERCASED_ANSWER), …]``."""
    try:
        data = puz_to_json(puz_path)
    except Exception:
        return []
    pairs: List[Tuple[str, str]] = []
    for direction in ("across", "down"):
        clues = data["clues"][direction]
        for num, (clue_text, answer_text) in clues.items():
            clue = preprocess_clue(clue_text)
            answer = "".join(
                c.upper() for c in answer_text if c.upper() in string.ascii_uppercase
            )
            if answer:
                pairs.append((clue, answer))
    return pairs


# ======================================================================
# Generate pattern-augmented training examples
# ======================================================================
def make_pattern(answer: str, reveal_frac: float) -> str:
    """Create a partial pattern like ``_A__E`` from an answer.

    ``reveal_frac`` in [0, 1] controls how many letters are revealed.
    """
    pattern = []
    for ch in answer:
        if random.random() < reveal_frac:
            pattern.append(ch)
        else:
            pattern.append("_")
    return "".join(pattern)


def format_input(clue: str, pattern: str) -> str:
    """Format the model input: ``Q: <clue> P: <pattern>``."""
    return f"Q: {clue} P: {pattern}"


def generate_examples(
    clue: str,
    answer: str,
    num_augmentations: int = 5,
) -> List[dict]:
    """Generate augmented training examples for one (clue, answer) pair."""
    examples = []
    # Always include a "no letters revealed" example
    blank_pattern = "_" * len(answer)
    examples.append(
        {
            "input": format_input(clue, blank_pattern),
            "target": answer.lower(),
            "clue": clue,
            "answer": answer,
            "reveal_frac": 0.0,
        }
    )
    # Generate random partial reveals
    for _ in range(num_augmentations - 1):
        frac = random.random()  # uniform [0, 1)
        pattern = make_pattern(answer, frac)
        examples.append(
            {
                "input": format_input(clue, pattern),
                "target": answer.lower(),
                "clue": clue,
                "answer": answer,
                "reveal_frac": frac,
            }
        )
    # Always include a "fully revealed" example
    examples.append(
        {
            "input": format_input(clue, answer),
            "target": answer.lower(),
            "clue": clue,
            "answer": answer,
            "reveal_frac": 1.0,
        }
    )
    return examples


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Build clue-answer training data")
    parser.add_argument(
        "--puz-dir",
        type=str,
        required=True,
        help="Directory containing .puz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write train.jsonl / val.jsonl",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of *puzzles* to hold out for validation",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=6,
        help="Number of pattern-augmented examples per (clue, answer) pair",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    random.seed(args.seed)

    puz_dir = Path(args.puz_dir)
    puz_files = sorted(puz_dir.glob("*.puz"))
    print(f"Found {len(puz_files)} .puz files in {puz_dir}")

    # Split by puzzle (not by example) to avoid leaking clues across splits
    random.shuffle(puz_files)
    n_val = max(1, int(len(puz_files) * args.val_fraction))
    val_files = set(puz_files[:n_val])
    train_files = puz_files[n_val:]

    os.makedirs(args.output_dir, exist_ok=True)

    stats = {"train_pairs": 0, "val_pairs": 0, "train_examples": 0, "val_examples": 0}

    for split_name, files in [("train", train_files), ("val", list(val_files))]:
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for puz_path in tqdm(files, desc=split_name):
                pairs = extract_clue_answer_pairs(puz_path)
                stats[f"{split_name}_pairs"] += len(pairs)
                for clue, answer in pairs:
                    examples = generate_examples(clue, answer, args.augmentations)
                    stats[f"{split_name}_examples"] += len(examples)
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
        print(f"Wrote {out_path}")

    # Also save unique (clue, answer) pairs as a simple TSV wordlist
    # for use as vocabulary during solving
    all_pairs: List[Tuple[str, str]] = []
    for puz_path in puz_files:
        all_pairs.extend(extract_clue_answer_pairs(puz_path))

    wordlist_path = os.path.join(args.output_dir, "wordlist.tsv")
    seen: set = set()
    with open(wordlist_path, "w") as f:
        f.write("id\tclue\tanswer\n")
        idx = 0
        for clue, answer in all_pairs:
            key = (clue, answer)
            if key not in seen:
                seen.add(key)
                f.write(f"{idx}\t{clue}\t{answer}\n")
                idx += 1
    print(f"Wrote {wordlist_path} ({idx} unique pairs)")

    # Save all unique answers as a vocabulary file
    all_answers = sorted({a for _, a in all_pairs})
    vocab_path = os.path.join(args.output_dir, "answer_vocab.txt")
    with open(vocab_path, "w") as f:
        for ans in all_answers:
            f.write(ans + "\n")
    print(f"Wrote {vocab_path} ({len(all_answers)} unique answers)")

    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
