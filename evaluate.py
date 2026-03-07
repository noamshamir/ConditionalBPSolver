"""Evaluate the Conditional BP Solver on a directory of puzzles.

Produces per-puzzle and aggregate accuracy statistics, matching
Berkeley's evaluation format for direct comparison.

Usage
-----
    python evaluate.py --puzzle-dir puzzles/ --config config/solve.yaml
    python evaluate.py --puzzle-dir puzzles/ --config config/solve.yaml --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver.crossword import Crossword
from solver.belief_propagation import ConditionalBPSolver
from solver.utils import puz_to_json, print_grid


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_puzzle(puzzle_path: Path, cfg: dict) -> dict:
    """Evaluate solver on a single puzzle."""
    try:
        if puzzle_path.suffix == ".puz":
            data = puz_to_json(puzzle_path)
        elif puzzle_path.suffix == ".json":
            with open(puzzle_path, "r") as f:
                data = json.load(f)
        else:
            return {"file": str(puzzle_path), "error": f"unsupported: {puzzle_path.suffix}"}
    except Exception as e:
        return {"file": str(puzzle_path), "error": str(e)}

    crossword = Crossword(data)
    print(f"\n{'='*60}")
    print(f"Puzzle: {puzzle_path.name} ({crossword})")

    t0 = time.time()
    solver = ConditionalBPSolver(
        crossword,
        model_checkpoint=cfg.get("model_checkpoint", "google/byt5-base"),
        reranker_checkpoint=cfg.get("reranker_checkpoint", "google/byt5-small"),
        answer_set_path=cfg.get("answer_set_path", "data/clue_answer_pairs/answer_vocab.txt"),
        max_candidates=cfg.get("max_candidates", 500),
        num_beams=cfg.get("num_beams", 50),
        device=cfg.get("device", "cuda:0"),
        belief_threshold=cfg.get("belief_threshold", 0.6),
    )

    grid = solver.solve(
        num_iters=cfg.get("num_iters", 10),
        requery_every=cfg.get("requery_every", 3),
        iterative_improvement_steps=cfg.get("iterative_improvement_steps", 5),
    )
    elapsed = time.time() - t0

    letter_acc, word_acc = solver.evaluate(grid)

    # Per-word breakdown
    word_results = []
    for var_id, var in crossword.variables.items():
        cells = var["cells"]
        predicted = "".join(grid[c[0]][c[1]] for c in cells)
        gold = var["gold"]
        correct = predicted == gold
        word_results.append({
            "id": var_id,
            "clue": var["clue"],
            "gold": gold,
            "predicted": predicted,
            "correct": correct,
        })

    return {
        "file": str(puzzle_path),
        "rows": crossword.rows,
        "cols": crossword.cols,
        "num_words": crossword.num_words,
        "num_cells": crossword.num_cells,
        "letter_accuracy": letter_acc,
        "word_accuracy": word_acc,
        "solve_time_seconds": elapsed,
        "words": word_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate solver on puzzle set")
    parser.add_argument("--puzzle-dir", type=str, required=True, help="Directory of .puz/.json files")
    parser.add_argument("--config", type=str, default="config/solve.yaml")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results to JSON")
    parser.add_argument("--limit", type=int, default=None, help="Max puzzles to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    puzzle_dir = Path(args.puzzle_dir)
    puzzles = sorted(puzzle_dir.glob("*.puz")) + sorted(puzzle_dir.glob("*.json"))
    if args.limit:
        puzzles = puzzles[: args.limit]
    print(f"Evaluating {len(puzzles)} puzzles from {puzzle_dir}")

    results = []
    total_letters_correct = 0
    total_letters = 0
    total_words_correct = 0
    total_words = 0
    total_puzzles_perfect = 0

    for puzzle_path in puzzles:
        result = evaluate_puzzle(puzzle_path, cfg)
        results.append(result)

        if "error" not in result:
            n_cells = result["num_cells"]
            n_words = result["num_words"]
            lc = int(result["letter_accuracy"] * n_cells)
            wc = int(result["word_accuracy"] * n_words)
            total_letters_correct += lc
            total_letters += n_cells
            total_words_correct += wc
            total_words += n_words
            if result["word_accuracy"] == 1.0:
                total_puzzles_perfect += 1

    # Aggregate
    n_solved = len([r for r in results if "error" not in r])
    agg = {
        "num_puzzles": len(puzzles),
        "num_solved": n_solved,
        "num_perfect": total_puzzles_perfect,
        "perfect_rate": total_puzzles_perfect / max(n_solved, 1),
        "aggregate_letter_accuracy": total_letters_correct / max(total_letters, 1),
        "aggregate_word_accuracy": total_words_correct / max(total_words, 1),
    }

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Puzzles: {n_solved}/{len(puzzles)} evaluated successfully")
    print(f"Perfect solves: {total_puzzles_perfect}/{n_solved} ({agg['perfect_rate']*100:.1f}%)")
    print(f"Letter accuracy: {total_letters_correct}/{total_letters} ({agg['aggregate_letter_accuracy']*100:.1f}%)")
    print(f"Word accuracy:   {total_words_correct}/{total_words} ({agg['aggregate_word_accuracy']*100:.1f}%)")

    if args.output:
        output = {"aggregate": agg, "per_puzzle": results}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
