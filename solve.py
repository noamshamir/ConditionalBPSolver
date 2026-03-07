"""Solve a crossword puzzle using the Conditional BP Solver.

Usage
-----
    python solve.py --puzzle path/to/puzzle.puz --config config/solve.yaml
    python solve.py --puzzle path/to/puzzle.json --config config/solve.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def solve_puzzle(puzzle_path: str, cfg: dict) -> tuple:
    """Load and solve a single puzzle. Returns (grid, letter_acc, word_acc)."""
    path = Path(puzzle_path)
    if path.suffix == ".puz":
        data = puz_to_json(path)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    crossword = Crossword(data)
    print(f"\nSolving {path.name}: {crossword}")

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

    print("\n*** Solver Output ***")
    print_grid(grid)
    print("\n*** Gold Solution ***")
    print_grid(crossword.letter_grid)

    letter_acc, word_acc = solver.evaluate(grid)
    return grid, letter_acc, word_acc


def main():
    parser = argparse.ArgumentParser(description="Solve a crossword puzzle")
    parser.add_argument("--puzzle", type=str, required=True, help=".puz or .json file")
    parser.add_argument("--config", type=str, default="config/solve.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    solve_puzzle(args.puzzle, cfg)


if __name__ == "__main__":
    main()
