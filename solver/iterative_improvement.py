"""Iterative improvement — post-BP local search.

Mirrors Berkeley's ``BPSolver.iterative_improvement`` and related methods
so that the comparison is fair.  Uses the T5 reranker to decide whether
proposed character flips improve the grid.
"""
from __future__ import annotations

import string
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from solver.utils import get_word_flips
from models.clue_answerer import reranker_score


def get_uncertain_answers(solver, grid: list) -> dict:
    """Identify words in the grid that are NOT in the answer set.

    Returns ``{clue_text: predicted_word}`` for uncertain words.
    """
    original_qa: dict = {}
    for var_id, var in solver.crossword.variables.items():
        cells = var["cells"]
        word = "".join(grid[c[0]][c[1]] for c in cells)
        clue_text = var["clue"]
        original_qa[clue_text] = word
        # Also tag BP cells
        for cell in solver.bp_cells:
            if cell.position in cells:
                cell.prediction[clue_text] = word

    uncertain = {}
    for clue_text, word in original_qa.items():
        if word not in solver.answer_set:
            uncertain[clue_text] = word
    return uncertain


def get_candidate_replacements(solver, uncertain_answers: dict, grid: list) -> list:
    """Build a list of proposed character-flip replacements."""
    candidate_replacements = []
    replacement_id_set = set()

    # Dictionary-based flips
    for clue, initial_word in uncertain_answers.items():
        flips = get_word_flips(initial_word, 10)
        # Find which clue positions have this clue
        clue_positions = [
            k
            for k, v in solver.crossword.variables.items()
            if v["clue"] == clue
        ]
        for clue_pos in clue_positions:
            cells = sorted(
                [c for c in solver.bp_cells if clue_pos in c.crossing_clues],
                key=lambda c: c.position,
            )
            if len(cells) == len(initial_word):
                break
        for flip in flips:
            if len(flip) != len(cells):
                continue
            for i in range(len(flip)):
                if flip[i] != initial_word[i]:
                    candidate_replacements.append([(cells[i], flip[i])])
                    break

    # Probability-based flips
    for cell in solver.bp_cells:
        probs = np.exp(cell.log_probs)
        above_thresh = [
            string.ascii_uppercase[i] for i in range(26) if probs[i] > 0.01
        ]
        current_letter = grid[cell.position[0]][cell.position[1]]
        new_chars = [c for c in above_thresh if c != current_letter]
        for nc in new_chars:
            rid = f"{cell.position}_{nc}"
            if rid not in replacement_id_set:
                candidate_replacements.append([(cell, nc)])
            replacement_id_set.add(rid)

    # Composite flips (two cells sharing a word)
    composites = []
    for i in range(len(candidate_replacements)):
        for j in range(i + 1, len(candidate_replacements)):
            f1, f2 = candidate_replacements[i], candidate_replacements[j]
            if f1[0][0] != f2[0][0]:
                shared = set(f1[0][0].crossing_clues + f2[0][0].crossing_clues)
                if len(shared) < 4:
                    composites.append(f1 + f2)
    candidate_replacements += composites

    return candidate_replacements


def iterative_improvement(solver, grid: list) -> Tuple[list, bool]:
    """Run one round of iterative improvement.

    Returns ``(new_grid, did_edit)``.
    """
    uncertain = get_uncertain_answers(solver, grid)
    replacements = get_candidate_replacements(solver, uncertain, grid)

    if not replacements:
        return grid, False

    original_score = solver.score_grid(grid)
    possible_edits = []

    for rep in replacements:
        modified = deepcopy(grid)
        for cell, letter in rep:
            modified[cell.position[0]][cell.position[1]] = letter
        modified_score = solver.score_grid(modified)
        if modified_score - original_score > 0.5:
            possible_edits.append((modified, modified_score, rep))

    if not possible_edits:
        return grid, False

    # Apply non-overlapping edits in order of decreasing score
    possible_edits.sort(key=lambda x: x[1], reverse=True)
    variables_modified = set()
    selected = []
    for edit in possible_edits:
        rep = edit[2]
        variables = set()
        for cell, _ in rep:
            variables.update(cell.crossing_vars)
        if not variables_modified.intersection(variables):
            variables_modified.update(variables)
            selected.append(edit)

    new_grid = deepcopy(grid)
    for edit in selected:
        for cell, letter in edit[2]:
            new_grid[cell.position[0]][cell.position[1]] = letter

    return new_grid, True
