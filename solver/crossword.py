"""
Crossword data structure — closely matches Berkeley-Crossword-Solver/solver/Crossword.py
so that evaluation is directly comparable.
"""
from __future__ import annotations

import re
import string
from typing import Dict, List, Optional, Tuple


def clean(text: str) -> str:
    """Remove line breaks and collapse whitespace."""
    return " ".join(text.strip().split())


class Crossword:
    """In-memory representation of a crossword puzzle.

    Attributes
    ----------
    letter_grid : list[list[str]]
        Gold solution grid — empty string for black cells.
    number_grid : list[list[str]]
        Cell numbers (empty string if un-numbered).
    variables : dict[str, dict]
        ``{word_id: {"clue", "gold", "cells", "crossing"}}``
        ``word_id`` is e.g. ``"1A"`` or ``"3D"``.
    grid_cells : dict[tuple[int,int], list[str]]
        ``{(row,col): [word_id, …]}``
    rows, cols : int
        Puzzle dimensions.
    """

    def __init__(self, data: dict):
        self.rows: int = data["metadata"]["rows"]
        self.cols: int = data["metadata"]["cols"]
        self._init_grids(data["grid"])
        self._init_clues(data["clues"])
        self._init_variables()

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------
    def _init_grids(self, grid: list):
        self.letter_grid: List[List[str]] = []
        self.number_grid: List[List[str]] = []
        for j in range(len(grid)):
            letter_row, number_row = [], []
            for i in range(len(grid[0])):
                cell = grid[j][i]
                if isinstance(cell, list):
                    number_row.append(cell[0])
                    letter_row.append(cell[1])
                else:
                    number_row.append("")
                    letter_row.append("")
            self.letter_grid.append(letter_row)
            self.number_grid.append(number_row)
        self.grid_cells: Dict[Tuple[int, int], List[str]] = {}

    # ------------------------------------------------------------------
    # Clue parsing
    # ------------------------------------------------------------------
    def _init_clues(self, clues: dict):
        self.across: dict = clues["across"]
        self.down: dict = clues["down"]

    # ------------------------------------------------------------------
    # Variable (word slot) construction
    # ------------------------------------------------------------------
    def _init_variable(self, position: Tuple[int, int], clues: dict, across: bool):
        row, col = position
        cell_number = self.number_grid[row][col]
        assert cell_number in clues, f"Missing clue for cell {cell_number}"
        word_id = cell_number + ("A" if across else "D")
        clue = clean(clues[cell_number][0])
        answer = clean(clues[cell_number][1])
        for idx in range(len(answer)):
            cell = (row, col + idx) if across else (row + idx, col)
            self.grid_cells.setdefault(cell, []).append(word_id)
            if word_id in self.variables:
                self.variables[word_id]["cells"].append(cell)
            else:
                self.variables[word_id] = {
                    "clue": clue,
                    "gold": answer,
                    "cells": [cell],
                    "crossing": [],
                }

    def _init_crossing(self):
        for word_id, var in self.variables.items():
            crossing = []
            for cell in var["cells"]:
                crossing += [wid for wid in self.grid_cells[cell] if wid != word_id]
            var["crossing"] = crossing

    def _init_variables(self):
        self.variables: Dict[str, dict] = {}
        for row in range(len(self.number_grid)):
            for col in range(len(self.number_grid[0])):
                cn = self.number_grid[row][col]
                if cn != "":
                    if cn in self.across:
                        self._init_variable((row, col), self.across, across=True)
                    if cn in self.down:
                        self._init_variable((row, col), self.down, across=False)
        self._init_crossing()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_cells(self) -> int:
        return sum(1 for r in self.letter_grid for c in r if c != "")

    @property
    def num_words(self) -> int:
        return len(self.variables)

    def __repr__(self) -> str:
        return f"Crossword({self.rows}x{self.cols}, {self.num_words} words, {self.num_cells} cells)"
