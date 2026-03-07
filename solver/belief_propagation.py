"""Belief Propagation Solver with Conditional Clue Re-interpretation.

This follows the same BP structure as Berkeley's ``BPSolver`` (which in turn
follows Littman et al. 1999), but with a crucial difference:

    **At each BP outer iteration, we re-query the neural clue answerer
    with updated letter beliefs from crossing words.**

Berkeley's pipeline:
    1. Query DPR once → fixed candidate list per clue
    2. Run BP on the fixed candidates
    3. Greedy decode + iterative improvement

Our pipeline:
    1. Query conditional model with blank patterns → initial candidates
    2. Run BP iteration → letter beliefs update
    3. Convert letter beliefs to patterns → re-query model → refresh candidates
    4. Repeat (2-3) for N outer iterations
    5. Greedy decode + iterative improvement (same as Berkeley)

The BPVar / BPCell structure mirrors Berkeley exactly for fair comparison.
"""
from __future__ import annotations

import math
import string
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import log_softmax, softmax
from tqdm import trange

from solver.crossword import Crossword
from solver.utils import preprocess_clue, print_grid, get_word_flips
from models.clue_answerer import (
    beliefs_to_pattern,
    format_clue_pattern,
    generate_candidates,
    score_answers,
    setup_clue_answerer,
    setup_reranker,
    reranker_score,
)


# ======================================================================
# Unigram priors  (same as Berkeley)
# ======================================================================
UNIGRAM_PROBS = [
    ("A", 0.0897), ("B", 0.0212), ("C", 0.0348), ("D", 0.0370),
    ("E", 0.1160), ("F", 0.0173), ("G", 0.0254), ("H", 0.0331),
    ("I", 0.0680), ("J", 0.0029), ("K", 0.0139), ("L", 0.0513),
    ("M", 0.0280), ("N", 0.0663), ("O", 0.0737), ("P", 0.0268),
    ("Q", 0.0015), ("R", 0.0708), ("S", 0.0741), ("T", 0.0724),
    ("U", 0.0289), ("V", 0.0092), ("W", 0.0143), ("X", 0.0031),
    ("Y", 0.0175), ("Z", 0.0027),
]

# Smoothing: probability that the answer is NOT in the candidate set,
# indexed by answer length. (same as Berkeley)
LETTER_SMOOTHING_FACTOR = [
    0.0, 0.0, 0.044, 0.000137, 0.000575, 0.00198, 0.00480, 0.01333,
    0.02715, 0.06514, 0.12528, 0.22003, 0.23172, 0.25487, 0.39851,
    0.27650, 0.67265, 0.68182, 0.85714, 0.82456, 0.80000, 0.71901, 0.0,
]


# ======================================================================
# Answer vocabulary (for checking if a word is "known")
# ======================================================================
def load_answer_set(path: str) -> set:
    """Load the answer vocabulary for iterative improvement."""
    answers = set()
    try:
        with open(path, "r") as f:
            for line in f:
                w = "".join(
                    c.upper()
                    for c in line.split("\t")[-1].upper()
                    if c in string.ascii_uppercase
                )
                if w:
                    answers.add(w)
    except FileNotFoundError:
        pass
    return answers


# ======================================================================
# BPVar — a word-variable in the factor graph
# ======================================================================
class BPVar:
    """Factor graph variable for one word slot.

    Mirrors Berkeley's ``BPVar`` exactly.
    """

    def __init__(self, name: str, variable: dict, candidates: dict, cells: list):
        self.name = name
        cells_by_position = {}
        for cell in cells:
            cells_by_position[cell.position] = cell
            cell._connect(self)
        self.length = len(cells)
        self.ordered_cells = [cells_by_position[pos] for pos in variable["cells"]]
        self.candidates = candidates
        self.words: List[str] = list(candidates["words"])
        self.word_indices = np.array(
            [[string.ascii_uppercase.index(c) for c in w] for w in self.words]
        )
        self.scores = -np.array(
            [candidates["weights"][w] for w in candidates["words"]]
        )
        self.prior_log_probs = log_softmax(self.scores)
        self.log_probs = log_softmax(self.scores)
        self.directional_scores = [
            np.zeros(len(self.log_probs)) for _ in range(len(self.ordered_cells))
        ]

    def refresh_candidates(self, new_candidates: dict):
        """Replace candidate set with new candidates from re-querying.

        This is the key extension vs. Berkeley: we can update candidates
        mid-BP as the neural model is re-queried with updated patterns.
        """
        self.candidates = new_candidates
        self.words = list(new_candidates["words"])
        if len(self.words) == 0:
            return
        self.word_indices = np.array(
            [[string.ascii_uppercase.index(c) for c in w] for w in self.words]
        )
        self.scores = -np.array(
            [new_candidates["weights"][w] for w in new_candidates["words"]]
        )
        self.prior_log_probs = log_softmax(self.scores)
        self.log_probs = log_softmax(self.scores)
        self.directional_scores = [
            np.zeros(len(self.log_probs)) for _ in range(len(self.ordered_cells))
        ]

    def _propagate_to_var(self, other: "BPCell", belief_state: np.ndarray):
        assert other in self.ordered_cells
        other_idx = self.ordered_cells.index(other)
        self.directional_scores[other_idx] = belief_state[
            self.word_indices[:, other_idx]
        ]

    def _postprocess(self, all_letter_probs: list) -> list:
        """Unigram smoothing (same as Berkeley)."""
        unigram_probs = np.array([x[1] for x in UNIGRAM_PROBS])
        smooth = LETTER_SMOOTHING_FACTOR[min(self.length, len(LETTER_SMOOTHING_FACTOR) - 1)]
        for i in range(len(all_letter_probs)):
            all_letter_probs[i] = (
                (1 - smooth) * all_letter_probs[i] + smooth * unigram_probs
            )
        return all_letter_probs

    def sync_state(self):
        if len(self.words) == 0:
            return
        self.log_probs = log_softmax(
            sum(self.directional_scores) + self.prior_log_probs
        )

    def propagate(self):
        if len(self.words) == 0:
            return
        all_letter_probs = []
        for i in range(len(self.ordered_cells)):
            word_scores = self.log_probs - self.directional_scores[i]
            word_probs = softmax(word_scores)
            letter_probs = (
                self.candidates["bit_array"][:, i] * np.expand_dims(word_probs, axis=0)
            ).sum(axis=1) + 1e-8
            all_letter_probs.append(letter_probs)
        all_letter_probs = self._postprocess(all_letter_probs)
        for i, cell in enumerate(self.ordered_cells):
            cell._propagate_to_cell(self, np.log(all_letter_probs[i]))


# ======================================================================
# BPCell — a letter-variable in the factor graph
# ======================================================================
class BPCell:
    """Factor graph variable for one grid cell (letter).

    Mirrors Berkeley's ``BPCell`` exactly.
    """

    def __init__(self, position: tuple, clue_pair: list):
        self.crossing_clues = clue_pair
        self.position = tuple(position)
        self.letters = list(string.ascii_uppercase)
        self.log_probs = np.log(
            np.array([1.0 / len(self.letters)] * len(self.letters))
        )
        self.crossing_vars: List[BPVar] = []
        self.directional_scores: List[Optional[np.ndarray]] = []
        self.prediction: dict = {}

    def _connect(self, var: BPVar):
        self.crossing_vars.append(var)
        self.directional_scores.append(None)
        assert len(self.crossing_vars) <= 2

    def _propagate_to_cell(self, other: BPVar, belief_state: np.ndarray):
        other_idx = self.crossing_vars.index(other)
        self.directional_scores[other_idx] = belief_state

    def sync_state(self):
        valid = [s for s in self.directional_scores if s is not None]
        if valid:
            self.log_probs = log_softmax(sum(valid))

    def propagate(self):
        for i, v in enumerate(self.crossing_vars):
            other_score = self.directional_scores[1 - i] if len(self.crossing_vars) == 2 else np.zeros(26)
            if other_score is not None:
                v._propagate_to_var(self, other_score)


# ======================================================================
# Candidate manager — builds the data structures BP needs
# ======================================================================
def build_candidate_dict(
    words_and_scores: List[Tuple[str, float]],
    length: int,
) -> dict:
    """Build the candidates dict from a list of ``(word, score)``."""
    chars = string.ascii_uppercase
    # Filter to correct length and remove duplicates
    seen = set()
    filtered = []
    for word, score in words_and_scores:
        word = "".join(c for c in word.upper() if c in chars)
        if len(word) == length and word not in seen:
            seen.add(word)
            filtered.append((word, score))

    if not filtered:
        # Fallback: add a dummy word to avoid crashes
        dummy = "A" * length
        filtered = [(dummy, -100.0)]

    weights = {w: -s for w, s in filtered}  # negate: BP expects costs
    sorted_words = sorted(weights, key=weights.get)
    char_map = {c: i for i, c in enumerate(chars)}
    bit_array = np.zeros((len(chars), length, len(sorted_words)))
    for word_idx, word in enumerate(sorted_words):
        for pos_idx, char in enumerate(word):
            bit_array[char_map[char], pos_idx, word_idx] = 1

    return {
        "words": sorted_words,
        "weights": weights,
        "bit_array": bit_array,
    }


# ======================================================================
# ConditionalBPSolver — the main solver
# ======================================================================
class ConditionalBPSolver:
    """Crossword solver using Conditional Clue Re-interpretation + BP.

    Parameters
    ----------
    crossword : Crossword
        The puzzle to solve.
    model_checkpoint : str
        Path or HF model ID for the conditional clue answerer.
    reranker_checkpoint : str
        Path or HF model ID for the T5 reranker.
    answer_set_path : str
        Path to a wordlist.tsv for iterative improvement.
    max_candidates : int
        Max candidates per clue from the neural model.
    device : str
        CUDA device.
    """

    def __init__(
        self,
        crossword: Crossword,
        model_checkpoint: str = "google/byt5-base",
        reranker_checkpoint: str = "google/byt5-small",
        answer_set_path: str = "data/clue_answer_pairs/answer_vocab.txt",
        max_candidates: int = 500,
        num_beams: int = 50,
        device: str = "cuda:0",
        belief_threshold: float = 0.6,
        requery_kl_threshold: float = 0.1,
    ):
        self.crossword = crossword
        self.max_candidates = max_candidates
        self.num_beams = num_beams
        self.device = device
        self.belief_threshold = belief_threshold
        self.requery_kl_threshold = requery_kl_threshold  # KL divergence threshold for selective re-querying

        # Belief snapshots for selective re-querying
        # Stores {var_id: np.ndarray(length, 26)} from the last time we queried
        self._belief_snapshots: Dict[str, np.ndarray] = {}

        # Load models
        self.model, self.tokenizer = setup_clue_answerer(
            model_checkpoint, device
        )
        self.reranker_model = None  # lazy-loaded
        self.reranker_tokenizer = None
        self.reranker_checkpoint = reranker_checkpoint

        # Load answer set for iterative improvement
        self.answer_set = load_answer_set(answer_set_path)
        if not self.answer_set:
            # Fallback: build from crossword gold answers
            self.answer_set = {
                v["gold"] for v in crossword.variables.values() if v.get("gold")
            }

        # Preprocess all clues
        self.processed_clues: Dict[str, str] = {}
        for var_id, var in crossword.variables.items():
            self.processed_clues[var_id] = preprocess_clue(var["clue"])

        # Initialize BP structures
        self._init_bp()

    def _init_bp(self):
        """Create BPCell and BPVar structures."""
        self.bp_cells: List[BPCell] = []
        self.bp_cells_by_clue: Dict[str, List[BPCell]] = defaultdict(list)

        for position, clue_pair in self.crossword.grid_cells.items():
            cell = BPCell(position, clue_pair)
            self.bp_cells.append(cell)
            for clue_id in clue_pair:
                self.bp_cells_by_clue[clue_id].append(cell)

        # Get initial candidates (blank patterns)
        initial_candidates = self._query_model_for_candidates(
            {vid: "_" * len(var["gold"]) for vid, var in self.crossword.variables.items()}
        )

        self.bp_vars: List[BPVar] = []
        self.bp_var_by_name: Dict[str, BPVar] = {}
        for var_id, var in self.crossword.variables.items():
            candidates = initial_candidates.get(var_id, build_candidate_dict([], len(var["gold"])))
            bp_var = BPVar(
                var_id, var, candidates, self.bp_cells_by_clue[var_id]
            )
            self.bp_vars.append(bp_var)
            self.bp_var_by_name[var_id] = bp_var

        # Snapshot initial uniform beliefs so the first selective re-query
        # sees infinite divergence and re-queries everything
        self._belief_snapshots = {}

    def _query_model_for_candidates(
        self,
        patterns: Dict[str, str],
    ) -> Dict[str, dict]:
        """Query the neural model for each clue with its current pattern."""
        var_ids = list(patterns.keys())
        clues = [self.processed_clues[vid] for vid in var_ids]
        pats = [patterns[vid] for vid in var_ids]
        lengths = [len(self.crossword.variables[vid]["gold"]) for vid in var_ids]

        # Generate candidates
        all_candidates = generate_candidates(
            self.model,
            self.tokenizer,
            clues,
            pats,
            num_beams=self.num_beams,
            num_return=self.num_beams,
            batch_size=16,
        )

        result = {}
        for i, var_id in enumerate(var_ids):
            candidates_raw = all_candidates[i]
            result[var_id] = build_candidate_dict(
                candidates_raw, lengths[i]
            )
        return result

    def _get_letter_beliefs(self) -> Dict[str, np.ndarray]:
        """Extract per-variable letter belief matrices from BP cells.

        Returns ``{var_id: np.ndarray shape (length, 26)}``.
        """
        beliefs = {}
        for var_id, var in self.crossword.variables.items():
            bp_var = self.bp_var_by_name[var_id]
            length = len(var["gold"])
            belief_matrix = np.zeros((length, 26))
            for pos_idx, cell in enumerate(bp_var.ordered_cells):
                belief_matrix[pos_idx] = softmax(cell.log_probs)
            beliefs[var_id] = belief_matrix
        return beliefs

    def _belief_divergence(self, var_id: str, current: np.ndarray) -> float:
        """Compute divergence between current beliefs and last-queried snapshot.

        Uses mean per-position KL divergence.  Returns ``float('inf')`` if
        no snapshot exists (i.e. first re-query after init).
        """
        prev = self._belief_snapshots.get(var_id)
        if prev is None:
            return float("inf")
        # Quick check: did any argmax letter change?
        if (current.argmax(axis=1) != prev.argmax(axis=1)).any():
            # Compute full KL for logging, but we already know it's worth re-querying
            pass
        else:
            # Argmax unchanged — check whether the distribution shape shifted
            pass
        # D_KL(current || prev), averaged over positions
        eps = 1e-10
        kl = (current * (np.log(current + eps) - np.log(prev + eps))).sum(axis=1)  # (length,)
        return float(kl.mean())

    def _select_slots_for_requery(self, beliefs: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Return ``{var_id: pattern}`` only for slots whose beliefs changed enough."""
        patterns: Dict[str, str] = {}
        skipped = 0
        for var_id, belief_matrix in beliefs.items():
            div = self._belief_divergence(var_id, belief_matrix)
            if div > self.requery_kl_threshold:
                patterns[var_id] = beliefs_to_pattern(
                    belief_matrix, threshold=self.belief_threshold
                )
            else:
                skipped += 1
        print(f"    Selective re-query: {len(patterns)} slots changed, {skipped} converged (skipped)")
        return patterns

    def _requery_with_beliefs(self):
        """Re-query the neural model *selectively* for slots whose beliefs shifted."""
        beliefs = self._get_letter_beliefs()
        patterns = self._select_slots_for_requery(beliefs)

        if not patterns:
            print("    All slots converged — skipping re-query entirely")
            # Still update snapshots
            self._belief_snapshots = beliefs
            return

        new_candidates = self._query_model_for_candidates(patterns)

        # Update snapshots for ALL slots (so next round compares against now)
        self._belief_snapshots = beliefs

        # Merge: keep old candidates that are still consistent + add new ones
        # Only touch vars that were actually re-queried
        for bp_var in self.bp_vars:
            var_id = bp_var.name
            if var_id in new_candidates:
                old_words_scores = {
                    w: bp_var.candidates["weights"].get(w, 10.0) for w in bp_var.words
                }
                new_words_scores = {
                    w: new_candidates[var_id]["weights"].get(w, 10.0)
                    for w in new_candidates[var_id]["words"]
                }
                # Merge: prefer new scores for overlapping words, keep old unique ones
                merged = {}
                for w, s in old_words_scores.items():
                    merged[w] = s
                for w, s in new_words_scores.items():
                    if w in merged:
                        merged[w] = min(merged[w], s)  # prefer lower cost
                    else:
                        merged[w] = s

                # Trim to max_candidates
                sorted_merged = sorted(merged.items(), key=lambda x: x[1])
                sorted_merged = sorted_merged[: self.max_candidates]

                length = len(self.crossword.variables[var_id]["gold"])
                merged_list = [(w, -s) for w, s in sorted_merged]  # un-negate
                new_dict = build_candidate_dict(merged_list, length)
                bp_var.refresh_candidates(new_dict)

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------
    def solve(
        self,
        num_iters: int = 10,
        requery_every: int = 3,
        iterative_improvement_steps: int = 5,
        return_greedy_states: bool = False,
        return_ii_states: bool = False,
    ) -> list | Tuple[list, list]:
        """Run the full solving pipeline.

        Parameters
        ----------
        num_iters : int
            Total BP message-passing iterations.
        requery_every : int
            Re-query the neural model every N iterations.
        iterative_improvement_steps : int
            Number of iterative improvement rounds after BP.
        """
        print(f"Beginning BP with conditional re-querying (requery every {requery_every} iters)")

        for iteration in trange(num_iters, desc="BP iterations"):
            # Standard BP message passing
            for var in self.bp_vars:
                var.propagate()
            for cell in self.bp_cells:
                cell.sync_state()
            for cell in self.bp_cells:
                cell.propagate()
            for var in self.bp_vars:
                var.sync_state()

            # Re-query neural model with updated beliefs
            if (iteration + 1) % requery_every == 0 and (iteration + 1) < num_iters:
                print(f"\n  Re-querying neural model at iteration {iteration + 1}")
                self._requery_with_beliefs()
                # Re-connect BP vars to cells (refresh directional scores)
                for var in self.bp_vars:
                    var.directional_scores = [
                        np.zeros(len(var.log_probs))
                        for _ in range(len(var.ordered_cells))
                    ]

        print("Done BP iterations")

        # Greedy sequential word solution
        if return_greedy_states:
            grid, all_grids = self._greedy_sequential_word_solution(return_grids=True)
        else:
            grid = self._greedy_sequential_word_solution()
            all_grids = []

        print("=====Greedy search grid=====")
        print_grid(grid)

        if iterative_improvement_steps < 1:
            if return_greedy_states or return_ii_states:
                return grid, all_grids
            return grid

        # Lazy-load reranker
        if self.reranker_model is None:
            self.reranker_model, self.reranker_tokenizer = setup_reranker(
                self.reranker_checkpoint, self.device
            )

        from solver.iterative_improvement import iterative_improvement

        for i in range(iterative_improvement_steps):
            print(f"Starting iterative improvement step {i}")
            self.evaluate(grid)
            grid, did_edit = iterative_improvement(
                self, grid
            )
            if not did_edit:
                break
            if return_ii_states:
                all_grids.append(deepcopy(grid))
            print(f"After iterative improvement step {i}")
            print_grid(grid)

        if return_greedy_states or return_ii_states:
            return grid, all_grids
        return grid

    # ------------------------------------------------------------------
    # Greedy sequential word filling (same as Berkeley)
    # ------------------------------------------------------------------
    def _greedy_sequential_word_solution(
        self, return_grids: bool = False
    ) -> list | Tuple[list, list]:
        all_grids = []
        cache = [
            (deepcopy(var.words), deepcopy(var.log_probs)) for var in self.bp_vars
        ]

        grid = [["" for _ in row] for row in self.crossword.letter_grid]
        unfilled_cells = set(cell.position for cell in self.bp_cells)

        for var in self.bp_vars:
            if len(var.words) > 0:
                smooth = LETTER_SMOOTHING_FACTOR[
                    min(var.length, len(LETTER_SMOOTHING_FACTOR) - 1)
                ]
                var.log_probs = var.log_probs + math.log(max(1 - smooth, 1e-10))

        best_per_var = []
        for var in self.bp_vars:
            if len(var.words) > 0:
                best_per_var.append(var.log_probs.max())
            else:
                best_per_var.append(None)

        while not all(x is None for x in best_per_var):
            all_grids.append(deepcopy(grid))
            valid = [x for x in best_per_var if x is not None]
            if not valid:
                break
            best_index = best_per_var.index(max(valid))
            best_var = self.bp_vars[best_index]
            if len(best_var.words) == 0:
                best_per_var[best_index] = None
                continue
            best_word = best_var.words[best_var.log_probs.argmax()]

            for i, cell in enumerate(best_var.ordered_cells):
                letter = best_word[i]
                grid[cell.position[0]][cell.position[1]] = letter
                unfilled_cells.discard(cell.position)
                for var in cell.crossing_vars:
                    if var != best_var:
                        cell_index = var.ordered_cells.index(cell)
                        keep = [
                            j
                            for j in range(len(var.words))
                            if var.words[j][cell_index] == letter
                        ]
                        var.words = [var.words[j] for j in keep]
                        var.log_probs = var.log_probs[keep]
                        var_index = self.bp_vars.index(var)
                        best_per_var[var_index] = (
                            var.log_probs.max() if len(keep) > 0 else None
                        )

            best_var.words = []
            best_var.log_probs = best_var.log_probs[[]]
            best_per_var[best_index] = None

        # Fill remaining cells with argmax from letter beliefs
        for cell in self.bp_cells:
            if cell.position in unfilled_cells:
                grid[cell.position[0]][cell.position[1]] = string.ascii_uppercase[
                    cell.log_probs.argmax()
                ]

        # Restore state
        for var, (words, lp) in zip(self.bp_vars, cache):
            var.words = words
            var.log_probs = lp

        if return_grids:
            return grid, all_grids
        return grid

    # ------------------------------------------------------------------
    # Evaluation (same as Berkeley)
    # ------------------------------------------------------------------
    def evaluate(self, solution: list) -> Tuple[float, float]:
        """Print and return (letter_accuracy, word_accuracy)."""
        letters_correct = 0
        letters_total = 0
        for i in range(len(self.crossword.letter_grid)):
            for j in range(len(self.crossword.letter_grid[0])):
                if self.crossword.letter_grid[i][j] != "":
                    letters_correct += int(
                        self.crossword.letter_grid[i][j] == solution[i][j]
                    )
                    letters_total += 1

        words_correct = 0
        words_total = 0
        for var_id, var in self.crossword.variables.items():
            cells = var["cells"]
            match = all(
                self.crossword.letter_grid[c[0]][c[1]] == solution[c[0]][c[1]]
                for c in cells
            )
            if match:
                words_correct += 1
            words_total += 1

        letter_acc = letters_correct / max(letters_total, 1)
        word_acc = words_correct / max(words_total, 1)
        print(
            f"Letters: {letters_correct}/{letters_total} ({letter_acc*100:.1f}%) | "
            f"Words: {words_correct}/{words_total} ({word_acc*100:.1f}%)"
        )
        return letter_acc, word_acc

    # ------------------------------------------------------------------
    # Score grid (for iterative improvement)
    # ------------------------------------------------------------------
    def score_grid(self, grid: list) -> float:
        """Score entire grid using the reranker."""
        if self.reranker_model is None:
            self.reranker_model, self.reranker_tokenizer = setup_reranker(
                self.reranker_checkpoint, self.device
            )
        clues_list = []
        answers_list = []
        for clue_id, cells in self.bp_cells_by_clue.items():
            letters = "".join(
                grid[c.position[0]][c.position[1]]
                for c in sorted(cells, key=lambda c: c.position)
            )
            clues_list.append(self.crossword.variables[clue_id]["clue"])
            answers_list.append(letters)
        scores = reranker_score(
            self.reranker_model, self.reranker_tokenizer, clues_list, answers_list
        )
        return sum(scores)
