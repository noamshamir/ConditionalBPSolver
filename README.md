# Conditional Clue Re-interpretation with Belief Propagation

A crossword solver that combines a **conditional clue answerer** with **loopy belief propagation**. Unlike retrieval-based approaches (e.g., Berkeley Crossword Solver), the neural model re-interprets clues at each BP iteration, conditioned on the current letter beliefs from crossing words.

## Key Idea

Traditional neural crossword solvers:

1. Run a QA model once to get candidate answers per clue
2. Run BP to find the best consistent assignment

Our approach:

1. Run a conditional model that takes **(clue, letter_pattern)** → answer probabilities
2. At each BP iteration, re-query the model with updated letter beliefs
3. The model can "re-interpret" ambiguous clues as crossing evidence accumulates

## Architecture

- **Conditional Clue Answerer**: Fine-tuned ByT5 (byte-level T5) that scores `P(answer | clue, pattern)` where `pattern` encodes known/uncertain letters from crossings
- **Belief Propagation**: Factor graph with word variables and letter cells, same structure as Berkeley solver
- **Iterative Improvement**: Post-BP local search using T5 reranker (same as Berkeley for fair comparison)

## Comparison with Berkeley Crossword Solver

| Component             | Berkeley                   | Ours                                  |
| --------------------- | -------------------------- | ------------------------------------- |
| Clue Model            | DPR biencoder (retrieval)  | ByT5 conditional (generative scoring) |
| Candidate Source      | Fixed vocabulary + FAISS   | Beam search + vocabulary scoring      |
| BP Integration        | One-shot candidates → BP   | Re-query each BP iteration            |
| Reranker              | ByT5-small                 | ByT5-small (same)                     |
| Iterative Improvement | Character flips + reranker | Character flips + reranker (same)     |

## Install

```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
# Build training dataset from .puz files
python -m data.build_dataset \
    --puz-dir ../TinyRecursiveModels/nyt_puz_flat \
    --output-dir data/clue_answer_pairs \
    --val-fraction 0.05
```

## Training

```bash
# Train the conditional clue answerer
python train.py --config config/train_clue_model.yaml
```

## Solving

```bash
# Solve a single puzzle
python solve.py --puzzle path/to/puzzle.puz --config config/solve.yaml

# Evaluate on a test set
python evaluate.py --puzzle-dir path/to/puzzles --config config/solve.yaml
```

## Project Structure

```
ConditionalBPSolver/
├── data/
│   ├── build_dataset.py          # Build (clue, pattern, answer) training triples
│   └── crossword_data.py         # Crossword loading from .puz
├── models/
│   ├── clue_answerer.py          # Conditional Clue Answerer (ByT5)
│   └── reranker.py               # T5 reranker (same as Berkeley)
├── solver/
│   ├── crossword.py              # Crossword data structure
│   ├── belief_propagation.py     # BP solver with neural re-interpretation
│   ├── iterative_improvement.py  # Post-BP local search
│   └── utils.py                  # Grid utils, puz conversion
├── config/
│   ├── train_clue_model.yaml     # Training config
│   └── solve.yaml                # Solving config
├── train.py                      # Training entry point
├── solve.py                      # Solving entry point
└── evaluate.py                   # Evaluation entry point
```
