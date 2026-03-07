#!/bin/bash
# Compare our solver vs Berkeley on the same puzzle set
# Runs both solvers and prints side-by-side results
#
# Prerequisites:
#   - ConditionalBPSolver trained model in checkpoints/clue_answerer/final
#   - Berkeley-Crossword-Solver set up with their checkpoints
#
# Usage:
#   bash scripts/compare.sh [puzzle_dir] [limit]

set -euo pipefail

PUZZLE_DIR="${1:-../TinyRecursiveModels/nyt_puz_flat}"
LIMIT="${2:-10}"

mkdir -p results

echo "============================================================"
echo "  COMPARISON: Conditional BP Solver vs Berkeley Solver"
echo "============================================================"
echo "Puzzle directory: $PUZZLE_DIR"
echo "Limit: $LIMIT puzzles"
echo

# --- Run our solver ---
echo ">>> Running Conditional BP Solver..."
python evaluate.py \
    --puzzle-dir "$PUZZLE_DIR" \
    --config config/solve.yaml \
    --output results/conditional_bp.json \
    --limit "$LIMIT"

echo
echo ">>> Conditional BP Solver results saved to results/conditional_bp.json"

# --- Run Berkeley solver (if available) ---
BERKELEY_DIR="../Berkeley-Crossword-Solver"
if [ -d "$BERKELEY_DIR" ]; then
    echo
    echo ">>> Running Berkeley Solver..."
    pushd "$BERKELEY_DIR" > /dev/null
    # Note: Berkeley solver expects .json files; convert .puz first
    python -c "
import json, sys, os
sys.path.insert(0, '.')
from utils import puz_to_json
from solver.Crossword import Crossword
from solver.BPSolver import BPSolver
from solver.Utils import print_grid
import glob

puzzle_dir = '$PUZZLE_DIR'
limit = $LIMIT
puzzles = sorted(glob.glob(os.path.join(puzzle_dir, '*.puz')))[:limit]

results = []
for puz_file in puzzles:
    try:
        data = puz_to_json(puz_file)
        cw = Crossword(data)
        solver = BPSolver(cw, max_candidates=500000)
        grid = solver.solve(num_iters=10, iterative_improvement_steps=5)
        solver.evaluate(grid)
    except Exception as e:
        print(f'Error on {puz_file}: {e}')
"
    popd > /dev/null
    echo ">>> Berkeley Solver evaluation complete"
else
    echo
    echo ">>> Berkeley solver not found at $BERKELEY_DIR — skipping comparison"
fi

echo
echo "============================================================"
echo "  COMPARISON COMPLETE"
echo "============================================================"
echo "Our results:      results/conditional_bp.json"
echo "Use 'jq .aggregate results/conditional_bp.json' for summary"
