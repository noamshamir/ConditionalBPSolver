#!/bin/bash
# Evaluate on a set of puzzles and compare with Berkeley
# Run from the ConditionalBPSolver directory:
#   bash scripts/evaluate.sh [puzzle_dir] [limit]

set -euo pipefail

PUZZLE_DIR="${1:-../TinyRecursiveModels/nyt_puz_flat}"
LIMIT="${2:-10}"
CONFIG="${3:-config/solve.yaml}"
OUTPUT="${4:-results/evaluation.json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Evaluating on puzzles from $PUZZLE_DIR (limit: $LIMIT)"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT"
echo

python evaluate.py \
    --puzzle-dir "$PUZZLE_DIR" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --limit "$LIMIT"
