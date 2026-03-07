#!/bin/bash
# Build the training dataset from .puz files
# Run from the ConditionalBPSolver directory:
#   bash scripts/build_data.sh

set -euo pipefail

PUZ_DIR="${1:-../TinyRecursiveModels/nyt_puz_flat}"
OUTPUT_DIR="${2:-data/clue_answer_pairs}"
VAL_FRACTION="${3:-0.05}"
AUGMENTATIONS="${4:-6}"

echo "Building training data..."
echo "  PUZ dir:        $PUZ_DIR"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Val fraction:   $VAL_FRACTION"
echo "  Augmentations:  $AUGMENTATIONS"
echo

python -m data.build_dataset \
    --puz-dir "$PUZ_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --val-fraction "$VAL_FRACTION" \
    --augmentations "$AUGMENTATIONS" \
    --seed 42

echo
echo "Done! Dataset ready in $OUTPUT_DIR"
echo "Next step: python train.py --config config/train_clue_model.yaml"
