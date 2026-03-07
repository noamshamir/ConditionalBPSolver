#!/bin/bash
# Train the conditional clue answerer
# Run from the ConditionalBPSolver directory:
#   bash scripts/train.sh

set -euo pipefail

CONFIG="${1:-config/train_clue_model.yaml}"

echo "Training conditional clue answerer..."
echo "  Config: $CONFIG"
echo

python train.py --config "$CONFIG"
