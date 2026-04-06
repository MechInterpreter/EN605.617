#!/bin/bash
set -e

echo "============================================"
echo "  CUDA Batched Causal Ablation Engine"
echo "  EN605.617 — GPU Computing"
echo "============================================"
echo ""

cd "$(dirname "$0")/.."

LAYERS=${LAYERS:-2}
WEIGHTS_DIR=${WEIGHTS_DIR:-""}

echo "[1/3] Cleaning previous build..."
make clean
echo ""

echo "[2/3] Building project..."
make
echo ""

echo "[3/3] Running validation + benchmark..."
echo "  Layers: $LAYERS"

if [ -n "$WEIGHTS_DIR" ]; then
    echo "  Weights: $WEIGHTS_DIR"
    ./ablation_engine --all \
        --layers "$LAYERS" \
        --weights "$WEIGHTS_DIR" \
        2>&1 | tee results/output.log
else
    ./ablation_engine --all \
        --layers "$LAYERS" \
        2>&1 | tee results/output.log
fi

echo ""
echo "============================================"
echo "  Complete. Output: results/output.log"
echo "============================================"
