#!/bin/bash
set -e

echo "============================================"
echo "  CUDA Batched Causal Ablation Engine"
echo "  EN605.617 — GPU Computing"
echo "============================================"
echo ""

cd "$(dirname "$0")/.."

echo "[1/3] Cleaning previous build..."
make clean
echo ""

echo "[2/3] Building project..."
make
echo ""

echo "[3/3] Running validation + benchmark..."
echo ""
./ablation_engine --all 2>&1 | tee results/output.log

echo ""
echo "============================================"
echo "  Complete. Output: results/output.log"
echo "============================================"
