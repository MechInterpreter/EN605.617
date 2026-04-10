set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"

# Parse script-level arguments
DO_CLEAN=0
ENGINE_ARGS=()
PAST_SEPARATOR=0

for arg in "$@"; do
    if [ "$PAST_SEPARATOR" -eq 1 ]; then
        ENGINE_ARGS+=("$arg")
    elif [ "$arg" = "--" ]; then
        PAST_SEPARATOR=1
    elif [ "$arg" = "--clean" ]; then
        DO_CLEAN=1
    else
        ENGINE_ARGS+=("$arg")
    fi
done

cd "$PROJECT_DIR"

# Clean if requested
if [ "$DO_CLEAN" -eq 1 ]; then
    echo "=== Cleaning ==="
    make clean 2>/dev/null || true
fi

# Build
echo "=== Building ==="
make -j$(nproc 2>/dev/null || echo 4) all

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Run
echo ""
echo "=== Running ==="
echo "  Args: ${ENGINE_ARGS[*]}"
echo ""

./ablation_engine "${ENGINE_ARGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/run_output.txt"

echo ""
echo "=== Output saved to results/run_output.txt ==="

# Run with different configs for demo
echo ""
echo "=== Demo: Device Info ==="
./ablation_engine --info \
    2>&1 | tee "$RESULTS_DIR/device_info.txt"

echo ""
echo "=== Demo: Small config with profiling ==="
./ablation_engine \
    --embed-dim 64 \
    --seq-len 16 \
    --interventions 32 \
    --batch-size 16 \
    --iterations 3 \
    --profile \
    2>&1 | tee "$RESULTS_DIR/small_profile.txt"

echo ""
echo "=== Demo: Large config ==="
./ablation_engine \
    --embed-dim 256 \
    --seq-len 64 \
    --interventions 256 \
    --batch-size 64 \
    --iterations 5 \
    --profile \
    2>&1 | tee "$RESULTS_DIR/large_config.txt"

echo ""
echo "=== Demo: Map/unmap mode ==="
./ablation_engine \
    --interventions 64 \
    --batch-size 32 \
    --use-map \
    --profile \
    2>&1 | tee "$RESULTS_DIR/map_mode.txt"

echo ""
echo "=== All runs complete. Results in: ==="
ls -la "$RESULTS_DIR/"
