set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Detecting OpenCL ==="

# Try common locations
if [ -d "/usr/local/cuda/include" ]; then
    INC="/usr/local/cuda/include"
    LIB="/usr/local/cuda/lib64"
    echo "  Found: NVIDIA CUDA toolkit"
elif [ -f "/usr/include/CL/cl.h" ]; then
    INC="/usr/include"
    LIB="/usr/lib/x86_64-linux-gnu"
    echo "  Found: System OpenCL"
else
    echo "  WARNING: OpenCL headers not found"
    echo "  Install: apt install opencl-headers"
    INC="/usr/include"
    LIB="/usr/lib/x86_64-linux-gnu"
fi

echo "=== Building ==="
make OPENCL_INC="$INC" OPENCL_LIB="$LIB" \
    -j$(nproc 2>/dev/null || echo 4) all

echo "=== Build complete ==="
echo "  Binary: ./ablation_engine"
