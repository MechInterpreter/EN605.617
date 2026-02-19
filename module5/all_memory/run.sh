#!/usr/bin/env bash
set -euo pipefail

make clean
make

echo "=== 64 threads (minimum) ==="
./mem_all --blocks 64 --threads 64 --mode global
./mem_all --blocks 64 --threads 64 --mode shared

echo "=== additional threads ==="
./mem_all --blocks 64 --threads 128 --mode shared
./mem_all --blocks 64 --threads 256 --mode shared

echo "=== additional blocks ==="
./mem_all --blocks 128 --threads 256 --mode shared
./mem_all --blocks 256 --threads 256 --mode shared

echo "=== timing compare (same math) ==="
./mem_all --blocks 256 --threads 256 --mode global
./mem_all --blocks 256 --threads 256 --mode shared