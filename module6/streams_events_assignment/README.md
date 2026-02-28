# CUDA Streams and Events Assignment

This repository demonstrates **CUDA streams and events** with a **non-trivial GPU workload**, plus timing comparisons across **different threads and blocks**.

## What the program does

Each "batch element" simulates an *interpretability intervention variant*:

1) **Patch** one value in a vector (like swapping one activation element).
2) Run **non-trivial math** on every element (sin/cos/sqrt/log).
3) Compute a **metric per variant** (dot product with a fixed direction), similar to computing a logit score.

Two modes are benchmarked:

- **Baseline (sequential)**: one stream, no overlap.
- **Pipelined (async)**: two streams + double buffering so transfers overlap with compute.

**CUDA events** are used to time total runtime and stage time.

## Build

### CMake
```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

### nvcc
```bash
nvcc -O3 -std=c++17 --use_fast_math main.cu -o streams_events
```

## Run

Default run includes a sweep over multiple thread/block settings:

```bash
./streams_events
```

Specific configuration:

```bash
./streams_events --threads 256 --blocks 240 --microB 2048 --dim 1024 --batches 64 --runs 2 --verbose
```

Disable sweep:

```bash
./streams_events --no-compare --threads 256 --blocks 240
```

## Extra credit (final project)

This code is a reusable "micro-batch execution engine" for the final project that batches many intervention tests:
- patch specs represent interventions
- streams overlap copies with compute
- events provide reliable timing