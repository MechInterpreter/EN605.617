# CUDA Batched Causal Ablation Engine

A CUDA-accelerated batched causal ablation engine for transformer-style mechanistic interpretability experiments. Uses **cuBLAS** for batched linear-algebra operations and **Thrust** for data-parallel mask generation, scoring, and ranking (all on GPU).

## Architecture

```
src/
  main.cu             CLI entry point, model setup
  transformer.cu/.h   Multi-layer transformer forward pass
                      (RMSNorm, causal SDPA, GeLU MLP)
  ablation.cu/.h      Mask generation, scoring, ranking (Thrust)
  batched_engine.cu   Micro-batched intervention engine
  baseline.cu         Sequential one-at-a-time baseline
  benchmark.cu        Sweep over intervention/batch sizes
  validation.cu       Correctness check (seq vs batched)
  weight_io.cu/.h     Load/save weights in .bin tensor format

include/
  types.h             ModelConfig, LayerWeights, Intervention, etc.
  config.h            Default dimensions, sweep parameters
  cuda_utils.h        CUDA/cuBLAS error macros, GpuTimer

python/
  ablation_engine.cpp pybind11 Python bindings
  setup.py            Build via pip install -e .

scripts/
  run_all.sh          Clean, build, run all modes
  export_weights.py   Export HuggingFace model -> .bin files
  run_from_python.py  Example Python usage
```

## Features

- **Multi-layer transformer** with RMSNorm, scaled dot-product attention (causal mask + softmax), GeLU MLP, and output projection
- **Per-layer interventions**: ablate attention heads, MLP outputs, or residual streams at any layer and token position
- **Batched engine**: process interventions in configurable micro-batches using Thrust + cuBLAS
- **Real weight loading**: load pretrained weights from `.bin` files exported from PyTorch/HuggingFace
- **Python bindings**: call the CUDA engine from Python via pybind11
- **Validation**: correctness check (sequential vs batched must match within tolerance)
- **Benchmark**: sweep across intervention counts and batch sizes

## Quick Start

### Build and run (synthetic weights)

```bash
make clean && make
./ablation_engine --all
```

### With real weights (GPT-2)

```bash
# Export weights
pip install transformers torch
python scripts/export_weights.py --model gpt2 --output weights/

# Run with loaded weights
./ablation_engine --all --weights weights/ --layers 12
```

### Python usage

```bash
cd python && pip install -e .
python ../scripts/run_from_python.py
```

## CLI Options

```
./ablation_engine [mode] [options]

Modes:
  --baseline   Sequential baseline (64 interventions)
  --batched    Batched engine (64 interventions, bs=32)
  --benchmark  Full sweep over sizes and batch sizes
  --validate   Correctness check (seq vs batched)
  --all        Validate + benchmark (default)

Options:
  --weights <dir>   Load weights from directory
  --layers <N>      Number of transformer layers (default: 2)
```

## Weight Format

Each `.bin` file stores one tensor:
```
[int32 ndims] [int32 shape[0]] ... [float32 data...]
```

Expected naming convention:
```
layer{L}_W_Q.bin      Query projection [E×E]
layer{L}_W_K.bin      Key projection [E×E]
layer{L}_W_V.bin      Value projection [E×E]
layer{L}_W_O.bin      Output projection [E×E]
layer{L}_W_mlp1.bin   MLP up-projection [E×M]
layer{L}_W_mlp2.bin   MLP down-projection [M×E]
layer{L}_rms_attn.bin Pre-attention norm [E]
layer{L}_rms_mlp.bin  Pre-MLP norm [E]
W_out.bin             Final output projection [E×V]
rms_final.bin         Final layer norm [E]
```

## Build Variables

```
CUDA_PATH   /usr/local/cuda (default)
GPU_ARCH    auto-detected or e.g. sm_75
```

## Intervention Types

| Type | Enum | Description |
|------|------|-------------|
| `IV_ATTN_HEAD` | 0 | Zero out one attention head's contribution at a token position |
| `IV_MLP_OUT` | 1 | Zero out MLP output at a token position |
| `IV_RESID_STREAM` | 2 | Zero out full residual at a token position |

## Remaining Limitations / Future Work

- **Attention is single-threaded per (token, head)**: The SDPA kernel uses one warp per query-head pair. For longer sequences, a tiled attention kernel or FlashAttention would be needed.
- **No bias terms**: The transformer does not include bias parameters (consistent with modern architectures like Llama, but not GPT-2).
- **No positional encoding**: RoPE or learned position embeddings are not yet implemented.
- **No multi-token output**: Logit extraction targets only one token position at a time.
- **Weight format is simple**: Does not parse safetensors or PyTorch `.pt` files directly; requires the export script.
- **Batched engine is sequential per batch**: Each intervention in a micro-batch runs a separate forward pass. A truly batched multi-layer forward with per-element masks would require fused kernels.
- **Python bindings use global state**: Only one engine instance at a time.
- **Config must match weights**: embed_dim, num_heads, etc. must be specified at CLI to match the exported weights.
