#!/usr/bin/env python3
"""
Export PyTorch transformer weights to .bin format
for the CUDA Batched Causal Ablation Engine.

.bin format per tensor:
  [int32 ndims] [int32 shape[0]] ... [float32 data...]

Usage:
  python export_weights.py --model gpt2 --output weights/
  python export_weights.py --model <path> --output weights/
"""

import argparse
import os
import struct
import numpy as np

def save_tensor(path, tensor):
    """Save a numpy array to .bin format."""
    arr = tensor.astype(np.float32)
    shape = arr.shape
    ndims = len(shape)
    with open(path, 'wb') as f:
        f.write(struct.pack('i', ndims))
        for s in shape:
            f.write(struct.pack('i', s))
        f.write(arr.tobytes())
    print(f"  Saved {path}  shape={shape}  "
          f"({arr.nbytes} bytes)")

def export_gpt2(model_name, output_dir):
    """Export GPT-2 weights using HuggingFace."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("Error: pip install transformers")
        return

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(
        model_name)
    sd = model.state_dict()

    num_layers = model.config.n_layer
    embed_dim = model.config.n_embd
    print(f"  Layers: {num_layers}  "
          f"Embed: {embed_dim}")

    os.makedirs(output_dir, exist_ok=True)

    for l in range(num_layers):
        prefix = f"transformer.h.{l}."

        # Attention QKV is fused in GPT-2:
        # c_attn.weight = [E, 3*E]
        qkv = sd[prefix + "attn.c_attn.weight"]
        qkv = qkv.numpy().T  # [3*E, E] -> [E, 3E]
        E = embed_dim
        W_Q = qkv[:E, :].copy()
        W_K = qkv[E:2*E, :].copy()
        W_V = qkv[2*E:3*E, :].copy()

        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_Q.bin"), W_Q)
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_K.bin"), W_K)
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_V.bin"), W_V)

        # Output projection
        W_O = sd[prefix + "attn.c_proj.weight"]
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_O.bin"),
            W_O.numpy().T)

        # MLP
        W_mlp1 = sd[prefix + "mlp.c_fc.weight"]
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_mlp1.bin"),
            W_mlp1.numpy().T)
        W_mlp2 = sd[prefix + "mlp.c_proj.weight"]
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_W_mlp2.bin"),
            W_mlp2.numpy().T)

        # LayerNorm -> RMSNorm (use LN weights)
        ln1 = sd[prefix + "ln_1.weight"]
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_rms_attn.bin"),
            ln1.numpy())
        ln2 = sd[prefix + "ln_2.weight"]
        save_tensor(
            os.path.join(output_dir,
                         f"layer{l}_rms_mlp.bin"),
            ln2.numpy())

    # Final layer norm
    ln_f = sd["transformer.ln_f.weight"]
    save_tensor(
        os.path.join(output_dir, "rms_final.bin"),
        ln_f.numpy())

    # Output projection (lm_head)
    W_out = sd["lm_head.weight"]
    save_tensor(
        os.path.join(output_dir, "W_out.bin"),
        W_out.numpy().T)

    print(f"\nExported {num_layers} layers to "
          f"{output_dir}/")
    print(f"Run with: ./ablation_engine "
          f"--weights {output_dir} "
          f"--layers {num_layers}")

def main():
    parser = argparse.ArgumentParser(
        description="Export transformer weights "
                    "to .bin format")
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="HuggingFace model name or path")
    parser.add_argument(
        "--output", type=str, default="weights",
        help="Output directory for .bin files")
    args = parser.parse_args()

    export_gpt2(args.model, args.output)

if __name__ == "__main__":
    main()
