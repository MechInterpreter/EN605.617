#!/usr/bin/env python3
"""
Example: run the CUDA ablation engine from Python.

Requires: pip install -e python/
"""

import ablation_engine
import numpy as np

def main():
    print("=== CUDA Ablation Engine (Python) ===")
    print()

    # Initialize with synthetic weights
    # (or pass weights_dir="weights/" for real)
    ablation_engine.init(
        embed_dim=128,
        num_heads=4,
        seq_len=32,
        mlp_dim=512,
        vocab_size=64,
        num_layers=2,
        weights_dir=""
    )

    cl = ablation_engine.clean_logit()
    print(f"Clean logit: {cl:.6f}")

    # Run ablation
    scores = ablation_engine.run_ablation(
        num_interventions=64,
        batch_size=32
    )
    print(f"Ran {len(scores)} interventions")
    print(f"Mean |causal score|: "
          f"{np.mean(np.abs(scores)):.6f}")
    print(f"Max  |causal score|: "
          f"{np.max(np.abs(scores)):.6f}")

    # Top-5 by absolute score
    top_idx = np.argsort(-np.abs(scores))[:5]
    print("\nTop-5 most important interventions:")
    for rank, idx in enumerate(top_idx):
        print(f"  #{rank+1}: iv {idx}  "
              f"score={scores[idx]:.6f}")

    # Validate
    print("\nValidating seq vs batched...")
    ok = ablation_engine.validate(64, 32)
    print(f"Validation: {'PASS' if ok else 'FAIL'}")

    # Cleanup
    ablation_engine.cleanup()
    print("\nDone.")

if __name__ == "__main__":
    main()
