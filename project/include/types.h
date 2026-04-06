#ifndef TYPES_H
#define TYPES_H

#include <vector>

// Core data structures for the batched causal ablation
// engine.

/// Configuration for the transformer model.
struct ModelConfig {
    int embed_dim;      // hidden / embedding dimension
    int num_heads;      // number of attention heads
    int head_dim;       // per-head dim (embed/heads)
    int seq_len;        // sequence length
    int mlp_dim;        // MLP intermediate width
    int num_components; // ablatable (heads + 1 MLP)
    int vocab_size;     // vocab for output projection
    int num_layers;     // number of transformer layers
    float norm_eps;     // RMSNorm epsilon
};

/// Intervention type: which activation to ablate.
enum InterventionType {
    IV_ATTN_HEAD   = 0,  // ablate one attention head
    IV_MLP_OUT     = 1,  // ablate MLP output
    IV_RESID_STREAM = 2  // ablate residual stream
};

/// A single causal intervention: ablate a component
/// at a specific layer and token position.
/// component_idx in [0, num_heads) = attention head;
/// component_idx == num_heads = MLP block.
struct Intervention {
    int component_idx;        // which component
    int token_pos;            // which token position
    int layer_idx;            // which transformer layer
    InterventionType type;    // what kind of ablation
};

/// Result produced for one intervention.
struct AblationResult {
    int   intervention_id; // index into list
    float causal_score;    // clean - ablated logit
};

/// Per-layer weight set on GPU.
struct LayerWeights {
    float* W_Q;       // query   [E x E]
    float* W_K;       // key     [E x E]
    float* W_V;       // value   [E x E]
    float* W_O;       // output  [E x E]
    float* W_mlp1;    // MLP L1  [E x M]
    float* W_mlp2;    // MLP L2  [M x E]
    float* rms_attn;  // RMSNorm (pre-attn) [E]
    float* rms_mlp;   // RMSNorm (pre-MLP)  [E]
};

/// Holds all GPU-resident model weights (device ptrs).
struct ModelWeights {
    std::vector<LayerWeights> layers;
    float* W_out;       // final logit projection [E x V]
    float* rms_final;   // final RMSNorm [E]
};

/// Benchmark timing record.
struct BenchmarkRecord {
    int   num_interventions;
    int   batch_size;          // 0 = sequential
    float elapsed_ms;
    float interventions_per_sec;
    float speedup;             // vs sequential
};

#endif // TYPES_H
