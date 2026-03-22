#ifndef TYPES_H
#define TYPES_H

#include <vector>

// Core data structures for the batched causal ablation
// engine.

/// Configuration for the simplified transformer model.
struct ModelConfig {
    int embed_dim;      // hidden / embedding dimension
    int num_heads;      // number of attention heads
    int head_dim;       // per-head dim (embed/heads)
    int seq_len;        // sequence length
    int mlp_dim;        // MLP intermediate width
    int num_components; // ablatable (heads + 1 MLP)
    int vocab_size;     // vocab for output projection
};

/// A single causal intervention: ablate component
/// `component_idx` at token position `token_pos`.
/// component_idx in [0, num_heads) = attention head;
/// component_idx == num_heads = MLP block.
struct Intervention {
    int component_idx;  // which component to ablate
    int token_pos;      // which token position
};

/// Result produced for one intervention.
struct AblationResult {
    int   intervention_id; // index into list
    float causal_score;    // clean - ablated logit
};

/// Holds all GPU-resident model weights (device ptrs).
struct ModelWeights {
    float* W_Q;     // query   [E x E]
    float* W_K;     // key     [E x E]
    float* W_V;     // value   [E x E]
    float* W_O;     // output  [E x E]
    float* W_mlp1;  // MLP L1  [E x M]
    float* W_mlp2;  // MLP L2  [M x E]
    float* W_out;   // logit   [E x V]
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
