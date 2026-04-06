#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../include/types.h"
#include "../include/cuda_utils.h"
#include <cublas_v2.h>

// Allocate and initialize model weights on GPU
// (synthetic deterministic init when no file loaded)
void allocate_model_weights(
    const ModelConfig& cfg, ModelWeights& w);
void free_model_weights(ModelWeights& w);

// Generate deterministic input activations
void generate_input_activations(
    float* d_input, int seq_len, int embed_dim);

// Run multi-layer transformer forward pass.
// intervention_layer / intervention_mask: if >= 0,
// apply the mask at that layer after the specified
// sub-block (attn or mlp).  Pass -1 to skip.
void transformer_forward(
    cublasHandle_t  handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float*   d_input,
    float*         d_residual,
    float*         d_Q,
    float*         d_K,
    float*         d_V,
    float*         d_attn_out,
    float*         d_mlp_hidden,
    float*         d_mlp_out,
    float*         d_logits,
    float*         d_head_outs,
    int            target_token,
    float*         d_logit_out,
    int            intervention_layer = -1,
    InterventionType intervention_type
                       = IV_ATTN_HEAD,
    const float*   d_mask = nullptr
);

// Scratch buffer management (single forward pass)
void allocate_forward_scratch(
    const ModelConfig& cfg,
    float** d_residual, float** d_Q,
    float** d_K, float** d_V,
    float** d_attn_out, float** d_mlp_hidden,
    float** d_mlp_out, float** d_logits,
    float** d_head_outs);

void free_forward_scratch(
    float* d_residual, float* d_Q,
    float* d_K, float* d_V,
    float* d_attn_out, float* d_mlp_hidden,
    float* d_mlp_out, float* d_logits,
    float* d_head_outs);

#endif // TRANSFORMER_H
