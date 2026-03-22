#include "baseline.h"
#include "transformer.h"
#include "ablation.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"

#include <thrust/device_vector.h>
#include <cstdio>
#include <vector>

// Process one intervention: copy input, mask, forward.
// Returns the ablated logit.
static float run_single_intervention(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& weights,
    const float* d_input,
    const std::vector<Intervention>& ivs,
    int iv_idx,
    float* d_work, int slice,
    float* d_res, float* d_Q, float* d_K,
    float* d_V, float* d_attn,
    float* d_mlp_h, float* d_mlp_o,
    float* d_logits, float* d_heads,
    float* d_logit_out
) {
    // Copy clean input
    CUDA_CHECK(cudaMemcpy(
        d_work, d_input,
        slice * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // Generate + apply mask using Thrust
    thrust::device_vector<float> d_mask;
    generate_ablation_masks(
        cfg, ivs, iv_idx, 1, d_mask);

    thrust::device_ptr<float> dp(d_work);
    thrust::transform(
        dp, dp + slice,
        d_mask.begin(), dp,
        thrust::multiplies<float>());

    // Forward pass with ablated input
    transformer_forward(
        handle, cfg, weights, d_work,
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits,
        d_heads, 0, d_logit_out);

    // Copy logit back
    float ablated;
    CUDA_CHECK(cudaMemcpy(
        &ablated, d_logit_out,
        sizeof(float),
        cudaMemcpyDeviceToHost));
    return ablated;
}

// Run sequential baseline over all interventions.
void run_sequential_baseline(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& weights,
    const float* d_input,
    float clean_logit,
    const std::vector<Intervention>& ivs,
    std::vector<float>& out_scores,
    float* elapsed_ms
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int num_iv = (int)ivs.size();
    int slice  = S * E;

    out_scores.resize(num_iv);

    // Allocate scratch for one forward pass
    float *d_res, *d_Q, *d_K, *d_V;
    float *d_attn, *d_mlp_h, *d_mlp_o;
    float *d_logits, *d_heads;
    allocate_forward_scratch(
        cfg, &d_res, &d_Q, &d_K, &d_V,
        &d_attn, &d_mlp_h, &d_mlp_o,
        &d_logits, &d_heads);

    float* d_work;
    CUDA_CHECK(cudaMalloc(
        &d_work, slice * sizeof(float)));
    float* d_logit_out;
    CUDA_CHECK(cudaMalloc(
        &d_logit_out, sizeof(float)));

    GpuTimer timer;
    timer.tic();

    for (int i = 0; i < num_iv; i++) {
        float ablated = run_single_intervention(
            handle, cfg, weights, d_input,
            ivs, i, d_work, slice,
            d_res, d_Q, d_K, d_V, d_attn,
            d_mlp_h, d_mlp_o, d_logits,
            d_heads, d_logit_out);
        out_scores[i] = clean_logit - ablated;
    }

    *elapsed_ms = timer.toc();

    cudaFree(d_work);
    cudaFree(d_logit_out);
    free_forward_scratch(
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits, d_heads);
}
