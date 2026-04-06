#include "batched_engine.h"
#include "transformer.h"
#include "ablation.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// Process one micro-batch of interventions.
// For each intervention, runs a full multi-layer
// forward pass with the mask applied at the
// target layer.
static void process_micro_batch(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    const std::vector<Intervention>& ivs,
    int start, int cur_batch,
    std::vector<float>& out_scores
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;

    // Allocate scratch for one forward pass
    float *d_res, *d_Q, *d_K, *d_V;
    float *d_attn, *d_mlp_h, *d_mlp_o;
    float *d_logits, *d_heads;
    allocate_forward_scratch(
        cfg, &d_res, &d_Q, &d_K, &d_V,
        &d_attn, &d_mlp_h, &d_mlp_o,
        &d_logits, &d_heads);

    float* d_logit_out;
    CUDA_CHECK(cudaMalloc(
        &d_logit_out, sizeof(float)));

    std::vector<float> h_logits(cur_batch);

    for (int b = 0; b < cur_batch; b++) {
        int iv_idx = start + b;
        const Intervention& iv = ivs[iv_idx];

        // Generate mask for this intervention
        thrust::device_vector<float> d_mask;
        generate_ablation_masks(
            cfg, ivs, iv_idx, 1, d_mask);

        const float* mask_ptr =
            thrust::raw_pointer_cast(d_mask.data());

        // Forward pass with per-layer intervention
        transformer_forward(
            handle, cfg, w, d_input,
            d_res, d_Q, d_K, d_V, d_attn,
            d_mlp_h, d_mlp_o, d_logits,
            d_heads, 0, d_logit_out,
            iv.layer_idx, iv.type, mask_ptr);

        CUDA_CHECK(cudaMemcpy(
            &h_logits[b], d_logit_out,
            sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    // Thrust: compute causal scores
    thrust::device_vector<float> d_abl(
        h_logits.begin(), h_logits.end());
    thrust::device_vector<float> d_sc;
    compute_causal_scores(
        clean_logit, d_abl, d_sc, cur_batch);
    thrust::copy(
        d_sc.begin(), d_sc.end(),
        out_scores.begin() + start);

    cudaFree(d_logit_out);
    free_forward_scratch(
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits, d_heads);
}

// Run the batched ablation engine.
// Processes interventions in micro-batches.
void run_batched_engine(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    const std::vector<Intervention>& ivs,
    int batch_size,
    std::vector<float>& out_scores,
    float* elapsed_ms
) {
    int num_iv = (int)ivs.size();
    batch_size = std::min(batch_size, num_iv);
    out_scores.resize(num_iv);

    GpuTimer timer;
    timer.tic();

    for (int st = 0; st < num_iv;
         st += batch_size) {
        int cur = std::min(
            batch_size, num_iv - st);
        process_micro_batch(
            handle, cfg, w, d_input,
            clean_logit, ivs, st, cur,
            out_scores);
    }

    *elapsed_ms = timer.toc();
}
