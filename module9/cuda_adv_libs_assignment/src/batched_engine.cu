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

// ReLU kernel for batched activations
__global__ void batched_relu_kernel(
    float* data, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// Scratch buffers for batched forward pass
struct BatchScratch {
    float *Q, *K, *V, *attn, *res;
    float *mlp_h, *mlp_o, *logits;
};

static void alloc_batch_scratch(
    BatchScratch& s, int bs, int S,
    int E, int M, int V
) {
    int se = bs * S * E;
    CUDA_CHECK(cudaMalloc(&s.Q,      se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.K,      se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.V,      se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.attn,   se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.res,    se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &s.mlp_h, bs*S*M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &s.mlp_o, se*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &s.logits, bs*V*sizeof(float)));
}

static void free_batch_scratch(BatchScratch& s) {
    cudaFree(s.Q);     cudaFree(s.K);
    cudaFree(s.V);     cudaFree(s.attn);
    cudaFree(s.res);   cudaFree(s.mlp_h);
    cudaFree(s.mlp_o); cudaFree(s.logits);
}

// Batched Q/K/V/O projections via SgemmStridedBatched.
// strideA=0 broadcasts the shared weight matrix.
static void batched_qkvo(
    cublasHandle_t handle,
    const ModelWeights& w,
    const float* d_in, int E, int S,
    int slice, int n,
    BatchScratch& sc
) {
    float alpha = 1.0f, beta = 0.0f;

    // Q = input x W_Q
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_Q, E, 0,
        d_in, E, slice,
        &beta, sc.Q, E, slice, n));

    // K = input x W_K
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_K, E, 0,
        d_in, E, slice,
        &beta, sc.K, E, slice, n));

    // V = input x W_V
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_V, E, 0,
        d_in, E, slice,
        &beta, sc.V, E, slice, n));

    // attn_out = V x W_O
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_O, E, 0,
        sc.V, E, slice,
        &beta, sc.attn, E, slice, n));
}

// Add attention output to residual using cublasSgeam.
static void batched_residual_add(
    cublasHandle_t handle,
    float* d_dst, const float* d_src,
    int E, int S, int n
) {
    float one = 1.0f;
    CUBLAS_CHECK(cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E * S, n,
        &one, d_dst, E * S,
        &one, d_src, E * S,
        d_dst, E * S));
}

// Batched MLP: two layers + ReLU via SgemmStridedBatched.
static void batched_mlp(
    cublasHandle_t handle,
    const ModelWeights& w,
    int E, int S, int M,
    int slice, int n,
    BatchScratch& sc
) {
    float alpha = 1.0f, beta = 0.0f;

    // mlp_hidden = residual x W_mlp1
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, S, E, &alpha,
        w.W_mlp1, M, 0,
        sc.res, E, slice,
        &beta, sc.mlp_h, M, S*M, n));

    // ReLU
    int nr = n * S * M;
    batched_relu_kernel<<<(nr+255)/256, 256>>>(
        sc.mlp_h, nr);

    // mlp_out = mlp_hidden x W_mlp2
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, M, &alpha,
        w.W_mlp2, E, 0,
        sc.mlp_h, M, S*M,
        &beta, sc.mlp_o, E, slice, n));
}

// Per-element output projection and logit extraction.
static void batched_output_proj(
    cublasHandle_t handle,
    const ModelWeights& w,
    int E, int V, int slice, int n,
    BatchScratch& sc, int target_tok,
    std::vector<float>& h_logits
) {
    float alpha = 1.0f, beta = 0.0f;
    h_logits.resize(n);

    for (int b = 0; b < n; b++) {
        long long off =
            (long long)b * slice + target_tok * E;
        CUBLAS_CHECK(cublasSgemv(
            handle, CUBLAS_OP_T,
            E, V, &alpha,
            w.W_out, E,
            sc.res + off, 1,
            &beta,
            sc.logits + (long long)b * V, 1));
    }

    // Copy first logit dim from each batch element
    for (int b = 0; b < n; b++) {
        CUDA_CHECK(cudaMemcpy(
            &h_logits[b],
            sc.logits + (long long)b * V,
            sizeof(float),
            cudaMemcpyDeviceToHost));
    }
}

// Process one micro-batch of interventions.
static void process_micro_batch(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    const std::vector<Intervention>& ivs,
    int start, int cur_batch,
    BatchScratch& sc,
    thrust::device_vector<float>& d_masks,
    thrust::device_vector<float>& d_batched_in,
    std::vector<float>& out_scores
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int M = cfg.mlp_dim;
    int V = cfg.vocab_size;
    int slice = S * E;

    // Thrust: generate masks + replicate/mask input
    generate_ablation_masks(
        cfg, ivs, start, cur_batch, d_masks);
    replicate_and_mask(
        d_input, slice, cur_batch,
        d_masks, d_batched_in);

    // Copy masked input into residual
    const float* in_ptr =
        thrust::raw_pointer_cast(d_batched_in.data());
    CUDA_CHECK(cudaMemcpy(
        sc.res, in_ptr,
        cur_batch * slice * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // cuBLAS: batched Q/K/V/O projections
    batched_qkvo(
        handle, w, in_ptr, E, S,
        slice, cur_batch, sc);

    // residual += attn_out
    batched_residual_add(
        handle, sc.res, sc.attn,
        E, S, cur_batch);

    // cuBLAS: batched MLP
    batched_mlp(
        handle, w, E, S, M,
        slice, cur_batch, sc);

    // residual += mlp_out
    batched_residual_add(
        handle, sc.res, sc.mlp_o,
        E, S, cur_batch);

    // Output projection + logit extraction
    std::vector<float> h_logits;
    batched_output_proj(
        handle, w, E, V, slice,
        cur_batch, sc, 0, h_logits);

    // Thrust: compute causal scores
    thrust::device_vector<float> d_abl(
        h_logits.begin(), h_logits.end());
    thrust::device_vector<float> d_sc;
    compute_causal_scores(
        clean_logit, d_abl, d_sc, cur_batch);
    thrust::copy(
        d_sc.begin(), d_sc.end(),
        out_scores.begin() + start);
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
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int M = cfg.mlp_dim;
    int V = cfg.vocab_size;
    int num_iv = (int)ivs.size();
    batch_size = std::min(batch_size, num_iv);
    out_scores.resize(num_iv);

    // Allocate scratch
    BatchScratch sc;
    alloc_batch_scratch(sc, batch_size, S, E, M, V);

    thrust::device_vector<float> d_masks;
    thrust::device_vector<float> d_batched_in;

    GpuTimer timer;
    timer.tic();

    for (int st = 0; st < num_iv; st += batch_size) {
        int cur = std::min(batch_size, num_iv - st);
        process_micro_batch(
            handle, cfg, w, d_input,
            clean_logit, ivs, st, cur,
            sc, d_masks, d_batched_in,
            out_scores);
    }

    *elapsed_ms = timer.toc();
    free_batch_scratch(sc);
}
