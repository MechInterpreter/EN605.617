#include "transformer.h"
#include "../include/cuda_utils.h"
#include <cmath>
#include <cstdio>

// Simple ReLU kernel (used after MLP first layer)
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// Initialize one weight buffer with deterministic values
static void init_weight_buffer(
    float* d_ptr, int count,
    float scale, int seed_off
) {
    std::vector<float> h(count);
    for (int i = 0; i < count; i++) {
        h[i] = scale * sinf(
            (float)(i + seed_off) * 0.01f
        );
    }
    CUDA_CHECK(cudaMemcpy(
        d_ptr, h.data(),
        count * sizeof(float),
        cudaMemcpyHostToDevice));
}

// Allocate and init model weights on GPU
void allocate_model_weights(
    const ModelConfig& cfg, ModelWeights& w
) {
    int E = cfg.embed_dim;
    int M = cfg.mlp_dim;
    int V = cfg.vocab_size;

    CUDA_CHECK(cudaMalloc(&w.W_Q,    E*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_K,    E*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_V,    E*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_O,    E*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_mlp1, E*M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_mlp2, M*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_out,  E*V*sizeof(float)));

    float s = 1.0f / sqrtf((float)E);
    float sm = 1.0f / sqrtf((float)M);
    float sv = 1.0f / sqrtf((float)V);
    init_weight_buffer(w.W_Q,    E*E, s,  1);
    init_weight_buffer(w.W_K,    E*E, s,  2);
    init_weight_buffer(w.W_V,    E*E, s,  3);
    init_weight_buffer(w.W_O,    E*E, s,  4);
    init_weight_buffer(w.W_mlp1, E*M, sm, 5);
    init_weight_buffer(w.W_mlp2, M*E, sm, 6);
    init_weight_buffer(w.W_out,  E*V, sv, 7);
}

void free_model_weights(ModelWeights& w) {
    cudaFree(w.W_Q);
    cudaFree(w.W_K);
    cudaFree(w.W_V);
    cudaFree(w.W_O);
    cudaFree(w.W_mlp1);
    cudaFree(w.W_mlp2);
    cudaFree(w.W_out);
}

// Generate deterministic input activations on GPU
void generate_input_activations(
    float* d_input, int seq_len, int embed_dim
) {
    int n = seq_len * embed_dim;
    std::vector<float> h(n);
    for (int i = 0; i < n; i++) {
        h[i] = 0.1f * cosf((float)i * 0.007f);
    }
    CUDA_CHECK(cudaMemcpy(
        d_input, h.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice));
}

// Multi-head attention: Q/K/V projections, output
// projection, and add to residual stream.
// Uses cublasSgemm for projections and cublasSgeam
// for the residual addition.
static void compute_attention(
    cublasHandle_t handle, int S, int E,
    const ModelWeights& w,
    const float* d_in, float* d_residual,
    float* d_Q, float* d_K, float* d_V,
    float* d_attn_out, float* d_head_outs
) {
    float alpha = 1.0f, beta = 0.0f;

    // Q = input x W_Q  [S*E] x [E*E] = [S*E]
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_Q, E, d_in, E, &beta, d_Q, E));

    // K = input x W_K
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_K, E, d_in, E, &beta, d_K, E));

    // V = input x W_V
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_V, E, d_in, E, &beta, d_V, E));

    // attn_out = V x W_O  [S*E] x [E*E] = [S*E]
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        w.W_O, E, d_V, E,
        &beta, d_attn_out, E));

    // Store attention output for ablation
    CUDA_CHECK(cudaMemcpy(
        d_head_outs, d_attn_out,
        S * E * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // residual += attn_out (cublasSgeam)
    alpha = 1.0f;
    CUBLAS_CHECK(cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S,
        &alpha, d_residual, E,
        &alpha, d_attn_out, E,
        d_residual, E));
}

// MLP: two-layer feedforward with ReLU, add to residual.
// Uses cublasSgemm for both layers and cublasSgeam
// for the residual addition.
static void compute_mlp(
    cublasHandle_t handle, int S, int E, int M,
    const ModelWeights& w,
    float* d_residual,
    float* d_mlp_hidden, float* d_mlp_out
) {
    float alpha = 1.0f, beta = 0.0f;

    // mlp_hidden = residual x W_mlp1
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, S, E, &alpha,
        w.W_mlp1, M, d_residual, E,
        &beta, d_mlp_hidden, M));

    // ReLU activation
    int n_mlp = S * M;
    relu_kernel<<<(n_mlp+255)/256, 256>>>(
        d_mlp_hidden, n_mlp);

    // mlp_out = mlp_hidden x W_mlp2
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, M, &alpha,
        w.W_mlp2, E, d_mlp_hidden, M,
        &beta, d_mlp_out, E));

    // residual += mlp_out
    alpha = 1.0f; beta = 1.0f;
    CUBLAS_CHECK(cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S,
        &alpha, d_residual, E,
        &alpha, d_mlp_out, E,
        d_residual, E));
}

// Output projection: residual[target_token] -> logit.
// Uses cublasSgemv for the single-vector projection.
static void project_output_logit(
    cublasHandle_t handle, int E, int V,
    const ModelWeights& w,
    const float* d_residual, int target_token,
    float* d_logits, float* d_logit_out
) {
    float alpha = 1.0f, beta = 0.0f;
    int offset = target_token * E;

    CUBLAS_CHECK(cublasSgemv(
        handle, CUBLAS_OP_T,
        E, V, &alpha,
        w.W_out, E,
        d_residual + offset, 1,
        &beta, d_logits, 1));

    // Copy the first logit as target
    CUDA_CHECK(cudaMemcpy(
        d_logit_out, d_logits,
        sizeof(float),
        cudaMemcpyDeviceToDevice));
}

// Run a single transformer forward pass.
// Returns the logit for target_token in *d_logit_out.
void transformer_forward(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float* d_residual, float* d_Q,
    float* d_K, float* d_V,
    float* d_attn_out, float* d_mlp_hidden,
    float* d_mlp_out, float* d_logits,
    float* d_head_outs,
    int target_token,
    float* d_logit_out
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;

    // Copy input into residual stream
    CUDA_CHECK(cudaMemcpy(
        d_residual, d_input,
        S * E * sizeof(float),
        cudaMemcpyDeviceToDevice));

    compute_attention(
        handle, S, E, w, d_input,
        d_residual, d_Q, d_K, d_V,
        d_attn_out, d_head_outs);

    compute_mlp(
        handle, S, E, cfg.mlp_dim, w,
        d_residual, d_mlp_hidden, d_mlp_out);

    project_output_logit(
        handle, E, cfg.vocab_size, w,
        d_residual, target_token,
        d_logits, d_logit_out);
}

// Allocate scratch buffers for a single forward pass
void allocate_forward_scratch(
    const ModelConfig& cfg,
    float** d_res, float** d_Q, float** d_K,
    float** d_V, float** d_attn,
    float** d_mlp_h, float** d_mlp_o,
    float** d_logits, float** d_heads
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int M = cfg.mlp_dim;
    int V = cfg.vocab_size;
    CUDA_CHECK(cudaMalloc(d_res,   S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_Q,     S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_K,     S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_V,     S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_attn,  S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_mlp_h, S*M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_mlp_o, S*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_logits, V*sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_heads, S*E*sizeof(float)));
}

void free_forward_scratch(
    float* d_res, float* d_Q, float* d_K,
    float* d_V, float* d_attn,
    float* d_mlp_h, float* d_mlp_o,
    float* d_logits, float* d_heads
) {
    cudaFree(d_res);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_attn);
    cudaFree(d_mlp_h);
    cudaFree(d_mlp_o);
    cudaFree(d_logits);
    cudaFree(d_heads);
}
