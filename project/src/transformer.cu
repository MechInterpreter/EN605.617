#include "transformer.h"
#include "../include/cuda_utils.h"
#include <cmath>
#include <cstdio>

// -----------------------------------------------
// CUDA kernels
// -----------------------------------------------

// RMSNorm: y = x / rms(x) * weight
// Each block handles one token (row of length E).
__global__ void rmsnorm_kernel(
    float* out, const float* inp,
    const float* weight,
    int E, float eps
) {
    int tok = blockIdx.x;
    const float* x = inp + tok * E;
    float* y = out + tok * E;

    // Compute mean of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    // Warp reduction
    for (int offset = warpSize / 2;
         offset > 0; offset >>= 1) {
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    }
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        shared_ss = rsqrtf(ss / (float)E + eps);
    }
    __syncthreads();
    float scale = shared_ss;

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        y[i] = x[i] * scale * weight[i];
    }
}

// GeLU activation (approximate: tanh version)
__global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (x + 0.044715f * x*x*x);
        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Scaled dot-product attention with causal mask.
// Q, K, V: [S x E],  output: [S x E].
// Operates per-head: head_dim = E / num_heads.
// One block per (token, head) pair.
__global__ void sdpa_kernel(
    const float* Q, const float* K,
    const float* V, float* out,
    int S, int E, int num_heads
) {
    int head = blockIdx.y;
    int query = blockIdx.x;
    int hd = E / num_heads;
    int h_off = head * hd;

    // Compute attention scores for this query
    // against all keys up to query (causal).
    extern __shared__ float smem[];
    float* scores = smem;  // [S]

    float max_val = -1e30f;
    for (int k = threadIdx.x; k <= query;
         k += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) {
            dot += Q[query * E + h_off + d]
                 * K[k * E + h_off + d];
        }
        dot /= sqrtf((float)hd);
        scores[k] = dot;
        if (dot > max_val) max_val = dot;
    }
    // Fill future positions with -inf
    for (int k = query + 1 + threadIdx.x;
         k < S; k += blockDim.x) {
        scores[k] = -1e30f;
    }
    __syncthreads();

    // Reduce max across threads
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int k = 0; k < S; k++) {
            if (scores[k] > m) m = scores[k];
        }
        s_max = m;
    }
    __syncthreads();

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    for (int k = threadIdx.x; k < S;
         k += blockDim.x) {
        scores[k] = expf(scores[k] - s_max);
        sum_exp += scores[k];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < S; k++) s += scores[k];
        s_sum = s;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < S;
         k += blockDim.x) {
        scores[k] /= s_sum;
    }
    __syncthreads();

    // Weighted sum of values
    if (threadIdx.x == 0) {
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int k = 0; k <= query; k++) {
                acc += scores[k]
                     * V[k * E + h_off + d];
            }
            out[query * E + h_off + d] = acc;
        }
    }
}

// Element-wise mask application kernel.
// Multiplies activation[i] by mask[i].
__global__ void apply_mask_kernel(
    float* data, const float* mask, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= mask[idx];
    }
}

// -----------------------------------------------
// Weight initialization
// -----------------------------------------------

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

// Initialize RMSNorm weights to 1.0
static void init_norm_weights(
    float* d_ptr, int count
) {
    std::vector<float> h(count, 1.0f);
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
    int L = cfg.num_layers;

    w.layers.resize(L);

    for (int l = 0; l < L; l++) {
        LayerWeights& lw = w.layers[l];
        float s = 1.0f / sqrtf((float)E);
        float sm = 1.0f / sqrtf((float)M);
        int base = l * 100;  // offset per layer

        CUDA_CHECK(cudaMalloc(
            &lw.W_Q, E*E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.W_K, E*E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.W_V, E*E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.W_O, E*E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.W_mlp1, E*M*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.W_mlp2, M*E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.rms_attn, E*sizeof(float)));
        CUDA_CHECK(cudaMalloc(
            &lw.rms_mlp, E*sizeof(float)));

        init_weight_buffer(
            lw.W_Q, E*E, s, base + 1);
        init_weight_buffer(
            lw.W_K, E*E, s, base + 2);
        init_weight_buffer(
            lw.W_V, E*E, s, base + 3);
        init_weight_buffer(
            lw.W_O, E*E, s, base + 4);
        init_weight_buffer(
            lw.W_mlp1, E*M, sm, base + 5);
        init_weight_buffer(
            lw.W_mlp2, M*E, sm, base + 6);
        init_norm_weights(lw.rms_attn, E);
        init_norm_weights(lw.rms_mlp, E);
    }

    // Final output projection + norm
    float sv = 1.0f / sqrtf((float)V);
    CUDA_CHECK(cudaMalloc(
        &w.W_out, E*V*sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        &w.rms_final, E*sizeof(float)));
    init_weight_buffer(
        w.W_out, E*V, sv, L * 100 + 7);
    init_norm_weights(w.rms_final, E);
}

void free_model_weights(ModelWeights& w) {
    for (size_t l = 0; l < w.layers.size(); l++) {
        LayerWeights& lw = w.layers[l];
        cudaFree(lw.W_Q);
        cudaFree(lw.W_K);
        cudaFree(lw.W_V);
        cudaFree(lw.W_O);
        cudaFree(lw.W_mlp1);
        cudaFree(lw.W_mlp2);
        cudaFree(lw.rms_attn);
        cudaFree(lw.rms_mlp);
    }
    cudaFree(w.W_out);
    cudaFree(w.rms_final);
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

// -----------------------------------------------
// Sub-block helpers
// -----------------------------------------------

// Multi-head attention: RMSNorm -> Q/K/V projections ->
// scaled dot-product attention -> output projection ->
// add to residual stream.
static void compute_attention(
    cublasHandle_t handle, int S, int E,
    const LayerWeights& lw,
    const float* d_in, float* d_residual,
    float* d_Q, float* d_K, float* d_V,
    float* d_attn_out, float* d_head_outs,
    float norm_eps, int num_heads
) {
    float alpha = 1.0f, beta = 0.0f;

    // RMSNorm into d_attn_out as scratch
    int threads = min(E, 256);
    rmsnorm_kernel<<<S, threads>>>(
        d_attn_out, d_residual,
        lw.rms_attn, E, norm_eps);

    // Q = norm(x) x W_Q  [S*E] x [E*E] = [S*E]
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        lw.W_Q, E, d_attn_out, E,
        &beta, d_Q, E));

    // K = norm(x) x W_K
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        lw.W_K, E, d_attn_out, E,
        &beta, d_K, E));

    // V = norm(x) x W_V
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        lw.W_V, E, d_attn_out, E,
        &beta, d_V, E));

    // Scaled dot-product attention (causal)
    int smem = S * sizeof(float);
    dim3 grid(S, num_heads);
    sdpa_kernel<<<grid, 32, smem>>>(
        d_Q, d_K, d_V, d_head_outs,
        S, E, num_heads);

    // Output projection: heads -> d_attn_out
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, E, &alpha,
        lw.W_O, E, d_head_outs, E,
        &beta, d_attn_out, E));

    // residual += attn_out (cublasSgeam)
    alpha = 1.0f;
    CUBLAS_CHECK(cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S,
        &alpha, d_residual, E,
        &alpha, d_attn_out, E,
        d_residual, E));
}

// MLP: RMSNorm -> two-layer feedforward with GeLU ->
// add to residual.
static void compute_mlp(
    cublasHandle_t handle, int S, int E, int M,
    const LayerWeights& lw,
    float* d_residual,
    float* d_mlp_hidden, float* d_mlp_out,
    float norm_eps
) {
    float alpha = 1.0f, beta = 0.0f;

    // RMSNorm into d_mlp_out as scratch
    int threads = min(E, 256);
    rmsnorm_kernel<<<S, threads>>>(
        d_mlp_out, d_residual,
        lw.rms_mlp, E, norm_eps);

    // mlp_hidden = norm(x) x W_mlp1
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, S, E, &alpha,
        lw.W_mlp1, M, d_mlp_out, E,
        &beta, d_mlp_hidden, M));

    // GeLU activation
    int n_mlp = S * M;
    gelu_kernel<<<(n_mlp+255)/256, 256>>>(
        d_mlp_hidden, n_mlp);

    // mlp_out = mlp_hidden x W_mlp2
    CUBLAS_CHECK(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S, M, &alpha,
        lw.W_mlp2, E, d_mlp_hidden, M,
        &beta, d_mlp_out, E));

    // residual += mlp_out
    alpha = 1.0f;
    CUBLAS_CHECK(cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        E, S,
        &alpha, d_residual, E,
        &alpha, d_mlp_out, E,
        d_residual, E));
}

// Output projection: final RMSNorm ->
// residual[target_token] -> logit.
static void project_output_logit(
    cublasHandle_t handle, int E, int V, int S,
    const ModelWeights& w,
    float* d_residual, float* d_scratch,
    int target_token,
    float* d_logits, float* d_logit_out,
    float norm_eps
) {
    // Final RMSNorm
    int threads = min(E, 256);
    rmsnorm_kernel<<<S, threads>>>(
        d_scratch, d_residual,
        w.rms_final, E, norm_eps);

    float alpha = 1.0f, beta = 0.0f;
    int offset = target_token * E;

    CUBLAS_CHECK(cublasSgemv(
        handle, CUBLAS_OP_T,
        E, V, &alpha,
        w.W_out, E,
        d_scratch + offset, 1,
        &beta, d_logits, 1));

    // Copy the first logit as target
    CUDA_CHECK(cudaMemcpy(
        d_logit_out, d_logits,
        sizeof(float),
        cudaMemcpyDeviceToDevice));
}

// -----------------------------------------------
// Multi-layer forward pass
// -----------------------------------------------

// Run multi-layer transformer forward pass.
// If intervention_layer >= 0, apply the mask at
// that layer after the specified sub-block.
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
    float* d_logit_out,
    int intervention_layer,
    InterventionType intervention_type,
    const float* d_mask
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int slice = S * E;

    // Copy input into residual stream
    CUDA_CHECK(cudaMemcpy(
        d_residual, d_input,
        slice * sizeof(float),
        cudaMemcpyDeviceToDevice));

    // Process each transformer layer
    for (int l = 0; l < cfg.num_layers; l++) {
        const LayerWeights& lw = w.layers[l];

        compute_attention(
            handle, S, E, lw, d_input,
            d_residual, d_Q, d_K, d_V,
            d_attn_out, d_head_outs,
            cfg.norm_eps, cfg.num_heads);

        // Apply intervention after attention
        if (d_mask && l == intervention_layer
            && (intervention_type == IV_ATTN_HEAD
             || intervention_type == IV_RESID_STREAM)) {
            apply_mask_kernel
                <<<(slice+255)/256, 256>>>(
                    d_residual, d_mask, slice);
        }

        compute_mlp(
            handle, S, E, cfg.mlp_dim, lw,
            d_residual, d_mlp_hidden, d_mlp_out,
            cfg.norm_eps);

        // Apply intervention after MLP
        if (d_mask && l == intervention_layer
            && intervention_type == IV_MLP_OUT) {
            apply_mask_kernel
                <<<(slice+255)/256, 256>>>(
                    d_residual, d_mask, slice);
        }
    }

    project_output_logit(
        handle, E, cfg.vocab_size, S, w,
        d_residual, d_attn_out,
        target_token,
        d_logits, d_logit_out,
        cfg.norm_eps);
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
