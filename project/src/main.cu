#include "transformer.h"
#include "baseline.h"
#include "batched_engine.h"
#include "benchmark.h"
#include "validation.h"
#include "ablation.h"
#include "weight_io.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"
#include "../include/types.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// Print GPU info and application banner
static void print_banner() {
    printf("===================================="
           "================================\n");
    printf("  CUDA Batched Causal Ablation "
           "Engine\n");
    printf("  EN605.617 — GPU Computing\n");
    printf("  Libraries: cuBLAS + Thrust\n");
    printf("===================================="
           "================================\n");

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(
        cudaGetDeviceProperties(&prop, device));
    printf("  GPU: %s\n", prop.name);
    printf("  Compute: %d.%d\n",
           prop.major, prop.minor);
    printf("  SMs: %d  |  Mem: %.1f GB\n",
           prop.multiProcessorCount,
           (float)prop.totalGlobalMem
               / (1024.0f*1024.0f*1024.0f));
    printf("===================================="
           "================================\n");
}

// Print model configuration
static void print_config(const ModelConfig& cfg) {
    printf("\n  Model Configuration:\n");
    printf("    embed_dim      = %d\n",
           cfg.embed_dim);
    printf("    num_heads      = %d\n",
           cfg.num_heads);
    printf("    head_dim       = %d\n",
           cfg.head_dim);
    printf("    seq_len        = %d\n",
           cfg.seq_len);
    printf("    mlp_dim        = %d\n",
           cfg.mlp_dim);
    printf("    num_layers     = %d\n",
           cfg.num_layers);
    printf("    num_components = %d + 1 = %d\n",
           cfg.num_heads, cfg.num_components);
    printf("    vocab_size     = %d\n",
           cfg.vocab_size);
    printf("    norm_eps       = %.1e\n",
           cfg.norm_eps);
    printf("\n");
}

// Compute clean (un-ablated) logit for target token
static float compute_clean_logit(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input
) {
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

    transformer_forward(
        handle, cfg, w, d_input,
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits,
        d_heads, 0, d_logit_out);

    float clean;
    CUDA_CHECK(cudaMemcpy(
        &clean, d_logit_out,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    cudaFree(d_logit_out);
    free_forward_scratch(
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits, d_heads);
    return clean;
}

// Print usage
static void print_usage(const char* prog) {
    printf("Usage: %s [mode] [options]\n", prog);
    printf("Modes:\n");
    printf("  --baseline   Sequential (64 ivs)\n");
    printf("  --batched    Batched (64, bs=32)\n");
    printf("  --benchmark  Full sweep\n");
    printf("  --validate   Correctness check\n");
    printf("  --all        validate + benchmark\n");
    printf("Options:\n");
    printf("  --weights <dir>  Load weights from "
           "directory\n");
    printf("  --layers <N>     Num transformer "
           "layers\n");
}

// Run baseline mode
static void run_baseline_mode(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit
) {
    int num_iv = 64;
    std::vector<Intervention> ivs;
    generate_interventions(cfg, num_iv, ivs);

    std::vector<float> scores;
    float ms;
    run_sequential_baseline(
        handle, cfg, w, d_input,
        clean_logit, ivs, scores, &ms);

    float ips = num_iv / (ms * 1e-3f);
    printf("Sequential: %d ivs in %.2f ms "
           "(%.1f iv/sec)\n", num_iv, ms, ips);

    int top = std::min(10, num_iv);
    printf("\nFirst %d causal scores:\n", top);
    for (int i = 0; i < top; i++) {
        printf("  [%d] L%d comp=%d tok=%d "
               "sc=%.6f\n",
               i, ivs[i].layer_idx,
               ivs[i].component_idx,
               ivs[i].token_pos, scores[i]);
    }
}

// Run batched mode
static void run_batched_mode(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit
) {
    int num_iv = 64, bs = 32;
    std::vector<Intervention> ivs;
    generate_interventions(cfg, num_iv, ivs);

    std::vector<float> scores;
    float ms;
    run_batched_engine(
        handle, cfg, w, d_input,
        clean_logit, ivs, bs, scores, &ms);

    float ips = num_iv / (ms * 1e-3f);
    printf("Batched: %d ivs (bs=%d) in %.2f ms "
           "(%.1f iv/sec)\n",
           num_iv, bs, ms, ips);

    int top = std::min(10, num_iv);
    printf("\nFirst %d causal scores:\n", top);
    for (int i = 0; i < top; i++) {
        printf("  [%d] L%d comp=%d tok=%d "
               "sc=%.6f\n",
               i, ivs[i].layer_idx,
               ivs[i].component_idx,
               ivs[i].token_pos, scores[i]);
    }
}

// Initialize model: config, weights, input, clean logit
static float setup_model(
    cublasHandle_t handle,
    ModelConfig& cfg,
    ModelWeights& w,
    float** d_input,
    const char* weights_dir,
    int num_layers
) {
    cfg.embed_dim      = DEFAULT_EMBED_DIM;
    cfg.num_heads      = DEFAULT_NUM_HEADS;
    cfg.head_dim       = DEFAULT_HEAD_DIM;
    cfg.seq_len        = DEFAULT_SEQ_LEN;
    cfg.mlp_dim        = DEFAULT_MLP_DIM;
    cfg.num_components = DEFAULT_NUM_COMPONENTS;
    cfg.vocab_size     = DEFAULT_VOCAB_SIZE;
    cfg.num_layers     = num_layers;
    cfg.norm_eps       = DEFAULT_NORM_EPS;
    print_config(cfg);

    bool loaded = false;
    if (weights_dir) {
        printf("  Loading weights from %s...\n",
               weights_dir);
        loaded = load_model_weights_from_dir(
            weights_dir, cfg, w);
        if (!loaded) {
            printf("  Failed to load weights, "
                   "falling back to synthetic.\n");
        }
    }

    if (!loaded) {
        printf("  Initializing synthetic "
               "weights...\n");
        allocate_model_weights(cfg, w);
    }

    int n = cfg.seq_len * cfg.embed_dim;
    CUDA_CHECK(cudaMalloc(
        d_input, n * sizeof(float)));
    generate_input_activations(
        *d_input, cfg.seq_len, cfg.embed_dim);

    float cl = compute_clean_logit(
        handle, cfg, w, *d_input);
    printf("  Clean logit (tok 0): %.6f\n\n", cl);
    return cl;
}

// Main
int main(int argc, char** argv) {
    enum Mode {
        BASELINE, BATCHED, BENCHMARK,
        VALIDATE, ALL
    };
    Mode mode = ALL;
    const char* weights_dir = nullptr;
    int num_layers = DEFAULT_NUM_LAYERS;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!strcmp(a, "--baseline")) {
            mode = BASELINE;
        } else if (!strcmp(a, "--batched")) {
            mode = BATCHED;
        } else if (!strcmp(a, "--benchmark")) {
            mode = BENCHMARK;
        } else if (!strcmp(a, "--validate")) {
            mode = VALIDATE;
        } else if (!strcmp(a, "--all")) {
            mode = ALL;
        } else if (!strcmp(a, "--weights")
                   && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (!strcmp(a, "--layers")
                   && i + 1 < argc) {
            num_layers = atoi(argv[++i]);
        } else if (!strcmp(a, "-h") ||
                 !strcmp(a, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr,
                "Unknown option: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
    }

    print_banner();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    ModelConfig cfg;
    ModelWeights w;
    float* d_input;
    float cl = setup_model(
        handle, cfg, w, &d_input,
        weights_dir, num_layers);

    switch (mode) {
    case BASELINE:
        run_baseline_mode(
            handle, cfg, w, d_input, cl);
        break;
    case BATCHED:
        run_batched_mode(
            handle, cfg, w, d_input, cl);
        break;
    case BENCHMARK:
        run_benchmark(
            handle, cfg, w, d_input, cl);
        break;
    case VALIDATE:
        run_validation(
            handle, cfg, w, d_input,
            cl, 64, 32);
        break;
    case ALL:
        run_validation(
            handle, cfg, w, d_input,
            cl, 64, 32);
        run_benchmark(
            handle, cfg, w, d_input, cl);
        break;
    }

    cublasDestroy(handle);
    cudaFree(d_input);
    free_model_weights(w);
    CUDA_CHECK(cudaDeviceReset());

    printf("Done.\n");
    return 0;
}
