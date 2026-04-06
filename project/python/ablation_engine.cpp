// pybind11 Python bindings for the CUDA Batched
// Causal Ablation Engine.
//
// Build: pip install -e python/
// Usage: import ablation_engine

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../src/transformer.h"
#include "../src/baseline.h"
#include "../src/batched_engine.h"
#include "../src/ablation.h"
#include "../src/weight_io.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"
#include "../include/types.h"

#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <string>

namespace py = pybind11;

// Persistent engine state for Python interface
struct EngineHandle {
    cublasHandle_t cublas;
    ModelConfig cfg;
    ModelWeights weights;
    float* d_input;
    float clean_logit;
    bool initialized;
};

static EngineHandle g_engine = {
    nullptr, {}, {}, nullptr, 0.0f, false
};

// Initialize engine with config and optional
// weight directory
void engine_init(
    int embed_dim, int num_heads,
    int seq_len, int mlp_dim,
    int vocab_size, int num_layers,
    const std::string& weights_dir
) {
    if (g_engine.initialized) {
        throw std::runtime_error(
            "Engine already initialized. "
            "Call cleanup() first.");
    }

    g_engine.cfg.embed_dim = embed_dim;
    g_engine.cfg.num_heads = num_heads;
    g_engine.cfg.head_dim = embed_dim / num_heads;
    g_engine.cfg.seq_len = seq_len;
    g_engine.cfg.mlp_dim = mlp_dim;
    g_engine.cfg.num_components = num_heads + 1;
    g_engine.cfg.vocab_size = vocab_size;
    g_engine.cfg.num_layers = num_layers;
    g_engine.cfg.norm_eps = DEFAULT_NORM_EPS;

    CUBLAS_CHECK(
        cublasCreate(&g_engine.cublas));

    bool loaded = false;
    if (!weights_dir.empty()) {
        loaded = load_model_weights_from_dir(
            weights_dir.c_str(),
            g_engine.cfg, g_engine.weights);
    }
    if (!loaded) {
        allocate_model_weights(
            g_engine.cfg, g_engine.weights);
    }

    int n = seq_len * embed_dim;
    CUDA_CHECK(cudaMalloc(
        &g_engine.d_input,
        n * sizeof(float)));
    generate_input_activations(
        g_engine.d_input, seq_len, embed_dim);

    // Compute clean logit
    float *d_res, *d_Q, *d_K, *d_V;
    float *d_attn, *d_mlp_h, *d_mlp_o;
    float *d_logits, *d_heads;
    allocate_forward_scratch(
        g_engine.cfg,
        &d_res, &d_Q, &d_K, &d_V,
        &d_attn, &d_mlp_h, &d_mlp_o,
        &d_logits, &d_heads);

    float* d_logit_out;
    CUDA_CHECK(cudaMalloc(
        &d_logit_out, sizeof(float)));

    transformer_forward(
        g_engine.cublas, g_engine.cfg,
        g_engine.weights, g_engine.d_input,
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits,
        d_heads, 0, d_logit_out);

    CUDA_CHECK(cudaMemcpy(
        &g_engine.clean_logit, d_logit_out,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    cudaFree(d_logit_out);
    free_forward_scratch(
        d_res, d_Q, d_K, d_V, d_attn,
        d_mlp_h, d_mlp_o, d_logits, d_heads);

    g_engine.initialized = true;
}

// Run batched ablation and return scores
py::array_t<float> engine_run_ablation(
    int num_interventions, int batch_size
) {
    if (!g_engine.initialized) {
        throw std::runtime_error(
            "Engine not initialized.");
    }

    std::vector<Intervention> ivs;
    generate_interventions(
        g_engine.cfg, num_interventions, ivs);

    std::vector<float> scores;
    float ms;
    run_batched_engine(
        g_engine.cublas, g_engine.cfg,
        g_engine.weights, g_engine.d_input,
        g_engine.clean_logit, ivs,
        batch_size, scores, &ms);

    // Return as numpy array
    auto result = py::array_t<float>(
        num_interventions);
    auto buf = result.mutable_unchecked<1>();
    for (int i = 0; i < num_interventions; i++) {
        buf(i) = scores[i];
    }
    return result;
}

// Run validation (seq vs batched)
bool engine_validate(
    int num_interventions, int batch_size
) {
    if (!g_engine.initialized) {
        throw std::runtime_error(
            "Engine not initialized.");
    }

    std::vector<Intervention> ivs;
    generate_interventions(
        g_engine.cfg, num_interventions, ivs);

    std::vector<float> seq_sc, bat_sc;
    float seq_ms, bat_ms;

    run_sequential_baseline(
        g_engine.cublas, g_engine.cfg,
        g_engine.weights, g_engine.d_input,
        g_engine.clean_logit, ivs,
        seq_sc, &seq_ms);

    run_batched_engine(
        g_engine.cublas, g_engine.cfg,
        g_engine.weights, g_engine.d_input,
        g_engine.clean_logit, ivs,
        batch_size, bat_sc, &bat_ms);

    // Check match
    int mismatches = 0;
    for (int i = 0; i < num_interventions; i++) {
        if (fabsf(seq_sc[i] - bat_sc[i])
            > TOLERANCE) {
            mismatches++;
        }
    }
    return mismatches == 0;
}

// Get clean logit
float engine_clean_logit() {
    if (!g_engine.initialized) {
        throw std::runtime_error(
            "Engine not initialized.");
    }
    return g_engine.clean_logit;
}

// Cleanup
void engine_cleanup() {
    if (!g_engine.initialized) return;
    cublasDestroy(g_engine.cublas);
    cudaFree(g_engine.d_input);
    free_model_weights(g_engine.weights);
    g_engine.initialized = false;
}

PYBIND11_MODULE(ablation_engine, m) {
    m.doc() = "CUDA Batched Causal Ablation Engine";

    m.def("init", &engine_init,
        "Initialize the ablation engine",
        py::arg("embed_dim") = DEFAULT_EMBED_DIM,
        py::arg("num_heads") = DEFAULT_NUM_HEADS,
        py::arg("seq_len") = DEFAULT_SEQ_LEN,
        py::arg("mlp_dim") = DEFAULT_MLP_DIM,
        py::arg("vocab_size") = DEFAULT_VOCAB_SIZE,
        py::arg("num_layers") = DEFAULT_NUM_LAYERS,
        py::arg("weights_dir") = "");

    m.def("run_ablation",
        &engine_run_ablation,
        "Run batched ablation",
        py::arg("num_interventions") = 64,
        py::arg("batch_size") = 32);

    m.def("validate", &engine_validate,
        "Validate seq vs batched",
        py::arg("num_interventions") = 64,
        py::arg("batch_size") = 32);

    m.def("clean_logit", &engine_clean_logit,
        "Get clean (un-ablated) logit");

    m.def("cleanup", &engine_cleanup,
        "Release all GPU resources");
}
