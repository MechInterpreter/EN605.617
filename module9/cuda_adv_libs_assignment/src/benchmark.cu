#include "benchmark.h"
#include "baseline.h"
#include "batched_engine.h"
#include "ablation.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"

#include <cstdio>
#include <vector>

// Print benchmark table header
static void print_bench_header() {
    printf("\n");
    printf("========================================"
           "============================\n");
    printf("               BENCHMARK RESULTS\n");
    printf("========================================"
           "============================\n");
    printf("%-8s %-8s %-14s %-18s %-10s\n",
           "N_IV", "BATCH", "TIME (ms)",
           "IV/sec", "SPEEDUP");
    printf("----------------------------------------"
           "----------------------------\n");
}

// Benchmark the sequential baseline for a given
// intervention count. Returns avg time in ms.
static float bench_sequential(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    std::vector<Intervention>& ivs
) {
    std::vector<float> scores;
    float ms = 0.0f;

    // Warm up
    for (int i = 0; i < WARMUP_ITERS; i++) {
        run_sequential_baseline(
            handle, cfg, w, d_input,
            clean_logit, ivs, scores, &ms);
    }

    // Timed runs
    float total = 0.0f;
    for (int i = 0; i < BENCH_ITERS; i++) {
        run_sequential_baseline(
            handle, cfg, w, d_input,
            clean_logit, ivs, scores, &ms);
        total += ms;
    }
    return total / BENCH_ITERS;
}

// Benchmark batched engine at one batch size.
// Returns avg time in ms.
static float bench_batched(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    std::vector<Intervention>& ivs,
    int bs
) {
    std::vector<float> scores;
    float ms = 0.0f;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        run_batched_engine(
            handle, cfg, w, d_input,
            clean_logit, ivs, bs, scores, &ms);
    }

    float total = 0.0f;
    for (int i = 0; i < BENCH_ITERS; i++) {
        run_batched_engine(
            handle, cfg, w, d_input,
            clean_logit, ivs, bs, scores, &ms);
        total += ms;
    }
    return total / BENCH_ITERS;
}

// Run full benchmark sweep and print results table.
void run_benchmark(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit
) {
    print_bench_header();

    for (int si = 0; si < NUM_INTERVENTION_SIZES; si++) {
        int num_iv = INTERVENTION_SIZES[si];

        std::vector<Intervention> ivs;
        generate_interventions(cfg, num_iv, ivs);

        // Sequential baseline timing
        float seq_ms = bench_sequential(
            handle, cfg, w, d_input,
            clean_logit, ivs);
        float seq_ips =
            (float)num_iv / (seq_ms * 1e-3f);

        printf("%-8d %-8s %-14.2f %-18.1f %-10s\n",
               num_iv, "seq", seq_ms,
               seq_ips, "1.00x");

        // Batched engine at various batch sizes
        for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
            int bs = BATCH_SIZES[bi];
            if (bs > num_iv) continue;

            float bat_ms = bench_batched(
                handle, cfg, w, d_input,
                clean_logit, ivs, bs);
            float bat_ips =
                (float)num_iv / (bat_ms * 1e-3f);
            float speedup = seq_ms / bat_ms;

            printf(
                "%-8d %-8d %-14.2f %-18.1f %.2fx\n",
                num_iv, bs, bat_ms,
                bat_ips, speedup);
        }

        printf("----------------------------------------"
               "----------------------------\n");
    }
    printf("\n");
}
