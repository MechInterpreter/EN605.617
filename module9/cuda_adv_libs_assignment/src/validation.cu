#include "validation.h"
#include "baseline.h"
#include "batched_engine.h"
#include "ablation.h"
#include "../include/cuda_utils.h"
#include "../include/config.h"

#include <thrust/device_vector.h>
#include <cstdio>
#include <cmath>
#include <vector>

// Compare sequential vs. batched scores element-wise.
// Returns true if all scores match within tolerance.
static bool compare_scores(
    const std::vector<float>& seq,
    const std::vector<float>& bat,
    int n
) {
    float max_err = 0.0f, sum_err = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < n; i++) {
        float d = fabsf(seq[i] - bat[i]);
        sum_err += d;
        if (d > max_err) max_err = d;
        if (d > TOLERANCE) {
            mismatches++;
            if (mismatches <= 5) {
                printf(
                    "    MISMATCH [%d]: "
                    "seq=%.6f bat=%.6f "
                    "diff=%.6e\n",
                    i, seq[i], bat[i], d);
            }
        }
    }

    float mean_err = sum_err / (float)n;

    printf("------------------------------------"
           "--------------------------------\n");
    printf("  Max abs error  : %.6e\n", max_err);
    printf("  Mean abs error : %.6e\n", mean_err);
    printf("  Mismatches     : %d / %d\n",
           mismatches, n);

    bool pass = (mismatches == 0);
    printf("  Result         : %s\n",
           pass ? "PASS" : "FAIL");
    printf("===================================="
           "================================\n\n");
    return pass;
}

// Print top-N most causally important interventions
// using Thrust ranking (sort_by_key + copy).
static void print_top_interventions(
    const std::vector<float>& seq_scores,
    const std::vector<Intervention>& ivs,
    int n
) {
    // Mean |causal score| via Thrust
    thrust::device_vector<float> d_sc(
        seq_scores.begin(), seq_scores.end());
    float mean_sc =
        mean_absolute_causal_score(d_sc, n);
    printf("  Mean |causal score| (seq) : %.6f\n",
           mean_sc);

    // Rank via Thrust
    thrust::device_vector<float> d_rsc(
        seq_scores.begin(), seq_scores.end());
    thrust::device_vector<int> d_ri;
    rank_interventions(d_rsc, d_ri, n);

    int top = std::min(5, n);
    std::vector<float> h_rs(top);
    std::vector<int>   h_ri(top);
    thrust::copy(
        d_rsc.begin(), d_rsc.begin() + top,
        h_rs.begin());
    thrust::copy(
        d_ri.begin(), d_ri.begin() + top,
        h_ri.begin());

    printf("  Top-%d most important:\n", top);
    for (int i = 0; i < top; i++) {
        const Intervention& iv = ivs[h_ri[i]];
        printf(
            "    #%d: iv %d (comp=%d tok=%d)"
            " score=%.6f\n",
            i+1, h_ri[i],
            iv.component_idx,
            iv.token_pos, h_rs[i]);
    }
    printf("\n");
}

// Run correctness validation.
// Returns true if all scores match within tolerance.
bool run_validation(
    cublasHandle_t handle,
    const ModelConfig& cfg,
    const ModelWeights& w,
    const float* d_input,
    float clean_logit,
    int num_iv, int batch_size
) {
    printf("\n");
    printf("===================================="
           "================================\n");
    printf("           CORRECTNESS VALIDATION\n");
    printf("===================================="
           "================================\n");
    printf("  Interventions : %d\n", num_iv);
    printf("  Batch size    : %d\n", batch_size);
    printf("  Tolerance     : %.1e\n", TOLERANCE);
    printf("------------------------------------"
           "--------------------------------\n");

    std::vector<Intervention> ivs;
    generate_interventions(cfg, num_iv, ivs);

    // Run sequential baseline
    std::vector<float> seq_sc;
    float seq_ms;
    printf("  Running sequential baseline...\n");
    run_sequential_baseline(
        handle, cfg, w, d_input,
        clean_logit, ivs, seq_sc, &seq_ms);
    printf("    Sequential time: %.2f ms\n", seq_ms);

    // Run batched engine
    std::vector<float> bat_sc;
    float bat_ms;
    printf("  Running batched engine...\n");
    run_batched_engine(
        handle, cfg, w, d_input,
        clean_logit, ivs, batch_size,
        bat_sc, &bat_ms);
    printf("    Batched time:    %.2f ms\n", bat_ms);

    bool pass = compare_scores(seq_sc, bat_sc, num_iv);
    print_top_interventions(seq_sc, ivs, num_iv);
    return pass;
}
