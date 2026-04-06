#ifndef BASELINE_H
#define BASELINE_H

#include "../include/types.h"
#include <cublas_v2.h>
#include <vector>

// Run sequential one-at-a-time baseline
void run_sequential_baseline(
    cublasHandle_t       handle,
    const ModelConfig&   cfg,
    const ModelWeights&  weights,
    const float*         d_input,
    float                clean_logit,
    const std::vector<Intervention>& interventions,
    std::vector<float>&  out_scores,
    float*               elapsed_ms
);

#endif // BASELINE_H
