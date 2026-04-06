#ifndef BATCHED_ENGINE_H
#define BATCHED_ENGINE_H

#include "../include/types.h"
#include <cublas_v2.h>
#include <vector>

// Run batched ablation engine with micro-batching
void run_batched_engine(
    cublasHandle_t       handle,
    const ModelConfig&   cfg,
    const ModelWeights&  weights,
    const float*         d_input,
    float                clean_logit,
    const std::vector<Intervention>& interventions,
    int                  batch_size,
    std::vector<float>&  out_scores,
    float*               elapsed_ms
);

#endif // BATCHED_ENGINE_H
