#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "../include/types.h"
#include <cublas_v2.h>

void run_benchmark(
    cublasHandle_t       handle,
    const ModelConfig&   cfg,
    const ModelWeights&  weights,
    const float*         d_input,
    float                clean_logit
);

#endif // BENCHMARK_H
