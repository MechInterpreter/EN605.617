#ifndef VALIDATION_H
#define VALIDATION_H

#include "../include/types.h"
#include <cublas_v2.h>

bool run_validation(
    cublasHandle_t       handle,
    const ModelConfig&   cfg,
    const ModelWeights&  weights,
    const float*         d_input,
    float                clean_logit,
    int                  num_interventions,
    int                  batch_size
);

#endif // VALIDATION_H
