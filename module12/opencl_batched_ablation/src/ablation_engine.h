// ablation_engine.h -- Batched causal ablation
// engine orchestration.

#ifndef ABLATION_ENGINE_H
#define ABLATION_ENGINE_H

#include "cl_setup.h"
#include "../include/types.h"

// Run the full batched ablation pipeline
int run_ablation_engine(
    CLState *st,
    const RunConfig *cfg);

#endif // ABLATION_ENGINE_H
