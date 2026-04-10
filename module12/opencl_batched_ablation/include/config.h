// config.h -- Default configuration constants.

#ifndef CONFIG_H
#define CONFIG_H

// Model geometry
#define DEFAULT_EMBED_DIM   128
#define DEFAULT_SEQ_LEN      32
#define DEFAULT_NUM_HEADS     4
#define DEFAULT_HEAD_DIM \
    (DEFAULT_EMBED_DIM / DEFAULT_NUM_HEADS)

// Intervention defaults
#define DEFAULT_NUM_INTERVENTIONS  64
#define DEFAULT_BATCH_SIZE         32

// Benchmark defaults
#define DEFAULT_ITERATIONS   5
#define DEFAULT_WARMUP       2

// Platform / device
#define DEFAULT_PLATFORM     0
#define DEFAULT_DEVICE       0

// Scoring constants
#define SCORE_SCALE        1.0f
#define SCORE_BIAS         0.0f


// Kernel work-group defaults
#define DEFAULT_LOCAL_SIZE   64

// Random seed
#define RNG_SEED  42

#endif // CONFIG_H
