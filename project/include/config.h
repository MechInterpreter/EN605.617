#ifndef CONFIG_H
#define CONFIG_H

// Embedding / hidden dimension
#define DEFAULT_EMBED_DIM   128

// Number of attention heads
#define DEFAULT_NUM_HEADS   4

// Head dimension  (embed_dim / num_heads)
#define DEFAULT_HEAD_DIM \
    (DEFAULT_EMBED_DIM / DEFAULT_NUM_HEADS)

// Sequence length (number of token positions)
#define DEFAULT_SEQ_LEN     32

// MLP intermediate dimension (typically 4x embed_dim)
#define DEFAULT_MLP_DIM     512

// Ablatable components: num_heads + 1 (MLP)
#define DEFAULT_NUM_COMPONENTS \
    (DEFAULT_NUM_HEADS + 1)

// Vocabulary size for output projection
#define DEFAULT_VOCAB_SIZE  64

// Number of transformer layers
#define DEFAULT_NUM_LAYERS  2

// RMSNorm epsilon
#define DEFAULT_NORM_EPS    1e-5f

// Intervention counts to sweep
#define NUM_INTERVENTION_SIZES  4
static const int INTERVENTION_SIZES
    [NUM_INTERVENTION_SIZES] = {
        64, 256, 1024, 4096
    };

// Batch sizes (interventions per batched call)
#define NUM_BATCH_SIZES  4
static const int BATCH_SIZES
    [NUM_BATCH_SIZES] = {32, 64, 128, 256};

// Number of warm-up iterations before timing
#define WARMUP_ITERS  2

// Number of timed iterations for averaging
#define BENCH_ITERS   5

// Correctness tolerance
#define TOLERANCE  1e-3f

// Random seed for reproducibility
#define RNG_SEED  42

#endif // CONFIG_H
