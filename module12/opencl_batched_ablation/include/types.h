// types.h -- Data types for the ablation engine.

#ifndef TYPES_H
#define TYPES_H

// Intervention type: which activation to zero
typedef enum {
    IV_ATTN_HEAD    = 0,  // ablate one attn head
    IV_MLP_OUT      = 1,  // ablate MLP output
    IV_RESID_STREAM = 2   // ablate full residual
} InterventionType;

// Single intervention descriptor
typedef struct {
    int component_idx;   // which component to ablate
    int token_pos;       // which token position
    InterventionType type;
} Intervention;

// Result for one intervention
typedef struct {
    int   intervention_id;  // index into the list
    float causal_score;     // clean - ablated logit
} AblationResult;

// Runtime configuration (from CLI)
typedef struct {
    int  platform_idx;
    int  device_idx;
    int  seq_len;
    int  embed_dim;
    int  num_heads;
    int  num_interventions;
    int  batch_size;
    int  iterations;
    int  use_map;       // 1 = map/unmap, 0 = R/W
    int  profile;       // 1 = profiling output
    int  show_info;     // 1 = show device info only
    int  show_help;     // 1 = show usage and exit
} RunConfig;

#endif // TYPES_H
