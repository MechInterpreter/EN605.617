#include "ablation.h"
#include "../include/cuda_utils.h"

#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <cstdio>
#include <cmath>

// Generate all interventions: sweep (component, token_pos)
void generate_interventions(
    const ModelConfig& cfg,
    int num_interventions,
    std::vector<Intervention>& interventions
) {
    interventions.resize(num_interventions);
    int total = cfg.num_components * cfg.seq_len;
    for (int i = 0; i < num_interventions; i++) {
        int idx = i % total;
        interventions[i].component_idx =
            idx / cfg.seq_len;
        interventions[i].token_pos =
            idx % cfg.seq_len;
    }
}

// Functor: generate a mask over [S x E] that zeros out
// the targeted component at the targeted token position.

// Component layout in the residual stream:
//   head h -> columns [h*head_dim, (h+1)*head_dim)
//   MLP    -> all E columns at target token
struct MaskGenerator {
    int S, E, head_dim, num_heads;
    int target_component, target_token;

    __host__ __device__
    float operator()(int flat_idx) const {
        int token = flat_idx / E;
        int col   = flat_idx % E;

        if (token != target_token) return 1.0f;

        if (target_component < num_heads) {
            int start = target_component * head_dim;
            int end   = start + head_dim;
            if (col >= start && col < end)
                return 0.0f;
        } else {
            return 0.0f;  // Ablate MLP
        }
        return 1.0f;
    }
};

// Generate ablation masks for a batch of interventions.
// d_masks: [batch_size x S x E] mask buffer on GPU.
// Each mask: 1.0 = keep, 0.0 = ablate.
void generate_ablation_masks(
    const ModelConfig& cfg,
    const std::vector<Intervention>& interventions,
    int batch_start, int batch_size,
    thrust::device_vector<float>& d_masks
) {
    int S = cfg.seq_len;
    int E = cfg.embed_dim;
    int slice = S * E;

    d_masks.resize(batch_size * slice);

    for (int b = 0; b < batch_size; b++) {
        int iv_idx = batch_start + b;
        const Intervention& iv = interventions[iv_idx];

        MaskGenerator gen;
        gen.S = S;
        gen.E = E;
        gen.head_dim = cfg.head_dim;
        gen.num_heads = cfg.num_heads;
        gen.target_component = iv.component_idx;
        gen.target_token     = iv.token_pos;

        // thrust::tabulate fills via functor on GPU
        thrust::tabulate(
            d_masks.begin() + b * slice,
            d_masks.begin() + (b + 1) * slice,
            gen);
    }
}

// Apply masks: element-wise multiply activations by mask.
// Operates on [batch_size x S x E] tensors.
void apply_ablation_masks(
    thrust::device_vector<float>& d_activations,
    const thrust::device_vector<float>& d_masks,
    int total_elements
) {
    thrust::transform(
        d_activations.begin(),
        d_activations.begin() + total_elements,
        d_masks.begin(),
        d_activations.begin(),
        thrust::multiplies<float>());
}

// Compute causal scores: clean - ablated for each.
// Uses thrust::transform.
struct CausalScoreFunctor {
    float clean_logit;
    __host__ __device__
    float operator()(float ablated) const {
        return clean_logit - ablated;
    }
};

void compute_causal_scores(
    float clean_logit,
    const thrust::device_vector<float>& d_ablated,
    thrust::device_vector<float>& d_scores,
    int count
) {
    d_scores.resize(count);
    CausalScoreFunctor fn;
    fn.clean_logit = clean_logit;
    thrust::transform(
        d_ablated.begin(),
        d_ablated.begin() + count,
        d_scores.begin(), fn);
}

// Rank interventions by |causal_score| descending.
// Uses thrust::sort_by_key + thrust::sequence.
struct AbsGreater {
    __host__ __device__
    bool operator()(float a, float b) const {
        return fabsf(a) > fabsf(b);
    }
};

void rank_interventions(
    thrust::device_vector<float>& d_scores,
    thrust::device_vector<int>& d_indices,
    int count
) {
    d_indices.resize(count);
    thrust::sequence(
        d_indices.begin(), d_indices.end());

    thrust::sort_by_key(
        d_scores.begin(),
        d_scores.begin() + count,
        d_indices.begin(), AbsGreater());
}

// Mean |causal score| via thrust::transform_reduce.
struct AbsFunctor {
    __host__ __device__
    float operator()(float x) const {
        return fabsf(x);
    }
};

float mean_absolute_causal_score(
    const thrust::device_vector<float>& d_scores,
    int count
) {
    float sum = thrust::transform_reduce(
        d_scores.begin(),
        d_scores.begin() + count,
        AbsFunctor(), 0.0f,
        thrust::plus<float>());
    return sum / (float)count;
}

// Replicate base activations for a batch and apply masks.
// Uses cudaMemcpy for replication, thrust::transform
// for element-wise masking.
void replicate_and_mask(
    const float* d_base,
    int slice_elements,
    int batch_size,
    const thrust::device_vector<float>& d_masks,
    thrust::device_vector<float>& d_batched
) {
    d_batched.resize(batch_size * slice_elements);

    // Replicate base activations into each slot
    for (int b = 0; b < batch_size; b++) {
        float* dst = thrust::raw_pointer_cast(
            d_batched.data()) + b * slice_elements;
        CUDA_CHECK(cudaMemcpy(
            dst, d_base,
            slice_elements * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }

    // Apply masks via thrust::transform
    int total = batch_size * slice_elements;
    thrust::transform(
        d_batched.begin(),
        d_batched.begin() + total,
        d_masks.begin(),
        d_batched.begin(),
        thrust::multiplies<float>());
}
