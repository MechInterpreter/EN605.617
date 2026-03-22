#ifndef ABLATION_H
#define ABLATION_H

#include "../include/types.h"
#include <thrust/device_vector.h>
#include <vector>

// Generate a list of interventions covering component × token_pos
void generate_interventions(const ModelConfig& cfg,
                            int num_interventions,
                            std::vector<Intervention>& interventions);

// Generate ablation masks on GPU using Thrust
void generate_ablation_masks(const ModelConfig& cfg,
                             const std::vector<Intervention>& interventions,
                             int batch_start, int batch_size,
                             thrust::device_vector<float>& d_masks);

// Apply masks element-wise (Thrust transform)
void apply_ablation_masks(thrust::device_vector<float>& d_activations,
                          const thrust::device_vector<float>& d_masks,
                          int total_elements);

// Compute clean_logit - ablated_logit (Thrust transform)
void compute_causal_scores(float clean_logit,
                           const thrust::device_vector<float>& d_ablated_logits,
                           thrust::device_vector<float>& d_scores,
                           int count);

// Rank interventions by |causal_score| descending (Thrust sort_by_key)
void rank_interventions(thrust::device_vector<float>& d_scores,
                        thrust::device_vector<int>& d_indices,
                        int count);

// Mean |causal_score| (Thrust transform_reduce)
float mean_absolute_causal_score(const thrust::device_vector<float>& d_scores,
                                 int count);

// Replicate activations and apply masks (Thrust copy + transform)
void replicate_and_mask(const float* d_base,
                        int slice_elements,
                        int batch_size,
                        const thrust::device_vector<float>& d_masks,
                        thrust::device_vector<float>& d_batched);

#endif // ABLATION_H
