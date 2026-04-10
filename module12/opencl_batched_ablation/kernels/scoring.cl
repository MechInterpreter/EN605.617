// scoring.cl -- Causal score computation kernels.


// Causal score: dot(clean,probe) - dot(ablated,probe)
// Uses __local reduction across embed_dim.
__kernel void compute_scores(
    __global const float *clean_acts,
    __global const float *ablated_acts,
    __global const float *probe_vec,
    __global       float *scores,
    __local        float *scratch,
    const int embed_dim,
    const int num_interventions)
{
    int iv  = get_global_id(0);
    int lid = get_local_id(1);
    int lsz = get_local_size(1);

    if (iv >= num_interventions) return;

    float clean_dot  = 0.0f;
    float ablate_dot = 0.0f;
    int base = iv * embed_dim;

    for (int j = lid; j < embed_dim; j += lsz) {
        float p = probe_vec[j];
        clean_dot  += clean_acts[base + j] * p;
        ablate_dot += ablated_acts[base + j] * p;
    }

    scratch[lid] = clean_dot - ablate_dot;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int s = lsz / 2; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        scores[iv] = scratch[0];
    }
}

// Vectorized dot product (float4)
__kernel void dot_product_f4(
    __global const float *vec_a,
    __global const float *vec_b,
    __global       float *result,
    const int embed_dim,
    const int num_elements)
{
    int gid = get_global_id(0);
    if (gid >= num_elements) return;

    int ed4 = embed_dim / 4;
    int base = gid * embed_dim;
    float4 acc = (float4)(0.0f);

    for (int j = 0; j < ed4; ++j) {
        int off = base + j * 4;
        float4 a = (float4)(
            vec_a[off],     vec_a[off + 1],
            vec_a[off + 2], vec_a[off + 3]);
        float4 b = (float4)(
            vec_b[off],     vec_b[off + 1],
            vec_b[off + 2], vec_b[off + 3]);
        acc += a * b;
    }

    result[gid] = acc.x + acc.y + acc.z + acc.w;
}
