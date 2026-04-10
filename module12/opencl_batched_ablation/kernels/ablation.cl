// ablation.cl -- Intervention kernels that zero out
// components of the activation tensor.


// Zero one attention head's contribution
// gid(0)=intervention, gid(1)=float4 in head_dim
__kernel void ablate_attn_head(
    __global       float *activations,
    __global const int   *interv_comp,
    __global const int   *interv_tok,
    const int embed_dim,
    const int head_dim,
    const int batch_n,
    const int batch_offset)
{
    int iv  = get_global_id(0);
    int vec = get_global_id(1);
    if (iv >= batch_n) return;
    int hd4 = head_dim / 4;
    if (vec >= hd4) return;

    int global_iv = iv + batch_offset;
    int comp = interv_comp[global_iv];
    int tok  = interv_tok[global_iv];

    int base = global_iv * embed_dim
             + comp * head_dim + vec * 4;
    activations[base]     = 0.0f;
    activations[base + 1] = 0.0f;
    activations[base + 2] = 0.0f;
    activations[base + 3] = 0.0f;
}

// Zero MLP output (full embedding)
// gid(0)=intervention, gid(1)=float4 in embed_dim
__kernel void ablate_mlp(
    __global       float *activations,
    __global const int   *interv_tok,
    const int embed_dim,
    const int batch_n,
    const int batch_offset)
{
    int iv  = get_global_id(0);
    int vec = get_global_id(1);
    if (iv >= batch_n) return;
    int ed4 = embed_dim / 4;
    if (vec >= ed4) return;

    int global_iv = iv + batch_offset;

    int base = global_iv * embed_dim + vec * 4;
    activations[base]     = 0.0f;
    activations[base + 1] = 0.0f;
    activations[base + 2] = 0.0f;
    activations[base + 3] = 0.0f;
}

// Zero full residual stream
// Same layout as ablate_mlp; separate kernel for
// dispatch clarity.
__kernel void ablate_residual(
    __global       float *activations,
    __global const int   *interv_tok,
    const int embed_dim,
    const int batch_n,
    const int batch_offset)
{
    int iv  = get_global_id(0);
    int vec = get_global_id(1);
    if (iv >= batch_n) return;
    int ed4 = embed_dim / 4;
    if (vec >= ed4) return;

    int global_iv = iv + batch_offset;

    int base = global_iv * embed_dim + vec * 4;
    activations[base]     = 0.0f;
    activations[base + 1] = 0.0f;
    activations[base + 2] = 0.0f;
    activations[base + 3] = 0.0f;
}
