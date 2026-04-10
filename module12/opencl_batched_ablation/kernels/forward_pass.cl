// forward_pass.cl -- Linear projection and
// element-wise kernels (float4).


// Linear projection: out = input * weights
// gid(0)=token, gid(1)=output float4 index
__kernel void linear_forward(
    __global const float *input,
    __global const float *weights,
    __global       float *output,
    const int seq_len,
    const int embed_dim)
{
    int tok = get_global_id(0);
    int vec = get_global_id(1);
    if (tok >= seq_len) return;
    int out4 = embed_dim / 4;
    if (vec >= out4) return;

    float4 acc = (float4)(0.0f);
    int in_off = tok * embed_dim;

    for (int j = 0; j < embed_dim; ++j) {
        float x = input[in_off + j];
        int w_off = j * embed_dim + vec * 4;
        float4 w = (float4)(
            weights[w_off],
            weights[w_off + 1],
            weights[w_off + 2],
            weights[w_off + 3]);
        acc += x * w;
    }

    int o_off = tok * embed_dim + vec * 4;
    output[o_off]     = acc.x;
    output[o_off + 1] = acc.y;
    output[o_off + 2] = acc.z;
    output[o_off + 3] = acc.w;
}

// ReLU (float4, element-wise)
__kernel void relu_activate(
    __global float *data,
    const int total_floats)
{
    int gid = get_global_id(0);
    int idx = gid * 4;
    if (idx + 3 >= total_floats) return;

    float4 v = (float4)(
        data[idx],     data[idx + 1],
        data[idx + 2], data[idx + 3]);
    v = fmax(v, (float4)(0.0f));
    data[idx]     = v.x;
    data[idx + 1] = v.y;
    data[idx + 2] = v.z;
    data[idx + 3] = v.w;
}

// Residual add (float4)
__kernel void residual_add(
    __global const float *input,
    __global const float *residual,
    __global       float *output,
    const int total_floats)
{
    int gid = get_global_id(0);
    int idx = gid * 4;
    if (idx + 3 >= total_floats) return;

    float4 a = (float4)(
        input[idx],    input[idx + 1],
        input[idx + 2], input[idx + 3]);
    float4 b = (float4)(
        residual[idx],    residual[idx + 1],
        residual[idx + 2], residual[idx + 3]);
    float4 c = a + b;
    output[idx]     = c.x;
    output[idx + 1] = c.y;
    output[idx + 2] = c.z;
    output[idx + 3] = c.w;
}
