// ablation_engine.cpp -- Batched causal ablation
// engine orchestration.

#include "ablation_engine.h"
#include "../include/config.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>

// Event elapsed time in ms
static double event_time_ms(cl_event ev)
{
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(
        ev, CL_PROFILING_COMMAND_START,
        sizeof(start), &start, NULL);
    clGetEventProfilingInfo(
        ev, CL_PROFILING_COMMAND_END,
        sizeof(end), &end, NULL);
    return (double)(end - start) / 1.0e6;
}

// Fill array with random floats in [-1, 1]
static void fill_random(
    float *arr, int n, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n; ++i) {
        arr[i] = ((float)rand() / RAND_MAX)
                 * 2.0f - 1.0f;
    }
}

// Generate random intervention descriptors
static void generate_interventions(
    std::vector<Intervention> &ivs,
    int count, int seq_len, int num_heads)
{
    ivs.resize(count);
    srand(RNG_SEED + 1);
    for (int i = 0; i < count; ++i) {
        ivs[i].component_idx =
            rand() % num_heads;
        ivs[i].token_pos = rand() % seq_len;
        ivs[i].type =
            (InterventionType)(rand() % 3);
    }
}

// Create buffer (allocating pinned host memory)
static cl_mem make_buffer(
    cl_context ctx, cl_mem_flags flags,
    size_t bytes, void *host_ptr)
{
    cl_int err;
    cl_mem buf = clCreateBuffer(
        ctx, flags | CL_MEM_ALLOC_HOST_PTR,
        bytes, host_ptr, &err);
    cl_check(err, "clCreateBuffer");
    return buf;
}



// Upload data (write or map/unmap)
static void upload_data(
    cl_command_queue q, cl_mem buf,
    const void *src, size_t bytes,
    int use_map)
{
    cl_int err;
    if (use_map) {
        void *ptr = clEnqueueMapBuffer(
            q, buf, CL_TRUE, CL_MAP_WRITE,
            0, bytes, 0, NULL, NULL, &err);
        cl_check(err, "MapBuffer(write)");
        memcpy(ptr, src, bytes);
        clEnqueueUnmapMemObject(
            q, buf, ptr, 0, NULL, NULL);
    } else {
        err = clEnqueueWriteBuffer(
            q, buf, CL_TRUE,
            0, bytes, src, 0, NULL, NULL);
        cl_check(err, "WriteBuffer");
    }
}

// Download data (read or map/unmap)
static void download_data(
    cl_command_queue q, cl_mem buf,
    void *dst, size_t bytes, int use_map)
{
    cl_int err;
    if (use_map) {
        void *ptr = clEnqueueMapBuffer(
            q, buf, CL_TRUE, CL_MAP_READ,
            0, bytes, 0, NULL, NULL, &err);
        cl_check(err, "MapBuffer(read)");
        memcpy(dst, ptr, bytes);
        clEnqueueUnmapMemObject(
            q, buf, ptr, 0, NULL, NULL);
    } else {
        err = clEnqueueReadBuffer(
            q, buf, CL_TRUE,
            0, bytes, dst, 0, NULL, NULL);
        cl_check(err, "ReadBuffer");
    }
}

// Print profiling info
static void print_profile(
    const char *name, cl_event ev)
{
    double ms = event_time_ms(ev);
    std::cout << "  [Profile] " << name
              << ": " << ms << " ms"
              << std::endl;
}

// Run clean forward pass kernel
static cl_event run_forward(
    CLState *st, cl_mem input,
    cl_mem weights, cl_mem output,
    int seq_len, int embed_dim)
{
    cl_int err;
    cl_kernel k = clCreateKernel(
        st->program, "linear_forward", &err);
    cl_check(err, "CreateKernel(forward)");

    err  = clSetKernelArg(
        k, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(
        k, 1, sizeof(cl_mem), &weights);
    err |= clSetKernelArg(
        k, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(
        k, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(
        k, 4, sizeof(int), &embed_dim);
    cl_check(err, "SetKernelArg(forward)");

    size_t global[2] = {
        (size_t)seq_len,
        (size_t)(embed_dim / 4)
    };
    cl_event ev;
    err = clEnqueueNDRangeKernel(
        st->queue, k, 2, NULL,
        global, NULL, 0, NULL, &ev);
    cl_check(err, "NDRange(forward)");
    clReleaseKernel(k);
    return ev;
}

// Run ablation on one micro-batch
static cl_event run_ablation_batch(
    CLState *st, cl_mem acts_buf,
    cl_mem comp_buf, cl_mem tok_buf,
    int batch_n, int embed_dim,
    int head_dim, int batch_offset)
{
    cl_int err;
    cl_kernel k = clCreateKernel(
        st->program, "ablate_attn_head", &err);
    cl_check(err, "CreateKernel(ablate)");

    err  = clSetKernelArg(
        k, 0, sizeof(cl_mem), &acts_buf);
    err |= clSetKernelArg(
        k, 1, sizeof(cl_mem), &comp_buf);
    err |= clSetKernelArg(
        k, 2, sizeof(cl_mem), &tok_buf);
    err |= clSetKernelArg(
        k, 3, sizeof(int), &embed_dim);
    err |= clSetKernelArg(
        k, 4, sizeof(int), &head_dim);
    err |= clSetKernelArg(
        k, 5, sizeof(int), &batch_n);
    err |= clSetKernelArg(
        k, 6, sizeof(int), &batch_offset);
    cl_check(err, "SetKernelArg(ablate)");

    size_t global[2] = {
        (size_t)batch_n,
        (size_t)(head_dim / 4)
    };
    cl_event ev;
    err = clEnqueueNDRangeKernel(
        st->queue, k, 2, NULL,
        global, NULL, 0, NULL, &ev);
    cl_check(err, "NDRange(ablate)");
    clReleaseKernel(k);
    return ev;
}

// Run scoring kernel
static cl_event run_scoring(
    CLState *st, cl_mem clean,
    cl_mem ablated, cl_mem probe,
    cl_mem scores,
    int embed_dim, int n_iv)
{
    cl_int err;
    cl_kernel k = clCreateKernel(
        st->program, "compute_scores", &err);
    cl_check(err, "CreateKernel(scores)");

    int local_sz = 64;
    if (local_sz > embed_dim)
        local_sz = embed_dim;
    size_t scratch = local_sz * sizeof(float);

    err  = clSetKernelArg(
        k, 0, sizeof(cl_mem), &clean);
    err |= clSetKernelArg(
        k, 1, sizeof(cl_mem), &ablated);
    err |= clSetKernelArg(
        k, 2, sizeof(cl_mem), &probe);
    err |= clSetKernelArg(
        k, 3, sizeof(cl_mem), &scores);
    err |= clSetKernelArg(
        k, 4, scratch, NULL);
    err |= clSetKernelArg(
        k, 5, sizeof(int), &embed_dim);
    err |= clSetKernelArg(
        k, 6, sizeof(int), &n_iv);
    cl_check(err, "SetKernelArg(scores)");

    size_t global[2] = {
        (size_t)n_iv, (size_t)local_sz
    };
    size_t local[2] = {1, (size_t)local_sz};
    cl_event ev;
    err = clEnqueueNDRangeKernel(
        st->queue, k, 2, NULL,
        global, local, 0, NULL, &ev);
    cl_check(err, "NDRange(scores)");
    clReleaseKernel(k);
    return ev;
}

// Print top-K results
static void print_top_results(
    const std::vector<AblationResult> &res,
    int k)
{
    int n = std::min(k, (int)res.size());
    std::cout << "\n  Top " << n
              << " causal scores:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout
            << "    #" << (i + 1)
            << "  iv=" << res[i].intervention_id
            << "  score="
            << res[i].causal_score
            << std::endl;
    }
}

// Per-iteration device buffers
struct IterBufs {
    cl_mem input, weights, clean, probe;
    cl_mem ablated, comp, tok, scores;
};

static IterBufs alloc_buffers(
    CLState *st, const RunConfig *cfg,
    float *h_in, float *h_wt, float *h_pr,
    std::vector<int> &h_comp,
    std::vector<int> &h_tok)
{
    int E = cfg->embed_dim;
    int S = cfg->seq_len;
    int N = cfg->num_interventions;
    int act_sz = S * E;
    IterBufs b;

    b.input = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        act_sz * sizeof(float), h_in);
    b.weights = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        E * E * sizeof(float), h_wt);
    b.clean = make_buffer(st->context,
        CL_MEM_READ_WRITE,
        act_sz * sizeof(float), NULL);
    b.probe = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        E * sizeof(float), h_pr);
    b.ablated = make_buffer(st->context,
        CL_MEM_READ_WRITE,
        (size_t)N * E * sizeof(float), NULL);
    b.comp = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        N * sizeof(int), h_comp.data());
    b.tok = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        N * sizeof(int), h_tok.data());
    b.scores = make_buffer(st->context,
        CL_MEM_WRITE_ONLY,
        N * sizeof(float), NULL);
    return b;
}

// Release device buffers
static void free_buffers(IterBufs &b)
{
    clReleaseMemObject(b.input);
    clReleaseMemObject(b.weights);
    clReleaseMemObject(b.clean);
    clReleaseMemObject(b.probe);
    clReleaseMemObject(b.ablated);
    clReleaseMemObject(b.comp);
    clReleaseMemObject(b.tok);
    clReleaseMemObject(b.scores);
}

// Copy clean last-token acts into ablated buffer
static void prep_ablated(
    CLState *st, IterBufs &b,
    const RunConfig *cfg,
    std::vector<float> &h_clean)
{
    int E = cfg->embed_dim;
    int S = cfg->seq_len;
    int N = cfg->num_interventions;
    int act_sz = S * E;

    h_clean.resize(act_sz);
    download_data(st->queue, b.clean,
        h_clean.data(),
        act_sz * sizeof(float),
        cfg->use_map);

    int last = S - 1;
    std::vector<float> h_abl(N * E);
    for (int i = 0; i < N; ++i) {
        memcpy(&h_abl[i * E],
               &h_clean[last * E],
               E * sizeof(float));
    }
    upload_data(st->queue, b.ablated,
        h_abl.data(),
        (size_t)N * E * sizeof(float),
        cfg->use_map);
}

// Run ablation via zero-copy micro-batches
static double run_abl_loop(
    CLState *st, IterBufs &b,
    const RunConfig *cfg, bool prn)
{
    int E  = cfg->embed_dim;
    int N  = cfg->num_interventions;
    int BS = cfg->batch_size;
    int HD = E / cfg->num_heads;
    int nb = (N + BS - 1) / BS;

    std::vector<cl_event> evs;
    for (int i = 0; i < nb; ++i) {
        int st_ = i * BS;
        int cnt = std::min(BS, N - st_);

        cl_event ev = run_ablation_batch(
            st, b.ablated, b.comp, b.tok,
            cnt, E, HD, st_);
        evs.push_back(ev);
    }

    if (!evs.empty())
        clWaitForEvents(
            (cl_uint)evs.size(), evs.data());

    double ms = 0.0;
    for (int i = 0; i < nb; ++i) {
        ms += event_time_ms(evs[i]);
        if (prn && cfg->profile) {
            char lbl[48];
            snprintf(lbl, sizeof(lbl),
                "Ablation batch %d", i);
            print_profile(lbl, evs[i]);
        }
        clReleaseEvent(evs[i]);
    }
    return ms;
}

// Score and rank interventions
static double score_and_rank(
    CLState *st, IterBufs &b,
    const RunConfig *cfg,
    std::vector<float> &h_clean,
    std::vector<AblationResult> &res,
    bool prn)
{
    int E = cfg->embed_dim;
    int S = cfg->seq_len;
    int N = cfg->num_interventions;
    int last = S - 1;

    std::vector<float> hce(N * E);
    for (int i = 0; i < N; ++i) {
        memcpy(&hce[i * E],
               &h_clean[last * E],
               E * sizeof(float));
    }
    cl_mem bce = make_buffer(st->context,
        CL_MEM_READ_ONLY
        | CL_MEM_COPY_HOST_PTR,
        N * E * sizeof(float), hce.data());

    cl_event ev = run_scoring(
        st, bce, b.ablated,
        b.probe, b.scores, E, N);
    clWaitForEvents(1, &ev);
    if (prn && cfg->profile)
        print_profile("Scoring", ev);

    std::vector<float> hs(N);
    download_data(st->queue, b.scores,
        hs.data(), N * sizeof(float),
        cfg->use_map);

    res.resize(N);
    for (int i = 0; i < N; ++i) {
        res[i].intervention_id = i;
        res[i].causal_score    = hs[i];
    }
    std::sort(res.begin(), res.end(),
        [](const AblationResult &a,
           const AblationResult &b) {
            return fabs(a.causal_score)
                 > fabs(b.causal_score);
        });

    double ms = event_time_ms(ev);
    clReleaseEvent(ev);
    clReleaseMemObject(bce);
    return ms;
}

// Run one timed iteration
static double run_one_iteration(
    CLState *st, const RunConfig *cfg,
    float *h_in, float *h_wt, float *h_pr,
    std::vector<Intervention> &ivs,
    std::vector<AblationResult> &res,
    bool prn)
{
    int N = cfg->num_interventions;
    std::vector<int> hc(N), ht(N);
    for (int i = 0; i < N; ++i) {
        hc[i] = ivs[i].component_idx;
        ht[i] = ivs[i].token_pos;
    }

    IterBufs b = alloc_buffers(
        st, cfg, h_in, h_wt, h_pr, hc, ht);

    cl_event ev_fwd = run_forward(
        st, b.input, b.weights, b.clean,
        cfg->seq_len, cfg->embed_dim);
    clWaitForEvents(1, &ev_fwd);
    if (prn && cfg->profile)
        print_profile("Forward pass", ev_fwd);
    double t = event_time_ms(ev_fwd);
    clReleaseEvent(ev_fwd);

    std::vector<float> h_clean;
    prep_ablated(st, b, cfg, h_clean);

    t += run_abl_loop(st, b, cfg, prn);
    t += score_and_rank(
        st, b, cfg, h_clean, res, prn);

    free_buffers(b);
    return t;
}

// Print engine configuration
static void print_config(const RunConfig *cfg)
{
    int E = cfg->embed_dim;
    int S = cfg->seq_len;
    int N = cfg->num_interventions;
    std::cout
        << "\n  Configuration:"
        << "\n    Embed dim:       " << E
        << "\n    Seq len:         " << S
        << "\n    Num heads:       "
        << cfg->num_heads
        << "\n    Interventions:   " << N
        << "\n    Batch size:      "
        << cfg->batch_size
        << "\n    Iterations:      "
        << cfg->iterations
        << "\n    Use map/unmap:   "
        << (cfg->use_map ? "yes" : "no")
        << "\n    Profiling:       "
        << (cfg->profile ? "on" : "off")
        << std::endl;
}

// Main entry point
int run_ablation_engine(
    CLState *st, const RunConfig *cfg)
{
    int N = cfg->num_interventions;
    int S = cfg->seq_len;
    int E = cfg->embed_dim;

    std::cout
        << "\n=========================="
        << "=========================="
        << "\n  OpenCL Batched Causal "
        << "Ablation Engine"
        << "\n=========================="
        << "=========================="
        << std::endl;
    print_config(cfg);

    int act_sz = S * E;
    std::vector<float> h_in(act_sz);
    std::vector<float> h_wt(E * E);
    std::vector<float> h_pr(E);
    fill_random(h_in.data(), act_sz, RNG_SEED);
    fill_random(
        h_wt.data(), E * E, RNG_SEED + 10);
    fill_random(
        h_pr.data(), E, RNG_SEED + 20);

    std::vector<Intervention> ivs;
    generate_interventions(
        ivs, N, S, cfg->num_heads);

    std::cout << "\n  Warming up ("
        << DEFAULT_WARMUP << " iters)..."
        << std::endl;
    std::vector<AblationResult> res;
    for (int w = 0; w < DEFAULT_WARMUP; ++w)
        run_one_iteration(st, cfg,
            h_in.data(), h_wt.data(),
            h_pr.data(), ivs, res, false);

    std::cout << "  Running "
        << cfg->iterations
        << " timed iterations..."
        << std::endl;
    double tot = 0.0;
    for (int i = 0; i < cfg->iterations; ++i) {
        bool prn =
            (i == cfg->iterations - 1);
        double ms = run_one_iteration(
            st, cfg, h_in.data(),
            h_wt.data(), h_pr.data(),
            ivs, res, prn);
        tot += ms;
        std::cout << "    Iter " << (i + 1)
            << ": " << ms << " ms"
            << std::endl;
    }

    double avg = tot / cfg->iterations;
    double ips = N / (avg / 1000.0);
    std::cout
        << "\n  Results:"
        << "\n    Avg GPU time:    "
        << avg << " ms"
        << "\n    Throughput:      "
        << ips << " iv/sec"
        << std::endl;
    print_top_results(res, 10);

    std::cout
        << "\n=========================="
        << "==========================\n"
        << std::endl;
    return 0;
}
