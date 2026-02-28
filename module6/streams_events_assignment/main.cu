/*
  CUDA Streams and Events Assignment
  ---------------------------------
  This program demonstrates:
    - Non-trivial GPU math work (two kernels)
    - CUDA streams to overlap H2D/D2H transfers with compute (double-buffer pipeline)
    - CUDA events to coordinate dependencies and to time stages
    - A test harness that executes at least 2 runs of each distinct part:
        (1) baseline sequential (single stream)
        (2) pipelined async (two streams + double buffering)
    - CLI control of threads-per-block and blocks-per-grid, plus optional sweep

  How this relates to the final project idea:
    Each "batch" represents many independent interpretability intervention variants
    patchIdx/patchVal simulate an "activation patch" at a specific location
    then we compute a "metric" (dot product with a fixed direction), analogous
    to computing a logit or logit-difference score for each variant
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

struct Options {
  int threads = 256;           // threads per block
  int blocks  = 0;             // blocks per grid (0 => auto)
  int microB  = 2048;          // variants per batch (batch size)
  int dim     = 1024;          // hidden dimension per variant
  int batches = 64;            // number of batches to process
  int runs    = 2;             // number of runs per mode (must be >= 2)
  bool compare = true;         // print comparison table across multiple thread/block settings
  bool baseline = true;
  bool pipelined = true;
  bool verbose = false;
  bool validate = true;
  int device = 0;
};

static void print_usage(const char* prog) {
  std::cout <<
    "Usage: " << prog << " [options]\n"
    "Options:\n"
    "  --threads <int>     Threads per block (default 256)\n"
    "  --blocks <int>      Blocks per grid (default 0 => auto)\n"
    "  --microB <int>      Variants per batch (default 2048)\n"
    "  --dim <int>         Dimension per variant (default 1024)\n"
    "  --batches <int>     Number of batches (default 64)\n"
    "  --runs <int>        Runs per mode (default 2)\n"
    "  --device <int>      CUDA device id (default 0)\n"
    "  --no-compare        Disable built-in comparison sweep\n"
    "  --only-baseline     Run baseline only\n"
    "  --only-pipelined    Run pipelined only\n"
    "  --no-validate       Skip output validation\n"
    "  --verbose           More printing\n"
    "  --help              Show help\n"
    "\nExamples:\n"
    "  " << prog << " --threads 256 --blocks 320 --microB 2048 --dim 1024 --batches 64\n"
    "  " << prog << " (default) runs a sweep to show multiple thread/block settings\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };

    if (a == "--threads") opt.threads = std::stoi(need("--threads"));
    else if (a == "--blocks") opt.blocks = std::stoi(need("--blocks"));
    else if (a == "--microB") opt.microB = std::stoi(need("--microB"));
    else if (a == "--dim") opt.dim = std::stoi(need("--dim"));
    else if (a == "--batches") opt.batches = std::stoi(need("--batches"));
    else if (a == "--runs") opt.runs = std::stoi(need("--runs"));
    else if (a == "--device") opt.device = std::stoi(need("--device"));
    else if (a == "--no-compare") opt.compare = false;
    else if (a == "--only-baseline") { opt.baseline = true; opt.pipelined = false; }
    else if (a == "--only-pipelined") { opt.baseline = false; opt.pipelined = true; }
    else if (a == "--no-validate") opt.validate = false;
    else if (a == "--verbose") opt.verbose = true;
    else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
    else {
      std::cerr << "Unknown option: " << a << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  if (opt.threads <= 0 || opt.threads > 1024) { std::cerr << "Bad --threads\n"; std::exit(1); }
  if (opt.blocks < 0) { std::cerr << "Bad --blocks\n"; std::exit(1); }
  if (opt.microB <= 0 || opt.dim <= 0 || opt.batches <= 0) { std::cerr << "Bad sizes\n"; std::exit(1); }
  if (opt.runs < 2) { std::cerr << "--runs must be >= 2\n"; std::exit(1); }
  return opt;
}

// Kernel 1: apply a per-variant "patch" and then do non-trivial math transform
// Layout: x is [microB, dim] flattened row-major
// patchIdx[b] is an index in [0, dim)
// patchVal[b] is the replacement value for x[b, patchIdx[b]]
// Output y has same shape
__global__ void patch_and_transform_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          const int* __restrict__ patchIdx,
                                          const float* __restrict__ patchVal,
                                          int microB, int dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = microB * dim;

  for (int idx = tid; idx < n; idx += blockDim.x * gridDim.x) {
    int b = idx / dim;
    int j = idx - b * dim;

    float v = x[idx];

    // "Activation patch": overwrite one element per variant
    int pj = patchIdx[b];
    if (j == pj) v = patchVal[b];

    // Non-trivial math:
    float a = fabsf(v);
    float t1 = sinf(v) * cosf(v);
    float t2 = sqrtf(a + 1e-6f);
    float t3 = logf(1.0f + a);
    float out = (t1 + t2 + t3) / (1.0f + 0.1f * a);

    y[idx] = out;
  }
}

// Kernel 2: metric per variant: dot(y[b,:], w[:]) producing score[b]
// We let threads cover the flattened [microB*dim] space and atomic into scores[b]
// "blocks" is therefore a meaningful performance knob for the assignment
__global__ void metric_dot_kernel(const float* __restrict__ y,
                                 const float* __restrict__ w,
                                 float* __restrict__ scores,
                                 int microB, int dim) {
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  int n = microB * dim;

  for (int idx = global_tid; idx < n; idx += total_threads) {
    int b = idx / dim;
    int j = idx - b * dim;
    float contrib = y[idx] * w[j];
    atomicAdd(&scores[b], contrib);
  }
}

// CPU reference for validation
static void cpu_reference(const std::vector<float>& x,
                          const std::vector<int>& patchIdx,
                          const std::vector<float>& patchVal,
                          const std::vector<float>& w,
                          int microB, int dim,
                          std::vector<float>& out_scores) {
  out_scores.assign(microB, 0.0f);
  for (int b = 0; b < microB; b++) {
    float acc = 0.0f;
    for (int j = 0; j < dim; j++) {
      float v = x[b * dim + j];
      if (j == patchIdx[b]) v = patchVal[b];

      float a = std::fabs(v);
      float t1 = std::sin(v) * std::cos(v);
      float t2 = std::sqrt(a + 1e-6f);
      float t3 = std::log(1.0f + a);
      float out = (t1 + t2 + t3) / (1.0f + 0.1f * a);

      acc += out * w[j];
    }
    out_scores[b] = acc;
  }
}

static void init_batch(std::vector<float>& h_x,
                       std::vector<int>& h_patchIdx,
                       std::vector<float>& h_patchVal,
                       std::vector<float>& h_w,
                       int microB, int dim,
                       int seed) {
  uint32_t s = 0x12345678u ^ (uint32_t)seed;
  auto rnd = [&]() {
    s = 1664525u * s + 1013904223u;
    return s;
  };

  h_x.resize((size_t)microB * dim);
  h_patchIdx.resize((size_t)microB);
  h_patchVal.resize((size_t)microB);
  h_w.resize((size_t)dim);

  for (int j = 0; j < dim; j++) {
    float r = (rnd() & 0xFFFF) / 65535.0f;
    h_w[j] = (r - 0.5f) * 2.0f; // [-1, 1]
  }
  for (int b = 0; b < microB; b++) {
    h_patchIdx[b] = (int)(rnd() % (uint32_t)dim);
    float rv = (rnd() & 0xFFFF) / 65535.0f;
    h_patchVal[b] = (rv - 0.5f) * 4.0f; // [-2, 2]
    for (int j = 0; j < dim; j++) {
      float r = (rnd() & 0xFFFF) / 65535.0f;
      h_x[(size_t)b * dim + j] = (r - 0.5f) * 6.0f; // [-3, 3]
    }
  }
}

struct Timings {
  float total_ms = 0.0f;
  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  float variants_per_s = 0.0f;
};

static Timings run_baseline_one_batch(int threads, int blocks,
                                      int microB, int dim,
                                      const float* h_x,
                                      const int* h_patchIdx,
                                      const float* h_patchVal,
                                      const float* h_w,
                                      float* h_scores_out) {

  const size_t bytes_x = (size_t)microB * dim * sizeof(float);
  const size_t bytes_w = (size_t)dim * sizeof(float);
  const size_t bytes_patchIdx = (size_t)microB * sizeof(int);
  const size_t bytes_patchVal = (size_t)microB * sizeof(float);
  const size_t bytes_scores = (size_t)microB * sizeof(float);

  float *d_x = nullptr, *d_y = nullptr, *d_w = nullptr, *d_scores = nullptr;
  int *d_patchIdx = nullptr;
  float *d_patchVal = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
  CUDA_CHECK(cudaMalloc(&d_y, bytes_x));
  CUDA_CHECK(cudaMalloc(&d_w, bytes_w));
  CUDA_CHECK(cudaMalloc(&d_scores, bytes_scores));
  CUDA_CHECK(cudaMalloc(&d_patchIdx, bytes_patchIdx));
  CUDA_CHECK(cudaMalloc(&d_patchVal, bytes_patchVal));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t ev_start, ev_after_h2d, ev_after_compute, ev_after_d2h;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_after_h2d));
  CUDA_CHECK(cudaEventCreate(&ev_after_compute));
  CUDA_CHECK(cudaEventCreate(&ev_after_d2h));

  if (blocks == 0) {
    int n = microB * dim;
    blocks = std::min(1024, div_up(n, threads));
  }

  CUDA_CHECK(cudaEventRecord(ev_start, stream));

  CUDA_CHECK(cudaMemcpyAsync(d_x, h_x, bytes_x, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_w, h_w, bytes_w, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_patchIdx, h_patchIdx, bytes_patchIdx, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_patchVal, h_patchVal, bytes_patchVal, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(ev_after_h2d, stream));

  CUDA_CHECK(cudaMemsetAsync(d_scores, 0, bytes_scores, stream));
  patch_and_transform_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, d_patchIdx, d_patchVal, microB, dim);
  metric_dot_kernel<<<blocks, threads, 0, stream>>>(d_y, d_w, d_scores, microB, dim);
  CUDA_CHECK(cudaEventRecord(ev_after_compute, stream));

  CUDA_CHECK(cudaMemcpyAsync(h_scores_out, d_scores, bytes_scores, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(ev_after_d2h, stream));
  CUDA_CHECK(cudaEventSynchronize(ev_after_d2h));

  Timings t{};
  CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, ev_start, ev_after_d2h));
  CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, ev_start, ev_after_h2d));
  CUDA_CHECK(cudaEventElapsedTime(&t.compute_ms, ev_after_h2d, ev_after_compute));
  CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, ev_after_compute, ev_after_d2h));

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_after_h2d));
  CUDA_CHECK(cudaEventDestroy(ev_after_compute));
  CUDA_CHECK(cudaEventDestroy(ev_after_d2h));
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_patchIdx));
  CUDA_CHECK(cudaFree(d_patchVal));

  return t;
}

static Timings run_pipelined(const Options& opt,
                             const float* h_x_batches,
                             const int* h_patchIdx_batches,
                             const float* h_patchVal_batches,
                             const float* h_w,
                             float* h_scores_batches) {

  const int microB = opt.microB;
  const int dim = opt.dim;

  const size_t bytes_x = (size_t)microB * dim * sizeof(float);
  const size_t bytes_w = (size_t)dim * sizeof(float);
  const size_t bytes_patchIdx = (size_t)microB * sizeof(int);
  const size_t bytes_patchVal = (size_t)microB * sizeof(float);
  const size_t bytes_scores = (size_t)microB * sizeof(float);

  float *d_x[2] = {nullptr, nullptr};
  float *d_y[2] = {nullptr, nullptr};
  int   *d_patchIdx[2] = {nullptr, nullptr};
  float *d_patchVal[2] = {nullptr, nullptr};
  float *d_scores[2] = {nullptr, nullptr};
  float *d_w = nullptr;

  for (int k = 0; k < 2; k++) {
    CUDA_CHECK(cudaMalloc(&d_x[k], bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y[k], bytes_x));
    CUDA_CHECK(cudaMalloc(&d_patchIdx[k], bytes_patchIdx));
    CUDA_CHECK(cudaMalloc(&d_patchVal[k], bytes_patchVal));
    CUDA_CHECK(cudaMalloc(&d_scores[k], bytes_scores));
  }
  CUDA_CHECK(cudaMalloc(&d_w, bytes_w));

  cudaStream_t stream[2];
  CUDA_CHECK(cudaStreamCreate(&stream[0]));
  CUDA_CHECK(cudaStreamCreate(&stream[1]));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  cudaEvent_t ev_first_h2d, ev_last_h2d, ev_first_compute, ev_last_compute, ev_first_d2h, ev_last_d2h;
  CUDA_CHECK(cudaEventCreate(&ev_first_h2d));
  CUDA_CHECK(cudaEventCreate(&ev_last_h2d));
  CUDA_CHECK(cudaEventCreate(&ev_first_compute));
  CUDA_CHECK(cudaEventCreate(&ev_last_compute));
  CUDA_CHECK(cudaEventCreate(&ev_first_d2h));
  CUDA_CHECK(cudaEventCreate(&ev_last_d2h));

  int blocks = opt.blocks;
  if (blocks == 0) {
    int n = microB * dim;
    blocks = std::min(1024, div_up(n, opt.threads));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_w, h_w, bytes_w, cudaMemcpyHostToDevice, stream[0]));
  CUDA_CHECK(cudaStreamSynchronize(stream[0]));

  CUDA_CHECK(cudaEventRecord(ev_start, 0));

  bool first_h2d=false, first_compute=false, first_d2h=false;

  for (int i = 0; i < opt.batches; i++) {
    int k = i & 1;

    const float* h_x = h_x_batches + (size_t)i * microB * dim;
    const int* h_pi = h_patchIdx_batches + (size_t)i * microB;
    const float* h_pv = h_patchVal_batches + (size_t)i * microB;
    float* h_scores = h_scores_batches + (size_t)i * microB;

    if (!first_h2d) { CUDA_CHECK(cudaEventRecord(ev_first_h2d, stream[k])); first_h2d=true; }

    CUDA_CHECK(cudaMemcpyAsync(d_x[k], h_x, bytes_x, cudaMemcpyHostToDevice, stream[k]));
    CUDA_CHECK(cudaMemcpyAsync(d_patchIdx[k], h_pi, bytes_patchIdx, cudaMemcpyHostToDevice, stream[k]));
    CUDA_CHECK(cudaMemcpyAsync(d_patchVal[k], h_pv, bytes_patchVal, cudaMemcpyHostToDevice, stream[k]));
    CUDA_CHECK(cudaEventRecord(ev_last_h2d, stream[k]));

    if (!first_compute) { CUDA_CHECK(cudaEventRecord(ev_first_compute, stream[k])); first_compute=true; }

    CUDA_CHECK(cudaMemsetAsync(d_scores[k], 0, bytes_scores, stream[k]));
    patch_and_transform_kernel<<<blocks, opt.threads, 0, stream[k]>>>(d_x[k], d_y[k], d_patchIdx[k], d_patchVal[k], microB, dim);
    metric_dot_kernel<<<blocks, opt.threads, 0, stream[k]>>>(d_y[k], d_w, d_scores[k], microB, dim);
    CUDA_CHECK(cudaEventRecord(ev_last_compute, stream[k]));

    if (!first_d2h) { CUDA_CHECK(cudaEventRecord(ev_first_d2h, stream[k])); first_d2h=true; }

    CUDA_CHECK(cudaMemcpyAsync(h_scores, d_scores[k], bytes_scores, cudaMemcpyDeviceToHost, stream[k]));
    CUDA_CHECK(cudaEventRecord(ev_last_d2h, stream[k]));
  }

  CUDA_CHECK(cudaStreamSynchronize(stream[0]));
  CUDA_CHECK(cudaStreamSynchronize(stream[1]));

  CUDA_CHECK(cudaEventRecord(ev_stop, 0));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  Timings t{};
  CUDA_CHECK(cudaEventElapsedTime(&t.total_ms, ev_start, ev_stop));
  CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, ev_first_h2d, ev_last_h2d));
  CUDA_CHECK(cudaEventElapsedTime(&t.compute_ms, ev_first_compute, ev_last_compute));
  CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, ev_first_d2h, ev_last_d2h));

  double total_variants = (double)opt.batches * microB;
  t.variants_per_s = (float)(total_variants / (t.total_ms / 1000.0));

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_first_h2d));
  CUDA_CHECK(cudaEventDestroy(ev_last_h2d));
  CUDA_CHECK(cudaEventDestroy(ev_first_compute));
  CUDA_CHECK(cudaEventDestroy(ev_last_compute));
  CUDA_CHECK(cudaEventDestroy(ev_first_d2h));
  CUDA_CHECK(cudaEventDestroy(ev_last_d2h));

  CUDA_CHECK(cudaStreamDestroy(stream[0]));
  CUDA_CHECK(cudaStreamDestroy(stream[1]));

  for (int k = 0; k < 2; k++) {
    CUDA_CHECK(cudaFree(d_x[k]));
    CUDA_CHECK(cudaFree(d_y[k]));
    CUDA_CHECK(cudaFree(d_patchIdx[k]));
    CUDA_CHECK(cudaFree(d_patchVal[k]));
    CUDA_CHECK(cudaFree(d_scores[k]));
  }
  CUDA_CHECK(cudaFree(d_w));

  return t;
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

static void print_device_info(int device) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "GPU: " << prop.name << " (SMs=" << prop.multiProcessorCount
            << ", cc=" << prop.major << "." << prop.minor << ")\n";
}

static void print_config(const Options& opt) {
  std::cout << "Config: threads=" << opt.threads
            << " blocks=" << opt.blocks
            << " microB=" << opt.microB
            << " dim=" << opt.dim
            << " batches=" << opt.batches
            << " runs=" << opt.runs
            << " compare=" << (opt.compare ? "on" : "off")
            << "\n";
}

static void print_timing_line(const std::string& label, const Timings& t) {
  std::cout << std::fixed << std::setprecision(3)
            << label
            << " total=" << t.total_ms << " ms"
            << " (H2D~" << t.h2d_ms << " ms"
            << ", compute~" << t.compute_ms << " ms"
            << ", D2H~" << t.d2h_ms << " ms)"
            << " throughput=" << std::setprecision(1) << t.variants_per_s << " variants/s\n";
}

static Timings avg_timings(const std::vector<Timings>& v) {
  Timings a{};
  for (auto& t : v) {
    a.total_ms += t.total_ms;
    a.h2d_ms += t.h2d_ms;
    a.compute_ms += t.compute_ms;
    a.d2h_ms += t.d2h_ms;
    a.variants_per_s += t.variants_per_s;
  }
  float n = (float)v.size();
  a.total_ms /= n;
  a.h2d_ms /= n;
  a.compute_ms /= n;
  a.d2h_ms /= n;
  a.variants_per_s /= n;
  return a;
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));
  print_device_info(opt.device);

  // Pinned host buffers so cudaMemcpyAsync can overlap
  const size_t bytes_x_all = (size_t)opt.batches * opt.microB * opt.dim * sizeof(float);
  const size_t bytes_pi_all = (size_t)opt.batches * opt.microB * sizeof(int);
  const size_t bytes_pv_all = (size_t)opt.batches * opt.microB * sizeof(float);
  const size_t bytes_scores_all = (size_t)opt.batches * opt.microB * sizeof(float);

  float* h_x_all = nullptr;
  int* h_pi_all = nullptr;
  float* h_pv_all = nullptr;
  float* h_scores_all = nullptr;

  CUDA_CHECK(cudaMallocHost(&h_x_all, bytes_x_all));
  CUDA_CHECK(cudaMallocHost(&h_pi_all, bytes_pi_all));
  CUDA_CHECK(cudaMallocHost(&h_pv_all, bytes_pv_all));
  CUDA_CHECK(cudaMallocHost(&h_scores_all, bytes_scores_all));

  float* h_w = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_w, (size_t)opt.dim * sizeof(float)));

  for (int i = 0; i < opt.batches; i++) {
    std::vector<float> hx;
    std::vector<int> hpi;
    std::vector<float> hpv;
    std::vector<float> hw;
    init_batch(hx, hpi, hpv, hw, opt.microB, opt.dim, 1337 + i);

    std::memcpy(h_x_all + (size_t)i * opt.microB * opt.dim, hx.data(), (size_t)opt.microB * opt.dim * sizeof(float));
    std::memcpy(h_pi_all + (size_t)i * opt.microB, hpi.data(), (size_t)opt.microB * sizeof(int));
    std::memcpy(h_pv_all + (size_t)i * opt.microB, hpv.data(), (size_t)opt.microB * sizeof(float));
    if (i == 0) std::memcpy(h_w, hw.data(), (size_t)opt.dim * sizeof(float));
  }

  if (opt.validate) {
    std::cout << "Validating correctness on batch 0 (CPU reference vs GPU baseline)...\n";
    std::vector<float> hx((size_t)opt.microB * opt.dim);
    std::vector<int> hpi((size_t)opt.microB);
    std::vector<float> hpv((size_t)opt.microB);
    std::vector<float> hw((size_t)opt.dim);
    std::memcpy(hx.data(), h_x_all, (size_t)opt.microB * opt.dim * sizeof(float));
    std::memcpy(hpi.data(), h_pi_all, (size_t)opt.microB * sizeof(int));
    std::memcpy(hpv.data(), h_pv_all, (size_t)opt.microB * sizeof(float));
    std::memcpy(hw.data(), h_w, (size_t)opt.dim * sizeof(float));

    std::vector<float> ref_scores;
    cpu_reference(hx, hpi, hpv, hw, opt.microB, opt.dim, ref_scores);

    std::vector<float> gpu_scores((size_t)opt.microB);
    Timings bt = run_baseline_one_batch(opt.threads, opt.blocks, opt.microB, opt.dim,
                                        hx.data(), hpi.data(), hpv.data(), hw.data(), gpu_scores.data());
    (void)bt;

    float diff = max_abs_diff(ref_scores.data(), gpu_scores.data(), (size_t)opt.microB);
    std::cout << "Max abs diff: " << std::scientific << diff << "\n";
    if (diff > 1e-2f) {
      std::cerr << "Validation failed: diff too large.\n";
      return 2;
    }
    std::cout << "Validation OK.\n";
  }

  print_config(opt);

  auto run_one_config = [&](int threads, int blocks) {
    Options cfg = opt;
    cfg.threads = threads;
    cfg.blocks = blocks;

    std::cout << "\n=== Benchmark: threads=" << threads << " blocks=" << blocks
              << " microB=" << cfg.microB << " dim=" << cfg.dim << " batches=" << cfg.batches << " ===\n";

    if (cfg.baseline) {
      std::vector<Timings> runs;
      std::vector<float> scores((size_t)cfg.microB);

      for (int r = 0; r < cfg.runs; r++) {
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0, 0));

        float h2d=0, comp=0, d2h=0;
        for (int i = 0; i < cfg.batches; i++) {
          const float* hx = h_x_all + (size_t)i * cfg.microB * cfg.dim;
          const int* hpi = h_pi_all + (size_t)i * cfg.microB;
          const float* hpv = h_pv_all + (size_t)i * cfg.microB;

          Timings t = run_baseline_one_batch(threads, blocks, cfg.microB, cfg.dim,
                                             hx, hpi, hpv, h_w, scores.data());
          h2d += t.h2d_ms;
          comp += t.compute_ms;
          d2h += t.d2h_ms;
        }

        CUDA_CHECK(cudaEventRecord(ev1, 0));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        Timings agg{};
        agg.total_ms = total_ms;
        agg.h2d_ms = h2d;
        agg.compute_ms = comp;
        agg.d2h_ms = d2h;
        double total_variants = (double)cfg.batches * cfg.microB;
        agg.variants_per_s = (float)(total_variants / (total_ms / 1000.0));
        runs.push_back(agg);

        if (cfg.verbose) print_timing_line("Baseline run", agg);
      }

      Timings avg = avg_timings(runs);
      print_timing_line("Baseline avg", avg);
    }

    if (cfg.pipelined) {
      std::vector<Timings> runs;
      for (int r = 0; r < cfg.runs; r++) {
        Timings t = run_pipelined(cfg, h_x_all, h_pi_all, h_pv_all, h_w, h_scores_all);
        runs.push_back(t);
        if (cfg.verbose) print_timing_line("Pipelined run", t);
      }
      Timings avg = avg_timings(runs);
      print_timing_line("Pipelined avg", avg);
    }
  };

  if (opt.compare) {
    std::vector<int> thread_list = {128, 256, 512};
    std::vector<int> block_list  = {120, 240, 480};

    if (std::find(thread_list.begin(), thread_list.end(), opt.threads) == thread_list.end())
      thread_list.push_back(opt.threads);

    int user_blocks = opt.blocks == 0 ? 240 : opt.blocks;
    if (std::find(block_list.begin(), block_list.end(), user_blocks) == block_list.end())
      block_list.push_back(user_blocks);

    std::sort(thread_list.begin(), thread_list.end());
    std::sort(block_list.begin(), block_list.end());

    std::cout << "\nRunning built-in comparison sweep (threads x blocks)...\n";
    for (int th : thread_list) {
      for (int bl : block_list) {
        run_one_config(th, bl);
      }
    }
  } else {
    run_one_config(opt.threads, opt.blocks);
  }

  CUDA_CHECK(cudaFreeHost(h_x_all));
  CUDA_CHECK(cudaFreeHost(h_pi_all));
  CUDA_CHECK(cudaFreeHost(h_pv_all));
  CUDA_CHECK(cudaFreeHost(h_scores_all));
  CUDA_CHECK(cudaFreeHost(h_w));

  std::cout << "\nDone.\n";
  return 0;
}