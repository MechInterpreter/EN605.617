#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int kDefaultBlocks = 256;
static const int kDefaultThreads = 256;
static const int kDefaultIters = 200;
static const size_t kDefaultN = 1u << 24;  // ~16M floats

__constant__ float c_coeff[4];

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

struct Args {
  int blocks;
  int threads;
  int iters;
  size_t n;
  bool use_shared;
};

static void usage(const char* prog) {
  std::printf(
      "Usage: %s [--blocks B] [--threads T] [--n N] [--iters I] "
      "[--mode global|shared]\n"
      "  --blocks   gridDim.x\n"
      "  --threads  blockDim.x (>= 64 recommended)\n"
      "  --n        number of elements\n"
      "  --iters    loop count for timing stability\n"
      "  --mode     global or shared\n",
      prog);
}

static Args parse_args(int argc, char** argv) {
  Args a;
  a.blocks = kDefaultBlocks;
  a.threads = kDefaultThreads;
  a.iters = kDefaultIters;
  a.n = kDefaultN;
  a.use_shared = true;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      a.blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
      a.threads = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
      a.n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      a.iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
      const char* m = argv[++i];
      if (std::strcmp(m, "shared") == 0) a.use_shared = true;
      else if (std::strcmp(m, "global") == 0) a.use_shared = false;
      else {
        usage(argv[0]);
        std::exit(1);
      }
    } else if (!std::strcmp(argv[i], "--help")) {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (a.blocks < 1 || a.threads < 1 || a.iters < 1 || a.n < 1) {
    usage(argv[0]);
    std::exit(1);
  }
  if (a.threads < 64) {
    std::printf("Warning: threads < 64 (rubric wants >= 64).\n");
  }
  return a;
}

static void init_host(float* in, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    in[i] = static_cast<float>((i % 2048) - 1024) * 0.001f;
    out[i] = 0.0f;
  }
}

static float checksum_sparse(const float* out, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i += 4096) sum += out[i];
  return static_cast<float>(sum);
}

// Global-load kernel: reads input directly from global memory
__global__ void mem_kernel_global(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  float* __restrict__ block_sums,
                                  size_t n, int iters) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int stride = gridDim.x * blockDim.x;

  // Register variables (locals)
  float acc = 0.0f;
  float a = c_coeff[0];
  float b = c_coeff[1];
  float d = c_coeff[2];
  float e = c_coeff[3];

  for (int it = 0; it < iters; ++it) {
    for (size_t i = static_cast<size_t>(gid); i < n; i += stride) {
      float x = in[i];                 // global read
      float y = (a * x + b) * d + e;   // same math
      out[i] = y;                      // global write
      acc += y;
    }
  }

  // Shared-memory reduction (shows shared memory use here too)
  extern __shared__ float sdata[];
  sdata[tid] = acc;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sdata[tid] += sdata[tid + off];
    __syncthreads();
  }
  if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

__global__ void mem_kernel_shared(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  float* __restrict__ block_sums,
                                  size_t n, int iters) {
  int tid = threadIdx.x;

  // Dynamic shared memory: one tile per block
  extern __shared__ float tile[];

  // Register vars
  float acc = 0.0f;
  float a = c_coeff[0];
  float b = c_coeff[1];
  float d = c_coeff[2];
  float e = c_coeff[3];

  size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (int it = 0; it < iters; ++it) {
    // Each block walks over the array in block-sized tiles
    for (size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x;
         base < n;
         base += grid_stride) {

      size_t idx = base + static_cast<size_t>(tid);

      // Load tile from global -> shared
      if (idx < n) tile[tid] = in[idx];
      __syncthreads();

      // Compute from shared -> write to global
      if (idx < n) {
        float x = tile[tid];
        float y = (a * x + b) * d + e;
        out[idx] = y;
        acc += y;
      }
      __syncthreads();
    }
  }

  // Block reduction using shared memory
  tile[tid] = acc;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) tile[tid] += tile[tid + off];
    __syncthreads();
  }
  if (tid == 0) block_sums[blockIdx.x] = tile[0];
}

static float run_kernel(const Args& a, const float* d_in, float* d_out,
                        float* d_block_sums) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  dim3 grid(a.blocks);
  dim3 block(a.threads);
  size_t shmem_bytes = static_cast<size_t>(a.threads) * sizeof(float);

  CUDA_CHECK(cudaEventRecord(start));
  if (a.use_shared) {
    mem_kernel_shared<<<grid, block, shmem_bytes>>>(
        d_in, d_out, d_block_sums, a.n, a.iters);
  } else {
    mem_kernel_global<<<grid, block, shmem_bytes>>>(
        d_in, d_out, d_block_sums, a.n, a.iters);
  }
  CUDA_CHECK(cudaEventRecord(stop));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  std::printf("GPU: %s\n", prop.name);
  std::printf("Config: blocks=%d threads=%d n=%zu iters=%d mode=%s\n",
              a.blocks, a.threads, a.n, a.iters,
              a.use_shared ? "shared" : "global");

  // Host memory (pinned arrays)
  float* h_in = nullptr;
  float* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, a.n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, a.n * sizeof(float)));
  init_host(h_in, h_out, a.n);

  // Constant memory init
  float h_coeff[4] = {1.01f, 0.25f, 0.99f, 0.125f};
  CUDA_CHECK(cudaMemcpyToSymbol(c_coeff, h_coeff, sizeof(h_coeff)));

  // Global memory on device
  float* d_in = nullptr;
  float* d_out = nullptr;
  float* d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, a.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, a.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_block_sums, a.blocks * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, a.n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, a.n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_block_sums, 0, a.blocks * sizeof(float)));

  // Warmup + timed run
  float warm = run_kernel(a, d_in, d_out, d_block_sums);
  float ms = run_kernel(a, d_in, d_out, d_block_sums);

  CUDA_CHECK(cudaMemcpy(h_out, d_out, a.n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("Warmup: %.3f ms\n", warm);
  std::printf("Time:   %.3f ms\n", ms);
  std::printf("Checksum (sparse): %.6f\n", checksum_sparse(h_out, a.n));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_block_sums));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}