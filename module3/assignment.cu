#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <string.h>

// Simple CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Print basic GPU information
static void outputCardInfo() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s | SMs: %d | CC: %d.%d | GlobalMem: %.2f GB\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0));
}

// Fill input array with random values
static void fillInput(std::vector<float>& a) {
    unsigned int x = 123456789u;
    for (size_t i = 0; i < a.size(); i++) {
        x ^= x << 13; 
        x ^= x >> 17; 
        x ^= x << 5;
        a[i] = ((x & 0xFFFFu) / 32768.0f) - 1.0f;
    }
}

// Baseline per-element computation
__host__ __device__ __forceinline__
float baseline_op(float x) {
    float y = x * 1.37f + 0.11f;
    y = y * y + 0.05f * x;
    y = y / (1.0f + fabsf(x));
    return y;
}

// Same computation with a simple branch added
__host__ __device__ __forceinline__
float branching_op(float x) {
    float y = baseline_op(x);
    if (x > 0.0f) 
        y = y + 0.25f * x;
    else         
        y = y - 0.25f * x;
    return y;
}

// GPU baseline kernel
__global__
void baselineKernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        out[i] = baseline_op(in[i]);
}

// GPU branching kernel
__global__
void branchingKernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        out[i] = branching_op(in[i]);
}

// CPU baseline version
static void baselineHost(const std::vector<float>& in, std::vector<float>& out) {
    for (size_t i = 0; i < in.size(); i++)
        out[i] = baseline_op(in[i]);
}

// CPU branching version
static void branchingHost(const std::vector<float>& in, std::vector<float>& out) {
    for (size_t i = 0; i < in.size(); i++)
        out[i] = branching_op(in[i]);
}

// Compare CPU and GPU outputs
static float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv)
{
    // total threads and threads per block
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2) totalThreads = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);

    // data size and repeat count for timing
    int n = 4 * 1024 * 1024;
    int iters = 50;
    const char* csvPath = NULL;
    int verify = 1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) 
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) 
            iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) 
            csvPath = argv[++i];
        else if (strcmp(argv[i], "--no-verify") == 0) 
            verify = 0;
    }

    int numBlocks = totalThreads / blockSize;

    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    if (blockSize <= 0) blockSize = 256;
    if (blockSize > 1024) blockSize = 1024;
    if (numBlocks <= 0) numBlocks = 1;

    outputCardInfo();
    printf("Config: totalThreads=%d blockSize=%d numBlocks=%d n=%d iters=%d\n",
           totalThreads, blockSize, numBlocks, n, iters);

    std::vector<float> hIn(n), hCpu(n), hGpu(n);
    fillInput(hIn);

    float *dIn = NULL, *dOut = NULL;
    CUDA_CHECK(cudaMalloc((void**)&dIn, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, n * sizeof(float)));

    // CPU baseline timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; k++)
        baselineHost(hIn, hCpu);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    double cpuBaselineMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // warm-up
    baselineKernel<<<numBlocks, blockSize>>>(dIn, dOut, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU baseline timing
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int k = 0; k < iters; k++)
        baselineKernel<<<numBlocks, blockSize>>>(dIn, dOut, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float gpuBaselineKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuBaselineKernelMs, e0, e1));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hGpu.data(), dOut, n * sizeof(float), cudaMemcpyDeviceToHost));
    float baselineErr = verify ? maxAbsDiff(hCpu, hGpu) : 0.0f;

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // CPU branching timing
    cpuStart = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iters; k++)
        branchingHost(hIn, hCpu);
    cpuStop = std::chrono::high_resolution_clock::now();
    double cpuBranchMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // warm-up
    branchingKernel<<<numBlocks, blockSize>>>(dIn, dOut, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU branching timing
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int k = 0; k < iters; k++)
        branchingKernel<<<numBlocks, blockSize>>>(dIn, dOut, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float gpuBranchKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuBranchKernelMs, e0, e1));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hGpu.data(), dOut, n * sizeof(float), cudaMemcpyDeviceToHost));
    float branchErr = verify ? maxAbsDiff(hCpu, hGpu) : 0.0f;

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));

    printf("\nRESULTS (iters=%d)\n", iters);
    printf("Baseline CPU: %.3f ms | Baseline GPU kernel: %.3f ms\n", cpuBaselineMs, gpuBaselineKernelMs);
    printf("Branch   CPU: %.3f ms | Branch   GPU kernel: %.3f ms\n", cpuBranchMs, gpuBranchKernelMs);

    if (verify) {
        printf("Max abs diff (baseline): %.6g\n", baselineErr);
        printf("Max abs diff (branch):   %.6g\n", branchErr);
    }

    double cpuBranchPenalty = cpuBranchMs / cpuBaselineMs;
    double gpuBranchPenalty = (double)gpuBranchKernelMs / (double)gpuBaselineKernelMs;
    printf("Branch penalty (CPU): %.3fx\n", cpuBranchPenalty);
    printf("Branch penalty (GPU): %.3fx\n", gpuBranchPenalty);

    printf("\nCSV:\n");
    printf("totalThreads,blockSize,numBlocks,n,iters,cpu_baseline_ms,gpu_baseline_kernel_ms,cpu_branch_ms,gpu_branch_kernel_ms,cpu_branch_penalty,gpu_branch_penalty\n");
    printf("%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           totalThreads, blockSize, numBlocks, n, iters,
           cpuBaselineMs, gpuBaselineKernelMs, cpuBranchMs, gpuBranchKernelMs,
           cpuBranchPenalty, gpuBranchPenalty);

    if (csvPath) {
        FILE* f = fopen(csvPath, "a");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            if (size == 0) {
                fprintf(f, "totalThreads,blockSize,numBlocks,n,iters,cpu_baseline_ms,gpu_baseline_kernel_ms,cpu_branch_ms,gpu_branch_kernel_ms,cpu_branch_penalty,gpu_branch_penalty\n");
            }
            fprintf(f, "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    totalThreads, blockSize, numBlocks, n, iters,
                    cpuBaselineMs, gpuBaselineKernelMs, cpuBranchMs, gpuBranchKernelMs,
                    cpuBranchPenalty, gpuBranchPenalty);
            fclose(f);
        }
    }

    return 0;
}