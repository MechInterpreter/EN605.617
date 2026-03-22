#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error-checking macros for CUDA runtime and cuBLAS calls
#define CUDA_CHECK(call)                                   \
    do {                                                   \
        cudaError_t err = (call);                          \
        if (err != cudaSuccess) {                          \
            fprintf(stderr,                                \
                    "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__,                     \
                    cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

static const char* cublasGetErrorString(
    cublasStatus_t status
) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        default:
            return "UNKNOWN CUBLAS ERROR";
    }
}

#define CUBLAS_CHECK(call)                                 \
    do {                                                   \
        cublasStatus_t stat = (call);                      \
        if (stat != CUBLAS_STATUS_SUCCESS) {                \
            fprintf(stderr,                                \
                    "cuBLAS error at %s:%d: %s\n",         \
                    __FILE__, __LINE__,                     \
                    cublasGetErrorString(stat));            \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

// GPU timer helper using CUDA events
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    // Returns elapsed milliseconds
    float toc(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(
            cudaEventElapsedTime(&ms, start, stop)
        );
        return ms;
    }
};

#endif // CUDA_UTILS_H
