#ifndef INLCUDES_H_
#define INLCUDES_H_

#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)


// Error checking macro for CUBLAS calls
#define CUBLAS_CHECK(call) \
    do \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)
#endif
