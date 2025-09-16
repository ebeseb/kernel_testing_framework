#ifndef KERNEL_V5_CUH
#define KERNEL_V5_CUH

/* ----- vectorAdd_v5 ------------------------------------------------------------------------------
All previous increases in BW utilization or "bytes in flight" are bought by using more registers in
turn. This may be fine for kernels that are not compute-heavy. Otherwise this might cause a new
issue by not having enough registers left for good occupancy (check out CUDA 13.0 feature for
spilling registers into shared memory:
https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/).
A way to get around this are async methods of memory loads, because they can go directly got to
shared memory, skipping registers.
------------------------------------------------------------------------------------------------- */
#include <cuda/pipeline> // libcudacxx API


template <int BLKSZ, int NUM_STAGES>
__global__ void vectorAdd_v5(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int tid    = threadIdx.x;
    const int offset = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // NUM_STAGES buffers per input for a NUM_STAGES deep pipeline
    __shared__ float a_buf[NUM_STAGES][BLKSZ];
    __shared__ float b_buf[NUM_STAGES][BLKSZ];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

#pragma unroll NUM_STAGES
    for(int stage = 0; stage < NUM_STAGES; ++stage)
    {
        pipe.producer_acquire();
        const int idx = offset + stage * stride;
        if(idx < n)
        {
            cuda::memcpy_async(&a_buf[stage][tid], a + idx, sizeof(float), pipe);
            cuda::memcpy_async(&b_buf[stage][tid], b + idx, sizeof(float), pipe);
        }
        pipe.producer_commit();
    }

    int stage = 0;
#pragma unroll NUM_STAGES
    for (int i = offset; i < n; i += stride)
    {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

        c[i] = a_buf[stage][tid] + b_buf[stage][tid];

        pipe.consumer_release();


        pipe.producer_acquire();
        const int idx = i + NUM_STAGES * stride;
        if(idx < n)
        {
            cuda::memcpy_async(&a_buf[stage][tid], a + idx, sizeof(float), pipe);
            cuda::memcpy_async(&b_buf[stage][tid], b + idx, sizeof(float), pipe);
        }
        pipe.producer_commit();
        
        stage = (stage + 1) % NUM_STAGES;
    }
}

class VectorAddV5 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v5";
    }
    
    void execute(const std::vector<DeviceMemory>& input,
                 void* output,
                 const std::vector<size_t> &input_sizes,
                 const size_t &output_size,
                 cudaStream_t stream = 0) override
    {
        const float *A = (float*)(input[0].data());
        const float *B = static_cast<const float*>(input[1].data());
        float *C       = static_cast<float*>(output);
        
        constexpr int blockSize           = 1024;
        constexpr int num_pipeline_stages = 2;
        int numBlocks                     = 170; // Fixed number of blocks for grid-stride loop
        
        vectorAdd_v5<blockSize, num_pipeline_stages><<<numBlocks, blockSize, 0, stream>>>
                                                                             (A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size;
    }
};

#endif
