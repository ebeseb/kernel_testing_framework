#ifndef KERNEL_V6_CUH
#define KERNEL_V6_CUH

/* ----- vectorAdd_v6 ------------------------------------------------------------------------------
We can optimize memory loads a little further compared to kernel v5. For 16 bit aligned loads from
global, we can bypass the level 1 cache.
Synchronous (and non 16 byte aligned) copy:
Global Memory -> L2 Cache -> L1 Cache -> Registers -> Shared Memory/Scoreboard
Asynchronous copy:
Only for 16byte aligned datatypes we can BYPASS L1 cache!
Global Memory -> L2 Cache -> Shared Memory/Scoreboard
------------------------------------------------------------------------------------------------- */
#include <cuda/pipeline> // libcudacxx API


template <int BLKSZ, int NUM_STAGES>
__global__ void vectorAdd_v6(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int tid    = threadIdx.x;
    const int offset = blockIdx.x * BLKSZ;
    const int stride = BLKSZ * gridDim.x;
    constexpr int memcpy_threads = BLKSZ / 4;
  

    // NUM_STAGES buffers per input for a NUM_STAGES deep pipeline
    __shared__ float a_buf[NUM_STAGES][BLKSZ];
    __shared__ float b_buf[NUM_STAGES][BLKSZ];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

#pragma unroll NUM_STAGES
    for(int stage = 0; stage < NUM_STAGES; ++stage)
    {
        // Producer - Consumer pattern:
        // A subset of threads load the data, but all threads use the data
        if(tid < memcpy_threads)
        {
            const int global_idx = offset + stage * stride + 4 * tid;
            pipe.producer_acquire();
            if(global_idx < n)
            {
                const int smem_idx = 4 * tid;
                cuda::memcpy_async(&a_buf[stage][smem_idx], a + global_idx, cuda::aligned_size_t<16> (4 * sizeof(float)), pipe);
                cuda::memcpy_async(&b_buf[stage][smem_idx], b + global_idx, cuda::aligned_size_t<16> (4 * sizeof(float)), pipe);
            }
            pipe.producer_commit();
        }
    }

    int stage = 0;
#pragma unroll NUM_STAGES
    for (int i = offset; i < n; i += stride)
    {
        if(tid < memcpy_threads)
        {
            cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
        }        

        __syncthreads(); // need sync threads, because not all threads take part in the data loading
        if(i + tid < n)
        {
            c[i + tid] = a_buf[stage][tid] + b_buf[stage][tid];
        }
        __syncthreads();

        if(tid < memcpy_threads)
        {
            pipe.consumer_release();
        }

        if(tid < memcpy_threads)
        {
            const int global_idx = i + NUM_STAGES * stride + 4 * tid;
            pipe.producer_acquire();

            if(global_idx < n)
            {
                const int smem_idx = 4 * tid;
                cuda::memcpy_async(&a_buf[stage][smem_idx], a + global_idx, cuda::aligned_size_t<16> (4 * sizeof(float)), pipe);
                cuda::memcpy_async(&b_buf[stage][smem_idx], b + global_idx, cuda::aligned_size_t<16> (4 * sizeof(float)), pipe);
            }
            pipe.producer_commit();
        }

        stage = (stage + 1) % NUM_STAGES;
    }

    // We don't need to handle trailing elements here as we had to with kernel v4, because the CUDA runtime
    // will perform a partial load if less than 4 elements remain at the end of the arrayand by that
    // take care of it.
}

class VectorAddV6 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v6";
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
        
        constexpr int blockSize           = 768;
        constexpr int num_pipeline_stages = 2;
        int numBlocks                     = 170; // Fixed number of blocks for grid-stride loop
        
        vectorAdd_v6<blockSize, num_pipeline_stages><<<numBlocks, blockSize, 0, stream>>>
                                                                             (A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size;
    }
};

#endif
