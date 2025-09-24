#ifndef KERNEL_V4_CUH
#define KERNEL_V4_CUH

/* ----- vectorAdd_v4 ------------------------------------------------------------------------------
Grid stride loop and using vectorized loads and stores from global memory.
The kernel has 128kB of data "in flight" per SM at a given time.
This is mostly a demo kernel for manual loop unrolling in case the compiler isn't fully reordering
all loads infront of the computations. A problem potentially happening in kernel v2.
bytes in flight per SM = # loads / thread
                         # bytes / load
                         # threads / block
                         # blocks / SM
                         = 4 * 16 * 256 * 8 = 128Kb
------------------------------------------------------------------------------------------------- */
template <int BLKSZ, int UNROLL_FACTOR>
__global__ void vectorAdd_v4(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int tid    = blockIdx.x * BLKSZ * UNROLL_FACTOR + threadIdx.x;
    const int stride = BLKSZ * UNROLL_FACTOR * gridDim.x;

    const int nrd = n >> 2; // n >> 2 == n / 4
   
    // Process 8 elements per thread using float4
    for (int i = tid; i < nrd; i += stride)
    {
        // Manual unrolling for 2 * 4 elements (if unrolling factor is 2)
#pragma unroll UNROLL_FACTOR
        for(int ii = 0; ii < UNROLL_FACTOR; ++ii)
        {
            const int idx = i + ii * BLKSZ;
            if(idx < nrd)
            {
                const float4 a_vec = reinterpret_cast<const float4*>(a)[idx];
                const float4 b_vec = reinterpret_cast<const float4*>(b)[idx];
                
                float4 c_vec;
                c_vec.x = a_vec.x + b_vec.x;
                c_vec.y = a_vec.y + b_vec.y;
                c_vec.z = a_vec.z + b_vec.z;
                c_vec.w = a_vec.w + b_vec.w;
                
                reinterpret_cast<float4*>(c)[idx] = c_vec;
            }
        }
    }

    // Handle remaining elements (less than 4)
    const int remaining_idx = tid + (n / (4 * UNROLL_FACTOR)) * (4 * UNROLL_FACTOR);
    if (remaining_idx < n)
    {
        c[remaining_idx] = a[remaining_idx] + b[remaining_idx];
    }
}

class VectorAddV4 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v4";
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
        
        constexpr int blockSize     = 256;
        constexpr int unroll_factor = 2;
        int numBlocks               = 170; // Fixed number of blocks for grid-stride loop
        
        vectorAdd_v4<blockSize, unroll_factor><<<numBlocks, blockSize, 0, stream>>>(A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size;
    }
};

#endif
