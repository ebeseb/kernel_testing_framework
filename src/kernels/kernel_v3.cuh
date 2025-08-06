#ifndef KERNEL_V3_CUH
#define KERNEL_V3_CUH

/* ----- vectorAdd_v3 ------------------------------------------------------------------------------
Grid stride loop and using vectorized loads and stores from global memory.
The kernel has 64kB of data "in flight" per SM at a given time.
If each thread loads a single float coalasced from global memory, thats on cacheline (128bytes).
Now if we switch to float2 or float4, we can load 2 or 4 entire cachelines with a single
instruction. -> More transfer with the same number of instructions.
bytes in flight per SM = # loads / thread
                         # bytes / load
                         # threads / block
                         # blocks / SM
                         = 2 * 16 * 256 * 8 = 64Kb
------------------------------------------------------------------------------------------------- */
__global__ void vectorAdd_v3(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int nrd = n >> 2; // n >> 2 == n / 4
   
    // Process 4 elements per thread using float4
    for (int i = idx; i < nrd; i += stride)
    {
        const float4 a_vec = reinterpret_cast<const float4*>(a)[i];
        const float4 b_vec = reinterpret_cast<const float4*>(b)[i];
        
        float4 c_vec;
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        
        reinterpret_cast<float4*>(c)[i] = c_vec;
    }

    // Handle remaining elements (less than 4)
    const int remaining_idx = idx + nrd * 4;
    if (remaining_idx < n)
    {
        c[remaining_idx] = a[remaining_idx] + b[remaining_idx];
    }
}

class VectorAddV3 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v3";
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
        
        int blockSize = 256;
        int numBlocks = 170; // Fixed number of blocks for grid-stride loop
        
        vectorAdd_v3<<<numBlocks, blockSize, 0, stream>>>(A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size;
    }
};

#endif
