#ifndef KERNEL_V2_CUH
#define KERNEL_V2_CUH

/* ----- vectorAdd_v2 ------------------------------------------------------------------------------
Changed kernel to a grid stride loop and use automatic loop unrolling. This has 32Kb of data
"in flight" per SM at a given time.
bytes in flight per SM = # loads / thread
                         # bytes / load
                         # threads / block
                         # blocks / SM
                         = 4 * 4 * 256 * 8 = 32Kb
Most likely this is not reached as the compiler will not assume it is safe to unroll. So some loads
might be done inbetween the add operations. See kernel v4 for a solution to this behavior.
------------------------------------------------------------------------------------------------- */
__global__ void vectorAdd_v2(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

#pragma unroll 2
    for (int i = idx; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

class VectorAddV2 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v2";
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
        
        vectorAdd_v2<<<numBlocks, blockSize, 0, stream>>>(A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size;
    }
};

#endif // KERNEL_V2_CUH
