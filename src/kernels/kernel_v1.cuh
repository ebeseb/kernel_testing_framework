#ifndef KERNEL_V1_CUH
#define KERNEL_V1_CUH

/* ----- vectorAdd_v1 ------------------------------------------------------------------------------
Most simple version of vector add one can do. This has 16Kb of data "in flight" per SM at a given
time.
This is rather low to saturate modern GPU's bandwidth. On H100 this reaches around 80% peak BW,
on B200 only ~50%.
bytes in flight per SM = # loads / thread
                         # bytes / load
                         # threads / block
                         # blocks / SM
                         = 2 * 4 * 256 * 8 = 16Kb
------------------------------------------------------------------------------------------------- */
__global__ void vectorAdd_v1(const float* __restrict__  a,
                             const float* __restrict__  b,
                             float* __restrict__  c,
                             int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

class VectorAddV1 : public KernelTest::KernelVersion
{
public:
    std::string getName() const override
    {
        return "VectorAdd_v1";
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

        const int blockSize = 256;
        const int numBlocks = (output_size + blockSize - 1) / blockSize;

        vectorAdd_v1<<<numBlocks, blockSize, 0, stream>>>(A, B, C, output_size);
    }
    
    size_t getFlops(const size_t &size) const override
    {
        return size; // One addition per element
    }
};

#endif // KERNEL_V1_CUH
