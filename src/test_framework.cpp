#include "test_framework.hpp"

namespace KernelTest {

/* ----- addKernel ---------------------------------------------------------------------------------
Adds a kernel in to the list of kernels to be tested by the testAllKernels function.
------------------------------------------------------------------------------------------------- */
void KernelTestFramework::registerKernel(std::unique_ptr<KernelVersion> kernel)
{
    const std::string name = kernel->getName();
    _kernels[name] = std::move(kernel);
}

/* ----- setInputData ------------------------------------------------------------------------------
Function used to register all inputs that go to a kernel (all kernels same inputs).
------------------------------------------------------------------------------------------------- */
void KernelTestFramework::setInputData(const void* data,
                                       const size_t &size,
                                       const size_t &byte_size)
{
    if(data != nullptr && size > 0 && byte_size > 0)
    {
        _input_data.push_back(data);
        _input_size.push_back(size);
        _input_byte_size.push_back(byte_size);
    }
}

/* ----- setReferenceOutput ------------------------------------------------------------------------
Function used to register the reference solution.
------------------------------------------------------------------------------------------------- */
void KernelTestFramework::setReferenceOutput(void* reference,
                                             const size_t &size,
                                             const size_t &byte_size)
{
    if(reference != nullptr)
    {
        _reference_output = reference;
        _output_size      = size;
        _output_byte_size = byte_size;
    }
}

/* ----- testAllKernels ----------------------------------------------------------------------------
Goes through the kernel registry and runs all kernels.
------------------------------------------------------------------------------------------------- */
std::vector<TestResult> KernelTestFramework::testAllKernels(const TestConfig& config)
{
    std::vector<TestResult> results;
    
    for (const auto& [name, kernel] : _kernels)
    {
        results.push_back(testKernel(name, config));
    }
    
    return results;
}

/* ----- testKernel --------------------------------------------------------------------------------
Runs a single kernel. Uses the config to optionally run performance and correctness tests.
------------------------------------------------------------------------------------------------- */
TestResult KernelTestFramework::testKernel(const std::string& kernel_name,
                                           const TestConfig& config)
 {
    TestResult result(kernel_name);
    
    auto it = _kernels.find(kernel_name);
    if (it == _kernels.end())
    {
        result.error_message = "Kernel not found: " + kernel_name;
        result.is_correct = false;

        return result;
    }
    
    KernelVersion* kernel = it->second.get();
    
    try
    {
        if (config.test_correctness)
        {
            result.is_correct = checkCorrectness(kernel, config);
            if (!result.is_correct)
            {
                result.error_message = "Correctness check failed";
            }
        }

        if (config.measure_performance)
        {
            result.performance = measurePerformance(kernel, config);
        }
        
    }
    catch (const std::exception& e)
    {
        result.error_message = "Exception during testing: " + std::string(e.what());
        result.is_correct    = false;
    }
    
    return result;
}

/* ----- printResults ------------------------------------------------------------------------------
Print a table of correctness and performance results for all kernels.
------------------------------------------------------------------------------------------------- */
void KernelTestFramework::printResults(const std::vector<TestResult>& results,
                                       const bool &show_performance)
{
    std::cout << "\n=== Kernel Test Results ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Kernel Name" 
              << std::setw(12) << "Correctness";
    
    if (show_performance)
    {
        std::cout << std::setw(12) << "Mean (ms)" 
                  << std::setw(12) << "Min (ms)" 
                  << std::setw(12) << "Max (ms)" 
                  << std::setw(12) << "Std Dev" 
                  << std::setw(12) << "GFLOPS";
    }
    
    std::cout << std::setw(30) << "Error Message" << std::endl;
    std::cout << std::string(show_performance ? 130 : 62, '-') << std::endl;
    
    for (const auto& result : results)
    {
        std::cout << std::left << std::setw(20) << result.kernel_name
                  << std::setw(12) << (result.is_correct ? "PASS" : "FAIL");
        
        if (show_performance)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << result.performance.mean_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.performance.min_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.performance.max_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.performance.std_dev_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.performance.throughput_gflops;
        }
        
        std::cout << std::setw(30) << result.error_message << std::endl;
    }
    std::cout << std::endl;
}

/* ----- checkCorrectness --------------------------------------------------------------------------
Runs a kernel once and compares the result with the previously registered 
------------------------------------------------------------------------------------------------- */
bool KernelTestFramework::checkCorrectness(KernelVersion* kernel, const TestConfig& config)
{
    if (_reference_output == nullptr)
    {
        std::cerr << "No reference output provided for correctness check" << std::endl;

        return false;
    }
    
    std::vector<DeviceMemory> d_input;
    for(int i = 0; i < _input_data.size(); ++i)
    {
        d_input.push_back(DeviceMemory(_input_size[i], _input_byte_size[i]));
        d_input[i].copyToDevice(_input_data[i]);
    }

    DeviceMemory d_output(_output_size, _output_byte_size);
    
    kernel->execute(d_input, d_output.data(), _input_size, _output_size);
    
    std::vector<float> kernel_output(_output_size);
    d_output.copyToHost(kernel_output.data());
    
    // Compare with reference
    for (int i = 0; i < _output_size; ++i)
    {
        const float diff    = std::abs(kernel_output[i] - static_cast<float*>(_reference_output)[i]);
        const float max_val = std::max(std::abs(kernel_output[i]), 
                                       std::abs(static_cast<float*>(_reference_output)[i]));
        
        // Compute relative error, check max_val > 0 to prevent division by 0. If max_val == 0 then
        // both elements are 0, meaning equal. So no tolerance violated.
        if (max_val > 0)
        {
            const float relative_error = static_cast<float>(diff / max_val);
            if (relative_error > config.rtol)
            {
                if (config.verbose)
                {
                    std::cout << "Mismatch at index " << i << ": kernel=" << kernel_output[i] 
                              << ", reference=" << static_cast<float*>(_reference_output)[i] 
                              << ", relative_error=" << relative_error << std::endl;
                }
                return false;
            }
        }
    }
    
    return true;
}

/* ----- measurePerformance ------------------------------------------------------------------------
Runs a kernel in a loop for the number of times specified in the config and averages the runtime
of it.
------------------------------------------------------------------------------------------------- */
PerformanceResult KernelTestFramework::measurePerformance(KernelVersion* kernel,
                                                          const TestConfig& config)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(0.0f, config.number_trials);

    std::vector<DeviceMemory> d_input;
    for(int i = 0; i < _input_data.size(); ++i)
    {
        d_input.push_back(DeviceMemory(_input_size[i], _input_byte_size[i]));
        d_input[i].copyToDevice(_input_data[i]);
    }

    DeviceMemory d_output(_output_size, _output_byte_size);
   
    // 1. Warmup runs
    for (int i = 0; i < config.number_warmups; ++i)
    {
        kernel->execute(d_input, d_output.data(), _input_size, _output_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 2. Measurement runs
    for (int i = 0; i < config.number_trials; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        kernel->execute(d_input, d_output.data(), _input_size, _output_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize()); 
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);

        d_output.setZero();
    }
    
    // Calculate statistics
    float mean     = 0.0f;
    float min_time = times[0];
    float max_time = times[0];
    
    for (float time : times)
    {
        mean    += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    mean /= times.size();
    
    float variance = 0.0f;
    for (float time : times)
    {
        variance += (time - mean) * (time - mean);
    }
    variance     /= times.size();
    const float std_dev = std::sqrt(variance);
    
    // Calculate throughput
    const float flops      = static_cast<float>(kernel->getFlops(_output_size));
    const float throughput = (flops / (mean * 1e-3f)) / 1e9f; // GFLOPS
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return PerformanceResult(mean, min_time, max_time, std_dev, throughput);
}

}
