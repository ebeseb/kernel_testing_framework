#ifndef CUDA_KERNEL_FRAMEWORK2_HPP_
#define CUDA_KERNEL_FRAMEWORK2_HPP_

#include <map>
#include <cmath>
#include <vector>
#include <memory>
#include <iomanip>
#include <iostream>

#include "includes.h"
#include "device_mem.hpp"


namespace KernelTest {

// Configuration structure for test parameters
struct TestConfig {
    bool                            measure_performance = false;
    bool                            test_correctness    = false;
    int                             number_warmups      = 100;
    int                             number_trials       = 1000;
    float                           rtol                = 1e-6f;
    bool                            verbose             = false;
    
    TestConfig() = default;
                                // Set performance measurement options for running kernels
    TestConfig&                     withPerformance(const int &runs = 1000, const int &warmup = 100)
    {
        measure_performance = true;
        number_trials       = runs;
        number_warmups      = warmup;

        return *this;
    }
    
                                // Set options for running kernels
    TestConfig&                     withCorrectness(const float &tol = 1e-6)
    {
        test_correctness = true;
        rtol             = tol;

        return *this;
    }
    
                                // Set options for running kernels
    TestConfig&                     withVerbose(const bool &v = true)
    {
        verbose = v;

        return *this;
    }
};

// Performance measurement results
struct PerformanceResult {
    float                           mean_time_ms      = 0.0f;
    float                           min_time_ms       = 0.0f;
    float                           max_time_ms       = 0.0f;
    float                           std_dev_ms        = 0.0f;
    float                           throughput_gflops = 0.0f;
    
    PerformanceResult() = default;
    PerformanceResult(const float &mean, const float &min, const float &max , const float &std,
                      const float &tput)
        : mean_time_ms(mean)
        , min_time_ms(min)
        , max_time_ms(max)
        , std_dev_ms(std)
        , throughput_gflops(tput)
    {}
};

// Test result structure
struct TestResult {
    std::string                     kernel_name   = "";
    bool                            is_correct    = false;
    PerformanceResult               performance   = {};
    std::string                     error_message = "";
    
    TestResult(const std::string& name)
        : kernel_name(name)
        , is_correct(true) 
    {}
};

/* -------------------------------------------------------------------------------------------------
Class: KernelVersion

Abstract base class for kernel implementations
I keep this here in this file, because then I don't have to explicitly include this class in
the kernel implementations.
------------------------------------------------------------------------------------------------- */
class KernelVersion {
public:
    virtual                         ~KernelVersion() = default;
                                // Return the kernel name string
    virtual std::string             getName() const  = 0;
                                // This is the main function for executing kernels
    virtual void                    execute(const std::vector<DeviceMemory>& input,
                                            void* output,
                                            const std::vector<size_t> &input_sizes,
                                            const size_t &output_size,
                                            cudaStream_t stream = 0) = 0;
                                // If possible add a metric in this function's implementation
                                // The size parameter is the number of operations done
    virtual size_t                  getFlops(const size_t &size) const { return 0; }
};


/* -------------------------------------------------------------------------------------------------
Class: KernelTestFramework

Main class to test different implementations of GPU kernels. It uses TestConfig to configure it's
runs.
------------------------------------------------------------------------------------------------- */
class KernelTestFramework {
public:
                                    KernelTestFramework()
                                            : _reference_output(nullptr)
                                            , _output_size(0) {}
                                // Add a kernel version
    void                            registerKernel(std::unique_ptr<KernelVersion> kernel);
                                // Set input data
    void                            setInputData(const void* data,
                                                 const size_t &size,
                                                 const size_t &byte_size = sizeof(float));
                                // Set reference output for correctness checking
    void                            setReferenceOutput(void* reference,
                                                       const size_t &size,
                                                       const size_t &byte_size = sizeof(float));
                                // Test all kernels
    std::vector<TestResult>         testAllKernels(const TestConfig& config = TestConfig());
                                // Test a specific kernel
    TestResult                      testKernel(const std::string& kernel_name,
                                               const TestConfig& config = TestConfig());
                                // Print a summary of performance and correctness results
    static void                     printResults(const std::vector<TestResult>& results,
                                                 const bool &show_performance = true);
private:
                                // Correctness checking
    bool                            checkCorrectness(KernelVersion* kernel,
                                                     const TestConfig& config);
                                // Performance measurement
    PerformanceResult               measurePerformance(KernelVersion* kernel,
                                                       const TestConfig& config);

private:
    std::map<std::string, std::unique_ptr<KernelVersion>> _kernels; // kernel registry

    std::vector<const void*>        _input_data;      // pointers to input fields
    std::vector<size_t>             _input_size;      // number of elements per field
    std::vector<size_t>             _input_byte_size; // size per element in byte

    void*                           _reference_output; // pointer to output field
                                                       // (only support one output atm)
    size_t                          _output_size;      // number of elements in the field
    size_t                          _output_byte_size; // size per element in byte
};

}

#endif
