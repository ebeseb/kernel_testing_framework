/* ***** STD includes *************************************************************************** */
#include <vector>
#include <iostream>

/* ***** framework  and kernel includes ********************************************************* */
#include "test_framework.hpp"
#include "kernels/kernel_v1.cuh"
#include "kernels/kernel_v2.cuh"
#include "kernels/kernel_v3.cuh"
#include "kernels/kernel_v4.cuh"
#include "kernels/kernel_v5.cuh"
#include "kernels/kernel_v6.cuh"

// Compile with CMake:
// mkdir build
// cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Debug
// or
// cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Release
// cmake --build build/


int main()
{
    /* ***** Run Setup ************************************************************************** */
    const int number_warmups = 100;
    const int number_trials  = 1000;
    const float atol         = 1e-6f;    
    const size_t N           = (1 << 24) + 3;
    

    /* ***** Setup test environment and register kernels with it ******************************** */
    KernelTest::KernelTestFramework framework;
    
    framework.registerKernel(std::make_unique<VectorAddV1>());
    framework.registerKernel(std::make_unique<VectorAddV2>());
    framework.registerKernel(std::make_unique<VectorAddV3>());
    framework.registerKernel(std::make_unique<VectorAddV4>());
    framework.registerKernel(std::make_unique<VectorAddV5>());
    framework.registerKernel(std::make_unique<VectorAddV6>());

    auto config = KernelTest::TestConfig()
        .withPerformance(number_trials, number_warmups)
        .withCorrectness(atol)
        .withVerbose(true);
    

    /* ***** Input data and Reference Solution ************************************************** */
    std::vector<float> input_a(N);
    std::vector<float> input_b(N);
    std::vector<float> reference_output(N);
    
    // If this needs to be more complex, write a function and call it here.
    for (size_t i = 0; i < N; ++i)
    {
        input_a[i]          = static_cast<float>(i);     // vector a
        input_b[i]          = static_cast<float>(i + 1); // vector b
        reference_output[i] = input_a[i] + input_b[i];   // a + b
    }
    
    // Register host side data in framework
    framework.setInputData((void*)input_a.data(), input_a.size(), sizeof(float));
    framework.setInputData((void*)input_b.data(), input_b.size(), sizeof(float));

    framework.setReferenceOutput((void*)reference_output.data(), 
                                 reference_output.size(),
                                 sizeof(float));
    

    /* ***** Run all Kernels that were added **************************************************** */
    auto results = framework.testAllKernels(config);
    
    /* ***** Print Results ********************************************************************** */
    KernelTest::KernelTestFramework::printResults(results, true);
    
    return 0;
}
