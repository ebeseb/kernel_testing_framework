# Simple Kernel Testing Framework
## Installation and Usage
Simply navigate to the src folder and run
`nvcc -o kernel_test.x main.cu test_framework.cpp`
Then run kernel_test.x

## Overview
Minimal and easy to use testing framework for CUDA kernels.

Core Framework Components:

KernelVersion:       Abstract base class for kernel implementations
KernelTestFramework: Main testing framework class
TestConfig:          Configuration structure for test parameters
TestResult:          Structure to hold test results
DeviceMemory:        RAII wrapper for CUDA memory management

Key Features:
1. Easy Kernel Management
Add new kernel versions with registerKernel() in the framework
Select version(s) to test by name
Kernels implement the KernelVersion class

2. Performance Measurement
Configurable warmup runs and measurement runs
Statistical analysis (mean, min, max, standard deviation)
GFLOPS calculation support
CUDA events for precise timing

3. Correctness Checking
Reference output comparison
Configurable (relative) tolerance levels
