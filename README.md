# Simple Kernel Testing Framework
## Installation and Usage
To compile and run the framework, navigate to the src folder and run
`nvcc -o kernel_test.x main.cu test_framework.cpp`
Then run `./kernel_test.x`


### Docker and Dev Container
This repo includes a `Dockerfile` and `devcontainer.json` for portability.

Build the docker image with
`docker buildx build -t cuda-dev-container .`

The container is currently based on Ubuntu 24.04 LTS with CUDA 13.0. It has fully working NSight profilers installed, they should be able to run as CLI and GUI versions. Apart from Dev Containers, the container can be run standalone, but I would recommend running it using the Dev Containers Plugin in VSCode.

Run it manually from the root of your CUDA code:
`docker run --gpus all --runtime=nvidia -it --rm --network host -v $(pwd):/workspaces/code -v vscode:/vscode -e DISPLAY=$DISPLAY --cap-add=SYS_ADMIN --security-opt seccomp=unconfined cuda-dev-container /bin/bash`

If you want to run the GUI version of the profilers after manually launching the kernel, run
`xhost +`
before launching the container. Remember to run
`xhost -`
after you are done.


### Usage
The framework is designed to be easy to use and extend. To add a new kernel, you only need to create a class that inherits from KernelVersion and implement the execute() method (see any of the kernels in the kernels folder). Then, register the kernel with the framework using the registerKernel() method (see main.cu for an example). Adjust your kernel input and outputs to whatever your kernels need, see the kernel implementations an main.cu for how this can be done.


## Features Overview
Minimal and easy to use testing framework for CUDA kernels.

Core Framework Components:
- KernelVersion:       Abstract base class for kernel implementations
- KernelTestFramework: Main testing framework class
- TestConfig:          Configuration structure for test parameters
- TestResult:          Structure to hold test results
- DeviceMemory:        RAII wrapper for CUDA memory management

Key Features:
- Select version(s) to test by name
- Configurable warmup runs and measurement runs
- Statistical analysis (mean, min, max, standard deviation)
- GFLOPS calculation support
- CUDA events for precise timing
- Reference output comparison
- Configurable (relative) tolerance levels
