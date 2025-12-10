# project basics
cmake_minimum_required(VERSION 3.18)
project(kernel_testing_framework LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)

# Find all files .cu/.cpp in src/
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(kernel_test.x ${PROJECT_SOURCES})

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set GPU architecture
set_target_properties(kernel_test.x PROPERTIES
     CUDA_ARCHITECTURES "120")


# Link against the CUDA runtime
target_link_libraries(kernel_test.x PRIVATE CUDA::cudart)

# Configure different builds
# Debug
target_compile_options(kernel_test.x PRIVATE
    $<$<CONFIG:Debug>:-O0 -g -DDEBUG>                                # all languages
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -lineinfo > # CUDA
)

# Release
target_compile_options(kernel_test.x PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -lineinfo -ffast-math>
)

