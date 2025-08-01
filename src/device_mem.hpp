#ifndef DEVICE_MEM_HPP_
#define DEVICE_MEM_HPP_

#include "includes.h"

/* -------------------------------------------------------------------------------------------------
Class: DeviceMemory

Memory management helper. Allocation / Deallocation as well as copy to and from device. This
class takes the approach of using void pointers and an explicitly passed in size in bytes per
element in the buffer to be allocated, instead of templates.
------------------------------------------------------------------------------------------------- */
class DeviceMemory {
private:
    void*                           _ptr       = nullptr;       // pointer to the data buffer
    size_t                          _size      = 0;             // number of elements in the buffer
    size_t                          _byte_size = sizeof(float); // size of each element in the
                                                                // buffer in bytes
    
public:
                                    DeviceMemory(size_t size, size_t byte_size = sizeof(float))
        : _size(size)
        , _byte_size(byte_size)             
    {
        CUDA_CHECK(cudaMalloc((void**)&_ptr, size * sizeof(byte_size)));
        setZero();
    }
    
                                    ~DeviceMemory()
    {
        if (_ptr)
        {
            CUDA_CHECK(cudaFree((void*)_ptr));
            _ptr = nullptr;
        }
    }
    
                                // Move constructor
                                    DeviceMemory(DeviceMemory&& other) noexcept
        : _ptr(other._ptr)
        , _size(other._size)
    {
        other._ptr  = nullptr;
        other._size = 0;
    }
    
                                // Move assignment
    DeviceMemory&                   operator=(DeviceMemory&& other) noexcept
    {
        if (this != &other)
        {
            if (_ptr) cudaFree(_ptr);
            _ptr        = other._ptr;
            _size       = other._size;
            other._ptr  = nullptr;
            other._size = 0;
        }

        return *this;
    }
    
                                // Delete copy constructor and assignment
                                    DeviceMemory(const DeviceMemory&)            = delete;
                                    DeviceMemory& operator=(const DeviceMemory&) = delete;
                                // Modifiers
    void*                           data() const { return _ptr;  }
    size_t                          size() const { return _size; }
                                // Copy functions
    void                            copyToDevice(const void* host_data)
    {
        CUDA_CHECK(cudaMemcpy(_ptr, host_data, _size * _byte_size, cudaMemcpyHostToDevice));
    }
    
    void                            copyToHost(void* host_data)
    {
        CUDA_CHECK(cudaMemcpy(host_data, _ptr, _size * _byte_size, cudaMemcpyDeviceToHost));
    }
                                // Default initialization
    void                            setZero()
    {
        CUDA_CHECK(cudaMemset(_ptr, 0, _size * _byte_size));
    }
};

#endif
