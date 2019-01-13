
#include <assert.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda/api_wrappers.hpp"  // -Icuda-api-wrappers/src/

// ================================================================ utils

// #define GPU_ERR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess)
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }


// #define CUDA_MALLOC()



// ================================================================ kernels

// template<class DataT> __global__ void
// vectorAdd(const DataT* A, const DataT* B, DataT* C, int numElements)
__global__ void
vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

// template void vectorAdd<float>(const float* A, const float* B, float* C, int numElements);

// ================================================================ main

int main(int argc, const char* argv[]) {

    // cudaError_t err = cudaSuccess;
    using DataT = float;
    static const int nelements = 50*1000;
    static const int nbytes = nelements * sizeof(DataT);

    // DataT* a_h = (DataT*)malloc(nbytes);
    // DataT* b_h = (DataT*)malloc(nbytes);
    // DataT* c_h = (DataT*)malloc(nbytes);
    auto a_h = std::make_unique<DataT[]>(nbytes);
    auto b_h = std::make_unique<DataT[]>(nbytes);
    auto c_h = std::make_unique<DataT[]>(nbytes);

    // initialize inputs
    for (int i = 0; i < nelements; i++) {
        a_h[i] = 2 * i;
        b_h[i] = nelements - i;
    }

    // alloc device vecs

    // DataT* a_d = nullptr;
    // DataT* b_d = nullptr;
    // DataT* c_d = nullptr;
    // err = cudaMalloc((void**)&a_d, sz);
    // assert(err == cudaSuccess);
    // err = cudaMalloc((void**)&b_d, sz);
    // assert(err == cudaSuccess);

    // using namespace cuda::device; // does this compile?
    // using namespace cuda::device::current; // does this compile?

    auto dev = cuda::device::current::get();
    auto a_d = cuda::memory::device::make_unique<DataT[]>(dev, nelements);
    auto b_d = cuda::memory::device::make_unique<DataT[]>(dev, nelements);
    auto c_d = cuda::memory::device::make_unique<DataT[]>(dev, nelements);

    cuda::memory::copy(a_d.get(), a_h.get(), nbytes);
    cuda::memory::copy(b_d.get(), b_h.get(), nbytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (nelements + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid
        << " blocks of " << threadsPerBlock << " threads\n";

    cuda::launch(vectorAdd,
        {blocksPerGrid, threadsPerBlock},
        a_d.get(), b_d.get(), c_d.get(), nelements);

    cuda::memory::copy(c_h.get(), c_d.get(), nbytes);

    printf("vec_add: main done\n");
    return 0;
}
