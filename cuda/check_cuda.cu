#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned error code " << error_id
                  << " (" << cudaGetErrorString(error_id) << ")\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices detected.\n";
        return 0;
    }

    std::cout << "Detected " << deviceCount << " CUDA-capable device(s).\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": " << deviceProp.name << "\n";
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << "\n";
    }

    return 0;
}
