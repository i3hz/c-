#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;

// ---------------- CPU functions ----------------
void initWith(float num, float *a, int N)
{
    for(int i = 0; i < N; ++i)
        a[i] = num;
}

void addVectorsCPU(float *result, float *a, float *b, int N)
{
    for(int i = 0; i < N; i++)
        result[i] = a[i] + b[i];
}

void checkElementsAre(float target, float *array, int N)
{
    for(int i = 0; i < N; i++)
    {
        if(array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

// ---------------- GPU kernel ----------------
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridstride = gridDim.x * blockDim.x;

    for(int i = idx; i < N; i += gridstride)
    {
        result[i] = a[i] + b[i];
    }
}

// ---------------- Main ----------------
int main()
{
    const int N = 10 << 20; // ~10 million elements
    size_t size = N * sizeof(float);

    float *a, *b, *c;       // CPU arrays
    float *d_a, *d_b, *d_c; // GPU arrays

    // Allocate CPU memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize CPU arrays
    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    // ---------------- CPU Benchmark ----------------
    auto start_cpu = high_resolution_clock::now();
    addVectorsCPU(c, a, b, N);
    auto end_cpu = high_resolution_clock::now();
    double cpu_time = duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("CPU time: %.3f ms\n", cpu_time);

    // ---------------- GPU Benchmark ----------------
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy CPU arrays to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    addVectorsInto<<<numBlocks, blockSize>>>(d_c, d_a, d_b, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result back to CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("GPU time: %.3f ms\n", gpu_time);

    // ---------------- Verify results ----------------
    checkElementsAre(7, c, N);

    // ---------------- Cleanup ----------------
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
