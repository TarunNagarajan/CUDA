#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

#define N (1 << 20)

// GPU kernel: square elements in-place
__global__ void squareGPU(float* data, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= data[idx];
    }
}

// CPU version: square elements in-place
void squareCPU(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] *= data[i]; 
    }
}

int main() {
    size_t bytes = sizeof(float) * N;

    // Allocate host memory
    float* h_data_cpu = new float[N];
    float* h_data_gpu = new float[N];

    // Initialize both arrays with the same data
    for (int i = 0; i < N; i++) {
        h_data_cpu[i] = h_data_gpu[i] = 1.0f * i;  // 🔧 Fix: assign values element-wise
    }

    // CPU timing using clock()
    clock_t start_cpu = clock();
    squareCPU(h_data_cpu, N);
    clock_t end_cpu = clock();

    double time_cpu_ms = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    std::cout << "CPU Time: " << time_cpu_ms << " ms\n";

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data_gpu, bytes, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // GPU timing using events
    cudaEventRecord(start);
    squareGPU<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);  // 🔧 Fix: sync `stop`, not whole device

    float time_gpu_ms;
    cudaEventElapsedTime(&time_gpu_ms, start, stop);
    std::cout << "GPU Time: " << time_gpu_ms << " ms\n";

    // Copy result back to host
    cudaMemcpy(h_data_gpu, d_data, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    delete[] h_data_cpu;
    delete[] h_data_gpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
