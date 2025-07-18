#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

#define N (1 << 20) // 1M elements

__global__ void squareGPU(float* data, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= data[idx];
    }
}

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

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data_cpu[i] = h_data_gpu[i] = static_cast<float>(i);
    }

    // CPU timing
    clock_t start_cpu = clock();
    squareCPU(h_data_cpu, N);
    clock_t end_cpu = clock();

    double time_cpu_ms = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    std::cout << "CPU Time: " << time_cpu_ms << " ms\n";

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, bytes);

    // Create events
    cudaEvent_t startH2D, startKernel, startD2H, stop;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stop);

    // Launch config
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Segment 1: H2D
    cudaEventRecord(startH2D);
    cudaMemcpy(d_data, h_data_gpu, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(startKernel);

    // Segment 2: Kernel execution
    squareGPU<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(startD2H);

    // Segment 3: D2H
    cudaMemcpy(h_data_gpu, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    // Sync and measure
    cudaEventSynchronize(stop);

    float timeH2D, timeKernel, timeD2H;
    cudaEventElapsedTime(&timeH2D, startH2D, startKernel);
    cudaEventElapsedTime(&timeKernel, startKernel, startD2H);
    cudaEventElapsedTime(&timeD2H, startD2H, stop);

    std::cout << "H2D Transfer Time: " << timeH2D << " ms\n";
    std::cout << "Kernel Execution Time: " << timeKernel << " ms\n";
    std::cout << "D2H Transfer Time: " << timeD2H << " ms\n";

    for (int i = 0; i < N; ++i) {
        if (fabs(h_data_cpu[i] - h_data_gpu[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" 
                      << h_data_cpu[i] << ", GPU=" << h_data_gpu[i] << "\n";
            break;
        }
    }

    // Clean up
    cudaFree(d_data);
    delete[] h_data_cpu;
    delete[] h_data_gpu;

    cudaEventDestroy(startH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stop);

    return 0;
}
