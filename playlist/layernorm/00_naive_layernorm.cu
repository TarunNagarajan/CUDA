#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

/*

First 10 normalized values from the first row:
-1.66936 -1.63502 -1.60068 -1.56633 -1.53199 -1.49765 -1.46331 -1.42896 -1.39462 -1.36028 

NVIDIA GTX TITAN X
Host to Device Transfer: 0.632353 ms
Kernel Execution: 8.54171 ms
Device to Host Transfer: 0.718903 ms
Exit status: 0

*/

__global__ void naive_layernorm(float* matrix, float* output, int m, int n, float epsilon = 1e-6f) {
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row >= m) return;

    // PASS I: Mean
    float mean = 0.0f;
    for (int col = 0; col < n; col++) {
        int idx = row * n + col;
        mean += matrix[idx];
    }
    mean /= n;

    // PASS II: Variance
    float variance = 0.0f;
    for (int col = 0; col < n; col++) {
        int idx = row * n + col;
        float diff = matrix[idx] - mean;
        variance += diff * diff;
    }
    variance /= n;

    // PASS III: Layernorm
    float stdev = sqrtf(variance + epsilon);
    for (int col = 0; col < n; col++) {
        int idx = row * n + col;
        output[idx] = (matrix[idx] - mean) / stdev;
    }
}

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main() {
    int m = 1024;
    int n = 1024;
    int SIZE = m * n;

    cudaEvent_t startH2D, startKernel, startD2H, stop;
    CHECK_CUDA(cudaEventCreate(&startH2D)); 
    CHECK_CUDA(cudaEventCreate(&startKernel));
    CHECK_CUDA(cudaEventCreate(&startD2H));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);

    float* h_matrix = new float[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_matrix[i] = static_cast<float>(i % 100); // fix: i -> i % 100 for better test values
    }

    float* d_matrix;
    CHECK_CUDA(cudaMalloc((void**)&d_matrix, SIZE * sizeof(float)));

    float* d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_output, SIZE * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(startH2D));
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(startKernel));
    naive_layernorm<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_output, m, n, 1e-6f);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(startD2H));
    float* h_output = new float[SIZE];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop)); 
    CHECK_CUDA(cudaEventSynchronize(stop));

    std::cout << "First 10 normalized values from the first row:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    float h2dTime, kernelTime, d2hTime; 
    CHECK_CUDA(cudaEventElapsedTime(&h2dTime, startH2D, startKernel));
    CHECK_CUDA(cudaEventElapsedTime(&kernelTime, startKernel, startD2H));
    CHECK_CUDA(cudaEventElapsedTime(&d2hTime, startD2H, stop)); 

    std::cout << "\n";
    std::cout << "NVIDIA GTX TITAN X\n";
    std::cout << "Host to Device Transfer: " << h2dTime << " ms" << std::endl;
    std::cout << "Kernel Execution: " << kernelTime << " ms" << std::endl;
    std::cout << "Device to Host Transfer: " << d2hTime << " ms" << std::endl;

    delete[] h_matrix;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
