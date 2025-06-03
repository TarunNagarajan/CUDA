#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20) // 2^20 elements

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void verifyResult(const float *A, const float *B, const float *C, int n) {
    for (int i = 0; i < n; ++i) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": " << C[i] << " != " << expected << "\n";
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Result verified! ✅\n";
}

int main() {
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize inputs
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());  // check kernel launch
    CHECK_CUDA(cudaDeviceSynchronize());  // sync to check runtime errors

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    verifyResult(h_A, h_B, h_C, N);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
