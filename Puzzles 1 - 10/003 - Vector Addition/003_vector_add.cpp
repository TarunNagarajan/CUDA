#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20)          // 2^20 elements
#define THREADS_PER_BLOCK 256

// CUDA kernel: vector add and print one thread's info
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Hello from blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
    }

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Verifies results on the host
void verifyResult(const float* A, const float* B, const float* C, int n) {
    for (int i = 0; i < n; ++i) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": " << C[i] << " != " << expected << "\n";
            return;
        }
    }
    std::cout << "Result verified! ✅\n";
}

int main() {
    size_t bytes = N * sizeof(float);

    // Host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize inputs
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify
    verifyResult(h_A, h_B, h_C, N);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
