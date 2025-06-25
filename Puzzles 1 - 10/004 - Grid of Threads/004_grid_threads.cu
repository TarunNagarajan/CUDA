#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 20)                 // Total number of elements = 2^20 = 1,048,576
#define THREADS_PER_BLOCK 256       // Threads per block

// Kernel function: adds vectors A and B into C
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Print info for one sample thread
    if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("Sample thread at (blockIdx.x = %d, threadIdx.x = %d)\n", blockIdx.x, threadIdx.x);
    }

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host-side verification function
void verify(const float* A, const float* B, const float* C, int n) {
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (fabs(expected - C[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": got " << C[i] << ", expected " << expected << "\n";
            return;
        }
    }
    std::cout << "Result Verified.\n";
}

int main() {
    size_t bytes = sizeof(float) * N;

    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    verify(h_a, h_b, h_c, N);

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
