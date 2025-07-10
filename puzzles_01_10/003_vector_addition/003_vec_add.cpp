#include <iostream>
#include <cuda_runtime.h> 

#define N (1 << 20)     
#define THREADS_PER_BLOCK 256

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Hello NVIDIA GTX TITAN X from blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
    }

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void verify(const float* A, const float* B, const float* C, int n) {
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (fabs(expected - C[i]) > 1e-5f) {
            std::cerr << "Mismatch occured at the index " << i << " != " << expected << "\n";
            return; 
        } 
    }

    std::cout << "Result Verified.\n";
}

int main() {
    size_t bytes = sizeof(float) * N; 

    // host memory, on the cpu.
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // device memory, on the gpu
    float *d_a; 
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // launch kernel
    int blocks = (N - 1 + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); 
    
    verify(h_a, h_b, h_c, N);

    // memory cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
    
}
