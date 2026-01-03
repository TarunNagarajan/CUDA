#include <cuda_runtime.h>

__global__ void convkern(const float* A, const float* B, float* C, size_t N, size_t K, int PAD) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    for (int j = 0; j < K; j++) {
        int idx = i + j - PAD;
        if (idx >= 0 && idx < N) {
            sum += A[idx] * B[j];
        }
    }

    C[i] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int PAD = K / 2;
    int NUM_THREADS =  256;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    convkern<<<NUM_BLOCKS, NUM_THREADS>>>(A, B, C, N, K, PAD);
}
