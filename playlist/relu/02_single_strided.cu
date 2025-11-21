#include <cuda_runtime.h>

__global__ void relukern(const float* input, float* output, size_t totalsize) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < totalsize) {
        output[idx] = fmaxf(0.0f, input[idx]); 
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t totalsize = m * n; 

    int threads = 256;
    int blocks = (totalsize + threads - 1) / threads;
    
    relukern<<<blocks, threads>>>(input, output, totalsize);
}
