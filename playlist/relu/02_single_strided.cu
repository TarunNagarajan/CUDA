#include <cuda_runtime.h>

__global__ void relukern(const float* input, float* output, size_t totalsize) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < totalsize; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t totalsize = m * n; 

    int threads = 256;
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm_count * 32;
    
    relukern<<<blocks, threads>>>(input, output, totalsize);
}
