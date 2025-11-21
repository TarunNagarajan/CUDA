#include <cuda_runtime.h>

__device__ inline float4 reluvec(float4 v) {
    float4 result;
    result.x = fmaxf(0.0f, v.x);
    result.y = fmaxf(0.0f, v.y);
    result.z = fmaxf(0.0f, v.z);
    result.w = fmaxf(0.0f, v.w);

    return result;
}

__global__ void relukern(const float* input, float* output, size_t totalsize) {
    const float4* invector = reinterpret_cast<const float4*>(input);
    float4* outvector = reinterpret_cast<float4*>(output);

    int vectors = totalsize / 4;

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < vectors; i += stride) {
        float4 v = invector[i];
        v = reluvec(v);
        outvector[i] = v;
    }

    size_t startscalar = vectors * 4;

    for (size_t i = startscalar + idx; i < totalsize; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t totalsize = m * n; 

    int deviceID;
    cudaGetDevice(&deviceID);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, deviceID);

    int threads = 256;
    int blocks;

    blocks = numSM * 32;
    // cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, relukern);

    relukern<<<blocks, threads>>>(input, output, totalsize);
}
