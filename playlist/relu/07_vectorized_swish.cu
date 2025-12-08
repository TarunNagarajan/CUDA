#include <cuda_runtime.h>
#include <math.h>

__device__ inline float4 swishvec(float4 v) {
    float4 e;
    e.x = __expf(-v.x);
    e.y = __expf(-v.y);
    e.z = __expf(-v.z);
    e.w = __expf(-v.w);
    
    float4 result;
    result.x = v.x / (1 + e.x);
    result.y = v.y / (1 + e.y);
    result.z = v.z / (1 + e.z);
    result.w = v.w / (1 + e.w);

    return result;
}

__global__ void swishfn(const float* __restrict__ input, float* __restrict__ output, size_t totalsize) {
    const float4* inputvector = reinterpret_cast<const float4*>(input);
    float4* outputvector = reinterpret_cast<float4*>(output);
    
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t totalvectors = totalsize / 4;

    for (size_t i = idx; i < totalvectors; i += stride) {
        float4 v = inputvector[i];
        v = swishvec(v);
        outputvector[i] = v;
    }

    size_t startvector = totalvectors * 4;
    
    for (size_t i = startvector + idx; i < totalsize; i += stride) {
        float x = input[i];
        float e = __expf(-x);
        output[i] = x / (1 + e);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t totalsize = m * n;

    int threads = 256;
    int blocks;

    int deviceId;
    cudaGetDevice(&deviceId);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, deviceId);

    blocks = numSM * 32;
    swishfn<<<blocks, threads>>>(input, output, totalsize);
}
