#include <cuda_runtime.h>

__device__ inline float4 leakyvector(float alpha, float4 v) {
    float4 result;
    result.x = fmaxf(alpha * v.x, v.x);
    result.y = fmaxf(alpha * v.y, v.y);
    result.z = fmaxf(alpha * v.z, v.z);
    result.w = fmaxf(alpha * v.w, v.w);

    return result;
}

__global__ void leakyrelu(const float* __restrict__ input, float alpha, float* __restrict__ output, size_t totalsize) {
    const float4* inputvector = reinterpret_cast<const float4*>(input);
    float4* outputvector = reinterpret_cast<float4*>(output);

    size_t vectorcount = totalsize / 4;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < vectorcount; i += stride) {
        float4 v = inputvector[i];
        v = leakyvector(alpha, v);
        outputvector[i] = v;
    }

    size_t startscalar = vectorcount * 4;
    for (size_t i = startscalar + idx; i < totalsize; i += stride) {
        float val = input[i];
        output[i] = fmaxf(alpha * val, val);
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    size_t totalsize = m * n;

    int deviceID;
    cudaGetDevice(&deviceID);

    int threads = 256;
    int blocks;

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, deviceID);

    blocks = numSM * 32;
    leakyrelu<<<blocks, threads>>>(input, alpha, output, totalsize);
}
