#include <cuda_runtime.h>
#include <math.h>

__device__ inline float4 tanvec(float4 v) {
    float4 result;
    result.x = tanhf(v.x);
    result.y = tanhf(v.y);
    result.z = tanhf(v.z);
    result.w = tanhf(v.w);
    return result;
}

__global__ void tanh(const float* __restrict__ input, float* __restrict__ output, size_t totalsize) {
    const float4* inputvector = reinterpret_cast<const float4*>(input);
    float4* outputvector = reinterpret_cast<float4*>(output);

    size_t vectorcount = totalsize / 4;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < vectorcount; i += stride) {
        float4 v = inputvector[i];
        v = tanvec(v);
        outputvector[i] = v;
    }

    size_t startscalar = vectorcount * 4;
    for (size_t i = startscalar + idx; i < totalsize; i += stride) {
        float val = input[i];
        output[i] = tanhf(val);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t totalsize = m * n;

    int deviceID;
    cudaGetDevice(&deviceID);

    int threads = 256;
    int blocks;

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, deviceID);

    blocks = numSM * 32;
    tanh<<<blocks, threads>>>(input, output, totalsize);
}
