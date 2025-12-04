#include <cuda_runtime.h>
#include <math.h>

__device__ inline float eluscal(float x, float alpha) {
    float pos = fmaxf(x, 0.0f);
    float neg = fminf(x, 0.0f);
    return pos + alpha * (expf(neg) - 1.0f);
}

__device__ inline float4 eluvec(float4 v, float alpha) {
    float4 result;
    result.x = eluscal(v.x, alpha);
    result.y = eluscal(v.y, alpha);
    result.z = eluscal(v.z, alpha);
    result.w = eluscal(v.w, alpha);
    return result;
}

__global__ void elu(const float* __restrict__ input, float* __restrict__ output, float alpha, size_t totalsize) {
    const float4* inputvector = reinterpret_cast<const float4*>(input);
    float4* outputvector = reinterpret_cast<float4*>(output);

    size_t totalvectors = totalsize / 4;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < totalvectors; i += stride) {
        float4 v = inputvector[i];
        v = eluvec(v, alpha);
        outputvector[i] = v;
    }

    size_t startvector = totalvectors * 4;
    for (size_t i = startvector + idx; i < totalsize; i += stride) {
        float val = input[i];
        output[i] = eluscal(val, alpha);
    }
}


extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t totalsize = m * n;

    int deviceId;
    cudaGetDevice(&deviceId);

    int blocks;
    int threads = 256;

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, deviceId);

    blocks = numSM * 32;
    elu<<<blocks, threads>>>(input, output, alpha, totalsize);
}
