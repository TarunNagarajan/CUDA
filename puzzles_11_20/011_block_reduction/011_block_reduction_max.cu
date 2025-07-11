#include <iostream>
#include <cfloat>
#include <cuda_runtime.h>

#define N (1 << 20)
#define THREADS_PER_BLOCK 256

__global__ void maxReduceMulti(const float* input, float* partial_max) {
    __shared__ float smem[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (idx < N) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = smem[0];
    }
}

__global__ void maxReduceOnce(const float* partial_max, float* output, int n) {
    __shared__ float smem[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (idx < n) ? partial_max[idx] : -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

int main() {
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input;
    float* d_partial_max;
    float* d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int curr_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc(&d_partial_max, curr_blocks * sizeof(float));

    maxReduceMulti<<<curr_blocks, THREADS_PER_BLOCK>>>(d_input, d_partial_max);
    cudaDeviceSynchronize();

    float result = 0.0f;

    while (curr_blocks > 1) {
        int next_blocks = (curr_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaMalloc(&d_output, next_blocks * sizeof(float));

        maxReduceOnce<<<next_blocks, THREADS_PER_BLOCK>>>(d_partial_max, d_output, curr_blocks);
        cudaDeviceSynchronize();

        cudaFree(d_partial_max);
        d_partial_max = d_output;
        curr_blocks = next_blocks;
    }

    cudaMemcpy(&result, d_partial_max, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Final Max: " << result << std::endl;

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_partial_max); 

    return 0;
}
