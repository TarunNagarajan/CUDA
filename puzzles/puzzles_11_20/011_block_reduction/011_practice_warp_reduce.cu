#include <iostream>
#include <cuda_runtime.h> 
#include <cmath>
#include <cfloat>

#define N 1024
#define BLOCK_SIZE 256

__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void blockReduce(const float* input, float* partial_sums) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    smem[tid] = (idx < N) ? input[idx] : static_cast<float>(0);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(smem, tid);
    if (tid == 0) partial_sums[blockIdx.x] = smem[0];
}

__global__ void onceReduce(const float* partial_sums, float* output) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    smem[tid] = (idx < N) ? partial_sums[idx] : static_cast<float>(0);

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid < 32) warpReduce(smem, tid);
    if (tid == 0) output[blockIdx.x] = smem[0];
}

int main() {
    float* h_input = new float[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = sinf(i * 0.001f);
    }

    float* d_input;
    float* d_partial_sums;
    float* d_output;

    float h_output = static_cast<float>(0);

    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    int NUM_BLOCKS = N / BLOCK_SIZE;
    cudaMalloc(&d_partial_sums, sizeof(float) * NUM_BLOCKS);

    // Launch the block reduce kernel. 
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blockReduce<<<blocks, BLOCK_SIZE>>>(d_input, d_partial_sums);
    cudaDeviceSynchronize();

    cudaMalloc(&d_output, sizeof(float));

    onceReduce<<<1, blocks>>>(d_partial_sums, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final Sum = " << h_output << std::endl;

    return 0;
}

