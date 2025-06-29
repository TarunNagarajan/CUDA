#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat> 

#define N (1 << 20) 
#define BLOCK_SIZE 256 // effectively, THREADS_PER_BLOCK

__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduceSumBlock(const float* input, float* partial_sums) {
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

__global__ void reduceSumOnce(const float* input, float* output, int num_elements) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    smem[tid] = (idx < num_elements) ? input[idx] : static_cast<float>(0);
    __syncthreads(); 

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride]; 
        }
        __syncthreads(); 
    }

    if (tid < 32) warpReduce(smem, tid);
    if (tid == 0) output[blockIdx.x] = smem[0]; 
}

__global__ void reduceVarianceBlock(const float* input, float* partial_variance, float mean) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float diff = (idx < N) ? input[idx] - mean : static_cast<float>(0);
    smem[tid] = diff * diff; 
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads(); 
    }

    if (tid < 32) warpReduce(smem, tid);
    if (tid == 0) partial_variance[blockIdx.x] = smem[0];
}

__global__ void reduceVarianceOnce(const float* input, float* output, int num_elements) {
    __shared__ float smem[BLOCK_SIZE]; 

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    smem[tid] = (idx < num_elements) ? input[idx] : static_cast<float>(0);
    __syncthreads();

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
    // allocate and initialize host memory. 
    float* h_input = new float[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = sinf(i * 0.001f);
    }

    // allocate device memory.
    float* d_input;
    float* d_partial_buffer;        // partial result from the current stage of reduction.
    float* d_next_partial_buffer;   // stores the reduced output from the current stage of reduction for the next stage.

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Phase I: Compute Mean. 
    int num_elements = N;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_partial_buffer, num_blocks * sizeof(float));

    // Launch Kernel
    reduceSumBlock<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial_buffer);
    cudaDeviceSynchronize(); 

    float total_sum = static_cast<float>(0); // 0.0f

    // Recursive Reduction by Loop
    while (num_blocks > 1) {
        int next_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE; 
        cudaMalloc(&d_next_partial_buffer, next_blocks * sizeof(float));

        // Launch Kernel
        reduceSumOnce<<<next_blocks, BLOCK_SIZE>>>(d_partial_buffer, d_next_partial_buffer, num_elements);
        cudaDeviceSynchronize();

        cudaFree(d_partial_buffer);
        num_elements = num_blocks;
        num_blocks = next_blocks;

        d_partial_buffer = d_next_partial_buffer;
    }

    cudaMemcpy(&total_sum, d_partial_buffer, sizeof(float), cudaMemcpyDeviceToHost);

    float mean = total_sum / N; 
    
    std::cout << "Mean = " << mean << std::endl;

    // Phase II: Compute Variance
    num_elements = N;
    num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_partial_buffer, num_blocks * sizeof(float));

    // Launch Kernel
    reduceVarianceBlock<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial_buffer, mean);
    cudaDeviceSynchronize();    

    // Recursive Reduction by Loop (Variance)

    while (num_blocks > 1) {
        int next_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_next_partial_buffer, next_blocks * sizeof(float));

        // Launch Kernel.
        reduceVarianceOnce<<<next_blocks, BLOCK_SIZE>>>(d_partial_buffer, d_next_partial_buffer, num_elements); 
        cudaDeviceSynchronize();

        cudaFree(d_partial_buffer);
        d_partial_buffer = d_next_partial_buffer;
        num_elements = num_blocks;
        num_blocks = next_blocks;
    }

    float var_host;

    cudaMemcpy(&var_host, d_partial_buffer, sizeof(float), cudaMemcpyDeviceToHost);
    float stddev = sqrtf(var_host / N);

    std::cout << "Standard Deviation = " << stddev << std::endl; 

    // Memory Cleanup 
    delete[] h_input; 
    cudaFree(d_input);
    cudaFree(d_partial_buffer);

    return 0; 
}

