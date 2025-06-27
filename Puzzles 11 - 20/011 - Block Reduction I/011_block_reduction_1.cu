#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256 // Effectively, THREADS_PER_BLOCK

__global__ void reduction(const float* input, float* partial_sums) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    smem[tid] = input[idx];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        // wrote to the shared memory, so for safety in handling memory...
        __syncthreads(); 
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = smem[0]; 
    }
}

__global__ void final_reduction(const float* partial_sums, float* final_sum) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    smem[tid] = partial_sums[tid];
    
    // wrote to the shared memory, so for safety in handling memory...
    __syncthreads();

    // accumulate partial sums with the same ol' stride method. 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = smem[tid + stride];
        }
        // wrote to the shared memory, so for the safety of handling memory...
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *final_sum = smem[0]; // write to global memory address. 
    }
}

int main() {
    float* h_input = new float[N]; // allocate memory on the host. 

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(1);
    }

    int NUM_BLOCKS = N / BLOCK_SIZE; // number of blocks. 

    // device variables. 
    float *d_input;
    float *d_partial_sums;
    float *d_final_sum;

    // host variable. 
    float h_final_sum;

    // allocate memory on device. 
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial_sums, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_final_sum, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // first kernel launch.
    reduction<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_partial_sums);
    cudaDeviceSynchronize();

    // unifying kernel launch.
    final_reduction<<<1, NUM_BLOCKS>>>(d_partial_sums, d_final_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final Sum = " << h_final_sum << std::endl;

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_partial_sums);
    cudaFree(d_final_sum); 

    return 0; 
}
