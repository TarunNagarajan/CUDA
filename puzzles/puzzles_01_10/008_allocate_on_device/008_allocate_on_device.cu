#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 24)
#define THREADS_PER_BLOCK 256

__global__ void fill(float* nums, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx < n) {
        nums[idx] = static_cast<float>(idx * 2.0);
    }
}

int main() {
    size_t bytes = sizeof(float) * N;

    float* d_array; // array to be operated on the device. 
    float h_array[N]; // array which receives the operated array back to the host.

    // allocate device memory. 
    cudaMalloc((void**)&d_array, bytes);

    // launch kernel. 
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill<<<blocks, THREADS_PER_BLOCK>>>(d_array, N); 
    cudaDeviceSynchronize();

    // copy data from back to the host. 
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Array filled on device:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // free the device memory with cudaFree();
    cudaFree(d_array);

    return 0; 
}
