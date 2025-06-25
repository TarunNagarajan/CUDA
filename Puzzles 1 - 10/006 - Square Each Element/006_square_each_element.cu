#include <iostream> 
#include <cuda_runtime.h>

#define N 15
#define THREADS_PER_BLOCK 256

__global__ void squareIP(float* nums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        nums[idx] = nums[idx] * nums[idx];
    } 
}

int main() {
    int bytes = sizeof(float) * N;

    // allocate memory to host array.
    float* h_nums = new float[N];

    // initialize that array.
    for (int i = 0; i < N; ++i) {
        h_nums[i] = static_cast<float>(i + 1);
    } 

    std::cout << "Input: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_nums[i] << " ";
    }

    // allocate memory to the device array. 
    float* d_nums;
    cudaMalloc(&d_nums, bytes);

    // copy the host array to the device array.
    cudaMemcpy(d_nums, h_nums, bytes, cudaMemcpyHostToDevice);

    // launch the kernel.
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    squareIP<<<blocks, THREADS_PER_BLOCK>>>(d_nums, N);
    cudaDeviceSynchronize();

    // result copy back to host. 
    cudaMemcpy(h_nums, d_nums, bytes, cudaMemcpyDeviceToHost);

    // print the result. 
    std::cout << "\nOutput: "; 
    for (int i = 0; i < N; i++) {
        std::cout << h_nums[i] << " ";
    }

    // clean-up
    delete[] h_nums;
    cudaFree(d_nums);

    return 0; 

}
