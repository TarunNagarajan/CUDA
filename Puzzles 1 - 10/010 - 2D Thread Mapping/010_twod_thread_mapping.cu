#include <iostream>
#include <cuda_runtime.h>

__global__ void threadMap() {
    int global_row = threadIdx.y + blockDim.y * blockIdx.y;
    int global_col = threadIdx.x + blockDim.x * blockIdx.x;

    printf("Thread at global (row, col) -> (%d, %d)\n", global_row, global_col);
}

int main() {
    dim3 block(4, 4);
    dim3 grid(2, 2);

    // launch the kernel
    threadMap<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0; 
}
