#include <iostream>
#include <cuda_runtime.h>

__global__ void helloGPU() {
    printf("Hello World ft. NVIDIA GTX TITAN X\n");
}

int main() {
    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0; 
}
