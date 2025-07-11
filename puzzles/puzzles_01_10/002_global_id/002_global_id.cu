#include <iostream>
#include <cuda_runtime.h>

__global__ void writeGlobalID(int* d_array) {
    int globalID = blockIdx.x * blockDim.x + threadIdx.x; 
    d_array[globalID] = globalID;
}

int main() {
    const int numThreadsPerBlock = 8;
    const int numBlocks = 4;
    const int arraySize = numThreadsPerBlock * numBlocks;

    int *d_array; // device;
    int *h_array; // host;

    h_array = (int*)malloc(arraySize * sizeof(int)); // store the final value on the CPU, host.
    cudaMalloc((void**)&d_array, arraySize * sizeof(int)); // filled on the GPU, device by the kernel

    // kernel launch
    writeGlobalID<<<numBlocks, numThreadsPerBlock>>>(d_array);

    cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Global IDs from Device: \n";
    for (int i = 0; i < arraySize; i++) {
        std::cout << "h_array[" << i << "] = " << h_array[i] << "\n";
    }

    cudaFree(d_array);
    free(h_array);
    
    return 0; 
}
