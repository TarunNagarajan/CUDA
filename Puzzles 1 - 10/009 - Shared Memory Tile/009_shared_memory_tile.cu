#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 8

__global__ void sharedTile(float* out, const float* in) {
    __shared__ float tile[TILE_SIZE];
    int idx = threadIdx.x;

    if (idx < TILE_SIZE) {
        tile[idx] = in[idx];
        __syncthreads();

        tile[idx] *= tile[idx]; // transform: squaring each element of the array. 
        __syncthreads();

        out[idx] = tile[idx];
    }  
}

int main() {
    // we're allocating memory on-device for this one. 
    float h_in[TILE_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8}; // 8 TILE_SIZE
    float h_out[TILE_SIZE];

    float* d_in;
    float* d_out;

    // number of bytes
    size_t bytes = sizeof(float) * TILE_SIZE;

    // on-device memory allocation
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    sharedTile<<<1, TILE_SIZE>>>(d_out, d_in);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Transformed output:\n";
    for (int i = 0; i < TILE_SIZE; ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0; 


}
