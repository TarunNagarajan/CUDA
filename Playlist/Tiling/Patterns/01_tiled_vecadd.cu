// Playlist/Tiling/Patterns/01_tiled_vecadd.cu

#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 256

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "ERROR ON LINE " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(1); \
    }

__global__ void tiled_vecadd(float* a, float* b, float* c, int n) {
    __shared__ float tile_a[TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid; // global indexing

    for (int i = blockIdx.x * TILE_SIZE; i < n; i += gridDim.x * TILE_SIZE) {

        /*

        so in my understanding, we start from i = 0, and skip jump
        the TILE_SIZE = 256 times the number of blocks and the 
        threads take care of the rest.

        the loop allows each block to process multiple tiles strided
        across the entire array. 

        Block 0 does: i = 0, 1024, 2048, ...

        Block 1 does: i = 256, 1280, 2304, ...

        Block 2 does: i = 512, 1536, ...

        Block 3 does: i = 768, 1792, ...

        */

        if (i + tid < n) {
            tile_a[tid] = a[i + tid];
            tile_b[tid] = b[i + tid]; 
        }

        __syncthreads();

        if (i + tid < n) {
            c[i + tid] = tile_a[tid] + tile_b[tid];
        }

        __syncthreads();
    }
}

int main() {
    int n = 1024;
    size_t bytes = sizeof(float) * n; 

    // allocate host memory
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    float* d_a;
    float* d_b; 
    float* d_c;

    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int BLOCK_SIZE = TILE_SIZE;
    int BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch kernel
    tiled_vecadd<<<BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));


    // Verify Result (first 6)
    std::cout << "Vector a = ";
    
    for (int i = 0; i < 6; ++i) {
        std::cout << h_a[i] << " ";
    } 

    std::cout << std::endl;

    std::cout << "Vector b = "; 

    for (int i = 0; i < 6; ++i) {
        std::cout << h_b[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "Vector c [RESULT] = ";

    for (int i = 0; i < 6; ++i) {
        std::cout << h_c[i] << " ";
    }

    std::cout << std::endl; 

    // memory cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}

/*

Running in FUNCTIONAL mode...
Compiling...
Executing...
Vector a = 1.80429e+09 1.68169e+09 1.95775e+09 7.19885e+08 5.96517e+08 1.0252e+09 
Vector b = 8.46931e+08 1.71464e+09 4.24238e+08 1.64976e+09 1.18964e+09 1.35049e+09 
Vector c [RESULT] = 2.65122e+09 3.39633e+09 2.38199e+09 2.36965e+09 1.78616e+09 2.37569e+09 
Exit status: 0

*/
