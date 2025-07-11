// Playlist/Tiling/Implement/01_basic_matmul_tiled.cu

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "Error occured at line " << __LINE__ << " : " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(1); \
    }

#define TILE_SIZE 16 
// each thread is assigned one element of the tile

__global__ void tiled_matmul(float* A, float* B, float* C, int M, int N, int K) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // load into respective shared memory version
    for (int t = 0; t < (TILE_SIZE + K - 1) / TILE_SIZE; t++) {
        // reading the row slice of 'A' for the given tile
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // reading the col slice of 'B' for the given tile
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // here's the actual computation
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 32;
    int N = 32;
    int K = 32; 

    size_t size_A = M * K * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // host allocation
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];  
    float* h_C = new float[M * N]; 

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f; 
    } 

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;  
    }

    float* d_A;
    float* d_B;
    float* d_C;

    // device allocation
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_C)); 

    // host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // define block and grid sizes
    dim3 BLOCK_SIZE(TILE_SIZE, TILE_SIZE);
    dim3 BLOCKS((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // launch kernel
    tiled_matmul<<<BLOCKS, BLOCK_SIZE>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy result back to the host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Print result (for small M, N)
    std::cout << "Result C (partial):\n";

    for (int i = 0; i < std::min(6, M); ++i) {
        for (int j = 0; j < std::min(6, N); ++j)
            std::cout << h_C[i * N + j] << " ";
        std::cout << "\n";
    }

    delete[] h_A; 
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 

    return 0; 
}

/*

Running in FUNCTIONAL mode...
Compiling...
Executing...
Result C (partial):
32 32 32 32 32 32 
32 32 32 32 32 32 
32 32 32 32 32 32 
32 32 32 32 32 32 
32 32 32 32 32 32 
32 32 32 32 32 32 
Exit status: 0

*/
