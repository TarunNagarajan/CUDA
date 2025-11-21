#include <cuda_runtime.h>

__global__ void relukern(const float* input, float* output, size_t n, size_t m) {
    size_t global_row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t global_col = blockDim.x * blockIdx.x + threadIdx.x; 

    size_t strided_row = blockDim.y * gridDim.y;
    size_t strided_col = blockDim.x * gridDim.x;

    for (size_t row = global_row; row < m; row += strided_row) {
        for (size_t col = global_col; col < n; col += strided_col) {
            size_t idx = row * n + col;
            output[idx] = fmaxf(0, input[idx]);
        }
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    int TILES = 16;
    dim3 threads(TILES, TILES);
    dim3 blocks((n + TILES - 1) / TILES, (m + TILES - 1) / TILES);

    relukern<<<blocks, threads>>>(input, output, n, m);
}
