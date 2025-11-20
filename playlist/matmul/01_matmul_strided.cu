#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/*
00: NAIVE MATMUL
-----------------
Kernel Execution Time: 8.426 ms
Max Error: 0.000017 | Average Error: 0.000002

01: GRID STRIDED
-----------------
Kernel Execution Time: 6.316 ms
Max Error: 0.000017 | Average Error: 0.000002
*/

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ void matmul_naive(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= M || col >= N) {
    return;
  }

  float sum = 0.0f;
  for (size_t k = 0; k < K; ++k) {
    float a = A[row * K + k];
    float b = B[k * N + col];
    sum += a * b;
  }

  C[row * N + col] = sum;
}

__global__ void matmul_strided(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
  size_t strided_row = blockDim.y * gridDim.y;
  size_t strided_col = blockDim.x * gridDim.x;

  for (size_t row = blockDim.y * blockIdx.y + threadIdx.y; row < M; row += strided_row) {
    for (size_t col = blockDim.x * blockIdx.x + threadIdx.x; col < N; col += strided_col) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
      }

      C[row * N + col] = sum;
    }
  }
}

extern "C" void solution(const float* d_A, const float* d_B, float* d_C, size_t M, size_t K, size_t N) {
  const int TILE = 16;
  dim3 threads(TILE, TILE);
  dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  matmul_strided<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Kernel Execution Time: %.3f ms\n", ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  
}

void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float s = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        s += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = s;
    }
  }
}

int main() {
  const size_t M = 1024;
  const size_t K = 1024;
  const size_t N = 1024;

  size_t sizeA = M * K;
  size_t sizeB = K * N; 
  size_t sizeC = M * N;

  float* h_A = (float*)malloc(sizeA * sizeof(float));
  float* h_B = (float*)malloc(sizeB * sizeof(float));
  float* h_C = (float*)malloc(sizeC * sizeof(float));

  float* h_C_ref = (float*)malloc(sizeC * sizeof(float));

  srand(150);
  for (size_t i = 0; i < sizeA; ++i) {
    h_A[i] = (rand() % 100 - 50) / 50.0f;
  }

  for (size_t i = 0; i < sizeB; ++i) {
    h_B[i] = (rand() % 100 - 50) / 50.0f;
  }

  matmul(h_A, h_B, h_C_ref, M, K, N);

  
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));

  solution(d_A, d_B, d_C, M, K, N);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

  double maxerr = 0.0;
  double sumerr = 0.0;

  for (size_t i = 0; i < sizeC; ++i) {
    double err = fabs((double)h_C[i] - (double)h_C_ref[i]);
    sumerr += err;
    maxerr = fmax(maxerr, err);
  }

  printf("Max Error: %.6f | Average Error: %.6f\n", maxerr, sumerr / sizeC);
  
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}
