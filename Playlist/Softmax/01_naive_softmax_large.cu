#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cfloat>

__global__ void naive_softmax(const float* A, float* B, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float x_max = -FLT_MAX;

        // Step 1: Find max value in row
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            x_max = fmaxf(x_max, A[i]);
        }

        // Step 2: Compute denominator using double for better precision
        double norm = 0.0;
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            norm += exp((double)(A[i] - x_max));
        }

        // Step 3: Compute softmax output
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            B[i] = (float)(exp((double)(A[i] - x_max)) / norm);
        }
    }
}

void CHECK_CUDA(cudaError_t call) {
    if (call != cudaSuccess) {
        std::cerr << "Error at " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        exit(1);
    }
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 32768;

    const int SIZE = M * N * sizeof(float);
    float* h_input = new float[M * N];
    float* h_output = new float[M * N];

    // Safer random values: avoid extreme ranges
    srand(42);
    for (int i = 0; i < M * N; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f; // range [-5, 5]
    }

    float* d_A;
    float* d_B;

    cudaEvent_t startH2D, startKernel, startD2H, stop;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stop);

    CHECK_CUDA(cudaMalloc(&d_A, SIZE));
    CHECK_CUDA(cudaMalloc(&d_B, SIZE));

    int BLOCK_SIZE = 128;
    int BLOCKS = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // time window: 1 (copy input from host to device)
    cudaEventRecord(startH2D);
    CHECK_CUDA(cudaMemcpy(d_A, h_input, SIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(startKernel);

    // time window: 2 (launch kernel)
    naive_softmax<<<BLOCKS, BLOCK_SIZE>>>(d_A, d_B, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(startD2H);

    // time window: 3 (copy output back to host)
    CHECK_CUDA(cudaMemcpy(h_output, d_B, SIZE, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float h2dTime, kernelTime, d2hTime;
    cudaEventElapsedTime(&h2dTime, startH2D, startKernel);
    cudaEventElapsedTime(&kernelTime, startKernel, startD2H);
    cudaEventElapsedTime(&d2hTime, startD2H, stop);

    std::cout << "Softmax Output:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Row " << std::setw(2) << i << ": ";
        for (int j = 0; j < 5; j++) {
            float val = h_output[i * N + j];
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << val << " ";
        }
        std::cout << "\n";
    }

    // Validate softmax sum per row
    for (int i = 0; i < 4; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            row_sum += h_output[i * N + j];
        }
        std::cout << "Row " << i << " sum: " << row_sum << "\n";
    }

    std::cout << "\n";
    std::cout << "Host to Device Transfer: " << h2dTime << " ms" << std::endl;
    std::cout << "Kernel Execution: " << kernelTime << " ms" << std::endl;
    std::cout << "Device to Host Transfer: " << d2hTime << " ms" << std::endl;

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(startH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stop);

    return 0;
}

/*

Running in FUNCTIONAL mode...
Compiling...
Executing...
Host to Device Transfer: 49.0650 ms
Kernel Execution: 1408.4968 ms
Device to Host Transfer: 108.0687 ms
Exit status: 0

*/
