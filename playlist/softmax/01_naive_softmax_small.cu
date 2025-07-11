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
        float norm = 0.0f;

        // Step 1: Find max value in row
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            x_max = fmaxf(x_max, A[i]);
        }

        // Step 2: Compute denominator
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            norm += expf(A[i] - x_max);
        }

        // Step 3: Compute softmax output
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            B[i] = expf(A[i] - x_max) / norm;
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
    const int M = 4; // rows
    const int N = 5; // cols
    const int SIZE = M * N * sizeof(float);

    // allocate and initialize memory on host
    float h_input[M * N] = {
        1, 2, 3, 4, 5,
        3, 1, 0, 2, 4,
        2, 4, 6, 8, 10,
        0, 0, 0, 0, 1
    };

    float* h_output = new float[M * N];

    // allocate memory on device and initialize input array

    float* d_A; // input matrix
    float* d_B; // normalized, shifted softmax

    // create cuda events for time profiling    
    cudaEvent_t startH2D, startKernel, startD2H, stop;

    cudaEventCreate(&startH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stop);

    CHECK_CUDA(cudaMalloc(&d_A, SIZE));
    CHECK_CUDA(cudaMalloc(&d_B, SIZE));

    int BLOCK_SIZE = 128; 
    int BLOCKS = (M + BLOCK_SIZE - 1) / BLOCK_SIZE; // this is ceil()

    // time window: 1 (copy input from host to device)
    cudaEventRecord(startH2D);
    CHECK_CUDA(cudaMemcpy(d_A, h_input, SIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(startKernel);

    // time window: 2 (launch kernel)
    naive_softmax<<<BLOCKS, BLOCK_SIZE>>>(d_A, d_B, M, N);
    CHECK_CUDA(cudaDeviceSynchronize()); 
    cudaEventRecord(startD2H);

    // time window: 3 (copy output back to host)
    CHECK_CUDA(cudaMemcpy(h_output, d_B, SIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);

    // synchronize
    cudaEventSynchronize(stop);

    float h2dTime;
    float kernelTime;
    float d2hTime;

    cudaEventElapsedTime(&h2dTime, startH2D, startKernel);
    cudaEventElapsedTime(&kernelTime, startKernel, startD2H);
    cudaEventElapsedTime(&d2hTime, startD2H, stop);

    std::cout << "Softmax Output:\n";

    for (int i = 0; i < M; i++) {
        std::cout << "Row " << std::setw(2) << i << ": ";
        for (int j = 0; j < N; j++) {
            float val = h_output[i * N + j];
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    std::cout << "Host to Device Transfer: " << h2dTime << " ms" << std::endl;
    std::cout << "Kernel Execution: " << kernelTime << " ms" << std::endl;
    std::cout << "Device to Host Transfer: " << d2hTime << "ms" << std::endl;

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
Softmax Output:
Row  0:   0.0117   0.0317   0.0861   0.2341   0.6364 
Row  1:   0.2341   0.0317   0.0117   0.0861   0.6364 
Row  2:   0.0003   0.0021   0.0158   0.1170   0.8647 
Row  3:   0.1488   0.1488   0.1488   0.1488   0.4046 

Host to Device Transfer: 0.3432 ms
Kernel Execution: 1.1005 ms
Device to Host Transfer: 0.1522ms
Exit status: 0

*/
