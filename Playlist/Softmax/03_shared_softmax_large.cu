#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cfloat>

__global__ void shared_softmax(const float* A, float* B, int M, int N) {
    __shared__ float smem[1024]; 

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= M) return;

    const float* input_row = A + row * N;
    float* output_row = B + row * N;

    // Step 1: Thread-local max
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input_row[i]);
    }

    // Step 2: Block-wide max reduction
    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float global_max = smem[0];
    __syncthreads();

    // Step 3: Compute per-thread partial sum of exponentials
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum += expf(input_row[i] - global_max);
    }

    // Step 4: Block-wide reduction for normalization constant
    smem[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float norm = smem[0];
    __syncthreads();

    // Step 5: Compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - global_max) / norm;
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
    shared_softmax<<<BLOCKS, BLOCK_SIZE>>>(d_A, d_B, M, N);
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

    std::cout << "Softmax v3: Shared Memory Softmax Output:\n";
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
Host to Device Transfer: 63.6765 ms
Kernel Execution: 5.0476 ms
Device to Host Transfer: 16.2689 ms
Exit status: 0

*/
