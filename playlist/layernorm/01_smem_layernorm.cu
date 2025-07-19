#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>

__global__ void smem_layernorm(float* matrix, float* output, int m, int n, float epsilon) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = (row * n) + col;

    extern __shared__ float smem[];

    float val = matrix[idx];
    smem[col] = val;
    __syncthreads();

    // STRIDE I: MEAN
    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            smem[col] += smem[col + stride];
        }
        __syncthreads();
    }

    // thread 0 is allowed to calculate the mean and store it in smem[0] for broadcasting
    if (col == 0) {
        smem[0] = smem[0] / n;
    }
    __syncthreads();

    float mean = smem[0];

    // STRIDE II: VARIANCE
    float diff = val - mean;
    smem[col] = diff * diff;
    __syncthreads(); 

    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            smem[col] += smem[col + stride];
        }
        __syncthreads();
    }

    // thread 0 is allowed to calculate the mean (in this case, the sum of squared differences divided by n(samples)) and store it in smem[0]
    // for broadcasting
    if (col == 0) {
        smem[0] = smem[0] / n;
    }
    __syncthreads();

    float variance = smem[0];

    float stdev = sqrtf(variance + epsilon);
    output[idx] = (val - mean) / stdev;
}

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main() {
    int m = 1024;
    int n = 1024;
    size_t SIZE = m * n;

    cudaEvent_t startH2D, startKernel, startD2H, stop;
    CHECK_CUDA(cudaEventCreate(&startH2D)); 
    CHECK_CUDA(cudaEventCreate(&startKernel));
    CHECK_CUDA(cudaEventCreate(&startD2H));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(m);

    size_t SMEM_SIZE = threadsPerBlock.x * sizeof(float);

    float* h_matrix = new float[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_matrix[i] = static_cast<float>(i % 100); // fix: i -> i % 100 for better test values
    }

    float* d_matrix;
    CHECK_CUDA(cudaMalloc((void**)&d_matrix, SIZE * sizeof(float)));

    float* d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_output, SIZE * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(startH2D));
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(startKernel));
    smem_layernorm<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_output, m, n, 1e-6f);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(startD2H));
    float* h_output = new float[SIZE];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop)); 
    CHECK_CUDA(cudaEventSynchronize(stop));

    std::cout << "First 10 normalized values from the first row:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    float h2dTime, kernelTime, d2hTime; 
    CHECK_CUDA(cudaEventElapsedTime(&h2dTime, startH2D, startKernel));
    CHECK_CUDA(cudaEventElapsedTime(&kernelTime, startKernel, startD2H));
    CHECK_CUDA(cudaEventElapsedTime(&d2hTime, startD2H, stop)); 

    std::cout << "\n";
    std::cout << "NVIDIA GTX TITAN X\n";
    std::cout << "Host to Device Transfer: " << h2dTime << " ms" << std::endl;
    std::cout << "Kernel Execution: " << kernelTime << " ms" << std::endl;
    std::cout << "Device to Host Transfer: " << d2hTime << " ms" << std::endl;

    delete[] h_matrix;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
