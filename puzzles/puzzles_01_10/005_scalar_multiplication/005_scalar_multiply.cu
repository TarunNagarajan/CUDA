#include <iostream>
#include <cuda_runtime.h>

#define N 10
#define THREADS_PER_BLOCK 256

// Problem: | 5 | Scalar Multiply | Multiply each element of a float array by a scalar value on the GPU.    

__global__ void scalarMultiply(const float* input, float *output, const float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}

int main() {
    // allocate the host arrays, memory. (on the cpu)
    int bytes = N * sizeof(float);
    float scalar = 2.0f;

    float *h_input = new float[N];
    float *h_output = new float[N];

    // allocate the device arrays, memory. (gpu)
    float *d_input;
    float *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i + 1); // [1, 2, 3, 4, ...]
    }

    // input to device.
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // launch kernel.
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; 
    scalarMultiply<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, scalar, N);
    cudaDeviceSynchronize();

    // result to host.
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // print the result.
    std::cout << "Input: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }

    std::cout << "\nOutput: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }

    delete[] h_input;
    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0; 

}
