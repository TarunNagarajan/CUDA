#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cfloat>

#define WARP_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

/*

    I'm compiling my CUDA code under Heterogeneous Compute Interface for Portability
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

*/

__global__ void shuffle_softmax(const float* A, float* B, int M, int N) {
    extern __shared__ float smem[]; // used for partial warp reductions

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE; 

    if (row >= M) return; // out of bounds, early return

    const float* input_row = A + row * N;
    float* output_row = B + row * N;

    // Step 1: Thread-local max (each thread computes local max over the segment)
    // strided access with no thread overlap //
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input_row[i]);
    }

    // Step 2: intra-warp reduction 
    // thread holds the maximum in the warp // 
    // subordinate threads report to warp leader // 
    float val = local_max;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // I'm compiling my CUDA code under Heterogeneous Compute Interface for Portability (HIP)
        // val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        val = fmaxf(val, __shfl_down(val, offset));
    }

    // Step 3: inter-warp reduction using shared memory
    // every warp leader presents their maximum to smem //
    // in an attempt to minimize the sync barriers //
    if (blockDim.x > WARP_SIZE) {
        // check if we really need to do inter-warp reduction,
        // in the case where there are multiple warps in the block
        if (lane == 0) {
            smem[tid / WARP_SIZE] = val; // we just wrote the max of each warp
        }
        __syncthreads(); // first sync barrier

        if (tid < WARP_SIZE) {
            val = (tid < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tid] : -INFINITY; 
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down(val, offset));
            }
            if (tid == 0) smem[0] = val; 
        }
    } else {
        // the 'if' case was for multiple warps, otherwise, the beginning of
        // the shared memory takes the maximum of the single warp.
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    // sync barrier for the final reduction: if tid == 0, smem[0] = val

    float global_max = smem[0];
    __syncthreads();
    // after intra-warp reduction, (optional) inter-warp reduction

    // now, we do the same thing all over again, but for norm computation
    // instead of local_max, we compute partial_sum

    // Step 4: thread's partial sum for exp(x[i] - x_max)
    float thread_sum = 0.0f;
    // strided access
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum += expf(input_row[i] - global_max);
    }

    // Step 5: warp-level reduction, use a offset stride access to reduce and consolidate
    val = thread_sum;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }

    // Step 6: inter-warp reduction 
    // the subordinate threads of the warp report to the warp leader
    if (blockDim.x > WARP_SIZE) {
        // check if the number of threads constitutes for multiple warps
        // in case of which we will apply inter-warp reduction using smem
        // more than 1 warp = 32 threads?
        if (lane == 0) {
            smem[tid / WARP_SIZE] = val; 
        }
        __syncthreads(); 

        if (tid < WARP_SIZE) {
            // read from shared memory only if tid is in the valid range of warp outputs
            val = (tid < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tid] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down(val, offset);
            } 
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float final_norm = smem[0];
    __syncthreads();

    // final softmax computation, with strided access using threads ids 
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - global_max) / final_norm;
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

    int BLOCK_SIZE = 128;  // Back to working configuration
    size_t SMEM_SIZE = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);
    int BLOCKS = M;  // One block per row

    // time window: 1 (copy input from host to device)
    cudaEventRecord(startH2D);
    CHECK_CUDA(cudaMemcpy(d_A, h_input, SIZE, cudaMemcpyHostToDevice));
    cudaEventRecord(startKernel);

    // time window: 2 (launch kernel)
    shuffle_softmax<<<BLOCKS, BLOCK_SIZE, SMEM_SIZE>>>(d_A, d_B, M, N);
    CHECK_CUDA(cudaGetLastError());  // Check for launch errors
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

    std::cout << "Softmax v4: Warp Shuffle Softmax Output:\n";
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
Host to Device Transfer: 50.0030 ms
Kernel Execution: 8188.0381 ms
Device to Host Transfer: 194.5602 ms
Exit status: 0

*/
