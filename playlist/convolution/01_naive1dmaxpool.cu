#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void maxpool (
    const float* __restrict__ input,
          float* __restrict__ output,
    int H_out,
    int H,
    int kernel,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H_out) return;

    float max = -FLT_MAX;

    for (int n = 0; n < kernel; ++n) {
        int in_c = idx * stride + n * dilation - padding;
        if (in_c >= 0 && in_c < H) {
            max = fmaxf(max, input[in_c]);
        } 
    }

    output[idx] = max;
}
extern "C" void solution(const float* input, int kernel, int stride, int padding, int dilation, float* output, size_t H) {
    int H_out = (H + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    dim3 block(BLOCK_SIZE);
    dim3 grid((H_out + BLOCK_SIZE - 1) / BLOCK_SIZE);

    maxpool<<<grid, block>>> (
        input, output, H_out, H, kernel, stride, padding, dilation
    );
} 
