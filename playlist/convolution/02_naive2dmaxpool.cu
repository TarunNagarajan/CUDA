#include <cuda_runtime.h>
#include <cfloat>

__global__ void maxpool (
    const float* __restrict__ input,
          float* __restrict__ output,
          int H, int W,
          int H_out, int W_out,
          int kernel,
          int stride,
          int padding,
          int dilation
) {
    int out_j = blockIdx.x * blockDim.x + threadIdx.x;
    int out_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_i >= H_out || out_j >= W_out) return;

    float max = -FLT_MAX;

    for (int m = 0; m < kernel; ++m) {
        for (int n = 0; n < kernel; ++n) {
            int in_r = out_i * stride + m * dilation - padding;
            int in_c = out_j * stride + n * dilation - padding;
            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                max = fmaxf(max, input[in_r * W + in_c]);
            }
        }
    }

    output[out_i * W_out + out_j] = max;
}

extern "C" void solution(const float* input, int kernel, int stride, int padding, int dilation, float* output, size_t H, size_t W) {
    int H_out = (H + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y);

    maxpool<<<grid, block>>> (
        input, output, int(H), int(W), H_out, W_out, kernel, stride, padding, dilation
    );
}
