#include <cuda_runtime.h>

__global__ void avgpool(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W,
    int H_out, int W_out,
    int kernel_size,
    int stride,
    int padding
) {
    int out_j = blockIdx.x * blockDim.x + threadIdx.x;
    int out_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_i >= H_out || out_j >= W_out) return;

    float sum = 0.0f;

    for (int m = 0; m < kernel_size; m++) {
        for (int n = 0; n < kernel_size; n++) {
            int in_r = out_i * stride + m - padding;
            int in_c = out_j * stride + n - padding;

            if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                sum += input[in_r * W + in_c];
            }
        }
    }
    output[out_i * W_out + out_j] = sum / (float)(kernel_size * kernel_size);
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H, size_t W) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(16, 16);

    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y);

    avgpool<<<grid, block>>>(
        input, output, (int)H, (int)W, H_out, W_out, kernel_size, stride, padding
    );
}
