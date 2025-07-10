# CUDA

This repository documents my CUDA learning journey focused on AI inference and systems-level GPU programming.
Although I don’t have access to a GPU locally, I use LeetGPU.com to write and test CUDA kernels remotely.
I'm following the book *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu and the Official CUDA documentation.  

| S.No | Kernel Name        | Description                                  |
| ---- | ------------------ | -------------------------------------------- |
| 1    | `naive_softmax.cu` | Row-wise softmax using global memory access  |
| 2    | `bmm_layernorm.cu` | Batched matrix multiplication + LayerNorm    |
| 3    | `einsum_perf.cu`   | Einsum-style fused operations with profiling |
| 4    | `spmm_tiled.cu`    | Tiled sparse-dense matrix multiplication     |
| ...  |                    |                                              |

