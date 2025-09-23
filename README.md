# CUDA

This repository documents my CUDA learning journey focused on AI inference and systems-level GPU programming.
Although I don’t have access to a GPU locally, I use LeetGPU.com to write and test CUDA kernels.
I'm following the book *Programming Massively Parallel Processors* by David B. Kirk and Wen-mei W. Hwu, along with the Official CUDA documentation.  

## Playlist
| S.No | Topic Name        | Description                                  |
| ---- | ------------------ | -------------------------------------------- |
| 1    | [Softmax](playlist/softmax) | Multiple Iterations of Softmax (with Warp Level Reductions and Shuffling) |
| 2    | [Layernorm](playlist/layernorm) |  In Progress |

## Puzzles
| S.No | Kernel Name        | Description                                  |
| ---- | ------------------ | -------------------------------------------- |
| 1    | [Hello World ft. CUDA](puzzles/puzzles_01_10/001_hello_world_gpu/001_hello_world_gpu.cu) | 'Hello World' with multiple threads in parallel  |
| 2    | [Global Id](puzzles/puzzles_01_10/002_global_id/002_global_id.cu) | Global IDs of each thread |
| 3    | [Vector Addition](puzzles/puzzles_01_10/003_vector_addition/003_vec_add.cu) | Non-Strided Vector Addition |
| 4    | [Grid of Threads](puzzles/puzzles_01_10/004_grid_of_threads/004_grid_threads.cu) | Implement a grid of threads |
| 5    | [Scalar Multiplication](puzzles/puzzles_01_10/005_scalar_multiplication/005_scalar_multiply.cu) | Scalar Multiplication |
| 6    | [Squaring In-place](puzzles/puzzles_01_10/006_square_each_element/006_square_each_element.cu) | Squaring the elements of an array in-place via linear traversal |
| 7    | [Timing on Device](puzzles/puzzles_01_10/007_timing_device/007_timing.c) | Learnt to manage CUDA events |
| 8    | [](puzzles/puzzles_01_10/004_grid_of_threads/004_grid_threads.cu) |  |
| 9    | [](puzzles/puzzles_01_10/004_grid_of_threads/004_grid_threads.cu) |  |
| 10    | [](puzzles/puzzles_01_10/004_grid_of_threads/004_grid_threads.cu) |  |


