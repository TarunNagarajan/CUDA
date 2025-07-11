
## Days 1–10: Core Concepts & Kernel Basics  
| Day | Puzzle Title                       | Description                                                                                                                |
|-----|------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| 1   | Hello, Thread!                     | Launch a kernel where each thread prints “Hello from thread X”. Use `threadIdx.x`, `blockIdx.x`, `blockDim.x`.            |
| 2   | Global ID Calculation              | Write each thread’s global ID into a device array; copy back and verify on host.                                          |
| 3   | Vector Add (GPU)                   | Add two float arrays on the GPU one element per thread; verify result on the host.                                        |
| 4   | Grid of Threads                    | Expand vector add to multiple blocks; inside kernel print `(blockIdx.x, threadIdx.x)` for one sample thread.              |
| 5   | Scalar Multiply                    | Multiply each element of a float array by a scalar value on the GPU.                                                      |
| 6   | Square Each Element                | Square every element of an input array in‑place using a simple CUDA kernel.                                               |
| 7   | Host vs Device Timing              | Time a kernel vs. a CPU loop in your playground (e.g., using `clock()` or built‑in timers).                               |
| 8   | Allocate on Device                 | `cudaMalloc` a device array, fill it in a kernel, and copy back with `cudaMemcpy`.                                        |
| 9   | Shared Memory Tile                 | Load an 8‑element tile into `__shared__` memory, apply a transform, and write back to global memory.                       |
| 10  | 2D Thread Mapping                  | Launch a 2D grid/block, then print each thread’s 2D global coordinates (row, col).                                        |

---

## Days 11–20: Shared Memory & Reductions  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 11  | Block Reduction I                  | Sum elements within each block using shared memory and `__syncthreads()`. Output one partial sum per block.                         |
| 12  | Block Reduction II                 | Optimize the previous reduction with warp‑unrolling for the final 32 threads (no extra `__syncthreads()`).                         |
| 13  | Grid‑Wide Reduction                | Copy block sums to host and finish reduction there. Compare against `thrust::reduce`.                                               |
| 14  | Bank Conflict Demo                 | Create shared‑memory accesses that conflict; then refactor to eliminate bank conflicts and measure difference.                       |
| 15  | Parallel Histogram                 | Build a 256‑bin histogram in per‑block shared memory, then merge partial histograms on the host.                                     |
| 16  | Prefix Sum (Scan) I                | Implement an exclusive Blelloch scan within each block using shared memory.                                                         |
| 17  | Prefix Sum (Scan) II               | Generalize your scan to any block size; test on random inputs of various lengths.                                                   |
| 18  | Warp Shuffle Reduction             | Use `__shfl_up_sync`/`__shfl_down_sync` to do a warp‑only reduction without shared memory.                                         |
| 19  | Hybrid Block‑Warp Scan             | Combine warp shuffle and shared memory to build a faster full‑block prefix sum.                                                   |
| 20  | Two‑Stage Global Barrier           | Simulate a grid‑wide barrier: write to global memory in kernel1, then relaunch kernel2 to finalize a reduction.                     |

---

## Days 21–30: Memory Optimization & Data Locality  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 21  | Coalesced vs Strided               | Write two kernels: one with coalesced loads, one with large strides. Measure throughput.                                            |
| 22  | Pinned Host Memory                 | Use `cudaHostAlloc` for a large buffer; compare transfer times vs. pageable memory.                                                 |
| 23  | Texture Memory Lookup              | Bind a 2D array to a texture, sample in a kernel, and compare performance to global loads.                                         |
| 24  | Cache Behavior                     | Run repeated reads on a large array; inspect L1/L2 hit rates in Nsight Compute. Optimize for better caching.                       |
| 25  | Concurrent Streams                 | Launch two kernels in separate streams with overlapping `cudaMemcpyAsync` and compute. Track SM utilization.                        |
| 26  | Tiled Matrix Transpose             | Transpose a matrix using shared‑memory tiles to avoid uncoalesced writes.                                                           |
| 27  | Compiler Flags & Unrolling         | Benchmark a compute‑heavy kernel with/without `-use_fast_math` and manual loop unrolling.                                          |
| 28  | Async Prefetching                  | In stream A, prefetch data with `cudaMemcpyAsync` while stream B runs compute on previous batch.                                    |
| 29  | Unified Memory Profiling           | Allocate Unified Memory and measure page‑fault overhead when accessed on host vs device.                                           |
| 30  | Inspect SASS                       | Use `cuobjdump --dump-sass` on a kernel; analyze register usage and instruction mix.                                                |

---

## Days 31–40: Advanced Parallel Patterns & Warp Tricks  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 31  | Cooperative Groups Barrier         | Use Cooperative Groups (`–rdc=true`) to implement a grid‑wide barrier inside a single kernel.                                       |
| 32  | Dynamic Parallelism                | Launch child kernels from a device kernel to process work chunks; compare performance vs flat launch.                              |
| 33  | Multi‑Block Reduction CG           | Use `cg::reduce` across multiple blocks within a grid group API.                                                                    |
| 34  | Inline PTX Atomics                 | Write inline PTX for a 32‑bit atomic add and call it from CUDA C++.                                                                |
| 35  | WMMA Matrix Multiply                | Use NVIDIA’s WMMA API to multiply two 16×16 FP16 tiles and accumulate into FP32 within a warp.                                      |
| 36  | Butterfly Warp Shuffle             | Implement a butterfly‑style reduction among 32 threads using `__shfl_xor_sync`.                                                     |
| 37  | Occupancy Calculator               | Query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to pick block size for max occupancy.                                        |
| 38  | Producer‑Consumer Pipeline          | Build a two‑stage pipeline: kernel A produces data, signals an event, then kernel B consumes it in another stream.                  |
| 39  | Persistent Thread Blocks           | Write a kernel where each block loops and atomically fetches work items from a global queue until empty.                            |
| 40  | Async Alloc/Free                   | Use `cudaMallocAsync` and `cudaMemPool` APIs to manage many small buffers in a tight loop.                                          |

---

## Days 41–50: Performance Debugging & NVIDIA Libraries  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 41  | Nsight Systems Trace               | Capture end‑to‑end trace of host + kernels; identify and remove bottlenecks.                                                        |
| 42  | Nsight Compute Drill‑Down          | Profile a slow kernel for memory/maths throughput; apply optimizations based on metrics.                                         |
| 43  | cuBLAS Batched GEMM                | Use `cublasSgemmBatched` for N small matrix multiplies; compare vs looped single GEMMs.                                            |
| 44  | cuSPARSE COO→CSR                   | Convert a sparse matrix from COO to CSR on the GPU with cuSPARSE; verify and benchmark.                                            |
| 45  | cuFFT 1D & 2D                      | Perform real‑data FFTs with cuFFT; compare planning and in‑place vs out‑of‑place.                                                   |
| 46  | cuRAND Uniform                     | Generate floats with cuRAND; measure throughput and use in a simple Monte Carlo simulation.                                        |
| 47  | Tensor Core Profiling              | Compare WMMA‑based FP16 tensor core kernel vs FP32 loop for same compute work.                                                     |
| 48  | Multi‑GPU Peer Copy                | In a multi‑GPU playground, do `cudaMemcpyPeerAsync` and measure latency vs host‑mediated copy.                                      |
| 49  | Reduction Tree                     | Implement an N‑ary tree reduction across blocks: each stage halves active blocks until one remains.                                |
| 50  | CUB DeviceRadixSort                | Sort large key arrays with CUB’s `DeviceRadixSort`; compare to your own bitonic sort.                                              |

---

## Days 51–60: Framework Integration & Real‑World Patterns  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 51  | PyTorch Custom Op I                | Write a CUDA extension for PyTorch: simple element‑wise kernel, compile with `setup.py`, and call from Python.                     |
| 52  | PyTorch Custom Op II               | Extend previous op to use shared memory or warp shuffles internally for reduction.                                                 |
| 53  | TensorRT Inference Snippet         | Write a minimal TensorRT engine that loads an ONNX model, allocates buffers, and runs inference on random data.                    |
| 54  | cuDNN Convolution Demo             | Use cuDNN’s API to perform a 2D convolution; compare to your own naive CUDA kernel.                                                |
| 55  | Real‑Time Anomaly Kernel           | Implement a simple threshold‑based anomaly detector as a CUDA kernel on streaming data chunks.                                     |
| 56  | k‑Means Update Step in CUDA        | Parallelize the k‑means centroid update step on the GPU; test with synthetic data.                                                 |
| 57  | PageRank Push Kernel               | Implement one “push” iteration of PageRank on a sparse graph stored in CSR format.                                                 |
| 58  | Graph Coloring Heuristic           | Write a simple greedy graph coloring kernel using shared memory for color counts.                                                  |
| 59  | Bounding‑Box NMS                   | Implement non‑maximum suppression for bounding boxes in CUDA; compare speed vs CPU.                                                |
| 60  | Streaming Prefix Sum               | Combine asynchronous copy + prefix‑sum kernel to process a sliding window over streaming input.                                   |

---

## Days 61–70: Domain‑Specific Accelerations  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 61  | FIR Filter in CUDA                 | Implement a finite impulse response filter on 1D signal; benchmark varying filter lengths.                                         |
| 62  | Image Convolution Tile             | Convolve a 2D image with a small kernel (e.g., 3×3) using shared‑memory tiles.                                                     |
| 63  | Stereo Disparity SSD               | Compute sum of squared differences for stereo image blocks in parallel.                                                             |
| 64  | CUDA Ray Casting                   | Cast rays against a simple sphere; compute color and depth buffers on GPU.                                                         |
| 65  | Volume Rendering                   | Implement a basic ray marching kernel over a 3D volume; accumulate color with early‑exit.                                          |
| 66  | Monte Carlo Path Tracer            | Write a minimal stochastic path‑tracer kernel sampling random directions.                                                          |
| 67  | Neural Net Inference Loop          | Build a tiny fully‑connected layer inference in pure CUDA C++ (no libraries).                                                      |
| 68  | LSTM Cell Forward Pass             | Implement forward pass of an LSTM cell for one timestep in CUDA.                                                                   |
| 69  | Transformer QKV MatMuls            | Write three mat‑mul kernels for Q, K, V projections and combine outputs (batched).                                                 |
| 70  | Attention Softmax Kernel           | Implement the softmax over attention scores in CUDA with numerical stability.                                                      |

---

## Days 71–80: Advanced ML & Hybrid Workloads  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 71  | Fused GEMM + Activation            | Fuse a small GEMM and activation function (ReLU or Sigmoid) into one kernel.                                                       |
| 72  | Layer‑Norm Kernel                  | Implement layer normalization across features for a batch of vectors.                                                             |
| 73  | GroupNorm on GPU                   | Parallelize group normalization using shared memory and warp shuffles.                                                            |
| 74  | Dropout Mask Generation            | Generate dropout masks with cuRAND and apply element‑wise in a single kernel.                                                     |
| 75  | Custom BatchNorm                   | Implement forward and backward batch normalization kernels.                                                                        |
| 76  | FP8 Quantized MatMul               | Simulate FP8 quantized matrix multiply in CUDA, including dequant/dequant steps.                                                  |
| 77  | Sparse × Dense MatMul              | Using CSR format, multiply a sparse matrix by a dense vector in one kernel.                                                       |
| 78  | GraphSAGE Neighborhood Aggregate    | Implement one layer of GraphSAGE aggregator over CSR‑stored graph in CUDA.                                                         |
| 79  | All‑Reduce for Multi‑GPU           | Emulate a ring all‑reduce across 2–4 GPUs using `cudaMemcpyPeerAsync` and streams.                                                 |
| 80  | RL Policy Gradient Kernel          | Compute policy gradient update for a batch of trajectories in CUDA.                                                              |

---

## Days 81–90: Tuning, Testing & Pipeline Integration  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 81  | Unit‑Test Your Kernel              | Integrate a small test framework (e.g., GoogleTest) to automatically verify kernel outputs on random inputs.                       |
| 82  | CI‑Driven CUDA Build               | Write a CI script (GitHub Actions) that builds your CUDA code, runs tests, and reports coverage.                                   |
| 83  | Benchmark Harness                  | Create a harness to run multiple kernels with varying sizes; log timing to CSV for later analysis.                                 |
| 84  | Kernel Parameter Sweep             | Automate testing of different block sizes/grid sizes to find best configuration for a given kernel.                              |
| 85  | Cross‑Platform CUDA Check          | Verify kernel correctness and performance on two different GPU architectures (e.g., Turing vs. Ampere).                           |
| 86  | JIT Kernel Compilation             | Use NVRTC to compile a kernel from a string at runtime and launch it.                                                             |
| 87  | Multi‑Language Binding             | Expose a CUDA kernel via a C API and call it from another language (e.g., Python ctypes or Rust FFI).                              |
| 88  | Real‑Time Data Ingestion           | Integrate a CUDA kernel into a streaming pipeline (e.g., reading from TCP socket) in your playground.                              |
| 89  | Profiling Dashboard                | Parse CSV timing logs and visualize kernel performance trends (you can simulate data in playground).                              |
| 90  | Automated Regression Tests         | Detect performance regressions by comparing current timings against historical baselines.                                          |

---

## Days 91–100: Final Challenges & Showcase  
| Day | Puzzle Title                       | Description                                                                                                                        |
|-----|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| 91  | Custom Memory Allocator            | Build a simple GPU slab allocator on top of `cudaMallocAsync`/pools; track allocations.                                             |
| 92  | Live Kernel Configurator           | Implement a host API that tweaks block/grid dims at runtime based on input size heuristics.                                       |
| 93  | Fault‑Injection & Recovery         | Simulate a hardware fault (e.g., bad memory read) and recover gracefully in your kernel.                                           |
| 94  | Dynamic Workload Balancing         | Implement a work‑stealing queue between blocks to balance uneven workloads.                                                        |
| 95  | Kernel Fusion Planner              | Write a host‑side routine that fuses two simple kernels (e.g., element‑wise add+mul) into one launch.                              |
| 96  | Cross‑Kernel Dependency Graph      | Build a DAG of kernel dependencies on the host and launch kernels accordingly with streams/events.                                  |
| 97  | Self‑Profiling Kernel              | Have your kernel record per‑thread timing or counters (e.g., via clock64) and report statistics back to host.                       |
| 98  | Custom Tensor Core Emulator        | In CUDA C++, emulate a basic tensor‑core‑style 16×16 FP16 tile multiply in parallel threads.                                         |
| 99  | GitHub Portfolio Write‑Up          | Auto‑generate a Markdown summary of all your puzzles with links to code snippets and timing charts.                                 |
| 100 | Grand Finale Project               | Choose one previous puzzle or combine multiple to create a mini “CUDA app” showcase, and document it end‑to‑end in a README.         |
