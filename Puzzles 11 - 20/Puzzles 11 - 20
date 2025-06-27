Shared Memory & Reductions  
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
