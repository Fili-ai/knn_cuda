# High-Performance Data and Graph Analytics - Fall 2023 Contest

## Challenge

GPU-accelerated Exact K-Nearest Neighobor with Cosine as distance metric

__Author__: Bolshakova Liubov, Galli Filippo

## Tips & Tricks

### Logging - Debug

In __knn.cpp__ there a define of the LOG_LEVEL in this way we can set it and show some useful information about our code. Log levels are:

- 2 -> only error of cuda commands
- 1 -> + display results of our cuda kernel
- 0 -> + display results of CPU reference and CUDA

## Implementations

### Solution 1

The first attempt to resolve this challenge on GPU. The shortest path is to modify a bit the function on the file knn.cpp to run the same function, sequentially, on GPU.
This result has been achieved and as everyone could deduce the program is slower than run on CPU.

### Solution 2

The second attempt is focused on using the parallel power of GPU. The difference between the first trial is that in this case we define block size and grid size and each thread processed a query.
It works and is faster than knn_gpu_1_Block_Grid.
Initially we try to allocate distance and index array directly in each thread to avoid the problem of concurrent access to the memory. We fail beacuse right from the start with the smallest parameters we allocate too much memory per thread.
So we allocate in the global memory all the space to contain all the index's array and distance ones for each thread. In this way all work fine but we have a bit of over head to read from the global memory.

### Solution 3

Our third attempt is focus on the parallelization on the reference and not queries.

### Solution 4

Use stream to adjust the level of parallelization according to the task to run.
