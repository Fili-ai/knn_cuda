# High-Performance Data and Graph Analytics - Fall 2023 Contest

## Challenge

GPU-accelerated Exact K-Nearest Neighobor with Cosine as distance metric

__Author__: Bolshakova Liubov, Galli Filippo

## Implementations

### knn_gpu_1_Block_Grid

The first attempt to resolve this challenge on GPU. The shortest path is to modify a bit the function on the file knn.cpp to run the same function, sequentially, on GPU.
This result has been achieved and as everyone could deduce the program is slower than run on CPU.

### knn_gpu / knn_gpu_v2 - in progress

The second attempt is focused on using the parallel power of GPU. The difference between the first trial is that in this case we define block size and grid size and each thread processed a query.
It works and is faster than knn_gpu_1_Block_Grid.
The problem to resolve is that after the kernel is launched when cudaMemcpy from device to host, we have the following error "An illegal memory access was encountered". Although there is this error the function always returns "PASSED", a bit strange behaviour.
To fix it the query_nb in knn.cpp should be decreased to 128, so I think we are allocating too much space per thread.

Previous of knn_gpu is right, it has too much memory allocated per thread. So I created a function knn_gpu_v2 and insertion_sort_gpu_v2 that work on dist and index in the global memory. The error before doesn't appear but the there is a __problem__ on the Insertion_sort_gpu_v2 that from 3 query it doesn't work well
