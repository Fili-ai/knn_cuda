#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "Solution1.cu"
//#include "Solution2.cu"
//#include "Solution3.cu"
#include "Solution4.cu"

/**
* Error checking function;
*/
#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: line %d - %s\n", line, cudaGetErrorString(code));
        if (abort)
            exit(code);
    }
}

bool your_solution(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index) {


    /**
    * @input ref reference points
    * @input ref_nb number of reference points
    * @input query the data to process and classify
    * @input query_nb number of query
    * @input dim dimension of every single point
    * @input k number of neighbors
    * @input knn_dist array to save the distances between every query 
             and the reference point
    * @input knn_index array with the solution of the classification
    */

    // ---------------------------------- Creating data location on gpu -------------------------------
    // Location for all reference data
    float * ref_gpu;
    gpuErrchk(cudaMalloc(&ref_gpu, ref_nb*dim*sizeof(float)));

    // Location for all query data
    float * query_gpu;
    gpuErrchk(cudaMalloc(&query_gpu, query_nb*dim*sizeof(float)));

    // Location for the k-nearest distances
    float * knn_dist_gpu;
    gpuErrchk(cudaMalloc(&knn_dist_gpu, query_nb*k*sizeof(float)));

    // Location for the k-nearest index
    int * knn_index_gpu;
    gpuErrchk(cudaMalloc(&knn_index_gpu, query_nb*k*sizeof(int)));

    // Location for index and dist 
    int * index_gpu;
    gpuErrchk(cudaMalloc(&index_gpu, query_nb*ref_nb*sizeof(int)));
    float * dist_gpu;
    gpuErrchk(cudaMalloc(&dist_gpu, query_nb*ref_nb*sizeof(float)));
    

    // ---------------------------------- Transfering data on device -------------------------------

    gpuErrchk(cudaMemcpy(ref_gpu, ref, ref_nb*dim*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(query_gpu, query, query_nb*dim*sizeof(float), cudaMemcpyHostToDevice));

    //gpuErrchk(cudaDeviceSynchronize());

    // ---------------------------------- Kernel launching -------------------------------

    // Solution - 1
    //knn_gpu_1_Block_Grid<<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);
     
    // Solution - 2
    //dim3 block_size(32, 1, 1);
    //dim3 grid((query_nb + block_size.x - 1) / block_size.x, 1, 1);
    //knn_gpu<<<grid, block_size>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
 
    // Solution - 3
    //dim3 block_size(32, 1, 1);
    //dim3 grid((query_nb + block_size.x - 1) / block_size.x, 1, 1);
    //dim3 block_size_cosine_distance(32, 1, 1);
    //dim3 grid_cosine_distance((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1);
    //cosine_distance_gpu<<<grid_cosine_distance, block_size_cosine_distance>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, dist_gpu, index_gpu);
    //insertion_sort_gpu<<<grid, block_size>>>(ref_nb, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
    
    // Solution - 4
    // memory management 
    //float * dots;
    //gpuErrchk(cudaMallocManaged(&dots, query_nb*ref_nb*dim*sizeof(float)));
    //float * denom_a;
    //gpuErrchk(cudaMallocManaged(&denom_a, query_nb*ref_nb*dim*sizeof(float)));
    //float * denom_b;
    //gpuErrchk(cudaMallocManaged(&denom_b, query_nb*ref_nb*dim*sizeof(float)));
    //float * sum_dots;
    //gpuErrchk(cudaMallocManaged(&sum_dots, query_nb*ref_nb*sizeof(float)));
    //float * sum_denom_a;
    //gpuErrchk(cudaMallocManaged(&sum_denom_a, query_nb*ref_nb*sizeof(float)));
    //float * sum_denom_b;
    //gpuErrchk(cudaMallocManaged(&sum_denom_b, query_nb*ref_nb*sizeof(float)));

    // block and grid dimension 
    //dim3 block_size(1024, 1, 1);
    //dim3 grid((query_nb + block_size.x - 1) / block_size.x, 1, 1);
    //dim3 block_size_fill(1024, 1, 1);
    //dim3 grid_fill((ref_nb*query_nb*dim + block_size.x - 1) / block_size.x, 1, 1);
    //dim3 block_size_reduction(1024, 1, 1);
    //dim3 grid_reduction((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1); 
    //dim3 block_size_cosine_distance(1024, 1, 1);
    //dim3 grid_cosine_distance((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1);

    // kernel launching
    //fill_gpu<<<grid_fill, block_size_fill>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, dots, denom_a, denom_b);
    ////reduceDimension<<<grid_reduction, block_size_reduction>>>(dots, denom_a, denom_b, ref_nb, query_nb, dim, sum_dots, sum_denom_a, sum_denom_b); 
    //reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(dots, sum_dots, query_nb, ref_nb, dim);
    //reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(denom_a, sum_denom_a, query_nb, ref_nb, dim);
    //reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(denom_b, sum_denom_b, query_nb, ref_nb, dim);
    //cudaFree(dots);
    //cudaFree(denom_a);
    //cudaFree(denom_b);
    //cosine_distance_gpu<<<grid_cosine_distance, block_size_cosine_distance>>>(ref_nb, query_nb, dist_gpu, index_gpu, sum_dots, sum_denom_a, sum_denom_b);
    ////cudaFree(sum_dots);
    ////cudaFree(sum_denom_a);
    ////cudaFree(sum_denom_b);
    //insertion_sort_gpu<<<grid, block_size>>>(ref_nb, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
    
    // ---------------------------------- Transfering data on host -------------------------------

    gpuErrchk(cudaMemcpy(knn_dist, knn_dist_gpu, query_nb*k*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(knn_index, knn_index_gpu, query_nb*k*sizeof(int), cudaMemcpyDeviceToHost));
    
    // ---------------------------------- Free memory -------------------------------

    cudaFree(ref_gpu);
    cudaFree(query_gpu);
    cudaFree(knn_dist_gpu);
    cudaFree(knn_index_gpu);

    cudaFree(index_gpu);
    cudaFree(dist_gpu);

    return true;
}