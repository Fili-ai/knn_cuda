#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "variables.h"

//#include "Solution1.cu"
#include "Solution2.cu"
//#include "Solution4.cu"

/**
* Error checking function;
*/
#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line, bool abort = false) {
    if (LOG_LEVEL < 3 && code != cudaSuccess) {
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

    // ---------------------------------- Variables' declaration ------------------------------- 

    dim3 block_size(1024, 1, 1);
    dim3 grid((query_nb + block_size.x - 1) / block_size.x, 1, 1);
    
    dim3 block_size_cosine_distance(1024, 1, 1);
    dim3 grid_cosine_distance((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1);

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

    gpuErrchk(cudaDeviceSynchronize());

    // ---------------------------------- Kernel launching -------------------------------

    // Solution - 1
    //knn_gpu_1_Block_Grid<<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);
     
    // Solution - 2
    knn_gpu<<<grid, block_size>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);

    
    // Solution - 4
    // cosine_distance_gpu<<<grid_cosine_distance, block_size_cosine_distance>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, dist_gpu, index_gpu);
    // insertion_sort_gpu<<<grid, block_size>>>(ref_nb, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);


    // ---------------------------------- Transfering data on host -------------------------------

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(knn_dist, knn_dist_gpu, query_nb*k*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(knn_index, knn_index_gpu, query_nb*k*sizeof(int), cudaMemcpyDeviceToHost));

    // ---------------------------------- Debug section -------------------------------

    
    if(LOG_LEVEL < 2){   
        for(int i = 0; i < query_nb ; ++i){
            std::cout<< std::endl << i <<" query:" << std::endl; 
            for (int j = 0; j < ref_nb; ++j){
                std::cout << "\treference index: " << index_gpu[i + j*query_nb] << " dist: " << dist_gpu[i + j*query_nb] <<std::endl;
            }
        }
    }

    
    // ---------------------------------- Free memory -------------------------------

    cudaFree(ref_gpu);
    cudaFree(query_gpu);
    cudaFree(knn_dist_gpu);
    cudaFree(knn_index_gpu);

    cudaFree(index_gpu);
    cudaFree(dist_gpu);
    
    return true;
}