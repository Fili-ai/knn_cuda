#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include<cuda_runtime.h>

/**
* Error checking function;
*/

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line, bool abort = false, bool log = true) {
    if (log && code != cudaSuccess) {
        fprintf(stderr, "GPUassert: line %d - %s\n", line, cudaGetErrorString(code));
        if (abort)
            exit(code);
    }
}

/**
* Insertion sort given in knn.cpp
* It doesn't work on knn_gpu_v2 because we are saving data of ref i not consecutively respect i-1 but at a distance query_nb
*/
__device__ void insertion_sort_gpu(float *dist_sort, int *index_sort, int length, int k){
    // Initialise the first index
    index_sort[0] = 0;

    // Go through all points
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist_sort[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist_sort[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        //int j = std::min(i, k-1);
        int j = i < k-1 ? i : k-1; 
        while (j > 0 && dist_sort[j-1] > curr_dist) {
            dist_sort[j]  = dist_sort[j-1];
            index_sort[j] = index_sort[j-1];
            --j;
        }

        // Write the current distance and index at their position
        dist_sort[j]  = curr_dist;
        index_sort[j] = curr_index; 
    }
}

/**
* Insertion sort given with data coaliscent 
*/
__device__ void insertion_sort_gpu_v2(float *dist_sort, int *index_sort, int length, int k, int query_nb, int query_index){
    // Initialise the first index
    index_sort[0] = 0;

    // Go through all points
    for (int i=1; i<length; i++) {

        // Store current distance and associated index
        float curr_dist  = dist_sort[query_index + i*query_nb];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value    
        if (i >= k && curr_dist >= dist_sort[query_index + (k-1)*query_nb]) {
            continue;
        }
        

        // Shift values (and indexes) higher that the current distance to the right
        int j = i < k - 1 ? i : k - 1;
        while (j > 0 && dist_sort[query_index + (j - 1) * query_nb] > curr_dist) {
            dist_sort[query_index + j * query_nb] = dist_sort[query_index +  (j - 1) * query_nb];
            index_sort[query_index +  j * query_nb] = index_sort[(query_index +  j - 1) * query_nb];
            --j;
        }

        // Write the current distance and index at their position
        dist_sort[query_index + j * query_nb] = curr_dist;
        index_sort[query_index + j * query_nb] = curr_index;
    }
}

/**
 * Computes the Euclidean distance between a reference point and a query point.
 */
__device__ float cosine_distance_gpu(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {

    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int d = 0u; d < dim; ++d) {
        dot += ref[d * ref_nb + ref_index] * query[d * query_nb + query_index] ;
        denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b += query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
}

__global__ void knn_gpu_1_Block_Grid(const float *  ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           k,
                        float *       knn_dist,
                        int *         knn_index){

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

    // Process one query point at the time
    for (int i=0; i<query_nb; ++i) {

        // Compute all distances / indexes
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

        // Sort distances / indexes
        insertion_sort_gpu(dist, index, ref_nb, k);

        // Copy k smallest distances and their associated index
        for (int j=0; j<k; ++j) {
            knn_dist[j * query_nb + i]  = dist[j];
            knn_index[j * query_nb + i] = index[j];
        }
    }

    free(index);
    free(dist);
}

__global__ void knn_gpu(const float *  ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           k,
                        float *       knn_dist,
                        int *         knn_index){

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

    // Process one query point at the time
    for (int query_index = threadIdx.x; query_index < query_nb; query_index += blockDim.x) {

        
        // Compute all distances / indexes
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, j, query_index);
            index[j] = j;
        }
     
        // Sort distances / indexes
        insertion_sort_gpu(dist, index, ref_nb, k);

        // Copy k smallest distances and their associated index
        for (int j = 0; j < k; ++j) {
            knn_dist[j * query_nb + query_index]  = dist[j];
            knn_index[j * query_nb + query_index] = index[j];
        }
        
    }

    free(index);
    free(dist);
}

__global__ void knn_gpu_v2(const float *  ref,
                        int           ref_nb,
                        const float * query,
                        int           query_nb,
                        int           dim,
                        int           k,
                        float *       knn_dist,
                        int *         knn_index,
                        int *         index, 
                        float *       dist){

    // Process one query point at the time
    for (int query_index = blockIdx.x * blockDim.x + threadIdx.x; query_index < query_nb; query_index += blockDim.x * gridDim.x) {
        
        // Compute query distance from every reference
        for(int reference = 0; reference < ref_nb; reference++){
            index[query_index + reference*query_nb] =  reference;

            dist[query_index + reference*query_nb] = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, reference, query_index);
        } 

        //data coalescent insertion sort
        insertion_sort_gpu_v2(dist, index, ref_nb, k, query_nb, query_index);

        // from coalescent data (index and dist) to sequential data (knn_index and knn_dist)
        for(int reference = 0; reference < k; ++reference){
            knn_index[query_index + reference] = index[query_index + reference * query_nb];
            knn_dist[query_index + reference] = dist[query_index + reference * query_nb];
        } 
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

    std::cout << "\nStarting Gpu function\n";

    // ---------------------------------- Variables' declaration ------------------------------- 

    int block_size = 1024;
    int grid = (query_nb + block_size -1)/block_size;    

    // ---------------------------------- Creating data location on gpu -------------------------------
    // Location for all reference data
    float * ref_gpu;
    gpuErrchk(cudaMallocManaged(&ref_gpu, ref_nb*dim*sizeof(float)));

    // Location for all query data
    float * query_gpu;
    gpuErrchk(cudaMallocManaged(&query_gpu, query_nb*dim*sizeof(float)));

    // Location for the k-nearest distances
    float * knn_dist_gpu;
    gpuErrchk(cudaMallocManaged(&knn_dist_gpu, query_nb*k*sizeof(float)));

    // Location for the k-nearest index
    int * knn_index_gpu;
    gpuErrchk(cudaMallocManaged(&knn_index_gpu, query_nb*k*sizeof(int)));

    // Location for index and dist 
    int * index_gpu;
    gpuErrchk(cudaMallocManaged(&index_gpu, query_nb*ref_nb*sizeof(int)));
    float * dist_gpu;
    gpuErrchk(cudaMallocManaged(&dist_gpu, query_nb*ref_nb*sizeof(float)));

    // ---------------------------------- Transfering data on device -------------------------------

    gpuErrchk(cudaMemcpy(ref_gpu, ref, ref_nb*dim*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(query_gpu, query, query_nb*dim*sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaDeviceSynchronize());

    // ---------------------------------- Kernel launching -------------------------------

    //knn_gpu<<<grid, block_size>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);
    knn_gpu_v2<<<grid, block_size>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
    //knn_gpu_1_Block_Grid<<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);

    // ---------------------------------- Transfering data on host -------------------------------

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(knn_dist, knn_dist_gpu, query_nb*k*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(knn_index, knn_index_gpu, query_nb*k*sizeof(int), cudaMemcpyDeviceToHost));

    // ---------------------------------- Debug section -------------------------------

    
    for(int i = 0; i < query_nb; ++i){
        std::cout<< i <<" query:" << std::endl; 

        for (int j = 0; j < k; ++j){
            std::cout << "\treference index: " << knn_index[i + j] << " dist: " << knn_dist[i + j] <<std::endl;
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