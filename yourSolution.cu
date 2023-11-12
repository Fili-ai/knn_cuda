#include <stdio.h>
#include <iostream>
#include <cuda.h>

/**
 * Computes the Euclidean distance between a reference point and a query point.
 */
float cosine_distance_testGPU(const float * ref,
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

__device__ void insertion_sort_gpu(float *dist, int *index, int length, int k){
        // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        //int j = std::min(i, k-1);
        int j = i < k-1 ? i : k-1; 
        while (j > 0 && dist[j-1] > curr_dist) {
            dist[j]  = dist[j-1];
            index[j] = index[j-1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j]  = curr_dist;
        index[j] = curr_index; 
    }
}

__global__ void cosine_distance_gpu(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index,
                       int *         distance) {

    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int d = 0u; d < dim; ++d) {
        dot += ref[d * ref_nb + ref_index] * query[d * query_nb + query_index] ;
        denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b += query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
    *distance = dot / (sqrt(denom_a) * sqrt(denom_b)) ;
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
    for (int i=0; i<query_nb; ++i) {

        // Compute all distances / indexes
        for (int j=0; j<ref_nb; ++j) {
            //dist[j]  = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, j, i);
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
    cudaMallocManaged(&ref_gpu, ref_nb*dim*sizeof(float));

    // Location for all query data
    float * query_gpu;
    cudaMallocManaged(&query_gpu, query_nb*dim*sizeof(float));

    // Location for the k-nearest distances
    float * knn_dist_gpu;
    cudaMallocManaged(&knn_dist_gpu, query_nb*k*sizeof(float));

    // Location for the k-nearest index
    int * knn_index_gpu;
    cudaMallocManaged(&knn_index_gpu, query_nb*k*sizeof(int));

    // test
    int * distance_gpu;
    cudaMallocManaged(&distance_gpu, sizeof(int));
    int distance = 0;

    // ---------------------------------- Transfering data on device -------------------------------

    cudaMemcpy(ref_gpu, ref, ref_nb*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(query_gpu, query, query_nb*dim*sizeof(float), cudaMemcpyHostToDevice);

    // ---------------------------------- Kernel launching -------------------------------

    std::cout << "\n...launching kernel...\n";
    
    //knn_gpu<<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);
    cosine_distance_gpu <<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, 0, 1, distance_gpu);
    distance = cosine_distance_testGPU(ref, ref_nb, query, query_nb, dim, 0, 1);

    std::cout << "Distance on cpu: " << distance << " on gpu: " << *distance_gpu << std::endl;

    
    std::cout << "...kernel finished...\n";
    
    // ---------------------------------- Transfering data on host -------------------------------

    //cudaMemcpy(knn_dist, knn_dist_gpu, query_nb*k*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(knn_index, knn_index_gpu, query_nb*k*sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------------------------- Debug -------------------------------

    /*for(int i = 0; i < k*query_nb; ++i){
        std::cout << "indexes found: " << knn_index[i] <<" distances " << knn_dist[i] << std::endl;
    }*/

    // ---------------------------------- Free memory -------------------------------

    cudaFree(ref_gpu);
    cudaFree(query_gpu);
    cudaFree(knn_dist_gpu);
    cudaFree(knn_index_gpu);
    

    return true;
}

