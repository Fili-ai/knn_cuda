#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "variables.h"

__global__ void insertion_sort_gpu( const float *dist, 
                                    const int *index, 
                                    const int ref_nb, 
                                    const int k, 
                                    const int query_nb,  
                                    int * knn_index,
                                    float * knn_dist, 
                                    int dim){ 
    
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int unique_id = id_x + id_y * gridDim.x * blockDim.x;

    int query_index = unique_id / (ref_nb * dim);
    // int ref_index = unique_id / dim - query_index * ref_nb;


    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist_sorted  = (float *) malloc((k+1) * sizeof(float));
    int *   index_sorted = (int *)   malloc((k+1) * sizeof(int));
    float curr_dist;
    int  curr_index;
    
    for (int i=0; i<ref_nb; ++i) {

        // Store current distance and associated index
        curr_dist  = dist[query_index + i * query_nb];
        curr_index = index[query_index + i * query_nb];     
        
        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist_sorted[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = i < k-1 ? i : k-1;  
        while (j >= 0 && dist_sorted[j-1] > curr_dist) {
            dist_sorted[j]  = dist_sorted[j-1];
            index_sorted[j] = index_sorted[j-1];
            --j;
        }
    
        // Write the current distance and index at their position
        dist_sorted[j]  = curr_dist;
        index_sorted[j] = curr_index; 
            
    }
    

    for(int i = 0; i < k; ++i){
        /*
        // to save the k distances consecutively
        knn_index[query_index*k + i] = index_sorted[i];
        knn_dist[query_index*k + i] = dist_sorted[i];
        */
        // to save the k distances at distance query_nb
        knn_index[query_index + i * query_nb] = index_sorted[i];
        knn_dist[query_index + i * query_nb] = dist_sorted[i];
    }

    free(dist_sorted);
    free(index_sorted); 
}

__global__ void dots_to_dist(const double *      dots, 
                             const double *      denom_a,
                             const double *      denom_b,
                             const int           query_nb,
                             const int           dim,
                             float *             dist,
                             int *               index, 
                             const int           ref_nb){
    /**
     * @brief Each thread sum all the dim of the product between a reference and a query
    */


    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int unique_id = id_x + id_y * gridDim.x * blockDim.x;

    int query_index = unique_id / (ref_nb * dim);
    int ref_index = unique_id / dim - query_index * ref_nb;

    // double dot = 0;
    // double a = 0;
    // double b = 0;

    for(int i = 0; i < dim; i++){
        // dot += dots[unique_id + i];
        // a += denom_a[unique_id + i];
        // b += denom_b[unique_id + i];
    }

    dist[query_index + ref_index*query_nb] = ref_index;
    index[query_index + ref_index*query_nb] = ref_index;

}

__global__ void cosine_distance(const float *  ref,
                                const int           ref_nb,
                                const float * query,
                                const int           query_nb,
                                const int           dim, 
                                double *      dots, 
                                double *      denom_a,
                                double *      denom_b ){
    
    /**
     * work only for 1D block and 1D/2D grid 
    */
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int unique_id = id_x + id_y * gridDim.x * blockDim.x;

    if(unique_id < dim * query_nb * ref_nb){
        int query_index = unique_id / (ref_nb * dim);
        int ref_index = unique_id / dim - query_index * ref_nb;
        int d = unique_id % dim; // dimension

        // dots[unique_id] = ref_index;
        dots[unique_id] = ref[d * ref_nb + ref_index] * query[d * query_nb + query_index];

        // denom are the same
        denom_a[unique_id] = ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b[unique_id] = query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
}
