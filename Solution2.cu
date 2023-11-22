#pragma once

/**
* Insertion sort for the queries 
*/
__device__ void insertion_sort_gpu( const float *dist, 
                                    const int *index, 
                                    const int length, 
                                    const int k, 
                                    const int query_nb, 
                                    const int query_index, 
                                    int * knn_index,
                                    float * knn_dist){ 

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist_sorted  = (float *) malloc((k+1) * sizeof(float));
    int *   index_sorted = (int *)   malloc((k+1) * sizeof(int));
    float curr_dist;
    int  curr_index;
    
    for (int i=0; i<length; ++i) {

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


/**
 * Cosine distance
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

/**
 * Kernel to solve the problem. It works in parallel, each kernel work on a subset of all queries
*/
__global__ void knn_gpu(const float *  ref,
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
        insertion_sort_gpu(dist, index, ref_nb, k, query_nb, query_index, knn_index, knn_dist);
                
    }
}