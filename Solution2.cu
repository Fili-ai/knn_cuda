#pragma once

__device__ void insertion_sort_gpu(const float * dist, 
                                   const int   * index, 
                                   const int     length, 
                                   const int     k, 
                                   const int     query_nb, 
                                   const int     query_index, 
                                   int         * knn_index,
                                   float       * knn_dist){ 

    /**
     * @brief Insertion sort for the reference's distances
     * @param dist array with distances to sort
     * @param index array with the index of the reference distance
     * @param length number of references
     * @param k number of items of interest
     * @param query_nb number of queries
     * @param query_index index of the current query
     * @param knn_index array to store the first k reference's indexes
     * @param knn_dist array to store the first k reference's distances
    */

    // Allocate local array to store sorted distances and indexes  
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
    
    // save the first k sorted references and indexes
    for(int i = 0; i < k; ++i){
        knn_index[query_index + i * query_nb] = index_sorted[i];
        knn_dist[query_index + i * query_nb] = dist_sorted[i];
    }

    free(dist_sorted);
    free(index_sorted); 
}

__device__ float cosine_distance_gpu(const float * ref,
                                     const int     ref_nb,
                                     const float * query,
                                     const int     query_nb,
                                     const int     dim,
                                     const int     ref_index,
                                     const int     query_index) {

    /**
     * @brief function to calculate the cosine distance of a reference and a query
     * @param ref array containing all references
     * @param ref_nb number of references
     * @param query array containing all queries
     * @param query_nb number of queries
     * @param dim dimension of each point (same for queries and references)
     * @param ref_index index of the current reference
     * @param query_index index of the current query
     * @return cosine distance between the current query and the current reference
    */

    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int d = 0u; d < dim; ++d) {
        dot += ref[d * ref_nb + ref_index] * query[d * query_nb + query_index] ;
        denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b += query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
}

__global__ void knn_gpu(const float * ref,
                        const int     ref_nb,
                        const float * query,
                        const int     query_nb,
                        const int     dim,
                        const int     k,
                        float *       knn_dist,
                        int *         knn_index,
                        int *         index, 
                        float *       dist){

    /**
     * @brief function to calculate all the query-reference distance and sort them
     * @param ref array containing all references
     * @param ref_nb number of references
     * @param query array containing all queries
     * @param query_nb number of queries
     * @param dim dimension of each point (same for queries and references)
     * @param k number of items of interest 
     * @param knn_dist array containing the first k distances for each query
     * @param knn_dist array containing the first k references for each query
     * @param index array containing all reference's indexes 
     * @param dist array containing all reference's distances
    */

    // Each thread work on a small number of query 
    for (int query_index = blockIdx.x * blockDim.x + threadIdx.x; query_index < query_nb; query_index += blockDim.x * gridDim.x) {
        
        // Compute all query-reference distances
        for(int reference = 0; reference < ref_nb; reference++){
            index[query_index + reference*query_nb] =  reference;
            dist[query_index + reference*query_nb] = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, reference, query_index);
        } 

        insertion_sort_gpu(dist, index, ref_nb, k, query_nb, query_index, knn_index, knn_dist);               
    }
}