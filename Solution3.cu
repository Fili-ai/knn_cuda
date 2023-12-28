#pragma once

__global__ void cosine_distance_gpu(const float * ref,
                                    const int     ref_nb,
                                    const float * query,
                                    const int     query_nb,
                                    const int     dim,
                                    float *       dist,
                                    int *         index,  
                                    const int     chunk) {
    
    /**
     * @brief function to calculate the cosine distance of references and queries
     * @param ref array containing all references
     * @param ref_nb number of references
     * @param query array containing all queries
     * @param query_nb number of queries
     * @param dim dimension of each point (same for queries and references)
     * @param index array containing all reference's indexes
     * @param dist array containing all reference's distances
     * @param chunk number of query processed per iteration
    */

    // unique id of a thread
    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // query and reference index on which the thread have to work 
    int query_index = unique_id / (ref_nb);
    int ref_index = unique_id % ref_nb;
    
    //cosine distance calculations
    if(query_index < query_nb){

        double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;

        #pragma unroll
        for(unsigned int d = 0u; d < dim; ++d) {
            dot += ref[d * ref_nb + ref_index] * query[d * chunk + query_index];
            denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
            denom_b += query[d * chunk + query_index] * query[d * chunk + query_index] ;
        } 

        index[query_index + ref_index*chunk] = ref_index;
        dist[query_index + ref_index*chunk] = dot / (sqrt(denom_a) * sqrt(denom_b));

    }  
}

__global__ void insertion_sort_gpu( const int     ref_nb,
                                    const int     query_nb,
                                    const int     dim,
                                    const int     k,
                                    float *       knn_dist,
                                    int *         knn_index,
                                    const int *   index, 
                                    const float * dist){

    /**
     * @brief Insertion sort for the reference's distances
     * @param ref_nb number of references
     * @param query_nb number of queries
     * @param dim dimension of each element (same for each query and reference)
     * @param k number of items of interest
     * @param knn_index array to store the first k reference's indexes
     * @param knn_dist array to store the first k reference's distances
     * @param index array with the index of the reference distance
     * @param dist array with distances to sort
    */

    // unique id of a thread
    int query_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(query_index < query_nb){
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
            // to save the k distances at distance query_nb
            knn_index[query_index + i * query_nb] = index_sorted[i];
            knn_dist[query_index + i * query_nb] = dist_sorted[i];
        }

        free(dist_sorted);
        free(index_sorted); 
    }
}