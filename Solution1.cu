#pragma once

__device__ void insertion_sort_gpu(float *dist_sort, 
                                   int *index_sort, 
                                   const int length, 
                                   const int k){
    /**
     * @brief Insertion sort for the reference's distances
     * @param dist_sort array with distances to sort
     * @param index_sort array with the index of the reference distance
     * @param length number of references
     * @param k number of items of interest
    */

    // Initialise the first index
    index_sort[0] = 0;

    // Go through all distances
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist_sort[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist_sort[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = i < k-1 ? i : k-1; 
        while (j > 0 && dist_sort[j-1] > curr_dist) {
            dist_sort[j]  = dist_sort[j-1];
            index_sort[j] = index_sort[j-1];
            --j;
        }

        // Write the current distance and index at their right positions
        dist_sort[j]  = curr_dist;
        index_sort[j] = curr_index; 
    }
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

    // Temporary variable to store calculations
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;

    for(unsigned int d = 0u; d < dim; ++d) {
        dot += ref[d * ref_nb + ref_index] * query[d * query_nb + query_index] ;
        denom_a += ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index] ;
        denom_b += query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }

    return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
}

/**
 * Kernel to solve our problem. It elaborate all queries
*/
__global__ void knn_gpu(const float * ref,
                        const int     ref_nb,
                        const float * query,
                        const int     query_nb,
                        const int     dim,
                        const int     k,
                        float *       knn_dist,
                        int *         knn_index){
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
    */

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

    // loop for each query
    for (int i=0; i<query_nb; ++i) {

        // Compute all reference distances and indexes
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = cosine_distance_gpu(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

        // Sort distances and corrisponding indexes
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