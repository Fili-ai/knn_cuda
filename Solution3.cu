/**
* Insertion sort given with data coaliscent 
*/
__device__ void insertion_sort_gpu(float *dist, int *index, int length, int k, int query_nb, int query_index){
    
    /*
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
    */

    // store in a multimap as kay the dist and as value the index, to sort it automatically
    std::multimap<float, unsigned int> pairs;

    for(int i = query_index; i < length; i += query_nb){
        
        pairs.insert(pair<float, unsigned>(i, dist[i]));

        if(pairs.size() == k){
            // erase the last element, which has the highest distance
            mp.erase(prev(mp.end())); 
        }

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
        insertion_sort_gpu(dist, index, ref_nb, k, query_nb, query_index);

        // from coalescent data (index and dist) to sequential data (knn_index and knn_dist)
        for(int reference = 0; reference < k; ++reference){
            knn_index[query_index + reference] = index[query_index + reference * query_nb];
            knn_dist[query_index + reference] = dist[query_index + reference * query_nb];
        } 
    }
}