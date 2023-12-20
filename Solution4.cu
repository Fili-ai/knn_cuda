#pragma once

__global__ void fill_gpu(const float * ref,
                         const int     ref_nb,
                         const float * query,
                         const int     query_nb,
                         const int     dim,
                         float       * dots,
                         float       * denom_a,
                         float       * denom_b) {
    
    /**
     * @brief function to fill dots, denom_a, denom_b
     * @param ref array containing all references
     * @param ref_nb number of references
     * @param query array containing all queries
     * @param query_nb number of queries
     * @param dim dimension of each point (same for queries and references)
     * @param dots array containing each reference-query product for each dimension
     * @param denom_a array containing each reference-reference product for each dimension
     * @param denom_b array containing each query-query product for each dimension
     * 
    */

    __shared__ int size;
    if(threadIdx.x == 0)
        size = query_nb*dim*ref_nb;
    __syncthreads();

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = unique_id / (ref_nb*dim);
    int ref_index = unique_id / dim - query_index*ref_nb;
    int d = unique_id % dim;
    
    if(unique_id < size ){

        dots[unique_id] = ref[d * ref_nb + ref_index] * query[d * query_nb + query_index]; 
        denom_a[unique_id] = ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index];
        denom_b[unique_id] = query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
}

__global__ void reduce0(const float * g_idata, 
                        float       * g_odata, 
                        const int     query_nb, 
                        const int     ref_nb, 
                        const int     dim) {
    
    /**
     * @brief function to do a reduction of the input array g_idata and save result in g_odata
     * @param g_idata array with input
     * @param g_odata array to store output 
     * @param query_nb number of queries
     * @param ref_nb number of references
     * @param dim dimension of each point (same for queries and references)
    */

   //[Assumption] threadIdx.x % dim == 0

    extern __shared__ float sdata[];
    __shared__ int size;

    if (threadIdx.x == 0)
        size = query_nb * dim * ref_nb;

    __syncthreads();

    // index of the thread inside the block 
    unsigned int tid = threadIdx.x;

    unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    int query_index = unique_id / (ref_nb*dim);
    int ref_index = unique_id / dim - query_index*ref_nb;

    // each thread loads one element from global to shared mem
    if (unique_id < size)
        sdata[tid] = g_idata[unique_id];

    // do reduction in shared mem
    for(unsigned int s=1; s < dim; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid % dim == 0){
        g_odata[query_index + ref_index * query_nb] = sdata[tid];
    }

}

__global__ void reduceDimension(const float * dots, 
                                const float * denom_a, 
                                const float * denom_b,
                                const int     ref_nb, 
                                const int     query_nb, 
                                const int     dim,
                                float       * sum_dots, 
                                float       * sum_denom_a, 
                                float       * sum_denom_b) {
    
    /**
     * @brief first attempt to sum dots, denom_a, denom_b
     * @param dots array containing each reference-query product for each dimension
     * @param denom_a array containing each reference-reference product for each dimension
     * @param denom_b array containing each query-query product for each dimension
     * @param ref_nb number of references
     * @param query_nb number of queries
     * @param dim dimension of each point (same for queries and references)
     * @param sum_dots array to store result of the sums of dots
     * @param sum_denom_a array to store result of the sums of denom_a
     * @param sum_denom_b array to store result of the sums of denom_b
     * 
    */

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    int query_index = unique_id / ref_nb;
    int ref_index = unique_id % ref_nb;

    int starting_index = ref_index * dim + query_index * ref_nb * dim;

    float temp_dots = 0;
    float temp_denom_a = 0;
    float temp_denom_b = 0;
    unsigned it = 0;

    for (int d = 0; d < dim; ++d) {
        it = starting_index + d;
        temp_dots += dots[it];
        temp_denom_a += denom_a[it];
        temp_denom_b += denom_b[it];
    }

    sum_dots[query_index + ref_index * query_nb] = temp_dots;
    sum_denom_a[query_index + ref_index * query_nb] = temp_denom_a;
    sum_denom_b[query_index + ref_index * query_nb] = temp_denom_b;

}

__global__ void cosine_distance_gpu(const int     ref_nb,
                                    const int     query_nb,
                                    float       * dist,
                                    int         * index, 
                                    const float * sum_dots, 
                                    const float * sum_denom_a, 
                                    const float * sum_denom_b) {
    
    /**
     * @brief function to calculate the cosine distance of references and queries
     * @param ref_nb number of references
     * @param query_nb number of queries
     * @param dist array containing all reference's distances
     * @param index array containing all reference's indexes
     * @param sum_dots array to store result of the sums of dots
     * @param sum_denom_a array to store result of the sums of denom_a
     * @param sum_denom_b array to store result of the sums of denom_b
    */

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = unique_id / ref_nb;
    int ref_index = unique_id % ref_nb;;

    float temp_denom_a = sqrt(sum_denom_a[query_index + ref_index*query_nb]);
    float temp_denom_b = sqrt(sum_denom_b[query_index + ref_index*query_nb]);

    if(query_index < query_nb && ref_index < ref_nb){

        index[query_index + ref_index*query_nb] =  ref_index;
        dist[query_index + ref_index*query_nb] = sum_dots[query_index + ref_index*query_nb] / (temp_denom_a * temp_denom_b );
               
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