#pragma once

__device__ int get_query_id(int unique_id, int ref_nb){
    return unique_id / ref_nb;
}

__device__ int get_ref_id(int unique_id, int ref_nb){
    return unique_id % ref_nb;
}

__device__ int get_dim(int unique_id, int dim){
    return unique_id % dim;
}

/**
 * filling dots, denom_a, denom_b
 */
__global__ void fill_gpu(const float * ref,
                        const int     ref_nb,
                        const float * query,
                        const int     query_nb,
                        const int     dim,
                        float *       dots,
                        float *       denom_a,
                        float *       denom_b) {
    
    __shared__ int size;
    if(threadIdx.x == 0)
        size = query_nb*dim*ref_nb;
    __syncthreads();

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = get_query_id(unique_id, ref_nb);
    int ref_index = get_ref_id(unique_id, ref_nb);
    int d = get_dim(unique_id, dim);

    int it = query_index + ref_index*dim + d*ref_nb*dim;
    
    if(it < size ){

        //dots[it] = it;
        dots[it] = ref[d * ref_nb + ref_index] * query[d * query_nb + query_index]; 
        denom_a[it] = ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index];
        denom_b[it] = query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }
}

// Reduction kernel to sum values along the dimension
__global__ void reduceDimension(const float* dots, 
                                const float* denom_a, 
                                const float* denom_b,
                                const int ref_nb, 
                                const int query_nb, 
                                const int dim,
                                float* sum_dots, 
                                float* sum_denom_a, 
                                float* sum_denom_b) {
    
    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    int query_index = get_query_id(unique_id, ref_nb);
    int ref_index = get_ref_id(unique_id, ref_nb);
    
    double temp_dots = 0; 
    double temp_denom_a = 0;
    double temp_denom_b = 0;

    int it = query_index + ref_index*dim;

    if (query_index < query_nb && ref_index < ref_nb){
        

        for(int d = 0; d < dim; ++d){
            temp_dots += dots[it + ref_nb*dim*d];
            temp_denom_a += denom_a[it + ref_nb*dim*d];
            temp_denom_b += denom_b[it + ref_nb*dim*d];
        }
    
        sum_dots[query_index + ref_index*query_nb] = temp_dots;
        sum_denom_a[query_index + ref_index*query_nb] = temp_denom_a;
        sum_denom_b[query_index + ref_index*query_nb] = temp_denom_b;
    }
}

__global__ void cosine_distance_gpu(const int     ref_nb,
                                    const int     query_nb,
                                    float *       dist,
                                    int *         index, 
                                    const float* sum_dots, 
                                    const float* sum_denom_a, 
                                    const float* sum_denom_b) {
    
    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = get_query_id(unique_id, ref_nb);
    int ref_index = get_ref_id(unique_id, ref_nb);

    float temp_denom_a = sqrt(sum_denom_a[query_index + ref_index*query_nb]);
    float temp_denom_b = sqrt(sum_denom_b[query_index + ref_index*query_nb]);

    if(query_index < query_nb && ref_index < ref_nb){

        //dist[query_index + ref_index*query_nb] = query_index + ref_index*query_nb;
        index[query_index + ref_index*query_nb] =  ref_index;
        dist[query_index + ref_index*query_nb] = sum_dots[query_index + ref_index*query_nb] / (temp_denom_a * temp_denom_b );
               
    }  
}


/**
 * Insertion sort
*/
__global__ void insertion_sort_gpu( const int     ref_nb,
                                    const int     query_nb,
                                    const int     dim,
                                    const int     k,
                                    float *       knn_dist,
                                    int *         knn_index,
                                    const int *   index, 
                                    const float * dist){

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