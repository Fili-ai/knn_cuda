#pragma once

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
    

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = unique_id / (ref_nb);
    int ref_index = unique_id % ref_nb;
    int d = unique_id % dim;
    
    if(query_index < query_nb && ref_index < ref_nb){
        int it = query_index*ref_nb*dim + ref_index*dim + d;
        dots[it] = ref[d * ref_nb + ref_index] * query[d * query_nb + query_index];
        denom_a[it] = ref[d * ref_nb + ref_index] * ref[d * ref_nb + ref_index];
        denom_b[unique_id] = query[d * query_nb + query_index] * query[d * query_nb + query_index] ;
    }  
}

// Reduction kernel to sum values along the dimension
__global__ void reduceDimension(const float* dots, const float* denom_a, const float* denom_b,
                                const int ref_nb, const int query_nb, const int dim,
                                float* sum_dots, float* sum_denom_a, float* sum_denom_b) {
    extern __shared__ float sharedMem[];

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;
    int query_index = unique_id / ref_nb;
    int ref_index = unique_id % ref_nb;

    if (query_index < query_nb && ref_index < ref_nb) {
        // Initialize shared memory
        sharedMem[threadIdx.x] = dots[unique_id];
        sharedMem[blockDim.x + threadIdx.x] = denom_a[unique_id];
        sharedMem[2 * blockDim.x + threadIdx.x] = denom_b[unique_id];

        // Perform parallel reduction along the dimension
        for (int stride = blockDim.x; stride > 0; stride /= 2) {
            __syncthreads();
            if (threadIdx.x < stride) {
                sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
                sharedMem[blockDim.x + threadIdx.x] += sharedMem[blockDim.x + threadIdx.x + stride];
                sharedMem[2 * blockDim.x + threadIdx.x] += sharedMem[2 * blockDim.x + threadIdx.x + stride];
            }
        }

        // Write the result to output arrays
        if (threadIdx.x == 0) {
            int outputIndex = query_index * ref_nb + ref_index;
            sum_dots[outputIndex] = sharedMem[0];
            sum_denom_a[outputIndex] = sharedMem[blockDim.x];
            sum_denom_b[outputIndex] = sharedMem[2 * blockDim.x];
        }
    }
}

__global__ void cosine_distance_gpu(const float * ref,
                                    const int     ref_nb,
                                    const float * query,
                                    const int     query_nb,
                                    const int     dim,
                                    float *       dist,
                                    int *         index, 
                                    float * dots, 
                                    float * denom_a, 
                                    float * denom_b) {
    

    int unique_id = blockIdx.x * blockDim.x + threadIdx.x;

    int query_index = unique_id / (ref_nb);
    int ref_index = unique_id % ref_nb;
    
    if(query_index < query_nb && ref_index < ref_nb){

        index[query_index + ref_index*query_nb] =  ref_index;
        dist[query_index + ref_index*query_nb] = dot / (sqrt(denom_a) * sqrt(denom_b));
        //dist[query_index + ref_index*query_nb] = unique_id;

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