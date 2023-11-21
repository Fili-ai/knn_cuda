#pragma once

__device__ void bubbleSort(int* indexes, float* distances, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                // Swap distances
                float tempDist = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = tempDist;

                // Swap indexes accordingly
                int tempIndex = indexes[j];
                indexes[j] = indexes[j + 1];
                indexes[j + 1] = tempIndex;
            }
        }
    }
}


/**
* Insertion sort given with data coaliscent 
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
    
    
    for(int i = 0; i < length; ++i){
        if(i < k){
            index_sorted[i] = index[query_index + i * query_nb];
            dist_sorted[i] = dist[query_index + i * query_nb];
            if(i > 0)
                bubbleSort(index_sorted, dist_sorted, i+1);
        }
        
        else{
            if(dist[query_index + i * query_nb] < dist_sorted[k-1]){ 
                
                // add the more fitting point
                index_sorted[k] = index[query_index + i * query_nb];
                dist_sorted[k] = dist[query_index + i * query_nb];

                //sort the new array
                bubbleSort(index_sorted, dist_sorted, k + 1);                
            }
        }       
    }

    for(int i = 0; i < k; ++i){
        knn_index[query_index*k + i] = index_sorted[i];
        knn_dist[query_index*k + i] = dist_sorted[i];
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
        insertion_sort_gpu(dist, index, ref_nb, k, query_nb, query_index, knn_index, knn_dist);
                
    }
}