#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//#include "Solution1.cu"
#include "Solution2.cu"
//#include "Solution3.cu"
//#include "Solution4.cu"

/**
* Error checking function;
*/
#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: line %d - %s\n", line, cudaGetErrorString(code));
        if (abort)
            exit(code);
    }
}

bool your_solution(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index) {


    /**
    * @param ref reference points
    * @param ref_nb number of reference points
    * @param query the data to process and classify
    * @param query_nb number of query
    * @param dim dimension of every single point
    * @param k number of neighbors
    * @param knn_dist array to save the distances between every query and the reference point
    * @param knn_index array with the solution of the classification
    */

    
    int chunk = query_nb;
    bool use_chunk = false;

    //10240 is maximum chunk size to avoid memory allocation error
    if(chunk > 10240){
        chunk = 10240;
        use_chunk = true;
    }

    // ---------------------------------- Creating data location on gpu -------------------------------
    // Location for all reference data
    float * ref_gpu;
    gpuErrchk(cudaMalloc(&ref_gpu, ref_nb*dim*sizeof(float)));

    // Location for all query data
    float * query_gpu;
    gpuErrchk(cudaMallocManaged(&query_gpu, chunk*dim*sizeof(float)));

    // Location for the k-nearest distances
    float * knn_dist_gpu;
    gpuErrchk(cudaMallocManaged(&knn_dist_gpu, chunk*k*sizeof(float)));

    // Location for the k-nearest index
    int * knn_index_gpu;
    gpuErrchk(cudaMallocManaged(&knn_index_gpu, chunk*k*sizeof(int)));

    // Location for index and dist 
    int * index_gpu;
    gpuErrchk(cudaMallocManaged(&index_gpu, chunk*ref_nb*sizeof(int)));
    float * dist_gpu;
    gpuErrchk(cudaMallocManaged(&dist_gpu, chunk*ref_nb*sizeof(float)));


    for(int iter = 0; iter < query_nb; iter += chunk){          

        // ---------------------------------- Transfering data on device -------------------------------
        gpuErrchk(cudaMemcpy(ref_gpu, ref, ref_nb*dim*sizeof(float), cudaMemcpyHostToDevice));
        
        if(!use_chunk){
            gpuErrchk(cudaMemcpy(query_gpu, query, chunk*dim*sizeof(float), cudaMemcpyHostToDevice));
        }
        else{
            // Temporary location of queries of a specific chunk
            float * query_temp;
            gpuErrchk(cudaMallocHost(&query_temp, chunk*dim*sizeof(float))); 

            for(int d = 0; d < dim; ++d){
                for(int query_idx = 0; query_idx < chunk; ++query_idx){
                    query_temp[query_idx + d*chunk] = query[query_idx + iter + d*query_nb];
                }
            }

            gpuErrchk(cudaMemcpy(query_gpu, query_temp, chunk*dim*sizeof(float), cudaMemcpyHostToDevice));

            cudaFree(query_temp);
        }

        cudaDeviceSynchronize();
        // ---------------------------------- Kernel launching -------------------------------
        
        // Solution - 1
        //knn_gpu_1_Block_Grid<<<1, 1>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu);

        // Solution - 2
        
        dim3 block_size(32, 1, 1);
        dim3 grid((chunk + block_size.x - 1) / block_size.x, 1, 1);
        knn_gpu<<<grid, block_size>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu, chunk);
        

        // Solution - 3
        /*
        dim3 block_size(32, 1, 1);
        dim3 grid((chunk + block_size.x - 1) / block_size.x, 1, 1);
        dim3 block_size_cosine_distance(32, 1, 1);
        dim3 grid_cosine_distance((ref_nb*chunk + block_size.x - 1) / block_size.x, 1, 1);
        cosine_distance_gpu<<<grid_cosine_distance, block_size_cosine_distance>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, dist_gpu, index_gpu, chunk);
        insertion_sort_gpu<<<grid, block_size>>>(ref_nb, chunk, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
        */

        // Solution - 4
        /*

        // memory management 
        float * dots;
        gpuErrchk(cudaMallocManaged(&dots, query_nb*ref_nb*dim*sizeof(float)));
        float * denom_a;
        gpuErrchk(cudaMallocManaged(&denom_a, query_nb*ref_nb*dim*sizeof(float)));
        float * denom_b;
        gpuErrchk(cudaMallocManaged(&denom_b, query_nb*ref_nb*dim*sizeof(float)));
        float * sum_dots;
        gpuErrchk(cudaMallocManaged(&sum_dots, query_nb*ref_nb*sizeof(float)));
        float * sum_denom_a;
        gpuErrchk(cudaMallocManaged(&sum_denom_a, query_nb*ref_nb*sizeof(float)));
        float * sum_denom_b;
        gpuErrchk(cudaMallocManaged(&sum_denom_b, query_nb*ref_nb*sizeof(float)));

        // block and grid dimension 
        dim3 block_size(1024, 1, 1);
        dim3 grid((query_nb + block_size.x - 1) / block_size.x, 1, 1);
        dim3 block_size_fill(1024, 1, 1);
        dim3 grid_fill((ref_nb*query_nb*dim + block_size.x - 1) / block_size.x, 1, 1);
        dim3 block_size_reduction(1024, 1, 1);
        dim3 grid_reduction((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1); 
        dim3 block_size_cosine_distance(1024, 1, 1);
        dim3 grid_cosine_distance((ref_nb*query_nb + block_size.x - 1) / block_size.x, 1, 1);

        // kernel launching
        fill_gpu<<<grid_fill, block_size_fill>>>(ref_gpu, ref_nb, query_gpu, query_nb, dim, dots, denom_a, denom_b);
        //reduceDimension<<<grid_reduction, block_size_reduction>>>(dots, denom_a, denom_b, ref_nb, query_nb, dim, sum_dots, sum_denom_a, sum_denom_b); 
        reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(dots, sum_dots, query_nb, ref_nb, dim);
        reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(denom_a, sum_denom_a, query_nb, ref_nb, dim);
        reduce0<<<grid_fill, block_size_fill, block_size.x * sizeof(float)>>>(denom_b, sum_denom_b, query_nb, ref_nb, dim);
        cudaFree(dots);
        cudaFree(denom_a);
        cudaFree(denom_b);
        cosine_distance_gpu<<<grid_cosine_distance, block_size_cosine_distance>>>(ref_nb, query_nb, dist_gpu, index_gpu, sum_dots, sum_denom_a, sum_denom_b);
        //cudaFree(sum_dots);
        //cudaFree(sum_denom_a);
        //cudaFree(sum_denom_b);
        insertion_sort_gpu<<<grid, block_size>>>(ref_nb, query_nb, dim, k, knn_dist_gpu, knn_index_gpu, index_gpu, dist_gpu);
        */

        // ---------------------------------- Transfering data on host -------------------------------
        
        if(!use_chunk){
            gpuErrchk(cudaMemcpy(knn_dist, knn_dist_gpu, query_nb*k*sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(knn_index, knn_index_gpu, query_nb*k*sizeof(int), cudaMemcpyDeviceToHost));
        }
        else{
            // Temporary location of knn_index and knn_distance of a specific chunk
            float * knn_dist_host;
            gpuErrchk(cudaMallocHost(&knn_dist_host, chunk*k*sizeof(float)));
            int * knn_index_host;
            gpuErrchk(cudaMallocHost(&knn_index_host, chunk*k*sizeof(int))); 

            gpuErrchk(cudaMemcpy(knn_dist_host, knn_dist_gpu, chunk*k*sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(knn_index_host, knn_index_gpu, chunk*k*sizeof(int), cudaMemcpyDeviceToHost));
            
            for(int k_elems = 0; k_elems < k; ++k_elems){
                for(int idx = 0; idx < chunk; ++idx){
                    knn_dist[idx + k_elems*query_nb + iter] = knn_dist_host[idx + k_elems*chunk];
                    knn_index[idx + k_elems*query_nb + iter] = knn_index_host[idx + k_elems*chunk];
                }
            }
            cudaFree(knn_dist_host);
            cudaFree(knn_index_host);
        }
    }

    // ---------------------------------- Free memory -------------------------------

    cudaFree(ref_gpu);
    cudaFree(query_gpu);
    cudaFree(knn_dist_gpu);
    cudaFree(knn_index_gpu);

    cudaFree(index_gpu);
    cudaFree(dist_gpu);

    return true;
}