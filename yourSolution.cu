#include <stdio.h>
#include <iostream>
#include <cuda.h>

__global__ void fillArrayKernel(int *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = i;  // Filling the array with numbers from 0 to 10
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
    * @input ref reference points
    * @input ref_nb number of reference points
    * @input query the data to process and classify
    * @input query_nb number of query
    * @input dim dimension of every single point
    * @input k number of neighbors
    * @input knn_dist array to save the distances between every query 
             and the reference point
    * @input knn_index array with the solution of the classification
    */

    const int arraySize = 10;  // Change the array size as needed
    int hostArray[arraySize];   // Host array to store the result

    int *deviceArray;  // Device array

    cudaError_t cudaStatus;

    // Allocate memory on the GPU
    cudaStatus = cudaMalloc((void**)&deviceArray, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "\ncudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Launch the kernel with 1 grid and 1 block
    fillArrayKernel<<<1, 1>>>(deviceArray, arraySize);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "\nKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceArray);
        return 1;
    }
    // Synchronize the device to make sure the kernel has finished
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "\ncudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceArray);
        return 1;
    }

    // Copy the result back to the host
    cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "\ncudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceArray);
        return 1;
    }

    // Output the result
    std::cout << "\nResult: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory on the GPU
    cudaFree(deviceArray);

    return 0;

}

