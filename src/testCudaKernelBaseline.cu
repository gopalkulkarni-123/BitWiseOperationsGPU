#include <iostream>
#include <random>
#include <chrono>
#include <cuda.h>

__global__ void bitwiseAndKernel(const int* a, const int* b, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] & b[idx];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N_arr[6] = {10, 100, 1000, 10000, 100000, 1000000};  // 1 million elements

    for (int j = 0; j < 6; ++j){    
        float sum = 0.0f, avg = 0.0f;
        for (int iter = 0; iter < 11; ++iter){    

            size_t size = N_arr[j] * sizeof(int);

            // Host vectors
            int* h_a = new int[N_arr[j]];
            int* h_b = new int[N_arr[j]];
            int* h_result = new int[N_arr[j]];

            // Random bit generation
            std::mt19937 rng(std::random_device{}());
            std::uniform_int_distribution<int> bitDist(0, 1);
            for (int i = 0; i < N_arr[j]; ++i) {
                h_a[i] = bitDist(rng);
                h_b[i] = bitDist(rng);
            }

            // Device memory allocation
            int *d_a, *d_b, *d_result;
            checkCudaError(cudaMalloc(&d_a, size), "Allocating d_a");
            checkCudaError(cudaMalloc(&d_b, size), "Allocating d_b");
            checkCudaError(cudaMalloc(&d_result, size), "Allocating d_result");

            // Copy to device
            checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Copying h_a to d_a");
            checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Copying h_b to d_b");

            // Kernel launch configuration
            int threadsPerBlock = 256;
            int blocksPerGrid = (N_arr[j] + threadsPerBlock - 1) / threadsPerBlock;

            // Time kernel execution
            auto start = std::chrono::high_resolution_clock::now();

            bitwiseAndKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N_arr[j]);

            checkCudaError(cudaGetLastError(), "Kernel launch");
            checkCudaError(cudaDeviceSynchronize(), "Kernel sync");

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;

            //std::cout << "Kernel execution time: " << duration.count() << " ms\n";
            if (iter != 0){
                sum += duration.count();
            }

            // Copy result back
            checkCudaError(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost), "Copying d_result to h_result");

            // Optional: Print first 10 results
            
            /*std::cout << "First 10 AND results: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << h_result[i] << " ";
            }
            std::cout << "\n";*/

            // Cleanup
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_result);
            delete[] h_a;
            delete[] h_b;
            delete[] h_result;
        }
        avg = sum/10;
        std::cout << N_arr[j] << "," << avg << "ms \n";               
    }

    return 0;
}
