#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

// CUDA kernel for bitwise AND on bit-packed data
__global__ void bitwise_or_bool_kernel(
    const uint32_t* A, const uint32_t* B, uint32_t* C, int wordsPerRow, int height
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int word = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && word < wordsPerRow) {
        int idx = row * wordsPerRow + word;
        C[idx] = A[idx] || B[idx];  // OR: A[idx] | B[idx]
    }
}

// Packs a bool matrix into a uint32_t array
void packBoolMatrix(const char* in, uint32_t* out, int width, int height) {
    int wordsPerRow = (width + 31) / 32;
    for (int row = 0; row < height; ++row) {
        for (int w = 0; w < wordsPerRow; ++w) {
            uint32_t word = 0;
            for (int bit = 0; bit < 32; ++bit) {
                int col = w * 32 + bit;
                if (col < width && in[row * width + col]) {
                    word |= (1U << bit);
                }
            }
            out[row * wordsPerRow + w] = word;
        }
    }
}

// Unpacks a uint32_t array into a bool matrix
void unpackBoolMatrix(const uint32_t* in, char* out, int width, int height) {
    int wordsPerRow = (width + 31) / 32;
    for (int row = 0; row < height; ++row) {
        for (int w = 0; w < wordsPerRow; ++w) {
            uint32_t word = in[row * wordsPerRow + w];
            for (int bit = 0; bit < 32; ++bit) {
                int col = w * 32 + bit;
                if (col < width) {
                    out[row * width + col] = (word >> bit) & 1U;
                }
            }
        }
    }
}

// GPU execution pipeline with timing
float runBitwiseAndCUDA(const char* A_host, const char* B_host, char* C_host, int width, int height) {
    int wordsPerRow = (width + 31) / 32;
    int totalWords = wordsPerRow * height;

    std::vector<uint32_t> A_packed(totalWords);
    std::vector<uint32_t> B_packed(totalWords);
    std::vector<uint32_t> C_packed(totalWords);

    packBoolMatrix(A_host, A_packed.data(), width, height);
    packBoolMatrix(B_host, B_packed.data(), width, height);

    uint32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, totalWords * sizeof(uint32_t));
    cudaMalloc(&d_B, totalWords * sizeof(uint32_t));
    cudaMalloc(&d_C, totalWords * sizeof(uint32_t));

    cudaMemcpy(d_A, A_packed.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_packed.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((wordsPerRow + 15) / 16, (height + 15) / 16);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bitwise_or_bool_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, wordsPerRow, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C_packed.data(), d_C, totalWords * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    unpackBoolMatrix(C_packed.data(), C_host, width, height);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// Optional: for debugging small outputs
void printBoolMatrix(const char* mat, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (mat[i * width + j] ? "1 " : "0 ");
        }
        std::cout << "\n";
    }
}

int main() {
    const int width = 1;
    const int height[5] = {10, 100, 1000, 10000, 100000};
    float sum = 0.0f, avg = 0.0f;

    for (int size = 0; size < 5; ++size){

        std::vector<char> A(width * height[size]);
        std::vector<char> B(width * height[size]);
        std::vector<char> C(width * height[size], 0);

        // Fill inputs with random 0s and 1s
        for (int i = 0; i < width * height[size]; ++i) {
            A[i] = rand() % 2;
            B[i] = rand() % 2;
        }

        //std::cout << "Size of array is " << height[size] <<std::endl;

        for (int iter = 0; iter < 11; ++iter){

            auto cpu_start = std::chrono::high_resolution_clock::now();
            float gpu_time = runBitwiseAndCUDA(A.data(), B.data(), C.data(), width, height[size]);
            auto cpu_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> total_time = cpu_end - cpu_start;

            //std::cout << "\n--- Timing Results ---\n";
            //std::cout << gpu_time << " \n";
            //std::cout << "Total time (packing + kernel + unpacking): " << total_time.count() << " ms\n";

            if (iter !=0){
            sum += gpu_time;
            }
        }
        //std::cout << "----------------------------------" << std::endl;

        avg = sum /10;
        std::cout <<"Size of array: "<< height[size] <<  ", GPU kernel execution: " << avg << " ms\n";
        sum = 0.0f;
        }

    return 0;
}
