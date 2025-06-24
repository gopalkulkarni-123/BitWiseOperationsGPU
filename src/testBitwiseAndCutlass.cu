#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <iostream>
#include <random>
#include <chrono>

int main() {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;

    // Vector size
    int M_arr[5] = {10, 100, 1000, 10000}; // 6x6 flattened

    std::cout << "vector size,iteration,time \n";
    float sum = 0.0f, avg = 0.0f;
    for (int j = 0; j < 4; ++j){ //
        for (int iter = 0; iter < 11; ++iter){
            // A: (M x 1), B: (1 x M), C: (M x M)
            cutlass::HostTensor<ElementA, cutlass::layout::ColumnMajor> A({M_arr[j], 1});
            cutlass::HostTensor<ElementB, cutlass::layout::RowMajor> B({1, M_arr[j]});
            cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> C({M_arr[j], M_arr[j]});

            // Initialize A and B with 0 or 1
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 1);

            for (int i = 0; i < M_arr[j]; ++i) {
                A.at({i, 0}) = static_cast<ElementA>(dist(gen));
                B.at({0, i}) = static_cast<ElementB>(dist(gen));
            }

            // Sync to device
            A.sync_device();
            B.sync_device();
            C.sync_device();

            // Define GEMM configuration: A (Mx1) * B (1xM) = C (MxM)
            using Gemm = cutlass::gemm::device::Gemm<
                ElementA, cutlass::layout::ColumnMajor,
                ElementB, cutlass::layout::RowMajor,
                ElementC, cutlass::layout::RowMajor
            >;

            typename Gemm::Arguments args(
                {M_arr[j], M_arr[j], 1},
                {A.device_data(), A.stride(0)},
                {B.device_data(), B.stride(0)},
                {C.device_data(), C.stride(0)},
                {C.device_data(), C.stride(0)},
                {1, 0}
            );

            Gemm gemm_op;
            auto start = std::chrono::high_resolution_clock::now();

            cutlass::Status status = gemm_op(args);
            
            //auto end = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();  // ensure the operation has completed

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            if (status != cutlass::Status::kSuccess) {
                std::cerr << "GEMM failed\n";
                return -1;
            }

            std::cout << "GEMM time: " << M_arr[j] << " , " << elapsed.count() << " ms\n";
            if(iter != 0){
            
            sum += elapsed.count();
            //std::cout << M_arr[j]<< ","  << iter << "," << elapsed.count() << "ms\n";
            }
        }
        avg = sum/10;
        std::cout << M_arr[j]<< ","  << avg << "ms\n";
        avg = 0;
        sum = 0;
    }
    
    /*C.sync_host();

    // Extract and print diagonal (emulates element-wise multiplication)
    
    A.sync_host();
    B.sync_host();

    std::cout << "A_flat = {";
    for (int i = 0; i < M; ++i) {
        std::cout << static_cast<int>(A.at({i, 0})) << (i < M - 1 ? ", " : "");
    }
    std::cout << "};\n";

    std::cout << "B_flat = {";
    for (int i = 0; i < M; ++i) {
        std::cout << static_cast<int>(B.at({0, i})) << (i < M - 1 ? ", " : "");
    }
    std::cout << "};\n";
    

    std::cout << "Element-wise A[i] * B[i] = {";
    for (int i = 0; i < M; ++i) {
        std::cout << C.at({i, i}) << (i < M - 1 ? ", " : "");
    }
    std::cout << "};\n";

    std::cout << "PASS\n";*/
    return 0;
}
