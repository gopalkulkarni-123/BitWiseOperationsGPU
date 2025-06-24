# Benchmarking of Bitwise operations on GPU
## Introduction
This report briefly and precisely explains the benchmarking of bit-wise operations, such as AND & OR operations. The project aimed to explore ways to speed up bitwise operations by leveraging GPUs. Two such techniques are explored and benchmarked with the basic CUDA kernel that performs the same operations. At the outset, some relevant topics such as tensor cores, bit packing and CUDA kernel are explained.

### Tensor cores
Tensor Cores are specialised hardware units integrated into NVIDIA GPUs, designed to accelerate matrix operations that are fundamental to a wide range of high-performance computing tasks, particularly in deep learning and scientific simulations. Introduced with the Volta architecture and enhanced in subsequent generations, these cores perform fused multiply-add operations on small matrix tiles using mixed precision arithmetic, enabling significantly higher throughput compared to traditional CUDA cores. By offloading dense linear algebra computations to Tensor Cores, applications can achieve substantial performance gains, especially in workloads involving large-scale matrix multiplications. Their integration into modern GPU architectures has made them a critical component for accelerating AI models, data-intensive simulations, and other compute-heavy applications.

![Tensor core](https://github.com/gopalkulkarni-123/BitWiseOperationsGPU/blob/master/Images/tesnor_core_diagram.png)

Tensor Cores operate by accelerating the fused multiply-add (FMA) operation on small matrices, typically of fixed tile sizes such as 4 by 4, 8 by 8, or larger, depending on the GPU architecture. They take three matrices A, B, and C and compute D equals A multiplied by B plus C in a single instruction, significantly reducing latency and increasing throughput. These operations are executed using mixed precision, where inputs may be in lower precision formats like FP16, BF16, or INT8 to enhance speed and efficiency, while accumulation is typically performed in higher precision, such as FP32, to maintain numerical accuracy. By processing multiple matrix elements in parallel at the hardware level, Tensor Cores deliver substantial computational density, making them especially well suited for the matrix-heavy workloads common in AI, graphics, and scientific computing.

### CUTLASS
CUTLASS, which stands for CUDA Templates for Linear Algebra Subroutines, is an open source CUDA C++ template library developed by NVIDIA to help developers write high-performance matrix multiplication and related operations that make full use of Tensor Cores. It provides a flexible and highly configurable framework that abstracts the complexity of programming Tensor Cores directly, allowing users to define custom tile sizes, memory layouts, data types, and kernel behaviours. By organising computations into hierarchical building blocks such as thread blocks, warps, and instruction tiles, CUTLASS efficiently maps matrix operations onto the GPU hardware. It supports a wide range of precision formats, including FP32, FP16, BF16, INT8, and more, and includes optimised kernels for both dense and batched matrix multiplications. CUTLASS is widely used as a foundation for performance-critical libraries like cuBLAS and serves as a powerful tool for researchers and developers looking to optimise linear algebra routines on NVIDIA GPUs.

### Bit Packing
Bit packing is a technique used to store multiple smaller data values within a single larger data type by tightly organising bits, thereby reducing memory usage and improving data transfer efficiency. Instead of allocating a full byte or word for each value, bit packing assigns only the minimum number of bits necessary to represent each element, fitting several values into a single integer or memory word. This is particularly useful when working with binary or low-precision data such as boolean flags, binary masks, or quantised values in machine learning and graphics. While bit packing can greatly reduce memory bandwidth and storage requirements, it often requires additional computation to extract and manipulate individual values, typically involving bitwise operations like shifts and masks. It is commonly used in applications where large volumes of small data need to be processed efficiently, including compression algorithms, cryptography, and GPU-accelerated computing.

## Method
To evaluate the effectiveness of Tensor Cores and bitwise operations on GPUs using bit-packed representations, a baseline CUDA kernel performing standard bitwise AND and OR operations is used as a reference. The experiments involve two input vectors with logarithmically increasing lengths, ranging from 10 to 100000 elements, on which element-wise bitwise operations are performed. The computation time is recorded and compared across different approaches. Since Tensor Cores are specifically optimized for matrix multiplication and addition, performing element-wise operations such as bitwise AND directly is not straightforward. To utilize Tensor Cores for this purpose, the CUTLASS library is employed. In this approach, the outer product of the two vectors is computed using matrix multiplication, and the diagonal elements of the resulting matrix are extracted to approximate the element-wise bitwise AND operation. The data type used for the matrix product in this method is `int8_t`, allowing compatibility with Tensor Core operations and bitwise logic.

![Outer product](https://github.com/gopalkulkarni-123/BitWiseOperationsGPU/blob/master/Images/Screenshot%20from%202025-06-24%2012-09-34.png)

## Results
The compute time is measured for all three cases for 10 iterations each and the average is calculated. 


![Results](https://github.com/gopalkulkarni-123/BitWiseOperationsGPU/blob/master/Images/Screenshot%20from%202025-06-24%2013-33-10.png)


| Vector Length     | CUTLASS (AND) | CUDA (AND) | Bit Packing (AND) | CUDA (OR) | Bit Packing (OR) |
|------------------:|---------------:|------------:|--------------------:|-----------:|-------------------:|
| 10 elements       | 0.0162 ms      | 0.0092 ms   | 0.0060 ms           | 0.0099 ms  | 0.0060 ms          |
| 100 elements      | 0.0173 ms      | 0.0089 ms   | 0.0057 ms           | 0.0087 ms  | 0.0057 ms          |
| 1000 elements     | 0.0872 ms      | 0.0089 ms   | 0.0056 ms           | 0.0088 ms  | 0.0058 ms          |
| 10000 elements    | 0.6580 ms      | 0.0121 ms   | 0.0059 ms           | 0.0121 ms  | 0.0057 ms          |
| 100000 elements   | —              | 0.0457 ms   | 0.0234 ms           | 0.0497 ms  | 0.0238 ms          |


## Conclusion
As observed from the results, Bit Packing is the most efficient method for performing both AND and OR operations on vectors of all tested sizes. It consistently outperforms both naive CUDA and CUTLASS implementations. While naive CUDA performs acceptably for small to medium vectors, it scales less effectively than Bit Packing.

CUTLASS, designed for matrix-heavy workloads, is ill-suited for element-wise operations and should be avoided for such tasks unless batch-level tensor operations are required. Its performance is significantly worse primarily because it computes a full matrix multiplication, including non-diagonal elements, which are discarded in the final result. This leads to substantial unnecessary computation, especially for large vectors, where only the diagonal elements (i.e., pairwise AND or OR results) are actually needed.
