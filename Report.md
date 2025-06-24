# Benchmarking of Bitwise operations on GPU
## Introduction
This report briefly and precisely explains the benchmarking of bit-wise operations, such as AND & OR operations. The project aimed to explore ways to speed up bitwise operations by leveraging GPUs. Two such techniques are explored and benchmarked with the basic CUDA kernel that performs the same operations. At the outset, some relevant topics such as tensor cores, bit packing and CUDA kernel are explained.

## Tensor cores
Tensor Cores are specialised hardware units integrated into NVIDIA GPUs, designed to accelerate matrix operations that are fundamental to a wide range of high-performance computing tasks, particularly in deep learning and scientific simulations. Introduced with the Volta architecture and enhanced in subsequent generations, these cores perform fused multiply-add operations on small matrix tiles using mixed precision arithmetic, enabling significantly higher throughput compared to traditional CUDA cores. By offloading dense linear algebra computations to Tensor Cores, applications can achieve substantial performance gains, especially in workloads involving large-scale matrix multiplications. Their integration into modern GPU architectures has made them a critical component for accelerating AI models, data-intensive simulations, and other compute-heavy applications.

![Tensor core](https://github.com/gopalkulkarni-123/BitWiseOperationsGPU/blob/master/Images/tesnor_core_diagram.png)


