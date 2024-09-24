#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "kernel_float.h"
namespace kf = kernel_float;

void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

template<int N>
__global__ void my_kernel(
    int length,
    kf::vec_ptr<const half, N> input,
    double constant,
    kf::vec_ptr<half, N, float> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * N < length) {
        output(i) = kf::fma(input[i], input[i], kf::cast<__half>(constant));
    }
}

template<int items_per_thread>
void run_kernel(int n) {
    double constant = 1.0;
    std::vector<half> input(n);
    std::vector<float> output_expected;
    std::vector<float> output_result;

    // Generate input data
    for (int i = 0; i < n; i++) {
        input[i] = half(i);
        output_expected[i] = float(i + constant);
    }

    // Allocate device memory
    __half* input_dev;
    float* output_dev;
    cuda_check(cudaMalloc(&input_dev, sizeof(half) * n));
    cuda_check(cudaMalloc(&output_dev, sizeof(float) * n));

    // Copy device memory
    cuda_check(cudaMemcpy(input_dev, input.data(), sizeof(half) * n, cudaMemcpyDefault));

    // Launch kernel!
    int block_size = 256;
    int items_per_block = block_size * items_per_thread;
    int grid_size = (n + items_per_block - 1) / items_per_block;
    my_kernel<items_per_thread><<<grid_size, block_size>>>(
        n,
        kf::assert_aligned(input_dev),
        constant,
        kf::assert_aligned(output_dev));

    // Copy results back
    cuda_check(cudaMemcpy(output_dev, output_result.data(), sizeof(float) * n, cudaMemcpyDefault));

    // Check results
    for (int i = 0; i < n; i++) {
        float result = output_result[i];
        float answer = output_expected[i];

        if (result != answer) {
            std::stringstream msg;
            msg << "error: index " << i << " is incorrect: " << result << " != " << answer;
            throw std::runtime_error(msg.str());
        }
    }

    cuda_check(cudaFree(input_dev));
    cuda_check(cudaFree(output_dev));
}

int main() {
    int n = 84000;  // divisible by 1, 2, 3, 4, 5, 6, 7, 8
    cuda_check(cudaSetDevice(0));

    run_kernel<1>(n);
    run_kernel<2>(n);
    //    run_kernel<3>(n);
    run_kernel<4>(n);
    run_kernel<8>(n);

    std::cout << "result correct\n";
    return EXIT_SUCCESS;
}
