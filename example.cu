#include "kernel_float.h"
#include <cuda_fp16.h>

namespace kf = kernel_float;

__global__ void kernel(
    kf::vec_ptr<half, 4, const __nv_fp8_e5m2> input, 
    float constant, 
    kf::vec_ptr<half, 4> output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output(i) = input[i] + kf::cast<half>(constant);
}
