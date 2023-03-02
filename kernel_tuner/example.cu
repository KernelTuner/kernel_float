#include "kernel_float.h"

namespace kf = kernel_float;

template <size_t K>
__global__ void vector_add(int n, const kf::doubleX<K>* a, const kf::doubleX<K>* b, kf::doubleX<K>* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i * K < n) {
        output[i] = kf::cast<TYPE>(a[i]) + kf::cast<TYPE>(b[i]);
    }
}
