#include "kernel_float.h"
namespace kf = kernel_float;

__global__ void vector_add(kf::vec<float_type, 1>* c, const kf::vec<float_type, 1>* a, const kf::vec<float_type, 1>* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
