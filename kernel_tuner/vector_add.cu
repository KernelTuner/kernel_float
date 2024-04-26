#include "kernel_float.h"
namespace kf = kernel_float;

__global__ void vector_add(
        kf::vec<float_type, elements_per_thread>* c,
        const kf::vec<float_type, elements_per_thread>* a,
        const kf::vec<float_type, elements_per_thread>* b,
        int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * elements_per_thread < n) {
        c[i] = a[i] + b[i];
    }
}
