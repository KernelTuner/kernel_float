#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// A scalar (unaligned / size-1) load must never emit a vectorized instruction.
// CHECK: ld.global.f32
__global__ void scalar_load(const float *input, float *output) {
    output[0] = kf::read<1>(input);
}

// read without vectorization
// CHECK-COUNT-2: ld.global.f32
__global__ void unaligned_load(const float *input, float *output) {
    output[0] = kf::sum(kf::read<2>(input));
}

// aligned read of 2 elements, vectorized as a single v2 load
// CHECK: ld.global.v2.f32
__global__ void vector2_load(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<2>(input));
}

// aligned read of 4 elements, vectorized as a single v4 load
// CHECK: ld.global.v4.f32
__global__ void vector4_load(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<4>(input));
}

// aligned read of 8 elements; PTX has no wider vector load than v4, so this
// is vectorized as two separate v4 loads
// CHECK-COUNT-2: ld.global.v4.f32
__global__ void vector8_load(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<8>(input));
}

// aligned read of 2 elements with a cache modifier: there is no native .vN
// form for cache-modified loads, so both elements are packed into a single
// scalar 64-bit load instead
// CHECK: ld.global.ca.u64
__global__ void vector2_load_ca(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<2, kf::cache_modifier::cache_all>(input));
}

// aligned read of 2 elements with a cache modifier, packed into a single
// scalar 64-bit load (no native .vN form for cache-modified loads)
// CHECK: ld.global.cg.u64
__global__ void vector2_load_cg(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<2, kf::cache_modifier::cache_global>(input));
}

// aligned read of 2 elements with a cache modifier, packed into a single
// scalar 64-bit load (no native .vN form for cache-modified loads)
// CHECK: ld.global.cs.u64
__global__ void vector2_load_cs(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<2, kf::cache_modifier::streaming>(input));
}

// aligned read of 2 elements with a cache modifier, packed into a single
// scalar 64-bit load (no native .vN form for cache-modified loads)
// CHECK: ld.global.cv.u64
__global__ void vector2_load_cv(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<2, kf::cache_modifier::uncached>(input));
}

// aligned read of 4 elements with a cache modifier, vectorized as a v2.u64 load
// CHECK: ld.global.ca.v2.u64
__global__ void vector4_load_ca(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<4, kf::cache_modifier::cache_all>(input));
}

// aligned read of 4 elements with a cache modifier, vectorized as a v2.u64 load
// CHECK: ld.global.cg.v2.u64
__global__ void vector4_load_cg(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<4, kf::cache_modifier::cache_global>(input));
}

// aligned read of 4 elements with a cache modifier, vectorized as a v2.u64 load
// CHECK: ld.global.cs.v2.u64
__global__ void vector4_load_cs(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<4, kf::cache_modifier::streaming>(input));
}

// aligned read of 4 elements with a cache modifier, vectorized as a v2.u64 load
// CHECK: ld.global.cv.v2.u64
__global__ void vector4_load_cv(const float *input, float *output) {
    output[0] = kf::sum(kf::read_aligned<4, kf::cache_modifier::uncached>(input));
}


}