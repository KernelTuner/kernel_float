#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// CHECK: ld.global.u16
// CHECK: st.global.u16
__global__ void half1_copy(const kf::half1 *input, kf::half1 *output) {
    *output = *input;
}

// CHECK: ld.global.u32
// CHECK: st.global.u32
__global__ void half2_copy(const kf::half2 *input, kf::half2 *output) {
    *output = *input;
}

// CHECK: ld.global.v4.u16
// CHECK: st.global.v4.u16
__global__ void half4_copy(const kf::half4 *input, kf::half4 *output) {
    *output = *input;
}

// CHECK: ld.global.v4.u32
// CHECK: st.global.v4.u32
__global__ void half8_copy(const kf::half8 *input, kf::half8 *output) {
    *output = *input;
}

// half4 holds 2 packed half2 pairs, so each op below lowers to the instruction twice (once
// per pair).
// CHECK-COUNT-2: add.f16x2
__global__ void half_add(const kf::half4 *input, kf::half4 *output) {
    *output = *input + *input;
}

// CHECK-COUNT-2: sub.f16x2
__global__ void half_sub(const kf::half4 *input, kf::half4 *output) {
    *output = *input - *input;
}

// CHECK-COUNT-2: mul.f16x2
__global__ void half_mul(const kf::half4 *input, kf::half4 *output) {
    *output = *input * *input;
}

// CHECK-COUNT-2: fma.rn.f16x2
__global__ void half_fma(const kf::half4 *input, kf::half4 *output) {
    *output = kf::fma(*input, *input, *input);
}

// CHECK-COUNT-2: neg.f16x2
__global__ void half_neg(const kf::half4 *input, kf::half4 *output) {
    *output = -*input;
}

// CHECK-COUNT-2: min.f16x2
__global__ void half_min(const kf::half4 *input, kf::half4 *output) {
    *output = kf::min(*input, *input);
}

// CHECK-COUNT-2: max.f16x2
__global__ void half_max(const kf::half4 *input, kf::half4 *output) {
    *output = kf::max(*input, *input);
}

// Comparisons on half2 pairs are a genuine packed instruction, unlike `divide`/`sqrt` below.
// CHECK-COUNT-2: set.eq.f16x2.f16x2
__global__ void half_eq(const kf::half4 *input, bool *output) {
    output[0] = kf::all(*input == *input);
}

// `__h2div` is not a native packed instruction: it decomposes per lane into `cvt.f32.f16` +
// `rcp.approx.ftz.f32` plus a Newton-Raphson refinement step, not a single `div.f16x2`. With
// 4 lanes (2 per pair, x2 pairs) that's 2 `cvt.f32.f16` and 1 `rcp.approx.ftz.f32` per lane.
// CHECK-COUNT-8: cvt.f32.f16
// CHECK-COUNT-4: rcp.approx.ftz.f32
// CHECK-NOT: div.f16x2
__global__ void half_div(const kf::half4 *input, kf::half4 *output) {
    *output = *input / *input;
}

// CHECK-COUNT-4: ex2.approx.ftz.f32
__global__ void half_exp(const kf::half4 *input, kf::half4 *output) {
    *output = kf::exp(*input);
}

// `hsqrt`/`h2sqrt` are not native packed hardware either: they expand to per-lane
// `cvt.f32.f16` -> `sqrt.approx.ftz.f32` -> `cvt.rn.f16.f32`, not a single `sqrt.f16x2`.
// CHECK-COUNT-4: sqrt.approx.ftz.f32
// CHECK-NOT: sqrt.f16x2
__global__ void half_sqrt(const kf::half4 *input, kf::half4 *output) {
    *output = kf::sqrt(*input);
}


}