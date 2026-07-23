#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// CHECK: ld.global.u16
// CHECK: st.global.u16
__global__ void bfloat16x1_copy(const kf::bfloat16x1 *input, kf::bfloat16x1 *output) {
    *output = *input;
}

// CHECK: ld.global.u32
// CHECK: st.global.u32
__global__ void bfloat16x2_copy(const kf::bfloat16x2 *input, kf::bfloat16x2 *output) {
    *output = *input;
}

// CHECK: ld.global.v4.u16
// CHECK: st.global.v4.u16
__global__ void bfloat16x4_copy(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = *input;
}

// CHECK: ld.global.v4.u32
// CHECK: st.global.v4.u32
__global__ void bfloat16x8_copy(const kf::bfloat16x8 *input, kf::bfloat16x8 *output) {
    *output = *input;
}

// bfloat16x4 holds 2 packed bf16x2 pairs, so each op below lowers to the instruction twice
// (once per pair). Unlike half2, bf16x2 has no native packed `add` instruction on sm_80: it
// is emulated as `fma(x, 1.0, x)`, so this lowers to `fma.rn.bf16x2`, not `add.bf16x2`.
// CHECK-COUNT-2: fma.rn.bf16x2
// CHECK-NOT: add.bf16x2
__global__ void bfloat16_add(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = *input + *input;
}

// Same story as `add`: emulated as `fma(x, -1.0, x)`, lowering to `fma.rn.bf16x2` rather than
// a `sub.bf16x2` instruction.
// CHECK-COUNT-2: fma.rn.bf16x2
// CHECK-NOT: sub.bf16x2
__global__ void bfloat16_sub(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = *input - *input;
}

// Same story again: emulated as `fma(x, x, -0.0)`, lowering to `fma.rn.bf16x2` rather than a
// `mul.bf16x2` instruction.
// CHECK-COUNT-2: fma.rn.bf16x2
// CHECK-NOT: mul.bf16x2
__global__ void bfloat16_mul(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = *input * *input;
}

// This is a genuine, intentional packed FMA (not an add/sub/mul emulated via FMA like above).
// CHECK-COUNT-2: fma.rn.bf16x2
__global__ void bfloat16_fma(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = kf::fma(*input, *input, *input);
}

// CHECK-COUNT-2: neg.bf16x2
__global__ void bfloat16_neg(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = -*input;
}

// CHECK-COUNT-2: min.bf16x2
__global__ void bfloat16_min(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = kf::min(*input, *input);
}

// CHECK-COUNT-2: max.bf16x2
__global__ void bfloat16_max(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = kf::max(*input, *input);
}

// Unlike half2's clean `set.eq.f16x2.f16x2`, bf16x2 equality has no native packed comparison:
// it is emulated by splitting each of the 4 lanes out to `f32` and comparing with
// `set.eq.f32.f32`, one per lane.
// CHECK-COUNT-4: set.eq.f32.f32
__global__ void bfloat16_eq(const kf::bfloat16x4 *input, bool *output) {
    output[0] = kf::all(*input == *input);
}

// Division is not packed at all: each of the 4 lanes is extracted individually.
// CHECK-COUNT-4: div.(rn|approx).f32
// CHECK-NOT: div.bf16x2
__global__ void bfloat16_div(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = *input / *input;
}

// Like half's `sqrt`, this is not native packed hardware: it expands to per-lane
// `sqrt.approx.f32` (no `.ftz` here, unlike the half2 path) followed by `cvt.rn.bf16.f32`,
// one per lane.
// CHECK-COUNT-4: sqrt.approx.f32
// CHECK-NOT: sqrt.bf16x2
__global__ void bfloat16_sqrt(const kf::bfloat16x4 *input, kf::bfloat16x4 *output) {
    *output = kf::sqrt(*input);
}

}
