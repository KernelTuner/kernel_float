#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// A scalar (unaligned / size-1) store must never emit a vectorized instruction.
// CHECK: st.global.f32
__global__ void scalar_store(const float *input, float *output) {
    kf::write(output, kf::read<1>(input));
}

// A plain `write` (not `write_aligned`) stores each element individually and must not be
// vectorized, even when writing multiple contiguous elements.
// CHECK-COUNT-2: st.global.f32
__global__ void unaligned_store(const float *input, float *output) {
    kf::write(output, kf::read_aligned<2>(input));
}

// An 8-byte aligned store of 2 floats should be emitted as a single vectorized `v2.f32` store.
// CHECK: st.global.v2.f32
__global__ void vector2_store(const float *input, float *output) {
    kf::write_aligned<2>(output, kf::read_aligned<2>(input));
}

// A 16-byte aligned store of 4 floats should be emitted as a single vectorized `v4.f32` store.
// CHECK: st.global.v4.f32
__global__ void vector4_store(const float *input, float *output) {
    kf::write_aligned<4>(output, kf::read_aligned<4>(input));
}

// A store of 8 floats exceeds the widest PTX vector store (v4), so it should be split into
// two vectorized `v4.f32` stores rather than falling back to 8 scalar stores.
// CHECK-COUNT-2: st.global.v4.f32
__global__ void vector8_store(const float *input, float *output) {
    kf::write_aligned<8>(output, kf::read_aligned<8>(input));
}

// `cache_all` on a 2-float (8-byte) store lowers to `__stwb` on a single 64-bit word, i.e.
// a `wb.u64` store, not `ca.u64` (the load-side mnemonic).
// CHECK: st.global.wb.u64
__global__ void vector2_store_ca(const float *input, float *output) {
    kf::write_aligned<2, kf::cache_modifier::cache_all>(output, kf::read_aligned<2>(input));
}

// `cache_global` on a 2-float (8-byte) store lowers to `__stcg` on a single 64-bit word.
// CHECK: st.global.cg.u64
__global__ void vector2_store_cg(const float *input, float *output) {
    kf::write_aligned<2, kf::cache_modifier::cache_global>(output, kf::read_aligned<2>(input));
}

// `streaming` on a 2-float (8-byte) store lowers to `__stcs` on a single 64-bit word.
// CHECK: st.global.cs.u64
__global__ void vector2_store_cs(const float *input, float *output) {
    kf::write_aligned<2, kf::cache_modifier::streaming>(output, kf::read_aligned<2>(input));
}

// `uncached` on a 2-float (8-byte) store lowers to `__stwt` on a single 64-bit word, i.e.
// a `wt.u64` store, not `cv.u64` (the load-side mnemonic).
// CHECK: st.global.wt.u64
__global__ void vector2_store_cv(const float *input, float *output) {
    kf::write_aligned<2, kf::cache_modifier::uncached>(output, kf::read_aligned<2>(input));
}

// `cache_all` on a 4-float (16-byte) store lowers to `__stwb` on a pair of 64-bit words, i.e.
// a `wb.v2.u64` store, not `ca.v2.u64` (the load-side mnemonic).
// CHECK: st.global.wb.v2.u64
__global__ void vector4_store_ca(const float *input, float *output) {
    kf::write_aligned<4, kf::cache_modifier::cache_all>(output, kf::read_aligned<4>(input));
}

// `cache_global` on a 4-float (16-byte) store lowers to `__stcg` on a pair of 64-bit words.
// CHECK: st.global.cg.v2.u64
__global__ void vector4_store_cg(const float *input, float *output) {
    kf::write_aligned<4, kf::cache_modifier::cache_global>(output, kf::read_aligned<4>(input));
}

// `streaming` on a 4-float (16-byte) store lowers to `__stcs` on a pair of 64-bit words.
// CHECK: st.global.cs.v2.u64
__global__ void vector4_store_cs(const float *input, float *output) {
    kf::write_aligned<4, kf::cache_modifier::streaming>(output, kf::read_aligned<4>(input));
}

// `uncached` on a 4-float (16-byte) store lowers to `__stwt` on a pair of 64-bit words, i.e.
// a `wt.v2.u64` store, not `cv.v2.u64` (the load-side mnemonic).
// CHECK: st.global.wt.v2.u64
__global__ void vector4_store_cv(const float *input, float *output) {
    kf::write_aligned<4, kf::cache_modifier::uncached>(output, kf::read_aligned<4>(input));
}

}
