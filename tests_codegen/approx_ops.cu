#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// CHECK-NOT: approx.f32
// CHECK-COUNT-4: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sin1(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sin<1>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-6: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sin2(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sin<2>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-7: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sin3(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sin<3>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-8: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sin4(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sin<4>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-NOT: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rcp0(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rcp<0>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-2: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rcp1(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rcp<1>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-4: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rcp2(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rcp<2>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-6: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rcp3(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rcp<3>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-8: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rcp4(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rcp<4>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-1: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sqrt0(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sqrt<0>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-4: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sqrt1(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sqrt<1>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-9: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sqrt2(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sqrt<2>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-13: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sqrt3(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sqrt<3>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-17: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_sqrt4(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_sqrt<4>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-NOT: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rsqrt0(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rsqrt<0>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-4: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rsqrt1(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rsqrt<1>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-8: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rsqrt2(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rsqrt<2>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-12: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rsqrt3(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rsqrt<3>(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-16: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_rsqrt4(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_rsqrt<4>(*x);
}


// CHECK-NOT: approx.f32
// CHECK-COUNT-1: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_exp(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_exp(*x);
}

// CHECK-NOT: approx.f32
// CHECK-COUNT-1: (fma|mul|add)(.rn|.ftz|.sat)*.f16x2
__global__ void approx_half_log(kf::half2 *x, kf::half2* y) {
    *y = kf::approx_log(*x);
}

}