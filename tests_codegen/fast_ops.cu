#include "kernel_float.h"
namespace kf = kernel_float;

extern "C" {

// CHECK: div.approx.f32
__global__ void fast_math_div(const float *input, float *output) {
    *output = kf::divide<kf::fast_policy>(*input, *input);
}

// CHECK: rcp.approx.f32
__global__ void fast_math_rcp(const float *input, float *output) {
    *output = kf::rcp<kf::fast_policy>(*input);
}

// CHECK: rcp.approx.ftz.f64
__global__ void fast_math_rcp_f64(const double *input, double *output) {
    *output = kf::rcp<kf::fast_policy>(*input);
}

// CHECK: sin.approx.f32
__global__ void fast_math_sin(const float *input, float *output) {
    *output = kf::sin<kf::fast_policy>(*input);
}

// CHECK: cos.approx.f32
__global__ void fast_math_cos(const float *input, float *output) {
    *output = kf::cos<kf::fast_policy>(*input);
}

// CHECK: cos.approx.f32
// CHECK: sin.approx.f32
// CHECK: div.approx.f32
__global__ void fast_math_tan(const float *input, float *output) {
    *output = kf::tan<kf::fast_policy>(*input);
}

// CHECK: tanh.approx.f32
__global__ void fast_math_tanh(const float *input, float *output) {
    *output = kf::tanh<kf::fast_policy>(*input);
}

// CHECK: sqrt.approx.f32
__global__ void fast_math_sqrt(const float *input, float *output) {
    *output = kf::sqrt<kf::fast_policy>(*input);
}

// CHECK: rsqrt.approx.f32
__global__ void fast_math_rsqrt(const float *input, float *output) {
    *output = kf::rsqrt<kf::fast_policy>(*input);
}

// CHECK: rsqrt.approx.ftz.f64
__global__ void fast_math_rsqrt_f64(const double *input, double *output) {
    *output = kf::rsqrt<kf::fast_policy>(*input);
}

// CHECK: ex2.approx.f32
__global__ void fast_math_exp(const float *input, float *output) {
    *output = kf::exp<kf::fast_policy>(*input);
}

// CHECK: ex2.approx.f32
__global__ void fast_math_exp2(const float *input, float *output) {
    *output = kf::exp2<kf::fast_policy>(*input);
}

// CHECK: ex2.approx.f32
__global__ void fast_math_exp10(const float *input, float *output) {
    *output = kf::exp10<kf::fast_policy>(*input);
}

// CHECK: lg2.approx.f32
__global__ void fast_math_log(const float *input, float *output) {
    *output = kf::log<kf::fast_policy>(*input);
}

// CHECK: lg2.approx.f32
__global__ void fast_math_log2(const float *input, float *output) {
    *output = kf::log2<kf::fast_policy>(*input);
}

// CHECK: lg2.approx.f32
__global__ void fast_math_log10(const float *input, float *output) {
    *output = kf::log10<kf::fast_policy>(*input);
}

// CHECK-NOT: div.approx.f32
__global__ void accurate_math_div(const float *input, float *output) {
    *output = kf::divide<kf::accurate_policy>(*input, *input);
}

// CHECK-NOT: rcp.approx.f32
__global__ void accurate_math_rcp(const float *input, float *output) {
    *output = kf::rcp<kf::accurate_policy>(*input);
}

// CHECK-NOT: rcp.approx.ftz.f64
__global__ void accurate_math_rcp_f64(const double *input, double *output) {
    *output = kf::rcp<kf::accurate_policy>(*input);
}

// CHECK-NOT: sin.approx.f32
__global__ void accurate_math_sin(const float *input, float *output) {
    *output = kf::sin<kf::accurate_policy>(*input);
}

// CHECK-NOT: cos.approx.f32
__global__ void accurate_math_cos(const float *input, float *output) {
    *output = kf::cos<kf::accurate_policy>(*input);
}

// CHECK-NOT: cos.approx.f32
// CHECK-NOT: sin.approx.f32
// CHECK-NOT: div.approx.f32
__global__ void accurate_math_tan(const float *input, float *output) {
    *output = kf::tan<kf::accurate_policy>(*input);
}

// CHECK-NOT: tanh.approx.f32
__global__ void accurate_math_tanh(const float *input, float *output) {
    *output = kf::tanh<kf::accurate_policy>(*input);
}

// CHECK-NOT: sqrt.approx.f32
__global__ void accurate_math_sqrt(const float *input, float *output) {
    *output = kf::sqrt<kf::accurate_policy>(*input);
}

// This is an exception, CUDA does emit an approx operation for rsqrt
// CHECK: rsqrt.approx.f32
__global__ void accurate_math_rsqrt(const float *input, float *output) {
    *output = kf::rsqrt<kf::accurate_policy>(*input);
}

// CHECK-NOT: rsqrt.approx.ftz.f64
__global__ void accurate_math_rsqrt_f64(const double *input, double *output) {
    *output = kf::rsqrt<kf::accurate_policy>(*input);
}

// CHECK-NOT: ex2.approx.f32
__global__ void accurate_math_exp(const float *input, float *output) {
    *output = kf::exp<kf::accurate_policy>(*input);
}

// This is an exception, CUDA does emit an approx operation for exp2
// CHECK: ex2.approx.f32
__global__ void accurate_math_exp2(const float *input, float *output) {
    *output = kf::exp2<kf::accurate_policy>(*input);
}

// CHECK-NOT: ex2.approx.f32
__global__ void accurate_math_exp10(const float *input, float *output) {
    *output = kf::exp10<kf::accurate_policy>(*input);
}

// CHECK-NOT: lg2.approx.f32
__global__ void accurate_math_log(const float *input, float *output) {
    *output = kf::log<kf::accurate_policy>(*input);
}

// CHECK-NOT: lg2.approx.f32
__global__ void accurate_math_log2(const float *input, float *output) {
    *output = kf::log2<kf::accurate_policy>(*input);
}

// CHECK-NOT: lg2.approx.f32
__global__ void accurate_math_log10(const float *input, float *output) {
    *output = kf::log10<kf::accurate_policy>(*input);
}

}