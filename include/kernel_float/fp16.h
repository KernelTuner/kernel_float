#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include "macros.h"

#if KERNEL_FLOAT_FP16_AVAILABLE
//#define CUDA_NO_HALF (1)
//#define __CUDA_NO_HALF_OPERATORS__ (1)
//#define __CUDA_NO_HALF2_OPERATORS__ (1)
//#define __CUDA_NO_HALF_CONVERSIONS__ (1)

#if KERNEL_FLOAT_IS_CUDA
#include <cuda_fp16.h>
#elif KERNEL_FLOAT_IS_HIP
#include <hip/hip_fp16.h>
#endif

#include "vector.h"

namespace kernel_float {

template<>
struct preferred_vector_size<__half> {
    static constexpr size_t value = 2;
};

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__half)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __half)

template<>
struct into_vector_impl<__half2> {
    using value_type = __half;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__half, 2> call(__half2 input) {
        return {input.x, input.y};
    }
};

namespace detail {
template<>
struct allow_float_fallback<__half> {
    static constexpr bool value = true;
};
};  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                                              \
    namespace ops {                                                                                \
    template<>                                                                                     \
    struct NAME<__half> {                                                                          \
        KERNEL_FLOAT_INLINE __half operator()(__half input) {                                      \
            return FUN1(input);                                                                    \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    namespace detail {                                                                             \
    template<>                                                                                     \
    struct apply_impl<accurate_policy, ops::NAME<__half>, 2, __half, __half> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<__half>, __half* result, const __half* a) { \
            __half2 r = FUN2(__half2 {a[0], a[1]});                                                \
            result[0] = r.x, result[1] = r.y;                                                      \
        }                                                                                          \
    };                                                                                             \
    }
#else
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)
#endif

KERNEL_FLOAT_FP16_UNARY_FUN(sin, hsin, h2sin)
KERNEL_FLOAT_FP16_UNARY_FUN(cos, hcos, h2cos)

KERNEL_FLOAT_FP16_UNARY_FUN(exp, hexp, h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(exp2, hexp2, h2exp2)
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, hexp10, h2exp10)
KERNEL_FLOAT_FP16_UNARY_FUN(log, hlog, h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(log2, hlog2, h2log2)
KERNEL_FLOAT_FP16_UNARY_FUN(log10, hlog10, h2log2)

KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, hsqrt, h2sqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, hrsqrt, h2rsqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(rcp, hrcp, h2rcp)

KERNEL_FLOAT_FP16_UNARY_FUN(abs, __habs, __habs2)
KERNEL_FLOAT_FP16_UNARY_FUN(floor, hfloor, h2floor)
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, hceil, h2ceil)
KERNEL_FLOAT_FP16_UNARY_FUN(rint, hrint, h2rint)
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, htrunc, h2trunc)
KERNEL_FLOAT_FP16_UNARY_FUN(negate, __hneg, __hneg2)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                                   \
    namespace ops {                                                                      \
    template<>                                                                           \
    struct NAME<__half> {                                                                \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {         \
            return ops::cast<decltype(FUN1(left, right)), __half> {}(FUN1(left, right)); \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    namespace detail {                                                                   \
    template<>                                                                           \
    struct apply_impl<accurate_policy, ops::NAME<__half>, 2, __half, __half, __half> {   \
        KERNEL_FLOAT_INLINE static void                                                  \
        call(ops::NAME<__half>, __half* result, const __half* a, const __half* b) {      \
            __half2 r = FUN2(__half2 {a[0], a[1]}, __half2 {b[0], b[1]});                \
            result[0] = r.x, result[1] = r.y;                                            \
        }                                                                                \
    };                                                                                   \
    }
#else
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)
#endif

// There are not available in HIP
#if KERNEL_FLOAT_IS_CUDA
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)
#endif

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __hneu, __hneu2)
KERNEL_FLOAT_FP16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_FP16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater_equal, __hge, __hgt2)

#if KERNEL_FLOAT_IS_DEVICE
namespace ops {
template<>
struct fma<__half> {
    KERNEL_FLOAT_INLINE __half operator()(__half a, __half b, __half c) const {
        return __hfma(a, b, c);
    }
};
}  // namespace ops

namespace detail {
template<>
struct apply_impl<accurate_policy, ops::fma<__half>, 2, __half, __half, __half, __half> {
    KERNEL_FLOAT_INLINE static void
    call(ops::fma<__half>, __half* result, const __half* a, const __half* b, const __half* c) {
        __half2 r = __hfma2(__half2 {a[0], a[1]}, __half2 {b[0], b[1]}, __half2 {c[0], c[1]});
        result[0] = r.x, result[1] = r.y;
    }
};
}  // namespace detail
#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)    \
    namespace ops {                                      \
    template<>                                           \
    struct cast<T, __half> {                             \
        KERNEL_FLOAT_INLINE __half operator()(T input) { \
            return TO_HALF;                              \
        }                                                \
    };                                                   \
    template<>                                           \
    struct cast<__half, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__half input) { \
            return FROM_HALF;                            \
        }                                                \
    };                                                   \
    }

// Only CUDA has a special `__double2half` intrinsic
#if KERNEL_FLOAT_IS_HIP
#define KERNEL_FLOAT_FP16_CAST_FWD(T) \
    KERNEL_FLOAT_FP16_CAST(T, static_cast<_Float16>(input), static_cast<T>(input))

KERNEL_FLOAT_FP16_CAST_FWD(double)
KERNEL_FLOAT_FP16_CAST_FWD(float)

KERNEL_FLOAT_FP16_CAST_FWD(char)
KERNEL_FLOAT_FP16_CAST_FWD(signed char)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned char)

KERNEL_FLOAT_FP16_CAST_FWD(signed short)
KERNEL_FLOAT_FP16_CAST_FWD(signed int)
KERNEL_FLOAT_FP16_CAST_FWD(signed long)
KERNEL_FLOAT_FP16_CAST_FWD(signed long long)

KERNEL_FLOAT_FP16_CAST_FWD(unsigned short)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned int)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned long)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned long long)
#else
KERNEL_FLOAT_FP16_CAST(double, __double2half(input), double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float, __float2half(input), __half2float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_FP16_CAST(char, __int2half_rn(input), (char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed char, __int2half_rn(input), (signed char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned char, __int2half_rn(input), (unsigned char)__half2int_rz(input));

KERNEL_FLOAT_FP16_CAST(signed short, __short2half_rn(input), __half2short_rz(input));
KERNEL_FLOAT_FP16_CAST(signed int, __int2half_rn(input), __half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned short, __ushort2half_rn(input), __half2ushort_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned int, __uint2half_rn(input), __half2uint_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));
#endif

using half = __half;
KERNEL_FLOAT_VECTOR_ALIAS(half, __half)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
