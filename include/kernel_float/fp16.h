#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include "macros.h"

#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>

#include "vector.h"

namespace kernel_float {
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
    struct apply_impl<ops::NAME<__half>, 2, __half, __half> {                                      \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<__half>, __half* result, const __half* a) { \
            __half2 r = FUN2(__half2 {a[0], a[1]});                                                \
            result[0] = r.x, result[1] = r.y;                                                      \
        }                                                                                          \
    };                                                                                             \
    }
#else
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)
#endif

KERNEL_FLOAT_FP16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_FP16_UNARY_FUN(negate, ::__hneg, ::__hneg2)
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_FP16_UNARY_FUN(cos, ::hcos, ::h2cos)
KERNEL_FLOAT_FP16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_FP16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_FP16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(log10, ::hlog10, ::h2log2)
KERNEL_FLOAT_FP16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)

KERNEL_FLOAT_FP16_UNARY_FUN(fast_exp, ::hexp, ::h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_log, ::hlog, ::h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_cos, ::hcos, ::h2cos)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_sin, ::hsin, ::h2sin)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                              \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__half> {                                                           \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {    \
            return FUN1(left, right);                                               \
        }                                                                           \
    };                                                                              \
    }                                                                               \
    namespace detail {                                                              \
    template<>                                                                      \
    struct apply_impl<ops::NAME<__half>, 2, __half, __half, __half> {               \
        KERNEL_FLOAT_INLINE static void                                             \
        call(ops::NAME<__half>, __half* result, const __half* a, const __half* b) { \
            __half2 r = FUN2(__half2 {a[0], a[1]}, __half2 {b[0], b[1]});           \
            result[0] = r.x, result[1] = r.y;                                       \
        }                                                                           \
    };                                                                              \
    }
#else
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)
#endif

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)
KERNEL_FLOAT_FP16_BINARY_FUN(fast_div, __hdiv, __h2div)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __heq, __heq2)
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
struct apply_impl<ops::fma<__half>, 2, __half, __half, __half, __half> {
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

KERNEL_FLOAT_FP16_CAST(double, __double2half(input), double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float, __float2half(input), __half2float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_FP16_CAST(char, __int2half_rn(input), (char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed char, __int2half_rn(input), (signed char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned char, __int2half_rn(input), (unsigned char)__half2int_rz(input));

KERNEL_FLOAT_FP16_CAST(signed short, __half2short_rz(input), __short2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed int, __half2int_rz(input), __int2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned short, __half2ushort_rz(input), __ushort2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned int, __half2uint_rz(input), __uint2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

using half = __half;
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

#if KERNEL_FLOAT_IS_DEVICE
namespace detail {
template<>
struct dot_impl<__half, 0> {
    KERNEL_FLOAT_INLINE
    static __half call(const __half* left, const __half* right) {
        return __half(0);
    }
};

template<>
struct dot_impl<__half, 1> {
    KERNEL_FLOAT_INLINE
    static __half call(const __half* left, const __half* right) {
        return __hmul(left[0], right[0]);
    }
};

template<size_t N>
struct dot_impl<__half, N> {
    static_assert(N >= 2, "internal error");

    KERNEL_FLOAT_INLINE
    static __half call(const __half* left, const __half* right) {
        __half2 first_a = {left[0], left[1]};
        __half2 first_b = {right[0], right[1]};
        __half2 accum = __hmul2(first_a, first_b);

#pragma unroll
        for (size_t i = 2; i + 2 <= N; i += 2) {
            __half2 a = {left[i], left[i + 1]};
            __half2 b = {right[i], right[i + 1]};
            accum = __hfma2(a, b, accum);
        }

        __half result = __hadd(accum.x, accum.y);

        if (N % 2 != 0) {
            __half a = left[N - 1];
            __half b = right[N - 1];
            result = __hfma(a, b, result);
        }

        return result;
    }
};
}  // namespace detail
#endif

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
