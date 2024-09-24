#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include "macros.h"

#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>

#include "binops.h"
#include "reduce.h"
#include "vector.h"

namespace kernel_float {

template<>
struct preferred_vector_size<__nv_bfloat16> {
    static constexpr size_t value = 2;
};

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_bfloat16)

template<>
struct into_vector_impl<__nv_bfloat162> {
    using value_type = __nv_bfloat16;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__nv_bfloat16, 2> call(__nv_bfloat162 input) {
        return {input.x, input.y};
    }
};

namespace detail {
template<>
struct allow_float_fallback<__nv_bfloat16> {
    static constexpr bool value = true;
};
};  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                                   \
    namespace ops {                                                                     \
    template<>                                                                          \
    struct NAME<__nv_bfloat16> {                                                        \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) {             \
            return FUN1(input);                                                         \
        }                                                                               \
    };                                                                                  \
    }                                                                                   \
    namespace detail {                                                                  \
    template<>                                                                          \
    struct apply_impl<ops::NAME<__nv_bfloat16>, 2, __nv_bfloat16, __nv_bfloat16> {      \
        KERNEL_FLOAT_INLINE static void                                                 \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat16* result, const __nv_bfloat16* a) { \
            __nv_bfloat162 r = FUN2(__nv_bfloat162 {a[0], a[1]});                       \
            result[0] = r.x, result[1] = r.y;                                           \
        }                                                                               \
    };                                                                                  \
    }
#else
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)
#endif

#if KERNEL_FLOAT_CUDA_ARCH >= 800
KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2)
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos)
KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2)
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)
KERNEL_FLOAT_BF16_UNARY_FUN(rcp, ::hrcp, ::h2rcp)
#endif

#if KERNEL_FLOAT_CUDA_ARCH >= 800
#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                               \
    template<>                                                                                    \
    struct NAME<__nv_bfloat16> {                                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                                         \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {                               \
            return FUN1(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    namespace detail {                                                                            \
    template<>                                                                                    \
    struct apply_impl<ops::NAME<__nv_bfloat16>, 2, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> { \
        KERNEL_FLOAT_INLINE static void call(                                                     \
            ops::NAME<__nv_bfloat16>,                                                             \
            __nv_bfloat16* result,                                                                \
            const __nv_bfloat16* a,                                                               \
            const __nv_bfloat16* b) {                                                             \
            __nv_bfloat162 r = FUN2(__nv_bfloat162 {a[0], a[1]}, __nv_bfloat162 {b[0], b[1]});    \
            result[0] = r.x, result[1] = r.y;                                                     \
        }                                                                                         \
    };                                                                                            \
    }
#else
#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)
#endif

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __hneu, __hneu2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)

#if KERNEL_FLOAT_CUDA_ARCH >= 800
namespace ops {
template<>
struct fma<__nv_bfloat16> {
    KERNEL_FLOAT_INLINE __nv_bfloat16
    operator()(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) const {
        return __hfma(a, b, c);
    }
};
}  // namespace ops

namespace detail {
template<>
struct apply_impl<
    ops::fma<__nv_bfloat16>,
    2,
    __nv_bfloat16,
    __nv_bfloat16,
    __nv_bfloat16,
    __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static void call(
        ops::fma<__nv_bfloat16>,
        __nv_bfloat16* result,
        const __nv_bfloat16* a,
        const __nv_bfloat16* b,
        const __nv_bfloat16* c) {
        __nv_bfloat162 r = __hfma2(
            __nv_bfloat162 {a[0], a[1]},
            __nv_bfloat162 {b[0], b[1]},
            __nv_bfloat162 {c[0], c[1]});
        result[0] = r.x, result[1] = r.y;
    }
};
}  // namespace detail
#endif

namespace ops {
template<>
struct cast<double, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(double input) {
        return __double2bfloat16(input);
    };
};

template<>
struct cast<float, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(float input) {
        return __float2bfloat16(input);
    };
};

template<>
struct cast<__nv_bfloat16, float> {
    KERNEL_FLOAT_INLINE float operator()(__nv_bfloat16 input) {
        return __bfloat162float(input);
    };
};
}  // namespace ops

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)           \
    namespace ops {                                             \
    template<>                                                  \
    struct cast<T, __nv_bfloat16> {                             \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(T input) { \
            return TO_HALF;                                     \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<__nv_bfloat16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__nv_bfloat16 input) { \
            return FROM_HALF;                                   \
        }                                                       \
    };                                                          \
    }

#if KERNEL_FLOAT_CUDA_ARCH >= 800
// clang-format off
// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(signed char, __int2bfloat16_rn(input), (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned char, __int2bfloat16_rn(input), (unsigned char)__bfloat162int_rz(input));

KERNEL_FLOAT_BF16_CAST(signed short, __bfloat162short_rz(input), __short2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(signed int, __bfloat162int_rz(input), __int2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(signed long, __ll2bfloat16_rn(input), (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __bfloat162ushort_rz(input), __ushort2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __bfloat162uint_rz(input), __uint2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(unsigned long, __ull2bfloat16_rn(input), (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));
// clang-format on
#else
KERNEL_FLOAT_BF16_CAST(
    bool,
    __nv_bfloat16_raw {input ? (unsigned short)0 : (unsigned short)0x3C00},
    (__nv_bfloat16_raw(input).x & 0x7FFF) != 0);
#endif

using bfloat16 = __nv_bfloat16;
KERNEL_FLOAT_VECTOR_ALIAS(bfloat16x, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __nv_bfloat16)

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
template<>
struct promote_type<__nv_bfloat16, __half> {
    using type = float;
};

template<>
struct promote_type<__half, __nv_bfloat16> {
    using type = float;
};

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  //KERNEL_FLOAT_BF16_H
