#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include "macros.h"

#if KERNEL_FLOAT_BF16_AVAILABLE
//#define CUDA_NO_BFLOAT16 (1)
//#define __CUDA_NO_BFLOAT16_OPERATORS__ (1)
//#define __CUDA_NO_BFLOAT162_OPERATORS__ (1)
//#define __CUDA_NO_BFLOAT16_CONVERSIONS__ (1)

#if KERNEL_FLOAT_IS_CUDA
#include <cuda_bf16.h>
#elif KERNEL_FLOAT_IS_HIP
#include <hip/hip_bf16.h>
#endif

#include "binops.h"
#include "reduce.h"
#include "vector.h"

namespace kernel_float {

#if KERNEL_FLOAT_IS_CUDA
using bfloat16_t = __nv_bfloat16;
using bfloat16x2_t = __nv_bfloat162;
#elif KERNEL_FLOAT_IS_HIP
using bfloat16_t = __hip_bfloat16;
using bfloat16x2_t = __hip_bfloat162;
#endif

#if KERNEL_FLOAT_IS_CUDA && __CUDA_ARCH__ >= 800
#define KERNEL_FLOAT_BF16_OPS_SUPPORTED 1
#endif

template<>
struct preferred_vector_size<bfloat16_t> {
    static constexpr size_t value = 2;
};

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(bfloat16_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, bfloat16_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, bfloat16_t)

template<>
struct into_vector_impl<bfloat16x2_t> {
    using value_type = bfloat16_t;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<bfloat16_t, 2> call(bfloat16x2_t input) {
        return {input.x, input.y};
    }
};

namespace detail {
template<>
struct allow_float_fallback<bfloat16_t> {
    static constexpr bool value = true;
};
};  // namespace detail

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                                      \
    namespace ops {                                                                        \
    template<>                                                                             \
    struct NAME<bfloat16_t> {                                                              \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t input) {                      \
            return FUN1(input);                                                            \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    namespace detail {                                                                     \
    template<>                                                                             \
    struct apply_impl<accurate_policy, ops::NAME<bfloat16_t>, 2, bfloat16_t, bfloat16_t> { \
        KERNEL_FLOAT_INLINE static void                                                    \
        call(ops::NAME<bfloat16_t>, bfloat16_t* result, const bfloat16_t* a) {             \
            bfloat16x2_t r = FUN2(bfloat16x2_t {a[0], a[1]});                              \
            result[0] = r.x, result[1] = r.y;                                              \
        }                                                                                  \
    };                                                                                     \
    }

KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos)

KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(exp2, ::hexp2, ::h2exp2)
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(log2, ::hlog2, ::h2log2)
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2)

KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(rcp, ::hrcp, ::h2rcp)

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2)
#endif

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                                       \
    namespace ops {                                                                          \
    template<>                                                                               \
    struct NAME<bfloat16_t> {                                                                \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t left, bfloat16_t right) const { \
            return ops::cast<decltype(FUN1(left, right)), bfloat16_t> {}(FUN1(left, right)); \
        }                                                                                    \
    };                                                                                       \
    }                                                                                        \
    namespace detail {                                                                       \
    template<>                                                                               \
    struct apply_impl<                                                                       \
        accurate_policy,                                                                     \
        ops::NAME<bfloat16_t>,                                                               \
        2,                                                                                   \
        bfloat16_t,                                                                          \
        bfloat16_t,                                                                          \
        bfloat16_t> {                                                                        \
        KERNEL_FLOAT_INLINE static void call(                                                \
            ops::NAME<bfloat16_t>,                                                           \
            bfloat16_t* result,                                                              \
            const bfloat16_t* a,                                                             \
            const bfloat16_t* b) {                                                           \
            bfloat16x2_t r = FUN2(bfloat16x2_t {a[0], a[1]}, bfloat16x2_t {b[0], b[1]});     \
            result[0] = r.x, result[1] = r.y;                                                \
        }                                                                                    \
    };                                                                                       \
    }

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
#endif

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
namespace ops {
template<>
struct fma<bfloat16_t> {
    KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t a, bfloat16_t b, bfloat16_t c) const {
        return __hfma(a, b, c);
    }
};
}  // namespace ops

namespace detail {
template<>
struct apply_impl<
    accurate_policy,
    ops::fma<bfloat16_t>,
    2,
    bfloat16_t,
    bfloat16_t,
    bfloat16_t,
    bfloat16_t> {
    KERNEL_FLOAT_INLINE static void call(
        ops::fma<bfloat16_t>,
        bfloat16_t* result,
        const bfloat16_t* a,
        const bfloat16_t* b,
        const bfloat16_t* c) {
        bfloat16x2_t r = __hfma2(
            bfloat16x2_t {a[0], a[1]},
            bfloat16x2_t {b[0], b[1]},
            bfloat16x2_t {c[0], c[1]});
        result[0] = r.x, result[1] = r.y;
    }
};

// clang-format off
#define KERNEL_FLOAT_FAST_BF16_DISPATCH(OP)                                                                 \
    template<size_t N>                                                                                      \
    struct apply_impl<fast_policy, ops::OP<bfloat16_t>, N, bfloat16_t, bfloat16_t> {                        \
        KERNEL_FLOAT_INLINE static void                                                                     \
        call(ops::OP<bfloat16_t>, bfloat16_t* output, const bfloat16_t* input) {                            \
            float v[N];                                                                                     \
            map_impl<fast_policy, ops::cast<bfloat16_t, float>, N, float, bfloat16_t>::call({}, v, input);  \
            map_impl<fast_policy, ops::OP<float>, N, float, float>::call({}, v, v);                         \
            map_impl<fast_policy, ops::cast<float, bfloat16_t>, N, bfloat16_t, float>::call({}, output, v); \
        }                                                                                                   \
    };
// clang-format on

KERNEL_FLOAT_FAST_F32_MAP(KERNEL_FLOAT_FAST_BF16_DISPATCH)
}  // namespace detail
#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)        \
    namespace ops {                                          \
    template<>                                               \
    struct cast<T, bfloat16_t> {                             \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(T input) { \
            return TO_HALF;                                  \
        }                                                    \
    };                                                       \
    template<>                                               \
    struct cast<bfloat16_t, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(bfloat16_t input) { \
            return FROM_HALF;                                \
        }                                                    \
    };                                                       \
    }

KERNEL_FLOAT_BF16_CAST(float, __float2bfloat16(input), __bfloat162float(input))
KERNEL_FLOAT_BF16_CAST(double, __double2bfloat16(input), __bfloat162float(input))

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
// clang-format off
// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(signed char, __int2bfloat16_rn(input), (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned char, __int2bfloat16_rn(input), (unsigned char)__bfloat162int_rz(input));

KERNEL_FLOAT_BF16_CAST(signed short, __short2bfloat16_rn(input), __bfloat162short_rz(input));
KERNEL_FLOAT_BF16_CAST(signed int, __int2bfloat16_rn(input), __bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(signed long, __ll2bfloat16_rn(input), (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __ushort2bfloat16_rn(input), __bfloat162ushort_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __uint2bfloat16_rn(input), __bfloat162uint_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned long, __ull2bfloat16_rn(input), (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));
// clang-format on
#endif

#if KERNEL_FLOAT_IS_CUDA
//KERNEL_FLOAT_BF16_CAST(
//    bool,
//    __nv_bfloat16_raw {input ? (unsigned short)0 : (unsigned short)0x3C00},
//    (__nv_bfloat16_raw(input).x & 0x7FFF) != 0);
#elif KERNEL_FLOAT_IS_HIP
KERNEL_FLOAT_BF16_CAST(
    bool,
    __hip_bfloat16 {input ? (unsigned short)0 : (unsigned short)0x3C00},
    (__hip_bfloat16(input).data & 0x7FFF) != 0);
#endif

KERNEL_FLOAT_VECTOR_ALIAS(bfloat16x, bfloat16_t)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, bfloat16_t)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, bfloat16_t)

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
template<>
struct promote_type<bfloat16_t, half_t> {
    using type = float;
};

template<>
struct promote_type<half_t, bfloat16_t> {
    using type = float;
};

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  //KERNEL_FLOAT_BF16_H
