#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include "macros.h"

#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>

#include "binops.h"
#include "reduce.h"
#include "vector.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_bfloat16)

template<>
struct into_vector_traits<__nv_bfloat162> {
    using value_type = __nv_bfloat16;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__nv_bfloat16, 2> call(__nv_bfloat162 input) {
        return {input.x, input.y};
    }
};

namespace detail {
template<typename F>
struct map_bfloat16x2 {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 call(F fun, __nv_bfloat162 input) {
        __nv_bfloat16 a = fun(input.x);
        __nv_bfloat16 b = fun(input.y);
        return {a, b};
    }
};

template<typename F>
struct zip_bfloat16x2 {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 call(F fun, __nv_bfloat162 left, __nv_bfloat162 right) {
        __nv_bfloat16 a = fun(left.x, left.y);
        __nv_bfloat16 b = fun(right.y, right.y);
        return {a, b};
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __nv_bfloat16, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static vector_storage<__nv_bfloat16, N>
    call(F fun, const vector_storage<__nv_bfloat16, N>& input) {
        vector_storage<__nv_bfloat16, N> result;

#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {input.data()[i], input.data()[i + 1]};
            __nv_bfloat162 b = map_bfloat16x2<F>::call(fun, a);
            result.data()[i + 0] = b.x;
            result.data()[i + 1] = b.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(input.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static vector_storage<__nv_bfloat16, N> call(
        F fun,
        const vector_storage<__nv_bfloat16, N>& left,
        const vector_storage<__nv_bfloat16, N>& right) {
        vector_storage<__nv_bfloat16, N> result;
#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {left.data()[i], left.data()[i + 1]};
            __nv_bfloat162 b = {right.data()[i], right.data()[i + 1]};
            __nv_bfloat162 c = zip_bfloat16x2<F>::call(fun, a, b);
            result.data()[i + 0] = c.x;
            result.data()[i + 1] = c.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(left.data()[N - 1], right.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct reduce_helper<F, N, __nv_bfloat16, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __nv_bfloat16
    call(F fun, const vector_storage<__nv_bfloat16, N>& input) {
        __nv_bfloat162 accum = {input.data()[0], input.data()[1]};

#pragma unroll
        for (size_t i = 2; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {input.data()[i], input.data()[i + 1]};
            accum = zip_bfloat16x2<F>::call(fun, accum, a);
        }

        __nv_bfloat16 result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input.data()[N - 1]);
        }

        return result;
    }
};
}  // namespace detail

#define KERNEL_FLOAT_BF16_UNARY_FORWARD(NAME)                               \
    namespace ops {                                                         \
    template<>                                                              \
    struct NAME<__nv_bfloat16> {                                            \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) { \
            return __nv_bfloat16(ops::NAME<float> {}(float(input)));        \
        }                                                                   \
    };                                                                      \
    }

KERNEL_FLOAT_BF16_UNARY_FORWARD(tan)
KERNEL_FLOAT_BF16_UNARY_FORWARD(asin)
KERNEL_FLOAT_BF16_UNARY_FORWARD(acos)
KERNEL_FLOAT_BF16_UNARY_FORWARD(atan)
KERNEL_FLOAT_BF16_UNARY_FORWARD(expm1)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                       \
    namespace ops {                                                         \
    template<>                                                              \
    struct NAME<__nv_bfloat16> {                                            \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) { \
            return FUN1(input);                                             \
        }                                                                   \
    };                                                                      \
    }                                                                       \
    namespace detail {                                                      \
    template<>                                                              \
    struct map_halfx2<ops::NAME<__nv_bfloat16>> {                           \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                           \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 input) {              \
            return FUN2(input);                                             \
        }                                                                   \
    };                                                                      \
    }
#else
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2) KERNEL_FLOAT_BF16_UNARY_FORWARD(NAME)
#endif

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

KERNEL_FLOAT_BF16_UNARY_FUN(fast_exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_cos, ::hcos, ::h2cos)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_sin, ::hsin, ::h2sin)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                              \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__nv_bfloat16> {                                                    \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                           \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {                 \
            return FUN1(left, right);                                               \
        }                                                                           \
    };                                                                              \
    }                                                                               \
    namespace detail {                                                              \
    template<>                                                                      \
    struct zip_bfloat16x2<ops::NAME<__nv_bfloat16>> {                               \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                   \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 left, __nv_bfloat162 right) { \
            return FUN2(left, right);                                               \
        }                                                                           \
    };                                                                              \
    }
#else
#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                            \
    namespace ops {                                                               \
    template<>                                                                    \
    struct NAME<__nv_bfloat16> {                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                         \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {               \
            return __nv_bfloat16(ops::NAME<float> {}(float(left), float(right))); \
        }                                                                         \
    };                                                                            \
    }
#endif

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(fast_div, __hdiv, __h2div)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)

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

KERNEL_FLOAT_BF16_CAST(double, __double2bfloat16(input), double(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float, __float2bfloat16(input), __bfloat162float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    signed char,
    __int2bfloat16_rn(input),
    (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned char,
    __int2bfloat16_rn(input),
    (unsigned char)__bfloat162int_rz(input));

KERNEL_FLOAT_BF16_CAST(signed short, __bfloat162short_rz(input), __short2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(signed int, __bfloat162int_rz(input), __int2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    signed long,
    __ll2bfloat16_rn(input),
    (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __bfloat162ushort_rz(input), __ushort2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __bfloat162uint_rz(input), __uint2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned long,
    __ull2bfloat16_rn(input),
    (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));

using bfloat16 = __nv_bfloat16;
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __nv_bfloat16)

#if KERNEL_FLOAT_IS_DEVICE
namespace detail {
template<size_t N>
struct dot_helper<__nv_bfloat16, N> {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat16 call(
        const vector_storage<__nv_bfloat16, N>& left,
        const vector_storage<__nv_bfloat16, N>& right) {
        if (N == 0) {
            return __nv_bfloat16(0);
        } else if (N == 1) {
            return __hmul(left.data()[0], right.data()[0]);
        } else {
            __nv_bfloat162 first_a = {left.data()[0], left.data()[1]};
            __nv_bfloat162 first_b = {right.data()[0], right.data()[1]};
            __nv_bfloat162 accum = __hmul2(first_a, first_b);

#pragma unroll
            for (size_t i = 2; i + 2 <= N; i += 2) {
                __nv_bfloat162 a = {left.data()[i], left.data()[i + 1]};
                __nv_bfloat162 b = {right.data()[i], right.data()[i + 1]};
                accum = __hfma2(a, b, accum);
            }

            __nv_bfloat16 result = __hadd(accum.x, accum.y);

            if (N % 2 != 0) {
                __nv_bfloat16 a = left.data()[N - 1];
                __nv_bfloat16 b = right.data()[N - 1];
                result = __hfma(a, b, result);
            }

            return result;
        }
    }
};
}  // namespace detail
#endif

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));

template<>
struct promote_type<__nv_bfloat16, __half> {
    using type = float;
}

template<>
struct promote_type<__half, __nv_bfloat16> {
    using type = float;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  //KERNEL_FLOAT_BF16_H
