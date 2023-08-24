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
struct into_vector_traits<__half2> {
    using value_type = __half;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__half, 2> call(__half2 input) {
        return {input.x, input.y};
    }
};

namespace detail {
template<typename F>
struct map_halfx2 {
    KERNEL_FLOAT_INLINE
    static __half2 call(F fun, __half2 input) {
        __half a = fun(input.x);
        __half b = fun(input.y);
        return {a, b};
    }
};

template<typename F>
struct zip_halfx2 {
    KERNEL_FLOAT_INLINE
    static __half2 call(F fun, __half2 left, __half2 right) {
        __half a = fun(left.x, left.y);
        __half b = fun(right.y, right.y);
        return {a, b};
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __half, __half> {
    KERNEL_FLOAT_INLINE static vector_storage<__half, N>
    call(F fun, const vector_storage<__half, N>& input) {
        vector_storage<__half, N> result;

#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __half2 a = {input.data()[i], input.data()[i + 1]};
            __half2 b = map_halfx2<F>::call(fun, a);
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
struct apply_impl<F, N, __half, __half, __half> {
    KERNEL_FLOAT_INLINE static vector_storage<__half, N>
    call(F fun, const vector_storage<__half, N>& left, const vector_storage<__half, N>& right) {
        vector_storage<__half, N> result;
#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __half2 a = {left.data()[i], left.data()[i + 1]};
            __half2 b = {right.data()[i], right.data()[i + 1]};
            __half2 c = zip_halfx2<F>::call(fun, a, b);
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
struct reduce_helper<F, N, __half, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __half call(F fun, const vector_storage<__half, N>& input) {
        __half2 accum = {input.data()[0], input.data()[1]};

#pragma unroll
        for (size_t i = 2; i + 2 <= N; i += 2) {
            __half2 a = {input.data()[i], input.data()[i + 1]};
            accum = zip_halfx2<F>::call(fun, accum, a);
        }

        __half result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input.data()[N - 1]);
        }

        return result;
    }
};

};  // namespace detail

#define KERNEL_FLOAT_FP16_UNARY_FORWARD(NAME)                 \
    namespace ops {                                           \
    template<>                                                \
    struct NAME<__half> {                                     \
        KERNEL_FLOAT_INLINE __half operator()(__half input) { \
            return __half(ops::NAME<float> {}(float(input))); \
        }                                                     \
    };                                                        \
    }

KERNEL_FLOAT_FP16_UNARY_FORWARD(tan)
KERNEL_FLOAT_FP16_UNARY_FORWARD(asin)
KERNEL_FLOAT_FP16_UNARY_FORWARD(acos)
KERNEL_FLOAT_FP16_UNARY_FORWARD(atan)
KERNEL_FLOAT_FP16_UNARY_FORWARD(expm1)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                               \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__half> {                                                           \
        KERNEL_FLOAT_INLINE __half operator()(__half input) {                       \
            return FUN1(input);                                                     \
        }                                                                           \
    };                                                                              \
    }                                                                               \
    namespace detail {                                                              \
    template<>                                                                      \
    struct map_halfx2<ops::NAME<__half>> {                                          \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 input) { \
            return FUN2(input);                                                     \
        }                                                                           \
    };                                                                              \
    }
#else
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2) KERNEL_FLOAT_FP16_UNARY_FORWARD(NAME)
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
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                               \
    template<>                                                                                    \
    struct NAME<__half> {                                                                         \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {                  \
            return FUN1(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    namespace detail {                                                                            \
    template<>                                                                                    \
    struct zip_halfx2<ops::NAME<__half>> {                                                        \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 left, __half2 right) { \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }
#else
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                           \
    namespace ops {                                                              \
    template<>                                                                   \
    struct NAME<__half> {                                                        \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const { \
            return __half(ops::NAME<float> {}(float(left), float(right)));       \
        }                                                                        \
    };                                                                           \
    }
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
template<size_t N>
struct dot_helper<__half, N> {
    KERNEL_FLOAT_INLINE
    static __half
    call(const vector_storage<__half, N>& left, const vector_storage<__half, N>& right) {
        if (N == 0) {
            return __half(0);
        } else if (N == 1) {
            return __hmul(left.data()[0], right.data()[0]);
        } else {
            __half2 first_a = {left.data()[0], left.data()[1]};
            __half2 first_b = {right.data()[0], right.data()[1]};
            __half2 accum = __hmul2(first_a, first_b);

#pragma unroll
            for (size_t i = 2; i + 2 <= N; i += 2) {
                __half2 a = {left.data()[i], left.data()[i + 1]};
                __half2 b = {right.data()[i], right.data()[i + 1]};
                accum = __hfma2(a, b, accum);
            }

            __half result = __hadd(accum.x, accum.y);

            if (N % 2 != 0) {
                __half a = left.data()[N - 1];
                __half b = right.data()[N - 1];
                result = __hfma(a, b, result);
            }

            return result;
        }
    }
};
}  // namespace detail
#endif

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
