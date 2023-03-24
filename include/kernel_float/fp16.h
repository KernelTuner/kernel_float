#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include "macros.h"

#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>

#include "interface.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__half, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __half)

template<>
struct vector_traits<__half2> {
    using value_type = __half;
    static constexpr size_t size = 2;

    KERNEL_FLOAT_INLINE
    static __half2 fill(__half value) {
#if KERNEL_FLOAT_ON_DEVICE
        return __half2half2(value);
#else
        return {value, value};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __half2 create(__half low, __half high) {
#if KERNEL_FLOAT_ON_DEVICE
        return __halves2half2(low, high);
#else
        return {low, high};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __half get(__half2 self, size_t index) {
#if KERNEL_FLOAT_ON_DEVICE
        if (index == 0) {
            return __low2half(self);
        } else {
            return __high2half(self);
        }
#else
        if (index == 0) {
            return self.x;
        } else {
            return self.y;
        }
#endif
    }

    KERNEL_FLOAT_INLINE
    static void set(__half2& self, size_t index, __half value) {
        if (index == 0) {
            self.x = value;
        } else {
            self.y = value;
        }
    }
};

template<size_t N>
struct default_storage<__half, N, Alignment::Maximum, enabled_t<(N >= 2)>> {
    using type = nested_array<__half2, N>;
};

template<size_t N>
struct default_storage<__half, N, Alignment::Packed, enabled_t<(N >= 2 && N % 2 == 0)>> {
    using type = nested_array<__half2, N>;
};

#if KERNEL_FLOAT_ON_DEVICE
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
    struct map_helper<ops::NAME<__half>, __half2, __half2> {                        \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 input) { \
            return FUN2(input);                                                     \
        }                                                                           \
    };                                                                              \
    }

KERNEL_FLOAT_FP16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_FP16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_FP16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_FP16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_FP16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_FP16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_FP16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_FP16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

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
    struct zip_helper<ops::NAME<__half>, __half2, __half2, __half2> {                             \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 left, __half2 right) { \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_FP16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater_equal, __hge, __hgt2)

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

KERNEL_FLOAT_FP16_CAST(signed short, __short2half_rn(input), __half2short_rz(input));
KERNEL_FLOAT_FP16_CAST(signed int, __int2half_rn(input), __half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned int, __uint2half_rn(input), __half2uint_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned short, __ushort2half_rn(input), __half2ushort_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

using half = __half;
using float16 = __half;
//KERNEL_FLOAT_TYPE_ALIAS(half, __half)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
