#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include "macros.h"

#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>

#include "binops.h"
#include "cast.h"
#include "interface.h"
#include "storage.h"
#include "unops.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__nv_bfloat16, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_bfloat16)

template<>
struct vector_traits<__nv_bfloat162> {
    using value_type = __nv_bfloat16;
    static constexpr size_t size = 2;

    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 fill(__nv_bfloat16 value) {
#if KERNEL_FLOAT_ON_DEVICE
        return __bfloat162bfloat162(value);
#else
        return {value, value};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 create(__nv_bfloat16 low, __nv_bfloat16 high) {
#if KERNEL_FLOAT_ON_DEVICE
        return __halves2bfloat162(low, high);
#else
        return {low, high};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __nv_bfloat16 get(__nv_bfloat162 self, size_t index) {
#if KERNEL_FLOAT_ON_DEVICE
        if (index == 0) {
            return __low2bfloat16(self);
        } else {
            return __high2bfloat16(self);
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
    static void set(__nv_bfloat162& self, size_t index, __nv_bfloat16 value) {
        if (index == 0) {
            self.x = value;
        } else {
            self.y = value;
        }
    }
};

template<size_t N>
struct default_storage<__nv_bfloat16, N, Alignment::Maximum, enabled_t<(N >= 2)>> {
    using type = nested_array<__nv_bfloat162, N>;
};

template<size_t N>
struct default_storage<__nv_bfloat16, N, Alignment::Packed, enabled_t<(N >= 2 && N % 2 == 0)>> {
    using type = nested_array<__nv_bfloat162, N>;
};

#if KERNEL_FLOAT_ON_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                             \
    namespace ops {                                                               \
    template<>                                                                    \
    struct NAME<__nv_bfloat16> {                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) {       \
            return FUN1(input);                                                   \
        }                                                                         \
    };                                                                            \
    }                                                                             \
    namespace detail {                                                            \
    template<>                                                                    \
    struct map_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat162, __nv_bfloat162> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                 \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 input) {                    \
            return FUN2(input);                                                   \
        }                                                                         \
    };                                                                            \
    }

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

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
    struct zip_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat162, __nv_bfloat162, __nv_bfloat162> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                                 \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 left, __nv_bfloat162 right) {               \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)

#endif

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
//KERNEL_FLOAT_TYPE_ALIAS(half, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __nv_bfloat16)

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  //KERNEL_FLOAT_BF16_H
