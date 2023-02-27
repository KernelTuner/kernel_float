#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include "macros.h"

#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>

#include "interface.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_bfloat16)

struct vector_bfloat16x2 {
    static_assert(sizeof(__nv_bfloat16) * 2 == sizeof(__nv_bfloat162), "invalid size");
    static_assert(alignof(__nv_bfloat16) <= alignof(__nv_bfloat162), "invalid alignment");

    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat16 v = {}) noexcept : vector_ {v, v} {}
    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat16 x, __nv_bfloat16 y) noexcept :
        vector_ {x, y} {}
    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat162 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __nv_bfloat162() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(const_index<0>) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(const_index<1>) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(const_index<0>, __nv_bfloat16 v) {
        *this = vector_bfloat16x2(v, get(const_index<1> {}));
    }

    KERNEL_FLOAT_INLINE void set(const_index<1>, __nv_bfloat16 v) {
        *this = vector_bfloat16x2(get(const_index<0> {}), v);
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(size_t index) const {
        if (index == 0) {
            return get(const_index<0> {});
        } else {
            return get(const_index<1> {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __nv_bfloat16 value) const {
        if (index == 0) {
            set(const_index<0> {}, value);
        } else {
            set(const_index<1> {}, value);
        }
    }

  private:
    __nv_bfloat162 vector_;
};

template<>
struct vector_traits<vector_bfloat16x2>:
    default_vector_traits<vector_bfloat16x2, __nv_bfloat16, 2> {};

template<>
struct vector_traits<__nv_bfloat16>: vector_traits<vector_scalar<__nv_bfloat16>> {};

template<>
struct vector_traits<__nv_bfloat162>: vector_traits<vector_bfloat16x2> {};

template<>
struct default_vector_storage<__nv_bfloat16, 2> {
    using type = vector_bfloat16x2;
};

#if KERNEL_FLOAT_ON_DEVICE
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
    struct map_helper<ops::NAME<__nv_bfloat16>, vector_bfloat16x2, vector_bfloat16x2> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                       \
        call(ops::NAME<__nv_bfloat16>, const __nv_bfloat162& input) {                   \
            return FUN2(input);                                                         \
        }                                                                               \
    };                                                                                  \
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
    struct zip_helper<                                                                            \
        ops::NAME<__nv_bfloat16>,                                                                 \
        vector_bfloat16x2,                                                                        \
        vector_bfloat16x2,                                                                        \
        vector_bfloat16x2> {                                                                      \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                                 \
        call(ops::NAME<__nv_bfloat16>, const __nv_bfloat162& left, const __nv_bfloat162& right) { \
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

namespace detail {
template<>
struct map_helper<ops::cast<__nv_bfloat16, float>, vector_storage<float, 2>, vector_bfloat16x2> {
    KERNEL_FLOAT_INLINE static vector_storage<float, 2>
    call(ops::cast<__nv_bfloat16, float>, __nv_bfloat162 input) noexcept {
        return __bfloat1622float2(input);
    }
};

template<>
struct map_helper<ops::cast<float, __nv_bfloat16>, vector_bfloat16x2, vector_storage<float, 2>> {
    KERNEL_FLOAT_INLINE static vector_bfloat16x2
    call(ops::cast<float, __nv_bfloat16>, const vector_storage<float, 2>& input) noexcept {
        return __float22bfloat162_rn(input);
    }
};
}  // namespace detail

using bfloat16 = __nv_bfloat16;
KERNEL_FLOAT_TYPE_ALIAS(bf16x, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bfloat16x, __nv_bfloat16)

}  // namespace kernel_float

#endif

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE

#endif  //KERNEL_FLOAT_BF16_H
