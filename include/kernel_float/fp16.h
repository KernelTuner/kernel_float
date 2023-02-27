#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include "macros.h"

#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>

#include "interface.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __half)

struct vector_half2 {
    static_assert(sizeof(__half) * 2 == sizeof(__half2), "invalid size");
    static_assert(alignof(__half) <= alignof(__half2), "invalid alignment");

    KERNEL_FLOAT_INLINE vector_half2(__half v = {}) noexcept : vector_ {v, v} {}
    KERNEL_FLOAT_INLINE vector_half2(__half x, __half y) noexcept : vector_ {x, y} {}
    KERNEL_FLOAT_INLINE vector_half2(__half2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __half2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __half get(const_index<0>) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __half get(const_index<1>) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(const_index<0>, __half v) {
        *this = vector_half2(v, get(const_index<1> {}));
    }

    KERNEL_FLOAT_INLINE void set(const_index<1>, __half v) {
        *this = vector_half2(get(const_index<0> {}), v);
    }

    KERNEL_FLOAT_INLINE __half get(size_t index) const {
        if (index == 0) {
            return get(const_index<0> {});
        } else {
            return get(const_index<1> {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __half value) const {
        if (index == 0) {
            set(const_index<0> {}, value);
        } else {
            set(const_index<1> {}, value);
        }
    }

  private:
    __half2 vector_;
};

template<>
struct vector_traits<vector_half2>: default_vector_traits<vector_half2, __half, 2> {};

template<>
struct vector_traits<__half>: vector_traits<vector_scalar<__half>> {};

template<>
struct vector_traits<__half2>: vector_traits<vector_half2> {};

template<>
struct default_vector_storage<__half, 2> {
    using type = vector_half2;
};

#if KERNEL_FLOAT_ON_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                                      \
    namespace ops {                                                                        \
    template<>                                                                             \
    struct NAME<__half> {                                                                  \
        KERNEL_FLOAT_INLINE __half operator()(__half input) {                              \
            return FUN1(input);                                                            \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    namespace detail {                                                                     \
    template<>                                                                             \
    struct map_helper<ops::NAME<__half>, vector_half2, vector_half2> {                     \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, const __half2& input) { \
            return FUN2(input);                                                            \
        }                                                                                  \
    };                                                                                     \
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

#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                               \
    namespace ops {                                                                  \
    template<>                                                                       \
    struct NAME<__half> {                                                            \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {     \
            return FUN1(left, right);                                                \
        }                                                                            \
    };                                                                               \
    }                                                                                \
    namespace detail {                                                               \
    template<>                                                                       \
    struct zip_helper<ops::NAME<__half>, vector_half2, vector_half2, vector_half2> { \
        KERNEL_FLOAT_INLINE static __half2                                           \
        call(ops::NAME<__half>, const __half2& left, const __half2& right) {         \
            return FUN2(left, right);                                                \
        }                                                                            \
    };                                                                               \
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

namespace detail {
template<>
struct map_helper<ops::cast<__half, float>, vector_storage<float, 2>, vector_half2> {
    KERNEL_FLOAT_INLINE static vector_storage<float, 2>
    call(ops::cast<__half, float>, __half2 input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float, __half>, vector_half2, vector_storage<float, 2>> {
    KERNEL_FLOAT_INLINE static vector_half2
    call(ops::cast<float, __half>, const vector_storage<float, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

}  // namespace detail

using half = __half;
using float16 = __half;
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
