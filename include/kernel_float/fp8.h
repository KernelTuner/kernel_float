#ifndef KERNEL_FLOAT_FP8_H
#define KERNEL_FLOAT_FP8_H

#include "macros.h"

#if KERNEL_FLOAT_FP8_AVAILABLE

#include <cuda_fp8.h>

#include "bf16.h"
#include "fp16.h"
#include "interface.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_fp8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_fp8_e5m2)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_fp8_e4m3)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__half, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__half, __nv_fp8_e5m2)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__nv_bfloat16, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__nv_bfloat16, __nv_fp8_e5m2)
#endif

using float8_e5m2 = __nv_fp8_e5m2;
using float8x2_e5m2 = vector_union<__nv_fp8_e5m2, 2, __nv_fp8x2_e5m2>;
using float8x4_e5m2 = vector_union<__nv_fp8_e5m2, 4, __nv_fp8x4_e5m2>;

using float8_e4m3 = __nv_fp8_e4m3;
using float8x2_e4m3 = vector_union<__nv_fp8_e4m3, 2, __nv_fp8x2_e4m3>;
using float8x4_e4m3 = vector_union<__nv_fp8_e4m3, 4, __nv_fp8x4_e4m3>;

template<>
struct vector_traits<float8_e5m2>: vector_traits<vector_scalar<float8_e5m2>> {};
template<>
struct vector_traits<float8_e4m3>: vector_traits<vector_scalar<float8_e4m3>> {};

template<>
struct vector_traits<__nv_fp8x2_e5m2>: vector_traits<float8x2_e5m2> {};
template<>
struct vector_traits<__nv_fp8x4_e5m2>: vector_traits<float8x4_e5m2> {};

template<>
struct vector_traits<__nv_fp8x2_e4m3>: vector_traits<float8x2_e4m3> {};
template<>
struct vector_traits<__nv_fp8x4_e4m3>: vector_traits<float8x4_e4m3> {};

template<>
struct default_vector_storage<float8_e5m2, 2> {
    using type = float8x2_e5m2;
};

template<>
struct default_vector_storage<float8_e5m2, 4> {
    using type = float8x4_e5m2;
};

template<>
struct default_vector_storage<float8_e4m3, 2> {
    using type = float8x2_e4m3;
};

template<>
struct default_vector_storage<float8_e4m3, 4> {
    using type = float8x4_e4m3;
};

namespace ops {
template<>
struct cast<float8_e5m2, float8_e4m3> {
    KERNEL_FLOAT_INLINE float8_e4m3 operator()(float8_e5m2 v) const {
        return float8_e4m3(__half(v));
    }
};

template<>
struct cast<float8_e4m3, float8_e5m2> {
    KERNEL_FLOAT_INLINE float8_e5m2 operator()(float8_e4m3 v) const {
        return float8_e5m2(__half(v));
    }
};
};  // namespace ops

#define KERNEL_FLOAT_FP8_CAST_VIA(T, BRIDGE)                    \
    namespace ops {                                             \
    template<>                                                  \
    struct cast<float8_e5m2, T> {                               \
        KERNEL_FLOAT_INLINE T operator()(float8_e5m2 v) const { \
            return (T)((BRIDGE)(v));                            \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<T, float8_e5m2> {                               \
        KERNEL_FLOAT_INLINE float8_e5m2 operator()(T v) const { \
            return float8_e5m2((BRIDGE)(v));                    \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<float8_e4m3, T> {                               \
        KERNEL_FLOAT_INLINE T operator()(float8_e4m3 v) const { \
            return (T)(BRIDGE)(v);                              \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<T, float8_e4m3> {                               \
        KERNEL_FLOAT_INLINE float8_e4m3 operator()(T v) const { \
            return float8_e4m3((BRIDGE)(v));                    \
        }                                                       \
    };                                                          \
    }

#define KERNEL_FLOAT_FP8_CAST(T) KERNEL_FLOAT_FP8_CAST_VIA(T, T)

KERNEL_FLOAT_FP8_CAST(float)
KERNEL_FLOAT_FP8_CAST(double)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_FP8_CAST(__half)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_FP8_CAST(__nv_bfloat16)
#endif

KERNEL_FLOAT_FP8_CAST(bool)
KERNEL_FLOAT_FP8_CAST_VIA(char, int)

KERNEL_FLOAT_FP8_CAST(signed char)
KERNEL_FLOAT_FP8_CAST(short)
KERNEL_FLOAT_FP8_CAST(int)
KERNEL_FLOAT_FP8_CAST_VIA(long, long long)
KERNEL_FLOAT_FP8_CAST(long long)

KERNEL_FLOAT_FP8_CAST(unsigned char)
KERNEL_FLOAT_FP8_CAST(unsigned short)
KERNEL_FLOAT_FP8_CAST(unsigned int)
KERNEL_FLOAT_FP8_CAST_VIA(unsigned long, unsigned long long)
KERNEL_FLOAT_FP8_CAST(unsigned long long)

#define KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(T, N)                   \
    namespace detail {                                            \
    template<>                                                    \
    struct map_helper<                                            \
        ops::cast<T, float8_e4m3>,                                \
        vector_storage<float8_e4m3, N>,                           \
        vector_storage<T, N>> {                                   \
        KERNEL_FLOAT_INLINE static vector_storage<float8_e4m3, N> \
        call(ops::cast<T, float8_e4m3>, T##N input) {             \
            return __nv_fp8x##N##_e4m3(input);                    \
        }                                                         \
    };                                                            \
    template<>                                                    \
    struct map_helper<                                            \
        ops::cast<T, float8_e5m2>,                                \
        vector_storage<float8_e5m2, N>,                           \
        vector_storage<T, N>> {                                   \
        KERNEL_FLOAT_INLINE static vector_storage<float8_e5m2, N> \
        call(ops::cast<T, float8_e5m2>, T##N input) {             \
            return __nv_fp8x##N##_e5m2(input);                    \
        }                                                         \
    };                                                            \
    }

#define KERNEL_FLOAT_FP8_CAST_VECTOR_TO(T, N)                        \
    namespace detail {                                               \
    template<>                                                       \
    struct map_helper<                                               \
        ops::cast<float8_e4m3, T>,                                   \
        vector_storage<T, N>,                                        \
        vector_storage<float8_e4m3, N>> {                            \
        KERNEL_FLOAT_INLINE static vector_storage<T, N>              \
        call(ops::cast<float8_e4m3, T>, __nv_fp8x##N##_e4m3 input) { \
            return (T##N)(input);                                    \
        }                                                            \
    };                                                               \
    template<>                                                       \
    struct map_helper<                                               \
        ops::cast<float8_e5m2, T>,                                   \
        vector_storage<T, N>,                                        \
        vector_storage<float8_e5m2, N>> {                            \
        KERNEL_FLOAT_INLINE static vector_storage<T, N>              \
        call(ops::cast<float8_e5m2, T>, __nv_fp8x##N##_e5m2 input) { \
            return (T##N)(input);                                    \
        }                                                            \
    };                                                               \
    }

// TODO: Some of these casts don't seem to work? Figure out why!
#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(__half, 2)
KERNEL_FLOAT_FP8_CAST_VECTOR_TO(__half, 2)
//    KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(__half, 4)
//    KERNEL_FLOAT_FP8_CAST_VECTOR_TO(__half, 4)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(__nv_bfloat16, 2)
//    KERNEL_FLOAT_FP8_CAST_VECTOR_TO(__nv_bfloat16, 2)
//    KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(__nv_bfloat16, 4)
//    KERNEL_FLOAT_FP8_CAST_VECTOR_TO(__nv_bfloat16, 4)
#endif

KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(float, 2)
KERNEL_FLOAT_FP8_CAST_VECTOR_TO(float, 2)
KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(float, 4)
KERNEL_FLOAT_FP8_CAST_VECTOR_TO(float, 4)

KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(double, 2)
KERNEL_FLOAT_FP8_CAST_VECTOR_TO(double, 2)
KERNEL_FLOAT_FP8_CAST_VECTOR_FROM(double, 4)
KERNEL_FLOAT_FP8_CAST_VECTOR_TO(double, 4)
}  // namespace kernel_float

#endif
#endif  //KERNEL_FLOAT_FP8_H
