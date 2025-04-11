#ifndef KERNEL_FLOAT_FP8_H
#define KERNEL_FLOAT_FP8_H

#include "macros.h"

#if KERNEL_FLOAT_FP8_AVAILABLE
#include <cuda_fp8.h>

#include "vector.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_fp8_e4m3)

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_fp8_e5m2)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_fp8_e5m2)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_fp8_e5m2)

namespace detail {
template<>
struct allow_float_fallback<__nv_fp8_e4m3> {
    static constexpr bool value = true;
};

template<>
struct allow_float_fallback<__nv_fp8_e5m2> {
    static constexpr bool value = true;
};
}  // namespace detail
}  // namespace kernel_float

#define KERNEL_FLOAT_FP8_CAST(T)                                  \
    namespace ops {                                               \
    template<>                                                    \
    struct cast<T, __nv_fp8_e4m3> {                               \
        KERNEL_FLOAT_INLINE __nv_fp8_e4m3 operator()(T v) const { \
            return __nv_fp8_e4m3(v);                              \
        }                                                         \
    };                                                            \
                                                                  \
    template<>                                                    \
    struct cast<__nv_fp8_e4m3, T> {                               \
        KERNEL_FLOAT_INLINE T operator()(__nv_fp8_e4m3 v) const { \
            return T(v);                                          \
        }                                                         \
    };                                                            \
                                                                  \
    template<>                                                    \
    struct cast<T, __nv_fp8_e5m2> {                               \
        KERNEL_FLOAT_INLINE __nv_fp8_e5m2 operator()(T v) const { \
            return __nv_fp8_e5m2(v);                              \
        }                                                         \
    };                                                            \
                                                                  \
    template<>                                                    \
    struct cast<__nv_fp8_e5m2, T> {                               \
        KERNEL_FLOAT_INLINE T operator()(__nv_fp8_e5m2 v) const { \
            return T(v);                                          \
        }                                                         \
    };                                                            \
    }

#define KERNEL_FLOAT_FP8_CAST2(T, FP8_TY, FP8_INTERP)                                            \
    namespace detail {                                                                           \
    template<>                                                                                   \
    struct apply_impl<accurate_policy, ops::cast<T, FP8_TY>, 2, FP8_TY, T> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::cast<T, FP8_TY>, FP8_TY* result, const T* v) { \
            __half2_raw x;                                                                       \
            memcpy(&x, v, 2 * sizeof(T));                                                        \
            __nv_fp8x2_storage_t y = __nv_cvt_halfraw2_to_fp8x2(x, __NV_NOSAT, FP8_INTERP);      \
            memcpy(result, &y, 2 * sizeof(FP8_TY));                                              \
        }                                                                                        \
    };                                                                                           \
    template<>                                                                                   \
    struct apply_impl<accurate_policy, ops::cast<FP8_TY, T>, 2, T, FP8_TY> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::cast<FP8_TY, T>, T* result, const FP8_TY* v) { \
            __nv_fp8x2_storage_t x;                                                              \
            memcpy(&x, v, 2 * sizeof(FP8_TY));                                                   \
            __half2_raw y = __nv_cvt_fp8x2_to_halfraw2(x, FP8_INTERP);                           \
            memcpy(result, &y, 2 * sizeof(T));                                                   \
        }                                                                                        \
    };                                                                                           \
    }

namespace kernel_float {
KERNEL_FLOAT_FP8_CAST(double)
}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half_t, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half_t, __nv_fp8_e5m2)

KERNEL_FLOAT_FP8_CAST(half_t)
KERNEL_FLOAT_FP8_CAST2(half_t, __nv_fp8_e4m3, __NV_E4M3)
KERNEL_FLOAT_FP8_CAST2(half_t, __nv_fp8_e5m2, __NV_E5M2)

}  // namespace kernel_float
#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#if KERNEL_FLOAT_BF16_AVAILABLE
#include "bf16.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(bfloat16_t, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(bfloat16_t, __nv_fp8_e5m2)

KERNEL_FLOAT_FP8_CAST(bfloat16_t)
KERNEL_FLOAT_FP8_CAST2(bfloat16_t, __nv_fp8_e4m3, __NV_E4M3)
KERNEL_FLOAT_FP8_CAST2(bfloat16_t, __nv_fp8_e5m2, __NV_E5M2)
}  // namespace kernel_float
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#endif  // KERNEL_FLOAT_FP8_AVAILABLE
#endif  // KERNEL_FLOAT_FP8_H
