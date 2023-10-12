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

KERNEL_FLOAT_FP8_CAST(double)
}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE
#include "fp16.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(__half, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(__half, __nv_fp8_e5m2)
KERNEL_FLOAT_FP8_CAST(__half)
}  // namespace kernel_float
#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#if KERNEL_FLOAT_BF16_AVAILABLE
#include "bf16.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(__nv_bfloat16, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(__nv_bfloat16, __nv_fp8_e5m2)
KERNEL_FLOAT_FP8_CAST(__nv_bfloat16)
}  // namespace kernel_float
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#endif  // KERNEL_FLOAT_FP8_AVAILABLE
#endif  // KERNEL_FLOAT_FP8_H
