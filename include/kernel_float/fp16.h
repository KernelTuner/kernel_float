#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include <cuda_fp16.h>

#include "all.h"

namespace kernel_float {
using float16 = __half;
using float16x2 = __half2;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float16)

namespace detail {
template<>
struct vec_storage<float16, 2> {
    static_assert(sizeof(float16) * 2 == sizeof(float16x2), "invalid size");
    static_assert(alignof(float16) <= alignof(float16x2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(float16 x, float16 y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(float16x2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator float16x2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE float16 get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE float16 get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, float16 v) {
        *this = vec_storage(v, get(I1 {}));
    }

    KERNEL_FLOAT_INLINE void set(I1, float16 v) {
        *this = vec_storage(get(I0 {}), v);
    }

    KERNEL_FLOAT_INLINE float16 get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(float16, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<0, 0>) const {
        return __low2half2(vector_);
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<1, 1>) const {
        return __high2half2(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, float16x2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, float16x2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    float16x2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_FP16_MONOP(NAME, FUN1, FUN2)               \
    namespace ops {                                             \
    template<>                                                  \
    struct NAME<float16> {                                      \
        KERNEL_FLOAT_INLINE float16 operator()(float16 input) { \
            return FUN1(input);                                 \
        }                                                       \
    };                                                          \
    }                                                           \
    template<>                                                  \
    struct map_helper<ops::NAME<float16>, float16, 2> {         \
        KERNEL_FLOAT_INLINE static vec<float16, 2>              \
        call(ops::NAME<float16>, float16x2 input) noexcept {    \
            return FUN2(input);                                 \
        }                                                       \
    };

KERNEL_FLOAT_FP16_MONOP(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_FP16_MONOP(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_FP16_MONOP(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_FP16_MONOP(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_FP16_MONOP(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_FP16_MONOP(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_FP16_MONOP(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_FP16_MONOP(log, ::hlog, ::h2log);
KERNEL_FLOAT_FP16_MONOP(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_FP16_MONOP(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_FP16_MONOP(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_FP16_MONOP(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_FP16_MONOP(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_FP16_MONOP(trunc, ::htrunc, ::h2trunc);
//    KERNEL_FLOAT_FP16_MONOP(rcp, hrcp, h2rcp);

#define KERNEL_FLOAT_FP16_BINOP(NAME, FUN1, FUN2)                          \
    namespace ops {                                                        \
    template<>                                                             \
    struct NAME<float16> {                                                 \
        KERNEL_FLOAT_INLINE float16 operator()(float16 lhs, float16 rhs) { \
            return FUN1(lhs, rhs);                                         \
        }                                                                  \
    };                                                                     \
    }                                                                      \
    template<>                                                             \
    struct zip_helper<ops::NAME<float16>, float16, float16, 2> {           \
        KERNEL_FLOAT_INLINE static float16x2                               \
        call(ops::NAME<float16>, float16x2 lhs, float16x2 rhs) {           \
            return FUN2(lhs, rhs);                                         \
        }                                                                  \
    };

KERNEL_FLOAT_FP16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_FP16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_FP16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_FP16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_FP16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_FP16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_FP16_RELOP(NAME, FUN1, FUN2)                       \
    namespace ops {                                                     \
    template<>                                                          \
    struct NAME<float16> {                                              \
        KERNEL_FLOAT_INLINE bool operator()(float16 lhs, float16 rhs) { \
            return FUN1(lhs, rhs);                                      \
        }                                                               \
    };                                                                  \
    }

KERNEL_FLOAT_FP16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_FP16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_FP16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_FP16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_FP16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_FP16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)     \
    namespace ops {                                       \
    template<>                                            \
    struct cast<T, float16> {                             \
        KERNEL_FLOAT_INLINE float16 operator()(T input) { \
            return TO_HALF;                               \
        }                                                 \
    };                                                    \
    template<>                                            \
    struct cast<float16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(float16 input) { \
            return FROM_HALF;                             \
        }                                                 \
    };                                                    \
    }

KERNEL_FLOAT_FP16_CAST(float64, __double2half(input), float64(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float32, __float2half(input), __half2float(input));

KERNEL_FLOAT_FP16_CAST(signed int, __half2int_rz(input), __int2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed short, __half2short_rz(input), __short2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned int, __half2uint_rz(input), __uint2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned short, __half2ushort_rz(input), __ushort2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

template<>
struct map_helper<ops::cast<float16, float32>, float16, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<float16, float32>, const vec<float16, 2>& input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, float16>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<float16, 2>
    call(ops::cast<float32, float16>, const vec<float32, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_FP16_H
