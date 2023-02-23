#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include <cuda_bf16.h>

#include "all.h"

namespace kernel_float {
using bfloat16 = __nv_bfloat16;
using bfloat16x2 = __nv_bfloat162;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, bfloat16)

namespace detail {
template<>
struct vec_storage<bfloat16, 2> {
    static_assert(sizeof(bfloat16) * 2 == sizeof(bfloat16x2), "invalid size");
    static_assert(alignof(bfloat16) <= alignof(bfloat16x2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(bfloat16 x, bfloat16 y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(bfloat16x2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator bfloat16x2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE bfloat16 get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE bfloat16 get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, bfloat16 v) {
        *this = vec_storage(v, __high2float(vector_));
    }

    KERNEL_FLOAT_INLINE void set(I1, bfloat16 v) {
        *this = vec_storage(__low2float(vector_), v);
    }

    KERNEL_FLOAT_INLINE bfloat16 get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, bfloat16 value) const {
        if (index == 0) {
            set(I0 {}, value);
        } else {
            set(I1 {}, value);
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(bfloat16, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<0, 0>) const {
        return {vector_.x, vector_.x};
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<1, 1>) const {
        return __high2bfloat162(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, bfloat16x2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, bfloat16x2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    bfloat16x2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_BF16_AVAILABLE && KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_BF16_MONOP(NAME, FUN1, FUN2)                 \
    namespace ops {                                               \
    template<>                                                    \
    struct NAME<bfloat16> {                                       \
        KERNEL_FLOAT_INLINE bfloat16 operator()(bfloat16 input) { \
            return FUN1(input);                                   \
        }                                                         \
    };                                                            \
    }                                                             \
    template<>                                                    \
    struct map_helper<ops::NAME<bfloat16>, bfloat16, 2> {         \
        KERNEL_FLOAT_INLINE static vec<bfloat16, 2>               \
        call(ops::NAME<bfloat16>, bfloat16x2 input) noexcept {    \
            return FUN2(input);                                   \
        }                                                         \
    };

KERNEL_FLOAT_BF16_MONOP(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_BF16_MONOP(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_BF16_MONOP(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_BF16_MONOP(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_BF16_MONOP(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_BF16_MONOP(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_BF16_MONOP(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_BF16_MONOP(log, ::hlog, ::h2log);
KERNEL_FLOAT_BF16_MONOP(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_BF16_MONOP(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_BF16_MONOP(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_BF16_MONOP(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_BF16_MONOP(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_BF16_MONOP(trunc, ::htrunc, ::h2trunc);
//    KERNEL_FLOAT_BF16_MONOP(rcp, hrcp, h2rcp);

#define KERNEL_FLOAT_BF16_BINOP(NAME, FUN1, FUN2)                             \
    namespace ops {                                                           \
    template<>                                                                \
    struct NAME<bfloat16> {                                                   \
        KERNEL_FLOAT_INLINE bfloat16 operator()(bfloat16 lhs, bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                            \
        }                                                                     \
    };                                                                        \
    }                                                                         \
    template<>                                                                \
    struct zip_helper<ops::NAME<bfloat16>, bfloat16, bfloat16, 2> {           \
        KERNEL_FLOAT_INLINE static bfloat16x2                                 \
        call(ops::NAME<bfloat16>, bfloat16x2 lhs, bfloat16x2 rhs) {           \
            return FUN2(lhs, rhs);                                            \
        }                                                                     \
    };

KERNEL_FLOAT_BF16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_BF16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_BF16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_BF16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_BF16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_BF16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_BF16_RELOP(NAME, FUN1, FUN2)                         \
    namespace ops {                                                       \
    template<>                                                            \
    struct NAME<bfloat16> {                                               \
        KERNEL_FLOAT_INLINE bool operator()(bfloat16 lhs, bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                        \
        }                                                                 \
    };                                                                    \
    }

KERNEL_FLOAT_BF16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_BF16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_BF16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_BF16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_BF16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_BF16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)      \
    namespace ops {                                        \
    template<>                                             \
    struct cast<T, bfloat16> {                             \
        KERNEL_FLOAT_INLINE bfloat16 operator()(T input) { \
            return TO_HALF;                                \
        }                                                  \
    };                                                     \
    template<>                                             \
    struct cast<bfloat16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(bfloat16 input) { \
            return FROM_HALF;                              \
        }                                                  \
    };                                                     \
    }

KERNEL_FLOAT_BF16_CAST(float64, __double2bfloat16(input), float64(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float32, __float2bfloat16(input), __bfloat162float(input));

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

template<>
struct map_helper<ops::cast<bfloat16, float32>, bfloat16, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<bfloat16, float32>, const vec<bfloat16, 2>& input) noexcept {
        return __bfloat1622float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, bfloat16>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<bfloat16, 2>
    call(ops::cast<float32, bfloat16>, const vec<float32, 2>& input) noexcept {
        return __float22bfloat162_rn(input);
    }
};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_BF16_H
