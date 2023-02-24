#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include "macros.h"

#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>

#include "all.h"

namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, __half)

namespace detail {
template<>
struct vec_storage<__half, 2> {
    static_assert(sizeof(__half) * 2 == sizeof(__half2), "invalid size");
    static_assert(alignof(__half) <= alignof(__half2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(__half x, __half y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(__half2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __half2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __half get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __half get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, __half v) {
        *this = vec_storage(v, get(I1 {}));
    }

    KERNEL_FLOAT_INLINE void set(I1, __half v) {
        *this = vec_storage(get(I0 {}), v);
    }

    KERNEL_FLOAT_INLINE __half get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __half value) const {
        if (index == 0) {
            set(I0 {}, value);
        } else {
            set(I1 {}, value);
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(__half, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<0, 0>) const {
        return __low2half2(vector_);
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<1, 1>) const {
        return __high2half2(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, __half2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, __half2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    __half2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_FP16_MONOP(NAME, FUN1, FUN2)             \
    namespace ops {                                           \
    template<>                                                \
    struct NAME<__half> {                                     \
        KERNEL_FLOAT_INLINE __half operator()(__half input) { \
            return FUN1(input);                               \
        }                                                     \
    };                                                        \
    }                                                         \
    template<>                                                \
    struct map_helper<ops::NAME<__half>, __half, 2> {         \
        KERNEL_FLOAT_INLINE static vec<__half, 2>             \
        call(ops::NAME<__half>, __half2 input) noexcept {     \
            return FUN2(input);                               \
        }                                                     \
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

#define KERNEL_FLOAT_FP16_BINOP(NAME, FUN1, FUN2)                                              \
    namespace ops {                                                                            \
    template<>                                                                                 \
    struct NAME<__half> {                                                                      \
        KERNEL_FLOAT_INLINE __half operator()(__half lhs, __half rhs) {                        \
            return FUN1(lhs, rhs);                                                             \
        }                                                                                      \
    };                                                                                         \
    }                                                                                          \
    template<>                                                                                 \
    struct zip_helper<ops::NAME<__half>, __half, __half, 2> {                                  \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 lhs, __half2 rhs) { \
            return FUN2(lhs, rhs);                                                             \
        }                                                                                      \
    };

KERNEL_FLOAT_FP16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_FP16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_FP16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_FP16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_FP16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_FP16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_FP16_RELOP(NAME, FUN1, FUN2)                     \
    namespace ops {                                                   \
    template<>                                                        \
    struct NAME<__half> {                                             \
        KERNEL_FLOAT_INLINE bool operator()(__half lhs, __half rhs) { \
            return FUN1(lhs, rhs);                                    \
        }                                                             \
    };                                                                \
    }

KERNEL_FLOAT_FP16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_FP16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_FP16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_FP16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_FP16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_FP16_RELOP(less_equal, __hle, __hle2);
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

KERNEL_FLOAT_FP16_CAST(float64, __double2half(input), float64(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float32, __float2half(input), __half2float(input));

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

template<>
struct map_helper<ops::cast<__half, float32>, __half, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<__half, float32>, const vec<__half, 2>& input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, __half>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<__half, 2>
    call(ops::cast<float32, __half>, const vec<float32, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

KERNEL_FLOAT_INTO_VEC(__half, __half, 1)
KERNEL_FLOAT_INTO_VEC(__half2, __half, 2)

}  // namespace kernel_float

#endif
#endif  //KERNEL_FLOAT_FP16_H
