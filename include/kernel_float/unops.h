#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "apply.h"

namespace kernel_float {

enum struct RoundingMode { ANY, DOWN, UP, NEAREST, TOWARD_ZERO };

namespace ops {

template<typename T, typename R, RoundingMode m = RoundingMode::ANY, typename = void>
struct cast;

template<typename T, RoundingMode m>
struct cast<T, T, m> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};

template<typename T>
struct cast<T, T, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};

template<typename T, typename R, typename = void>
struct cast_float_fallback;

template<typename T, typename R, typename>
struct cast_float_fallback {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

// clang-format off
template<typename T, typename R>
struct cast_float_fallback<
    T,
    R,
    enable_if_t<
        !is_same_type<T, float> &&
        !is_same_type<R, float> &&
        (detail::allow_float_fallback<T>::value || detail::allow_float_fallback<R>::value)
    >
> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return cast<float, R> {}(cast<T, float> {}(input));
    }
};
// clang-format on

template<typename T, typename R>
struct cast<T, R, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return cast_float_fallback<T, R> {}(input);
    }
};

}  // namespace ops

/**
 * Cast the elements of the given vector `input` to a different type `R`.
 *
 * This function casts each element of the input vector to a different data type specified by
 * template parameter `R`.
 *
 * Optionally, the rounding mode can be set using the `Mode` template parameter. The default mode is `ANY`, which
 * uses the fastest rounding mode available.
 *
 * Example
 * =======
 * ```
 * vec<float, 4> input {1.2f, 2.7f, 3.5f, 4.9f};
 * auto casted = cast<int>(input); // [1, 2, 3, 4]
 * ```
 */
template<typename R, RoundingMode Mode = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, vector_extent_type<V>> cast(const V& input) {
    using F = ops::cast<vector_value_type<V>, R, Mode>;
    return map(F {}, input);
}

#define KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)                                                        \
    template<typename Accuracy = default_policy, typename V>                                       \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> NAME(const V& input) { \
        using F = ops::NAME<vector_value_type<V>>;                                                 \
        return ::kernel_float::map<Accuracy>(F {}, input);                                         \
    }

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)       \
    namespace ops {                                 \
    template<typename T>                            \
    struct NAME {                                   \
        KERNEL_FLOAT_INLINE T operator()(T input) { \
            return T(EXPR);                         \
        }                                           \
    };                                              \
    }                                               \
                                                    \
    KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                           \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                      \
                                                                               \
    template<typename T, typename E, typename S>                               \
    KERNEL_FLOAT_INLINE vector<T, E> operator OP(const vector<T, E, S>& vec) { \
        return NAME(vec);                                                      \
    }

KERNEL_FLOAT_DEFINE_UNARY_OP(negate, -, -input)
KERNEL_FLOAT_DEFINE_UNARY_OP(bit_not, ~, ~input)
KERNEL_FLOAT_DEFINE_UNARY_OP(logical_not, !, (ops::cast<bool, T> {}(!ops::cast<T, bool> {}(input))))

#define KERNEL_FLOAT_DEFINE_UNARY_STRUCT(NAME, EXPR_F64, EXPR_F32)        \
    namespace ops {                                                       \
    template<typename T, typename = void>                                 \
    struct NAME;                                                          \
                                                                          \
    template<typename T>                                                  \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> { \
        KERNEL_FLOAT_INLINE T operator()(T input_arg) {                   \
            float input = float(input_arg);                               \
            return T(EXPR_F32);                                           \
        }                                                                 \
    };                                                                    \
                                                                          \
    template<>                                                            \
    struct NAME<double> {                                                 \
        KERNEL_FLOAT_INLINE double operator()(double input) {             \
            return double(EXPR_F64);                                      \
        }                                                                 \
    };                                                                    \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_MATH(NAME)                             \
    KERNEL_FLOAT_DEFINE_UNARY_STRUCT(NAME, ::NAME(input), ::NAME(input)) \
    KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)

KERNEL_FLOAT_DEFINE_UNARY_MATH(acos)
KERNEL_FLOAT_DEFINE_UNARY_MATH(abs)
KERNEL_FLOAT_DEFINE_UNARY_MATH(acosh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(asin)
KERNEL_FLOAT_DEFINE_UNARY_MATH(asinh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(atan)
KERNEL_FLOAT_DEFINE_UNARY_MATH(atanh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cbrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(ceil)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cos)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cosh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfc)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfcinv)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfcx)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfinv)
KERNEL_FLOAT_DEFINE_UNARY_MATH(exp)
KERNEL_FLOAT_DEFINE_UNARY_MATH(exp10)
KERNEL_FLOAT_DEFINE_UNARY_MATH(exp2)
KERNEL_FLOAT_DEFINE_UNARY_MATH(expm1)
KERNEL_FLOAT_DEFINE_UNARY_MATH(fabs)
KERNEL_FLOAT_DEFINE_UNARY_MATH(floor)
KERNEL_FLOAT_DEFINE_UNARY_MATH(ilogb)
KERNEL_FLOAT_DEFINE_UNARY_MATH(lgamma)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log10)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log2)
KERNEL_FLOAT_DEFINE_UNARY_MATH(nearbyint)
KERNEL_FLOAT_DEFINE_UNARY_MATH(normcdf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rcbrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(sin)
KERNEL_FLOAT_DEFINE_UNARY_MATH(sinh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tan)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tanh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tgamma)
KERNEL_FLOAT_DEFINE_UNARY_MATH(trunc)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rint)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(round)
KERNEL_FLOAT_DEFINE_UNARY_MATH(signbit)
KERNEL_FLOAT_DEFINE_UNARY_MATH(isinf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(isnan)

// CUDA offers special reciprocal functions (rcp), but only on the device.
#if KERNEL_FLOAT_IS_DEVICE
KERNEL_FLOAT_DEFINE_UNARY_STRUCT(rcp, __drcp_rn(input), __frcp_rn(input))
#else
KERNEL_FLOAT_DEFINE_UNARY_STRUCT(rcp, 1.0 / input, 1.0f / input)
#endif

KERNEL_FLOAT_DEFINE_UNARY_FUN(rcp)

#define KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(NAME)                                            \
    template<typename V>                                                                    \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> fast_##NAME(    \
        const V& input) {                                                                   \
        return ::kernel_float::map<fast_policy>(ops::NAME<vector_value_type<V>> {}, input); \
    }

KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(exp)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(log)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(rcp)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(sin)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(cos)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(tan)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(exp2)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(log2)

#if KERNEL_FLOAT_IS_DEVICE

#define KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(T, F, FAST_FUN)                       \
    namespace detail {                                                                \
    template<>                                                                        \
    struct apply_fastmath_impl<ops::F<T>, 1, T, T> {                                  \
        KERNEL_FLOAT_INLINE static void call(ops::F<T>, T* result, const T* inputs) { \
            *result = FAST_FUN(*inputs);                                              \
        }                                                                             \
    };                                                                                \
    }

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, exp, __expf)
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, log, __logf)

#define KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(T, F, INSTR, REG)                         \
    namespace detail {                                                                    \
    template<>                                                                            \
    struct apply_fastmath_impl<ops::F<T>, 1, T, T> {                                      \
        KERNEL_FLOAT_INLINE static void call(ops::F<T> fun, T* result, const T* inputs) { \
            asm(INSTR : "=" REG(*result) : REG(*inputs));                                 \
        }                                                                                 \
    };                                                                                    \
    }

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(double, rcp, "rcp.approx.ftz.f64 %0, %1;", "d")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(double, rsqrt, "rsqrt.approx.f64 %0, %1;", "d")

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, sqrt, "sqrt.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, rcp, "rcp.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, rsqrt, "rsqrt.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, sin, "sin.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, cos, "cos.approx.f32 %0, %1;", "f")

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, exp2, "ex2.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, log2, "lg2.approx.f32 %0, %1;", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, tanh, "tanh.approx.f32 %0, %1;", "f")

#endif

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
