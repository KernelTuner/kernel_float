#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "base.h"

namespace kernel_float {
namespace detail {

template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* result, const Args*... inputs) {
#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result[i] = fun(inputs[i]...);
        }
    }
};
}  // namespace detail

template<typename F, typename V>
using map_type = vector<result_t<F, vector_value_type<V>>, vector_extent_type<V>>;

/**
 * Apply the function `F` to each element from the vector `input` and return the results as a new vector.
 *
 * Examples
 * ========
 * ```
 * vec<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * vec<float, 4> squared = map([](auto x) { return x * x; }, input); // [1.0f, 4.0f, 9.0f, 16.0f]
 * ```
 */
template<typename F, typename V>
KERNEL_FLOAT_INLINE map_type<F, V> map(F fun, const V& input) {
    using Input = vector_value_type<V>;
    using Output = result_t<F, Input>;
    vector_storage<Output, vector_extent<V>> result;

    detail::apply_impl<F, vector_extent<V>, Output, Input>::call(
        fun,
        result.data(),
        into_vector_storage(input).data());

    return result;
}

namespace detail {
// Indicates that elements of type `T` offer less precision than floats, thus operations
// on elements of type `T` can be performed by upcasting them to ` float`.
template<typename T>
struct allow_float_fallback {
    static constexpr bool value = false;
};

template<>
struct allow_float_fallback<float> {
    static constexpr bool value = true;
};
}  // namespace detail

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

template<typename T, typename R>
struct cast<T, R, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return cast_float_fallback<T, R> {}(input);
    }
};

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
    template<typename V>                                                                           \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> NAME(const V& input) { \
        using F = ops::NAME<vector_value_type<V>>;                                                 \
        return map(F {}, input);                                                                   \
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

#define KERNEL_FLOAT_DEFINE_UNARY_MATH(NAME)                              \
    namespace ops {                                                       \
    template<typename T, typename = void>                                 \
    struct NAME;                                                          \
                                                                          \
    template<typename T>                                                  \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> { \
        KERNEL_FLOAT_INLINE T operator()(T input) {                       \
            return T(::NAME(float(input)));                               \
        }                                                                 \
    };                                                                    \
                                                                          \
    template<>                                                            \
    struct NAME<double> {                                                 \
        KERNEL_FLOAT_INLINE double operator()(double input) {             \
            return double(::NAME(input));                                 \
        }                                                                 \
    };                                                                    \
    }                                                                     \
                                                                          \
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
KERNEL_FLOAT_DEFINE_UNARY_MATH(cospi)
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
KERNEL_FLOAT_DEFINE_UNARY_MATH(logb)
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

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_DEFINE_UNARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_UNARY(FUN_NAME, ops::OP_NAME<T> {}(input))   \
    namespace ops {                                                  \
    template<>                                                       \
    struct OP_NAME<float> {                                          \
        KERNEL_FLOAT_INLINE float operator()(float input) {          \
            return FLOAT_FUN(input);                                 \
        }                                                            \
    };                                                               \
    }
#else
#define KERNEL_FLOAT_DEFINE_UNARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_UNARY(FUN_NAME, ops::OP_NAME<T> {}(input))
#endif

KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_exp, exp, __expf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_log, log, __logf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_cos, cos, __cosf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_sin, sin, __sinf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_tan, tan, __tanf)

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
