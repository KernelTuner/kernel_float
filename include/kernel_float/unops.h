#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "base.h"

namespace kernel_float {
namespace detail {

template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl {
    KERNEL_FLOAT_INLINE static vector_storage<Output, N>
    call(F fun, const vector_storage<Args, N>&... inputs) {
        vector_storage<Output, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = fun(inputs.data()[i]...);
        }

        return result;
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
    return detail::apply_impl<F, vector_extent<V>, Output, Input>::call(
        fun,
        into_vector_storage(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                                      \
    namespace ops {                                                                                \
    template<typename T>                                                                           \
    struct NAME {                                                                                  \
        KERNEL_FLOAT_INLINE T operator()(T input) {                                                \
            return T(EXPR);                                                                        \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    template<typename V>                                                                           \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> NAME(const V& input) { \
        using F = ops::NAME<vector_value_type<V>>;                                                 \
        return map(F {}, input);                                                                   \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                           \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                      \
    template<typename T, typename E, typename S>                               \
    KERNEL_FLOAT_INLINE vector<T, E> operator OP(const vector<T, E, S>& vec) { \
        return NAME(vec);                                                      \
    }

KERNEL_FLOAT_DEFINE_UNARY_OP(negate, -, -input)
KERNEL_FLOAT_DEFINE_UNARY_OP(bit_not, ~, ~input)
KERNEL_FLOAT_DEFINE_UNARY_OP(logical_not, !, !bool(input))

#define KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME) KERNEL_FLOAT_DEFINE_UNARY(NAME, ::NAME(input))

KERNEL_FLOAT_DEFINE_UNARY_FUN(acos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(abs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(acosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ceil)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cospi)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcx)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp2)
KERNEL_FLOAT_DEFINE_UNARY_FUN(expm1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(fabs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(floor)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ilogb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(lgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(logb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(nearbyint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(normcdf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rcbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(trunc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y0)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(yn)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(round)
KERNEL_FLOAT_DEFINE_UNARY_FUN(signbit)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isinf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isnan)

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
