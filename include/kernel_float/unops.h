#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "base.h"

namespace kernel_float {
namespace detail {
template<typename F, size_t N, typename Output, typename Input, typename = void>
struct map_helper {
    KERNEL_FLOAT_INLINE static tensor_storage<Output, N>
    call(F fun, const tensor_storage<Input, N>& input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static tensor_storage<Output, N>
    call(F fun, const tensor_storage<Input, N>& input, index_sequence<Is...>) {
        return {fun(input[Is])...};
    }
};
}  // namespace detail

template<typename F, typename V>
using map_type = tensor<result_t<F, tensor_value_type<V>>, tensor_extents<V>>;

template<typename F, typename V>
KERNEL_FLOAT_INLINE map_type<F, V> map(F fun, const V& input) {
    using Input = tensor_value_type<V>;
    using Output = result_t<F, Input>;
    return detail::map_helper<F, tensor_volume<V>, Output, Input>::call(
        fun,
        into_tensor(input).storage());
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                      \
    namespace ops {                                                \
    template<typename T>                                           \
    struct NAME {                                                  \
        KERNEL_FLOAT_INLINE T operator()(T input) {                \
            return T(EXPR);                                        \
        }                                                          \
    };                                                             \
    }                                                              \
    template<typename V>                                           \
    KERNEL_FLOAT_INLINE into_tensor_type<V> NAME(const V& input) { \
        using F = ops::NAME<tensor_value_type<V>>;                 \
        return map(F {}, input);                                   \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                        \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                   \
    template<typename T, typename D>                                        \
    KERNEL_FLOAT_INLINE tensor<T, D> operator OP(const tensor<T, D>& vec) { \
        return NAME(vec);                                                   \
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

enum struct RoundingMode { ANY, DOWN, UP, NEAREST, TOWARD_ZERO };

namespace ops {
template<typename T, typename R, RoundingMode m = RoundingMode::ANY, typename = void>
struct cast;

template<typename T, typename R>
struct cast<T, R, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

template<typename T, RoundingMode m>
struct cast<T, T, m> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};
}  // namespace ops

namespace detail {
template<size_t N, typename T>
struct map_helper<ops::cast<T, T>, N, T, T> {
    KERNEL_FLOAT_INLINE static tensor_storage<T, N>
    call(ops::cast<T, T> fun, const tensor_storage<T, N>& input) {
        return input;
    }
};
}  // namespace detail

template<typename R, RoundingMode Mode = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE tensor<R, tensor_extents<V>> cast(const V& input) {
    using F = ops::cast<tensor_value_type<V>, R, Mode>;
    return map(F {}, input);
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H