#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "cast.h"
#include "storage.h"

namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Input, typename = void>
struct map_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input) {
        return call(fun, input, make_index_sequence<vector_size<Input>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input, index_sequence<Is...>) {
        return vector_traits<Output>::create(fun(vector_get<Is>(input))...);
    }
};

template<typename F, typename V, size_t N>
struct map_helper<F, nested_array<V, N>, nested_array<V, N>> {
    KERNEL_FLOAT_INLINE static nested_array<V, N> call(F fun, const nested_array<V, N>& input) {
        return call(fun, input, make_index_sequence<nested_array<V, N>::num_packets> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static nested_array<V, N>
    call(F fun, const nested_array<V, N>& input, index_sequence<Is...>) {
        return {map_helper<F, V, V>::call(fun, input[Is])...};
    }
};
}  // namespace detail

template<typename F, typename Input>
using map_type = default_storage_type<result_t<F, vector_value_type<Input>>, vector_size<Input>>;

/**
 * Applies ``fun`` to each element from vector ``input`` and returns a new vector with the results.
 * This function is the basis for all unary operators like ``sin`` and ``sqrt``.
 *
 * Example
 * =======
 * ```
 * vector<int, 3> v = {1, 2, 3};
 * vector<int, 3> w = map([](auto i) { return i * 2; }); // 2, 4, 6
 * ```
 */
template<typename F, typename Input, typename Output = map_type<F, Input>>
KERNEL_FLOAT_INLINE Output map(F fun, const Input& input) {
    return detail::map_helper<F, Output, into_storage_type<Input>>::call(fun, into_storage(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                            \
    namespace ops {                                                                      \
    template<typename T>                                                                 \
    struct NAME {                                                                        \
        KERNEL_FLOAT_INLINE T operator()(T input) {                                      \
            return T(EXPR);                                                              \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    template<typename V>                                                                 \
    KERNEL_FLOAT_INLINE vector<into_storage_type<V>> NAME(const V& input) {              \
        return map<ops::NAME<vector_value_type<V>>, V, into_storage_type<V>>({}, input); \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                  \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                             \
    template<typename V>                                              \
    KERNEL_FLOAT_INLINE vector<V> operator OP(const vector<V>& vec) { \
        return NAME(vec);                                             \
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

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
